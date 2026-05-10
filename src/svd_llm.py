import torch
import re
import math
import gc
from typing import Dict, Optional, List, Union, Literal
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from torch.utils.data import DataLoader
from .utils import *

class LowRank(torch.nn.Module):
    def __init__(self, in_features, out_features, rank, bias):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W_v = nn.Linear(in_features, rank, bias=False)
        self.W_u = nn.Linear(rank, out_features, bias=bias)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output =  self.W_u(self.W_v(input))
        return output

def get_whitening_matrices(
        model: Qwen2ForCausalLM,
        model_name: str,
        loader: DataLoader,
        layers_str: List[str],
        layers_list: List,
        attributes: List[str],
        n_tokens: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        is_v2: bool = False,
        save_path: str = "./tmp"
):
    """
    Computes whitening matrices layer by layer and saves each one to disk
    immediately after computation. Returns Dict[layer_str -> file_path].
    Peak RAM usage is bounded by a single decoder block's XXT matrices.
    """
    version_str = "v2" if is_v2 else "v1"
    wm_dir = os.path.join(save_path, "whitening_matrices")
    wm_dir = os.path.join(wm_dir, model_name.replace("/", "_").replace("-", "_"))
    wm_dir = os.path.join(wm_dir, version_str)
    os.makedirs(wm_dir, exist_ok=True)
    print(f"[WHITENING] Streaming whitening matrices to: {wm_dir}")

    # Group weight matrices by layer index (e.g., "layers.5")
    decoder_groups: Dict[str, List] = defaultdict(list)
    for layer_obj, attr, lstr in zip(layers_list, attributes, layers_str):
        match = re.search(r'\.layers\.(\d+)\.', lstr)
        idx = int(match.group(1)) if match else -1
        decoder_groups[f"layer_{idx}"].append((layer_obj, attr, lstr))

    decoder_layers = model.model.layers
    num_decoder_layers = len(decoder_layers)

    # PHASE 1
    # Move only the embedding (+ optional global components) to GPU,
    # wrap the first layer (layer_0) in a Catcher to intercept its inputs, then run
    # the full calibration set through it.
    # Each forward pass aborts immediately after the embedding so only one layer is ever on GPU.

    model.model.embed_tokens = model.model.embed_tokens.to(device)
    # Qwen2.5 (and some other models) have a global rotary embedding
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb = model.model.rotary_emb.to(device)

    # Each entry is stored in the following format: {"inp": Tensor[B, S, H], "attention_mask": ..., "position_ids": ...}
    captured: List[Dict] = []

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def __getattr__(self, name: str):
            # Proxy any attribute not found on Catcher itself to the wrapped module.
            # This handles cases like `decoder_layer.attention_type` that transformers
            # reads directly on the layer object before calling forward().
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

        def forward(self, inp, **kwargs):
            pe = kwargs.get("position_embeddings", None)
            captured.append({
                "inp": inp.detach().cpu(),
                "attention_mask": kwargs.get("attention_mask",  None) ,
                "position_ids": kwargs.get("position_ids",    None),
                "position_embeddings": (pe[0].cpu(), pe[1].cpu()) if pe is not None else None,
            })
            # Move masks to CPU immediately so GPU memory is freed
            entry = captured[-1]
            if entry["attention_mask"] is not None:
                entry["attention_mask"] = entry["attention_mask"].cpu()
            if entry["position_ids"] is not None:
                entry["position_ids"] = entry["position_ids"].cpu()
            raise ValueError   # abort the forward pass early

    decoder_layers[0] = decoder_layers[0].to(device)
    original_layer0 = decoder_layers[0]
    decoder_layers[0] = Catcher(original_layer0)

    print("[WHITENING] Capturing layer_0 inputs...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Capturing layer_0 inputs"):
            try:
                batch = {k: v.to(device) for k, v in batch.items()
                         if k in ("input_ids", "attention_mask")}
                model(**batch)
            except ValueError:
                pass

    # Restore layer_0 and move everything back to CPU
    decoder_layers[0] = original_layer0
    decoder_layers[0] = decoder_layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()
    print(f"[WHITENING] Captured layer_0 inputs for {len(captured)} batches.")

    # PHASE 2
    # Process decoder blocks one at a time.
    # `inps` threads the activations from one block to the next.

    # Hook definition (accumulates XXT on CPU)
    def hook(module, input, output):
        inp = input[0].detach().to(dtype=torch.float32)
        act = torch.einsum('bsi,bsj->ij', inp, inp) # sum over batch and seq
        module.raw_xxt_matrix.add_(act.cpu())
        del inp, act

    whitening_matrices_paths: Dict[str, str] = {}

    # `inps` is a list parallel to `captured`, so each entry is a batch tensor
    inps: List[torch.Tensor] = [entry["inp"] for entry in captured]
    outs: List[Optional[torch.Tensor]] = [None] * len(inps)
    
    # Run calibration through one layer at a time
    for idx in tqdm(range(num_decoder_layers), desc="Computing whitening matrices..."):
        layer = decoder_layers[idx].to(device)

        # Register hooks on target sublayers inside this decoder block
        group = decoder_groups.get(f"layer_{idx}", [])
        handles = []
        for layer_obj, attr, _ in group:
            la = getattr(layer_obj, attr)
            if isinstance(la, nn.Linear):
                la.raw_xxt_matrix = torch.zeros(
                    la.in_features, la.in_features, dtype=torch.float32
                ) # XXT is initialized and stays on CPU
                handles.append(la.register_forward_hook(hook))

        # Run every calibration batch through this single layer
        with torch.no_grad():
            for j, entry in enumerate(captured):
                inp_j = inps[j].to(device)
                kwargs = {}
                if entry["attention_mask"] is not None:
                    kwargs["attention_mask"] = entry["attention_mask"].to(device)
                if entry["position_ids"] is not None:
                    kwargs["position_ids"]   = entry["position_ids"].to(device)
                if entry["position_embeddings"] is not None:
                    cos, sin = entry["position_embeddings"]
                    kwargs["position_embeddings"] = (cos.to(device), sin.to(device))

                out = layer(inp_j, **kwargs)
                # Decoder layers return a tuple, first element is the hidden state
                outs[j] = (out[0] if isinstance(out, tuple) else out).detach().cpu()
                del inp_j, out

        # Clean up hooks
        for h in handles:
            h.remove()

        # Compute (and immediately free) whitening matrices for this block
        for layer_obj, attr, lstr in group:
            la = getattr(layer_obj, attr)
            if not (isinstance(la, nn.Linear) and hasattr(la, 'raw_xxt_matrix')):
                continue
            
            # Move XXT to device (GPU) and normalize it
            raw_xxt = la.raw_xxt_matrix.to(device, dtype=torch.float64)
            raw_xxt /= n_tokens # pyright: ignore[reportOperatorIssue]

            if is_v2:
                # In v2 path we compute eigh immediately and save the tuple
                # This will be needed in scoring for heterogeneous path, and for compression
                # There's no need to save XXT
                L_s, U_s = torch.linalg.eigh(raw_xxt)
                L_s = L_s.flip(0).clamp(min=0.0)
                U_s = U_s.flip(1)
                
                wm = (U_s.to(torch.float32).cpu(), L_s.cpu())
                del U_s, L_s
            else:
                try:
                    wm = torch.linalg.cholesky(raw_xxt).cpu()
                except Exception:
                    print(f"[WARNING] Not positive-definite: {lstr}. Applying regularization.")
                    eigvals = torch.linalg.eigvalsh(raw_xxt)
                    raw_xxt += (-eigvals[0] + 1e-6) * torch.eye(
                        raw_xxt.shape[0], dtype=torch.float64, device=device
                    )
                    wm = torch.linalg.cholesky(raw_xxt).cpu()
                    eigvals = None
                    del eigvals

            # Sanitize layer name for use as filename
            fname = lstr.replace(".", "_") + ".pt"
            fpath = os.path.join(wm_dir, fname)
            torch.save(wm, fpath)
            whitening_matrices_paths[lstr] = fpath  # store path, not tensor
            print(f"[WHITENING] Saved {lstr} -> {fpath}")

            # Free accumulator
            la.raw_xxt_matrix = wm = None # pyright: ignore[reportArgumentType]
            del la.raw_xxt_matrix, raw_xxt, wm

        # Move layer back to CPU and free VRAM before next iteration
        decoder_layers[idx] = layer.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        # Thread activations forward: this block's outputs become next block's inputs
        inps, outs = outs, inps # pyright: ignore[reportAssignmentType]

    del inps, outs, captured
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[WHITENING] Done. {len(whitening_matrices_paths)} matrices saved to {wm_dir}")
    return whitening_matrices_paths

def allocate_ratios(
        group_criterion: Union[GroupBy, Literal["global", "decoder", "type"]],
        score_map: Dict,
        layers_str: List[str],
        target_ratio: float,
        group_patterns: Dict[str, List[str]] | None = None
) -> Dict[str, float]:
    """
    Redistributes compression budget within each weight group.
    Groups: MLP (gate, up, down), Q proj, K proj, V proj, Attention out proj.
    
    Within each group, matrices with higher score get a lower
    compression ratio and vice versa.

    This method also applies the iterative
    rank refinement from NIDA-SVD to guarantee the global
    compression target is met exactly and no ratio exceeds valid bounds.
    """
    # Coerce to enum
    if isinstance(group_criterion, str):
        try:
            group_criterion = GroupBy(group_criterion)
        except ValueError:
            raise ValueError(
                f"Invalid `group_criterion`: '{group_criterion}'. "
                f"Expected one of: {[e.value for e in GroupBy]}"
            )
        
    print(f"\n[BUDGET] Initializing redistribution using strategy: {group_criterion.value.upper()}")
        
    # Group weight matrices by desired criterion
    groups = defaultdict(list)
    unmatched_keys = []
    missing_score_keys = []

    match group_criterion:
        case GroupBy.GLOBAL:
            # Filter keys that have score data
            valid_keys = []
            for k in layers_str:
                if k in score_map: valid_keys.append(k)
                else: missing_score_keys.append(k)
            # All weight matrices in a single bucket named "global"
            groups["global"] = valid_keys

        case GroupBy.DECODER:
            # Group by layer index (e.g., "layers.5")
            # This regex looks for digits surrounded by dots or at the start/end
            for key in layers_str:
                if key not in score_map:
                    missing_score_keys.append(key)
                    continue
                match = re.search(r'\.layers\.(\d+)\.', key)
                layer_idx = match.group(1) if match else "remainder"
                groups[f"layer_{layer_idx}"].append(key)

        case GroupBy.TYPE:
            if group_patterns is None:
                raise ValueError("`group_patterns` required for GroupBy.TYPE")
            for key in layers_str:
                if key not in score_map:
                    missing_score_keys.append(key)
                    continue

                group_name = None
                for name, patterns in group_patterns.items():
                    if any(p in key for p in patterns):
                        group_name = name
                        break

                if group_name is not None:
                    groups[group_name].append(key)
                else:
                    unmatched_keys.append(key)

    # Redistribute budget within each group
    ratio_map = {}
    for group_name, keys in groups.items():
        if not keys:
            print(f"[WARNING] Group {group_name} is empty")
            continue

        # Get scores within the group
        scores = torch.tensor(
            [score_map[k] for k in keys],
            dtype=torch.float64
        )

        if len(keys) == 1:
            # Single-member group: no redistribution possible
            print(f"  [GROUP: {group_name}] Single member detected. Assigning target_ratio {target_ratio:.4f} to {keys[0]}")
            ratio_map[keys[0]] = target_ratio
            continue

        # Inverse-log normalization:
        #   high score  -> 1/log(scores) is small -> less compression (matrix is information-dense)
        #   low score   -> 1/log(scores) is large -> more compression (matrix is redundant)
        log_scores = torch.log(scores)
        inv_log_scores = 1.0 / log_scores
        normalized = inv_log_scores / inv_log_scores.sum()

        # Scale so that the mean ratio across the group equals `target_ratio`,
        # preserving the global memory budget
        ratios = inv_log_scores.shape[0] * target_ratio * normalized

        print(f"  [GROUP: {group_name}] Redistributing over {len(keys)} layers:")
        for key, r in zip(keys, ratios.tolist()):
            # Clamp ratio to (0, 1) as a safety measure before refinement
            ratio_map[key] = max(1e-2, min(r, 0.9))
            print(f"    - {key:<50} | Ratio: {max(1e-2, min(r, 0.9)):.4f} | Score: {score_map[key]:.6f}")

    # Fallback for any unmatched layers
    final_fallbacks = []
    for key in layers_str:
        if key not in ratio_map:
            ratio_map[key] = target_ratio
            # Identify the reason for the fallback
            reason = "Missing score data" if key in missing_score_keys else "No pattern match"
            final_fallbacks.append((key, reason))

    # --- Summary Report ---
    print(f"\n[BUDGET] Allocation Summary:")
    print(f"  - Successfully Redistributed:       {len(ratio_map) - len(final_fallbacks)} layers")
    print(f"  - Fallback (assigned target_ratio): {len(final_fallbacks)} layers")
    if final_fallbacks:
        print(f"[BUDGET] Fallback Details:")
        for key, reason in final_fallbacks:
            print(f"  - {key:<50} | Reason: {reason}")

    print("-" * 80 + "\n")
    return ratio_map

# Compress model with SVD-LLM
def compress_svd_llm(
        model_name: str,
        ratio: float, 
        dataset: Dict,
        max_length: int = 2048,
        is_v2: bool = False,
        dtype: str = "bfloat16",
        batch_size: int = 32,
        seed: Optional[int] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_path: Optional[str] = None,
        whitening_mat_path: Optional[str] = None,
        compress_mlp: bool = False,
        compress_att_qkv: bool = False,
        compress_att_out: bool = False,
        score_metric: Union[ScoreMetric, Literal["truncation", "entropy", "norm|p"]] = "truncation",
        heterogeneous: bool = False,
        group_criterion: Union[GroupBy, Literal["global", "decoder", "type"]] = "type",
        group_patterns: Dict[str, List[str]] | None = None,
        hf_token: Optional[str] = None
):
    # Load model and tokenizer
    vram_usage("Before loading original model")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=None, # WIP try to load model with auto, enforce cuda only in special cases
        use_safetensors=True,
        token=hf_token,
        trust_remote_code=True
    )
    ram_usage("After loading original model")
    vram_usage("After loading original model")
    # Avoid warning
    model.generation_config.pad_token_id = model.generation_config.eos_token_id # pyright: ignore[reportOptionalMemberAccess]
    model.eval()
    model.config.use_cache = False

    # Preprocess calibration dataset
    print("=== DATASET PREPROCESSING ===")
    vram_usage("Before loading dataset")
    calibration_dataset, dataset["max_samples"] = tokenize_dataset(
        dataset["name"],
        dataset["subset"],
        dataset["split"],
        tokenizer,
        dataset["max_samples"],
        batch_size,
        max_length,
        seed,
        save_path
    )
    print(calibration_dataset)
    print(calibration_dataset["input_ids"])
    print(calibration_dataset["input_ids"][0])
    print(len(calibration_dataset["input_ids"][0]))
    calibration_dataloader = DataLoader(
        calibration_dataset, # pyright: ignore[reportArgumentType]
        batch_size=batch_size,
        shuffle=False
    )
    ram_usage("After loading dataset")
    vram_usage("After loading dataset")
    print("=== FINAL DATASET STRUCTURE ===")
    print(calibration_dataset)

    # Get list of layers of interest
    layers_str = generate_paths(
        compress_mlp, 
        compress_att_qkv, 
        compress_att_out, 
        layers_number=model.config.num_hidden_layers
    )
    layers_list, attributes = get_layers(model, layers_str, True)

    # Compute/load whitening matrices for each layer
    vram_usage("Before loading whitening matrices")
    version_str = "v2" if is_v2 else "v1"
    if whitening_mat_path:
        whitening_mat_actual_path = os.path.join(whitening_mat_path, os.path.join(model_name.replace("/", "_").replace("-", "_"), version_str))
        if os.path.isdir(whitening_mat_actual_path):
            # Build the loading dictionary (individual .pt files saved by `get_whitening_matrices`)`
            print(f"[DEBUG] Loading whitening matrix paths from directory: {whitening_mat_actual_path}")
            whitening_matrices: Dict[str, str] = {
                lstr: os.path.join(whitening_mat_actual_path, lstr.replace(".", "_") + ".pt")
                for lstr in layers_str
            }
            # Validate all expected files exist
            missing = [p for p in whitening_matrices.values() if not os.path.exists(p)]
            if missing:
                raise FileNotFoundError(
                    f"[ERROR] {len(missing)} whitening matrix files missing from {whitening_mat_actual_path}:\n"
                    + "\n".join(missing[:5]) + ("..." if len(missing) > 5 else "")
                )
        else:
            raise FileNotFoundError(f"[ERROR] Whitening matrices for this model do not exist in this path: {whitening_mat_actual_path}")
    else:
        print("=== WHITENING MATRICES GENERATION ===")
        whitening_matrices = get_whitening_matrices(
            model, # pyright: ignore[reportArgumentType]
            model_name,
            calibration_dataloader,
            layers_str,
            layers_list,
            attributes,
            max(max_length * dataset["max_samples"], 1),
            device,
            is_v2,
            save_path or "./tmp"
        )
    ram_usage("After loading whitening matrices")
    vram_usage("After loading whitening matrices")

    print("=== LLM COMPRESSION ===")

    rank_map = {}
    steps: int = 1
    steps_counter: int = 1

    # Compression ratio allocation
    if heterogeneous:
        # Compute SVD for all layers and collect score metric
        vram_usage("Before performing scores calculation")

        # Coerce to enum
        if isinstance(score_metric, str):
            try:
                score_metric = ScoreMetric(score_metric)
            except ValueError:
                raise ValueError(
                    f"Invalid `score_metric`: '{score_metric}'. "
                    f"Expected one of: {[e.value for e in ScoreMetric]}"
                )
            
        print(f"\n[DEBUG] Score metric is: {score_metric.value.upper()}")

        score_map = {}
        steps: int = 2
        for i, (layer, attr) in tqdm(
            enumerate(zip(layers_list, attributes)),
            total=len(layers_list),
            desc=f"Step {steps_counter}/{steps}: Computing scores..."
        ):
            # Get weight and normalized whitening matrix
            layer_attr = getattr(layer, attr)
            W = layer_attr.weight.data.to(device, dtype=torch.float64)

            if is_v2:
                # Perform SVD on whitening matrix (S)
                U_s, L_s = load_whitening_data(whitening_matrices, layers_str[i], device, keep=True)

                # Auxiliary matrix - 1e-6 acts as a proper regularization due to normalization
                L_s_sqrt = torch.sqrt(L_s + 1e-6)

                # Perform SVD on W x U_s x sqrt(L_s)
                D = torch.matmul(W, U_s * L_s_sqrt.unsqueeze(0))
                # Calculate svdvals only
                L = torch.linalg.svdvals(D)
            else:
                whitening_matrix = load_whitening_data(whitening_matrices, layers_str[i], device, keep=True)
                WS = torch.matmul(W, whitening_matrix) # pyright: ignore[reportArgumentType]
                L = torch.linalg.svdvals(WS)

            # Compute a tentative rank under the uniform target ratio.
            rank = int((W.shape[0] * W.shape[1] * (1 - ratio)) / (W.shape[0] + W.shape[1]))
            rank = max(1, min(rank, L.shape[0] - 1))

            # Multiply by sqrt(n_tokens) to recover the unnormalized singular values.
            # This pushes the scores into the hundreds/thousands
            L = L * math.sqrt(max(max_length * dataset["max_samples"], 1))

            # Calculate score metric
            match score_metric:
                case "truncation":
                    # After whitening, theoretical truncation loss equals to the L2 norm of truncated singular values = 2-schatten norm = Frobenius norm
                    score_map[layers_str[i]] = torch.linalg.norm(L[rank:], ord=2).item()
                case "entropy":
                    # After whitening, entropy loss equals the sum of normalized singular values of the tail
                    norm_spectrum = L/L.sum()
                    raw_entropy = -(norm_spectrum[rank:] * torch.log(norm_spectrum[rank:] + 1e-9)).sum().item()
                    #raw_entropy = -(norm_spectrum[rank:] * torch.log(norm_spectrum[rank:])).sum().item() # OLD

                    # Shift the score to guarantee it is strictly > 1.0 before ratio allocation. Adding 1.5 provides a safe buffer away from the log(1) cliff.
                    score_map[layers_str[i]] = raw_entropy + 1.5
                case s if s.startswith('norm'):
                    if s.split("|")[1].startswith("-"):
                        p_norm_value = -float(s.split("|")[1][1:])
                    else:
                        p_norm_value = float(s.split("|")[1])
                    # After whitening, norm loss equals to the Lp norm of truncated singular values = p-schatten norm
                    # WARNING: if `p_norm_value`=2 this is the same of `truncation` case
                    score_map[layers_str[i]] = torch.linalg.norm(L[rank:], ord=p_norm_value).item()
                    

            # Free up vram and ram
            W = whitening_matrix = WS = L = U_s = L_s = L_s_sqrt = D = None
            del W, whitening_matrix, WS, L, U_s, L_s, L_s_sqrt, D

        # Allocate compression ratios to each layer based on score metric
        ratio_map = allocate_ratios(group_criterion, score_map, layers_str, ratio, group_patterns)
        torch.cuda.empty_cache()
        steps_counter += 1
        ram_usage("After performing scores calculation")
        vram_usage("After performing scores calculation")
    else:
        ratio_map = {k: ratio for k in layers_str}

    # Compress layers using the calculated compression ratios
    vram_usage("Before performing layer compression")
    for i, (layer, attr) in tqdm(
        enumerate(zip(layers_list, attributes)),
        total=len(layers_list),
        desc=f"Step {steps_counter}/{steps}: Compressing layers..."
    ):
        # Get weight matrix
        layer_attr = getattr(layer, attr)
        W = layer_attr.weight.data.to(device, dtype=torch.float64)
        
        # Compute rank from compression ratio
        layer_ratio = ratio_map[layers_str[i]]
        rank = int((W.shape[0] * W.shape[1] * (1 - layer_ratio)) / (W.shape[0] + W.shape[1]))
        
        if is_v2:
            # heterogeneous-v2 path - stream U_s and L_s calculated while generating the whitening matrices
            U_s, L_s = load_whitening_data(whitening_matrices, layers_str[i], device, keep=False)

            # Auxiliary matrix - 1e-6 acts as a proper regularization due to normalization
            L_s_sqrt = torch.sqrt(L_s + 1e-6)

            # Perform SVD on W x U_s x sqrt(L_s)
            D = torch.matmul(
                W, 
                U_s * L_s_sqrt.unsqueeze(0)
            )
            # Free W as soon as D is ready
            W = None
            del W

            U_ws, L_ws, V_wsT = torch.linalg.svd(D, full_matrices=False)
            # Free D as soon as U_ws, L_ws and V_wsT are ready
            D = None
            del D

            # Calculate sqrt(L_s) and U_s inverse matrices
            # 1e-6 acts as a proper regularization due to normalization
            L_s_sqrt_inv = (1.0 / (L_s_sqrt + 1e-6))
            U_s_inv_L_s_sqrt_inv = (U_s * L_s_sqrt_inv.unsqueeze(0)).T

            # Free U_s and L_s
            U_s = L_s = L_s_sqrt = None
            del U_s, L_s, L_s_sqrt

            # Calculate final rank and truncate matrices
            rank = max(1, min(rank, L_ws.shape[0] - 1))
            rank_map[layers_str[i]] = rank
            U_ws_r = U_ws[:, :rank]
            L_ws_r = L_ws[:rank]
            V_wsT_r = V_wsT[:rank, :]

            # Free full-rank matrices as soon as truncated slices are built
            U_ws = L_ws = V_wsT = L_s_sqrt_inv = None
            del U_ws, L_ws, V_wsT, L_s_sqrt_inv

            # Compute approximate weight matrix, split in two matrices
            L_ws_r_sqrt = torch.sqrt(L_ws_r)
            W_u = (U_ws_r * L_ws_r_sqrt.unsqueeze(0)).cpu().to(layer_attr.weight.dtype)
            W_v = torch.matmul(L_ws_r_sqrt.unsqueeze(1) * V_wsT_r, U_s_inv_L_s_sqrt_inv).cpu().to(layer_attr.weight.dtype)
            # Free low-rank matrices, leave only W_u and W_v
            U_ws_r = L_ws_r = V_wsT_r = L_ws_r_sqrt = U_s_inv_L_s_sqrt_inv = None
            del U_ws_r, L_ws_r, V_wsT_r, L_ws_r_sqrt, U_s_inv_L_s_sqrt_inv
        else:
            # Get normalized whitening matrix
            whitening_matrix = load_whitening_data(whitening_matrices, layers_str[i], device, keep=False)

            # Compute the inverse of the normalized whitening matrix
            try:
                whitening_matrix_inv = torch.linalg.inv(whitening_matrix)
            except Exception as e:
                print("[WARNING] whitening_matrix is not full rank!")
                # Because the matrix is normalized, 1e-6 * eye is statistically relevant
                whitening_matrix += 1e-6 * torch.eye(
                    whitening_matrix.shape[0], # type: ignore
                    dtype=whitening_matrix.dtype # pyright: ignore[reportAttributeAccessIssue]
                ).to(device)
                whitening_matrix_inv = torch.linalg.inv(whitening_matrix)

            # Perform SVD on W x S
            WS = torch.matmul(W, whitening_matrix)  # pyright: ignore[reportArgumentType]
            # Free whitening_matrix and W as soon as WS is ready
            W = whitening_matrix = None # pyright: ignore[reportArgumentType]
            del W, whitening_matrix

            U, L, VT = torch.linalg.svd(WS, full_matrices=False)
            # Free WS as soon as U, L and VT are ready
            WS = None
            del WS

            # Calculate final rank and truncate matrices
            rank = max(1, min(rank, L.shape[0] - 1))
            rank_map[layers_str[i]] = rank
            U_r = U[:, :rank]
            L_r_sqrt = torch.sqrt(L[:rank])
            VT_r = torch.matmul(VT[:rank, :], whitening_matrix_inv)
            # Free full-rank matrices as soon as truncated slices are built
            U = L = VT = whitening_matrix_inv = None
            del U, L, VT, whitening_matrix_inv

            # Compute approximate weight matrix, split in two matrices
            W_u = (U_r * L_r_sqrt.unsqueeze(0)).cpu().to(layer_attr.weight.dtype)
            W_v = (VT_r * L_r_sqrt.unsqueeze(1)).cpu().to(layer_attr.weight.dtype)
            # Free low-rank matrices, leave only W_u and W_v
            U_r = VT_r = L_r_sqrt = None
            del U_r, VT_r, L_r_sqrt

        # Replace the original nn.Linear with the LowRank module, which implements the forward pass as W_u(W_v(x)).
        van = LowRank(
            layer_attr.in_features,
            layer_attr.out_features,
            rank,
            layer_attr.bias is not None
        )
        van.W_u.weight.data = W_u
        van.W_v.weight.data = W_v
        if layer_attr.bias is not None:
            van.W_u.bias.data = layer_attr.bias.data

        setattr(layer, attr, van)

        # Free ram and vram from all leftover matrices
        W_u = W_v = None
        del W_u, W_v
    for name, param in model.named_parameters():
        if 'W_v' in name or 'W_u' in name:
            print(f"{name}: dtype={param.dtype}, device={param.device}, norm={param.norm():.4f}")
            break  # just check the first one
    ram_usage("After performing layer compression")
    vram_usage("After performing layer compression")

    if save_path:
        print("[DEBUG] Saving compressed model to disk...")
        # Create model directory
        save_path_model = save_path + \
                          "/models/" + \
                          model_name.replace("/", "_").replace("-", "_") + \
                          "/"
        if not os.path.exists(save_path_model):
            os.makedirs(save_path_model)
        # Save tokenizer
        tokenizer.save_pretrained(save_path_model)
        # Save model weights
        compress_att_qkv_str = "_qkv" if compress_att_qkv else ""
        compress_att_out_str = "_out" if compress_att_out else ""
        compress_mlp_str = "_mlp" if compress_mlp else ""
        heterogeneous_str = "_het" if heterogeneous else ""
        group_criterion_str = ("_" + group_criterion) if heterogeneous else ""
        score_metric_substr = score_metric.replace("|", "") if len(score_metric.split("|")) > 1 else score_metric
        score_metric_str = ("_" + score_metric_substr) if heterogeneous else ""
        v2_str = "_v2" if is_v2 else ""
        torch.save({
            "state_dict": model.state_dict(),
            "rank_map": rank_map,
        }, save_path_model + 
           model_name.replace("/", "_").replace("-", "_") + 
           compress_att_qkv_str + 
           compress_att_out_str + 
           compress_mlp_str + "_" +
           str(round(ratio, 2)) +
           heterogeneous_str + 
           group_criterion_str +
           score_metric_str +
           v2_str + 
           ".pt")
        print("[DEBUG] Compressed model saved succesfully")

    torch.cuda.empty_cache()
    gc.collect()
    return model, tokenizer

def apply_lowrank(model, rank_map):
    """
    Replace MLP linear layers with LowRank modules.
    rank_map: dict with keys like 'model.layers.0.mlp.down_proj', 'model.layers.0.mlp.gate_proj', etc.
    """

    for layer_name, rank in rank_map.items():
        # Get the old layer
        layer_path = layer_name.split('.')[:-1]
        layer = model
        for sub_layer in layer_path:
            layer = getattr(layer, sub_layer)

        # Update the layer
        attr_name = layer_name.split('.')[-1]
        attr = getattr(layer, attr_name)
        setattr(
            layer,
            attr_name,
            LowRank(
                in_features=attr.in_features,
                out_features=attr.out_features,
                rank=rank,
                bias=attr.bias is not None
            )
        )