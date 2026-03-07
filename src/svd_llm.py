from typing import Dict, Optional, List, Union, Literal
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
import torch
from torch.utils.data import DataLoader
from .utils import *

GROUP_PATTERNS = {
                "q_proj": ["self_attn.q_proj"],
                "k_proj": ["self_attn.k_proj"],
                "v_proj": ["self_attn.v_proj"],
                "o_proj": ["self_attn.o_proj"],
                "mlp": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
            }

class LowRank(torch.nn.Module):
    def __init__(self, in_features, out_features, rank, bias):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W_v = nn.Linear(in_features, rank, bias)
        self.W_u = nn.Linear(rank, out_features, bias)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output =  self.W_u(self.W_v(input))
        return output

def get_whitening_matrices(
        model: Qwen2ForCausalLM,
        loader: DataLoader,
        layers_str: List[str], 
        layers_list: List,
        attributes: List[str],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        is_v2: bool = False
):
    # Define pytorch forward hook to incrementally compute X^T @ X
    def hook(module, input, output):
        inp = input[0].detach().to(dtype=torch.float32) # Detach input activation from computational graph
        act = torch.matmul(inp.transpose(1,2), inp) # Calculate x^T @ x for the entire batch
        act_batch = torch.sum(act, dim=0) # Sum across batch dimension
        module.raw_xxt_matrix += act_batch
        del inp, act, act_batch
        torch.cuda.empty_cache()

    # Inizialize empty X^T @ X matrices and register forward hooks
    for layer, attr in tqdm(zip(layers_list, attributes), total=len(layers_list), desc="Registering hooks..."):
        layer_attr = getattr(layer, attr)
        if (isinstance(layer_attr, nn.Linear)):
            layer_attr.raw_xxt_matrix = torch.zeros(layer_attr.in_features, layer_attr.in_features, device=device, dtype=torch.float32)
            layer_attr.register_forward_hook(hook)

    # Run inference on calibration data
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Running inference on calibration data..."):
            # REMOVE final_input before forward pass
            batch = {
                k: v.to(device)
                for k, v in batch.items()
                if k in ("input_ids", "attention_mask")
            }
            model(**batch)

    for layer, attr in tqdm(zip(layers_list, attributes), total=len(layers_list), desc="Removing hooks..."):
        layer_attr = getattr(layer, attr)
        if (isinstance(layer_attr, nn.Linear)):
            # Move X^T @ X matrix to CPU if it exists
            if hasattr(layer_attr, 'raw_xxt_matrix'):
                layer_attr.raw_xxt_matrix = layer_attr.raw_xxt_matrix.cpu()
            # Remove hook
            layer_attr._forward_hooks.clear()

    # Empty cache and move model to CPU
    torch.cuda.empty_cache()
    model = model.cpu()
    print("DEBUG: Cache emptied succesfully after model execution on calibration data")

    # Build whitening matrices applying Cholesky decomposition to X^T @ X
    whitening_matrices = {}
    for i, (layer, attr) in tqdm(enumerate(zip(layers_list, attributes)), total=len(layers_list), desc="Generating whitening matrices..."):
        layer_attr = getattr(layer, attr)
        if (isinstance(layer_attr, nn.Linear)):
            if hasattr(layer_attr, 'raw_xxt_matrix'):
                raw_xxt_matrix = layer_attr.raw_xxt_matrix.to(device, dtype=torch.float64)
                if is_v2:
                    whitening_matrices[layers_str[i]]=raw_xxt_matrix.cpu()
                else:
                    try:
                        whitening_matrix = torch.linalg.cholesky(raw_xxt_matrix)
                    except Exception as e:
                        print("WARNING: eigen whitening_matrix is not positive!")
                        eigenvalues = torch.linalg.eigvalsh(raw_xxt_matrix)
                        rows: int = raw_xxt_matrix.shape[0] # type: ignore
                        raw_xxt_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(rows).to(device, dtype=torch.float64)
                        whitening_matrix = torch.linalg.cholesky(raw_xxt_matrix)
                        eigenvalues = None
                        del eigenvalues
                    whitening_matrices[layers_str[i]]=whitening_matrix.cpu()
                whitening_matrix = raw_xxt_matrix = layer_attr.raw_xxt_matrix = None # pyright: ignore[reportArgumentType]
                del whitening_matrix, raw_xxt_matrix, layer_attr.raw_xxt_matrix
                torch.cuda.empty_cache()

    return whitening_matrices

# TODO add "group_criterion" argument to decide wheter to group by weight type, layer or globally
def allocate_ratios(
        group_criterion: Union[GroupBy, Literal["global", "decoder", "type"]],
        loss_map: Dict,
        layers_str: List[str],
        target_ratio: float,
        group_patterns: Dict[str, List[str]] | None = None
) -> Dict[str, float]:
    """
    Redistributes compression budget within each weight group.
    Groups: MLP (gate, up, down), Q proj, K proj, V proj, Attention out proj.
    
    Within each group, matrices with higher truncation loss get a lower
    compression ratio and vice versa.
    """

    match group_criterion:
        case GroupBy.GLOBAL:
            return
        case GroupBy.DECODER:
            return
        case GroupBy.TYPE:
            if group_patterns is None:
                raise Exception("`group_patterns` must be a map of layer types into groups")
            # Define group membership by matching layer path suffixes - TODO make it an argument to provide with group_criterion
            # Assign each layer to its group
            groups = defaultdict(list)
            for key in layers_str:
                if key not in loss_map:
                    continue
                group = get_group(key, GROUP_PATTERNS) # TODO - replace constant with variable
                if group is not None:
                    groups[group].append(key)
                else:
                    print(f"WARNING: {key} did not match any group, using target ratio {target_ratio}")

            # Redistribute budget within each group
            ratio_map = {}
            for group_name, keys in groups.items():
                # Get losses within the group
                losses = torch.tensor(
                    [loss_map[k] for k in keys],
                    dtype=torch.float64
                )

                if len(keys) == 1:
                    # Single-member group: no redistribution possible
                    ratio_map[keys[0]] = target_ratio
                    continue

                # Inverse-log normalization:
                #   high loss  -> 1/log(loss) is small -> less compression (matrix is information-dense)
                #   low loss   -> 1/log(loss) is large -> more compression (matrix is redundant)
                log_losses = torch.log(losses + 1.0)   # +1 guards against log(0) or log(very small)
                inv_log_losses = 1.0 / log_losses
                normalized = inv_log_losses / inv_log_losses.sum()
                print(f"  [{group_name}] log_losses={log_losses}; inv_log_losses={inv_log_losses}; sum={inv_log_losses.sum()}; normalized={normalized}")

                # Scale so that the mean ratio across the group equals `target_ratio`,
                # preserving the global memory budget
                ratios = len(keys) * target_ratio * normalized

                for key, r in zip(keys, ratios.tolist()):
                    ratio_map[key] = r
                    print(f"  [{group_name}] {key}: ratio={r:.4f} (loss={loss_map[key]:.4f})")

            # Fallback for any unmatched layers
            for key in layers_str:
                if key not in ratio_map:
                    ratio_map[key] = target_ratio

            return ratio_map

# Compress model with SVD-LLM
def compress_svd_llm(
        model_name: str,
        ratio: float, 
        dataset: Dict,
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
        heterogeneous: bool = False,
        hf_token: Optional[str] = None
):
    # Load model and tokenizer
    vram_usage("Before loading original model")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device,
        use_safetensors=True,
        token=hf_token
    )
    vram_usage("After loading original model")
    # Avoid warning
    model.generation_config.pad_token_id = model.generation_config.eos_token_id # pyright: ignore[reportOptionalMemberAccess]

    # Preprocess calibration dataset
    print("=== DATASET PREPROCESSING ===")
    vram_usage("Before loading dataset")
    calibration_dataset = tokenize_dataset(
        dataset["dataset_name"],
        dataset["split"],
        tokenizer,
        dataset["max_samples"],
        batch_size,
        seed,
        save_path
    )
    calibration_dataloader = DataLoader(
        calibration_dataset, # pyright: ignore[reportArgumentType]
        batch_size=batch_size,
        shuffle=False
    )
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
    if whitening_mat_path:
        print("DEBUG: Loading whitening matrices from disk...")
        whitening_matrices = torch.load(whitening_mat_path, map_location="cpu")
    else:
        print("=== WHITENING MATRICES GENERATION ===")
        whitening_matrices = get_whitening_matrices(
            model, # pyright: ignore[reportArgumentType]
            calibration_dataloader,
            layers_str,
            layers_list,
            attributes,
            device,
            is_v2
        )
        if save_path:
            print("DEBUG: Saving whitening matrices to disk...")
            save_path_whitening = save_path + "/whitening_matrices/"
            v2_str = "v2" if is_v2 else ""
            if not os.path.exists(save_path_whitening):
                os.makedirs(save_path_whitening)
            torch.save(whitening_matrices, save_path_whitening + 
                                           model_name.replace("/", "_").replace("-", "_") + 
                                           '_whitening_'+ 
                                           dataset["dataset_name"].replace("/", "_").replace("-", "_").split("_")[-1] + 
                                           '_' + 
                                           str(dataset["max_samples"]) + 
                                           '_' + 
                                           str(seed) +
                                           '_' +
                                           v2_str +
                                           '.pt')
    vram_usage("After loading whitening matrices")

    print("=== LLM COMPRESSION ===")
    vram_usage("Before performing layer replacement")

    rank_map = {}
    us_cache = {}  # caches (U_s, L_s) per layer — only populated when is_v2 and heterogeneous
    steps: int = 1
    steps_counter: int = 1

    # Compression ratio allocation
    if heterogeneous:
        # Compute SVD for all layers and collect truncation losses
        loss_map = {}
        steps: int = 2
        vram_usage("Before performing truncation loss calculation")
        for i, (layer, attr) in tqdm(
            enumerate(zip(layers_list, attributes)),
            total=len(layers_list),
            desc=f"Step {steps_counter}/{steps}: Computing truncation losses..."
        ):
            # Get weight and whitening matrix
            layer_attr = getattr(layer, attr)
            W = layer_attr.weight.data.to(device, dtype=torch.float32)
            whitening_matrix = whitening_matrices[layers_str[i]].to(device, dtype=torch.float32)

            if is_v2:
                # Perform SVD on whitening matrix (S)
                U_s, L_s, _ = torch.linalg.svd(whitening_matrix, full_matrices=False)
                L_s_sqrt = torch.diag(torch.sqrt(L_s))

                # Perform SVD on W x U_s x sqrt(L_s)
                D = torch.matmul(W, torch.matmul(U_s, L_s_sqrt))
                # Calculate svdvals only
                L = torch.linalg.svdvals(D)

                # Cache U_s and L_s on CPU to avoid computing SVD of S twice
                # these are in_features × in_features, far smaller than caching the full SVD of D
                us_cache[layers_str[i]] = (U_s.cpu(), L_s.cpu())
            else:
                WS = torch.matmul(W, whitening_matrix)
                L = torch.linalg.svdvals(WS)

            # Compute a tentative rank under the uniform target ratio.
            rank = int((W.shape[0] * W.shape[1] * (1 - ratio)) / (W.shape[0] + W.shape[1]))
            print(f"DEBUG: L shape: {L.shape}")
            rank = max(1, min(rank, L.shape[0] - 1))

            # Calculate theoretical truncation loss
            # After whitening, this equals the sum of squared singular values
            loss_map[layers_str[i]] = torch.sum(L[rank:] ** 2).item()

            # Free up vram and ram
            W = whitening_matrix = WS = L = U_s = L_s = L_s_sqrt = D = None
            del W, whitening_matrix, WS, L, U_s, L_s, L_s_sqrt, D
            torch.cuda.empty_cache()

        # Allocate compression ratios to each layer based on theoretical truncation loss
        # TODO - replace constants with variables
        ratio_map = allocate_ratios("type", loss_map, layers_str, ratio, GROUP_PATTERNS)
        steps_counter += 1
        vram_usage("After performing truncation loss calculation")
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
        W = layer_attr.weight.data.to(device, dtype=torch.float32)

        # Compute rank from compression ratio
        layer_ratio = ratio_map[layers_str[i]]
        rank = int((W.shape[0] * W.shape[1] * (1 - layer_ratio)) / (W.shape[0] + W.shape[1]))
        
        if is_v2:
            # Get whitening matrix
            whitening_matrix = whitening_matrices[layers_str[i]].to(device, dtype=torch.float32)

            if layers_str[i] in us_cache:
                # Heterogeneous mode - get cached U_s and L_s from pass 1
                U_s, L_s = (t.to(device) for t in us_cache.pop(layers_str[i]))
                L_s_sqrt = torch.diag(torch.sqrt(L_s))
            else:
                # Homogeneous mode - perform SVD on whitening matrix (S)
                U_s, L_s, _ = torch.linalg.svd(whitening_matrix, full_matrices=False)
                L_s_sqrt = torch.diag(torch.sqrt(L_s))
            # Free whitening matrix as soon as U_s and L_s_sqrt are ready
            whitening_matrix = None
            del whitening_matrix
            torch.cuda.empty_cache()

            # Perform SVD on W x U_s x sqrt(L_s)
            D = torch.matmul(
                W, 
                torch.matmul(U_s, L_s_sqrt)
            )
            # Free W as soon as D is ready
            W = None
            del W
            torch.cuda.empty_cache()

            U_ws, L_ws, V_wsT = torch.linalg.svd(D, full_matrices=False)
            print(f"DEBUG: L_ws shape: {L_ws.shape}")
            # Free D as soon as U_ws, L_ws and V_wsT are ready
            D = None
            del D
            torch.cuda.empty_cache()

            # Calculate sqrt(L_s) and U_s inverse matrices
            L_s_sqrt_inv = torch.diag(1.0 / torch.sqrt(L_s))
            print(f"DEBUG: L_s_sqrt_inv shape: {L_s_sqrt_inv.shape}")
            U_s_inv = torch.transpose(U_s, 0, 1)

            # Calculate final rank and truncate matrices
            rank = max(1, min(rank, L_ws.shape[0] - 1))
            rank_map[layers_str[i]] = rank
            U_ws_r = U_ws[:, :rank]
            L_ws_r = L_ws[:rank]
            V_wsT_r = torch.matmul(
                V_wsT[:rank, :], 
                torch.matmul(L_s_sqrt_inv, U_s_inv)
            )

            # Free full-rank matrices as soon as truncated slices are built
            U_ws = L_ws = V_wsT = L_s = L_s_sqrt = L_s_sqrt_inv = U_s = U_s_inv = None
            del U_ws, L_ws, V_wsT, L_s, L_s_sqrt, L_s_sqrt_inv, U_s, U_s_inv
            torch.cuda.empty_cache()

            # Compute approximate weight matrix, split in two matrices
            L_ws_r_sqrt = torch.diag(torch.sqrt(L_ws_r))
            W_u = torch.matmul(U_ws_r, L_ws_r_sqrt).cpu().to(layer_attr.weight.dtype)
            W_v = torch.matmul(L_ws_r_sqrt, V_wsT_r).cpu().to(layer_attr.weight.dtype)
            # Free low-rank matrices, leave only W_u and W_v
            U_ws_r = L_ws_r = V_wsT_r = L_ws_r_sqrt = None
            del U_ws_r, L_ws_r, V_wsT_r, L_ws_r_sqrt
        else:
            # Get whitening matrix
            whitening_matrix = whitening_matrices[layers_str[i]].to(device, dtype=torch.float64)

            # Compute the inverse of the whitening matrix
            try:
                whitening_matrix_inv = torch.linalg.inv(whitening_matrix)
            except Exception as e:
                print("WARNING: whitening_matrix is not full rank!")
                whitening_matrix += 1e-6 * torch.eye(
                    whitening_matrix.shape[0],
                    dtype=whitening_matrix.dtype
                ).to(device)
                whitening_matrix_inv = torch.linalg.inv(whitening_matrix)

            # Cast whitening matrix (and inverse) to lower precision
            whitening_matrix = whitening_matrix.to(dtype=torch.float32)
            whitening_matrix_inv = whitening_matrix_inv.to(dtype=torch.float32)

            # Perform SVD on W x S
            WS = torch.matmul(W, whitening_matrix)
            # Free whitening_matrix and W as soon as WS is ready
            W = whitening_matrix = None
            del W, whitening_matrix
            torch.cuda.empty_cache()

            U, L, VT = torch.linalg.svd(WS, full_matrices=False)
            # Free WS as soon as U, L and VT are ready
            WS = None
            del WS
            torch.cuda.empty_cache()

            # Calculate final rank and truncate matrices
            rank = max(1, min(rank, L.shape[0] - 1))
            rank_map[layers_str[i]] = rank
            U_r = U[:, :rank]
            L_r = torch.diag(L[:rank])
            VT_r = torch.matmul(VT[:rank, :], whitening_matrix_inv)
            # Free full-rank matrices as soon as truncated slices are built
            U = L = VT = whitening_matrix_inv = None
            del U, L, VT, whitening_matrix_inv
            torch.cuda.empty_cache()

            # Compute approximate weight matrix, split in two matrices
            L_r_sqrt = torch.sqrt(L_r)
            W_u = torch.matmul(U_r, L_r_sqrt).cpu().to(layer_attr.weight.dtype)
            W_v = torch.matmul(L_r_sqrt, VT_r).cpu().to(layer_attr.weight.dtype)
            # Free low-rank matrices, leave only W_u and W_v
            U_r = L_r = VT_r = L_r_sqrt = None
            del U_r, L_r, VT_r, L_r_sqrt

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
        W_u = W_v = whitening_matrices[layers_str[i]] = None
        del W_u, W_v, whitening_matrices[layers_str[i]]
        torch.cuda.empty_cache()
    vram_usage("After performing layer compression")

    if save_path:
        print("DEBUG: Saving compressed model to disk...")
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
        compress_mlp_str = "mlp_" if compress_mlp else ""
        compress_att_qkv_str = "qkv_" if compress_att_qkv else ""
        compress_att_out_str = "out_" if compress_att_out else ""
        heterogeneous_str = "het_" if heterogeneous else ""
        v2_str = "v2_" if is_v2 else ""
        torch.save({
            "state_dict": model.state_dict(),
            "rank_map": rank_map,
        }, save_path_model + 
           model_name.replace("/", "_").replace("-", "_") + 
           "_" + 
           compress_att_qkv_str + 
           compress_att_out_str + 
           compress_mlp_str + 
           str(round(ratio, 2)) + "_" +
           heterogeneous_str + 
           v2_str +
           "compressed" + 
           ".pt")
        print("DEBUG: Compressed model saved succesfully")

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