import re
import math
import gc
import os
import torch
import torch.nn as nn
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)

import jax
import numpy as np
import scipy.linalg
from functools import partial
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

jax.config.update("jax_enable_x64", True)

@partial(jax.jit, donate_argnums=(0,))
def _jax_eigh_compiled(x):
    # jax.lax.linalg.eigh returns: eigenvectors, eigenvalues
    return jax.lax.linalg.eigh(x)


def eigh_jax_gpu_from_cpu(t_cpu: torch.Tensor):
    """
    Large-matrix V2 whitening eigensolver.

    Memory policy:
    - input tensor starts on CPU
    - JAX moves only this matrix to GPU
    - JAX computes eigh on GPU
    - eigenvectors/eigenvalues are moved back to CPU immediately
    - no PyTorch CUDA tensor is created in this function
    """
    if t_cpu.device.type != "cpu":
        raise ValueError("eigh_jax_gpu_from_cpu expects a CPU tensor.")

    # Ensure CPU-contiguous storage before exposing it to NumPy.
    t_cpu = t_cpu.detach().contiguous()

    # CPU NumPy view; no copy and no GPU allocation here.
    x_np = t_cpu.numpy()

    # JAX owns the GPU input allocation.
    gpu = jax.devices("gpu")[0]
    x = jax.device_put(x_np, gpu)

    eigvecs_jax, eigvals_jax = _jax_eigh_compiled(x)
    jax.block_until_ready((eigvecs_jax, eigvals_jax))

    # Copy results back to CPU as writable NumPy arrays.
    eigvals_np = np.array(jax.device_get(eigvals_jax), copy=True)
    eigvecs_np = np.array(jax.device_get(eigvecs_jax), copy=True)

    L_s = torch.from_numpy(eigvals_np)
    U_s = torch.from_numpy(eigvecs_np)

    # Explicitly delete JAX device buffers when possible.
    for arr in (eigvecs_jax, eigvals_jax, x):
        try:
            arr.delete()
        except Exception:
            pass

    del x_np, x
    del eigvecs_jax, eigvals_jax
    del eigvals_np, eigvecs_np

    gc.collect()
    cuda_cleanup()

    return L_s, U_s

def get_whitening_matrices(
        model: Qwen2ForCausalLM,
        model_name: str,
        loader: DataLoader,
        layers_str: List[str],
        n_tokens: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        is_v2: bool = False,
        save_path: str = "./tmp",
        start_layer: int = 0,
        end_layer: Optional[int] = None
):
    """
    Computes whitening matrices layer by layer and saves each one to disk.

    Supports activation checkpointing for chunked runs.

    Behavior:
        - start_layer is inclusive.
        - end_layer is exclusive.
        - If activation checkpoint for start_layer exists, it is loaded and
          layers before start_layer are skipped.
        - If no activation checkpoint exists, the function starts from layer 0
          and forwards until start_layer before accumulating whitening matrices.
        - At the end of the chunk, activations for end_layer are saved.
        - If end_layer is the final decoder layer, activation checkpoints are deleted.
    """

    version_str = "v2" if is_v2 else "v1"

    wm_dir = os.path.join(save_path, "whitening_matrices", model_name.replace("/", "_").replace("-", "_"), version_str)
    os.makedirs(wm_dir, exist_ok=True)

    act_ckpt_dir = os.path.join(save_path, "activation_checkpoints", model_name.replace("/", "_").replace("-", "_"), version_str)
    os.makedirs(act_ckpt_dir, exist_ok=True)

    print(f"[WHITENING] Streaming whitening matrices to: {wm_dir}")
    print(f"[WHITENING] Activation checkpoint dir: {act_ckpt_dir}")

    decoder_layers = model.model.layers
    num_decoder_layers = len(decoder_layers)

    if end_layer is None:
        end_layer = num_decoder_layers

    start_layer = max(0, int(start_layer))
    end_layer = min(num_decoder_layers, int(end_layer))

    if start_layer >= end_layer:
        raise ValueError(
            f"Invalid whitening layer range: start_layer={start_layer}, end_layer={end_layer}"
        )

    print(f"[WHITENING] Requested decoder layer range [{start_layer}, {end_layer})")

    # Group target weight matrices by decoder layer index.
    decoder_groups: Dict[int, List[tuple[str, str]]] = defaultdict(list)

    for lstr in layers_str:
        m = re.search(r"model\.layers\.(\d+)\.(.*)", lstr)
        if m is None:
            continue
        idx = int(m.group(1))
        local_path = m.group(2)
        decoder_groups[idx].append((local_path, lstr))

    # ---------------------------------------------------------------------
    # Try loading activation checkpoint for start_layer.
    # ---------------------------------------------------------------------

    loaded_ckpt, loaded_inps, loaded_captured, loaded_ckpt_path = try_load_activation_checkpoint(act_ckpt_dir, model_name, version_str, n_tokens, start_layer)

    if loaded_ckpt:
        inps: List[torch.Tensor] = loaded_inps  # type: ignore[assignment]
        captured: List[Dict] = loaded_captured  # type: ignore[assignment]
        loop_start_layer = start_layer

        print(f"[WHITENING] Starting directly from decoder layer {start_layer} using activation checkpoint.")

    else:
        # -----------------------------------------------------------------
        # PHASE 1: Capture layer_0 inputs.
        # -----------------------------------------------------------------

        print("[WHITENING] No usable activation checkpoint. Capturing layer_0 inputs...")

        model.model.embed_tokens = model.model.embed_tokens.to(device)

        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.to(device)

        captured: List[Dict] = []

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def __getattr__(self, name: str):
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return getattr(self.module, name)

            def forward(self, inp, **kwargs):
                pe = kwargs.get("position_embeddings", None)

                captured.append({
                    "inp": inp.detach().cpu(),
                    "attention_mask": kwargs.get("attention_mask", None),
                    "position_ids": kwargs.get("position_ids", None),
                    "position_embeddings": (
                        pe[0].detach().cpu(),
                        pe[1].detach().cpu(),
                    ) if pe is not None else None,
                })

                entry = captured[-1]

                if entry["attention_mask"] is not None:
                    entry["attention_mask"] = entry["attention_mask"].detach().cpu()

                if entry["position_ids"] is not None:
                    entry["position_ids"] = entry["position_ids"].detach().cpu()

                raise ValueError

        decoder_layers[0] = decoder_layers[0].to(device)
        original_layer0 = decoder_layers[0]
        decoder_layers[0] = Catcher(original_layer0)

        with torch.inference_mode():
            for batch in tqdm(loader, desc="Capturing layer_0 inputs"):
                try:
                    batch = {
                        k: v.to(device)
                        for k, v in batch.items()
                        if k in ("input_ids", "attention_mask")
                    }
                    model(**batch)
                except ValueError:
                    pass
                finally:
                    del batch

        decoder_layers[0] = original_layer0
        decoder_layers[0] = decoder_layers[0].cpu()

        model.model.embed_tokens = model.model.embed_tokens.cpu()

        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.cpu()

        cuda_cleanup()

        print(f"[WHITENING] Captured layer_0 inputs for {len(captured)} batches.")
        vram_usage("After layer 0 capture")

        inps = [entry["inp"] for entry in captured]

        for entry in captured:
            entry["inp"] = None

        loop_start_layer = 0

    outs: List[Optional[torch.Tensor]] = [None] * len(inps)

    # ---------------------------------------------------------------------
    # Hook definition.
    # ---------------------------------------------------------------------

    def hook(module, input, output):
        inp = input[0].detach().to(dtype=module.raw_xxt_matrix.dtype)
        act = torch.einsum("bsi,bsj->ij", inp, inp)
        module.raw_xxt_matrix.add_(act)
        del inp, act

    whitening_matrices_paths: Dict[str, str] = {}

    # ---------------------------------------------------------------------
    # PHASE 2: Replay decoder layers.
    # ---------------------------------------------------------------------

    for idx in tqdm(range(loop_start_layer, end_layer), desc="Computing whitening matrices..."):
        should_save_this_layer = start_layer <= idx < end_layer

        print(f"[WHITENING] Processing decoder layer {idx} | save={should_save_this_layer}")

        layer = decoder_layers[idx].to(device)

        group = decoder_groups.get(idx, []) if should_save_this_layer else []

        handles = []

        for local_path, lstr in group:
            la = get_submodule(layer, local_path)

            if isinstance(la, nn.Linear):
                #acc_dtype = torch.float32 if la.in_features > SOLVER_GPU_MAX_DIM else torch.float64

                la.raw_xxt_matrix = torch.zeros(
                    la.in_features,
                    la.in_features,
                    dtype=torch.float32,
                    device=device,
                )

                handles.append(la.register_forward_hook(hook))

        # Replay every calibration batch through this decoder layer.
        with torch.inference_mode():
            for j, entry in enumerate(captured):
                inp_j = inps[j].to(device)

                kwargs = {}

                if entry["attention_mask"] is not None:
                    kwargs["attention_mask"] = entry["attention_mask"].to(device)

                if entry["position_ids"] is not None:
                    kwargs["position_ids"] = entry["position_ids"].to(device)

                if entry["position_embeddings"] is not None:
                    cos, sin = entry["position_embeddings"]
                    kwargs["position_embeddings"] = (
                        cos.to(device),
                        sin.to(device),
                    )

                out = layer(inp_j, **kwargs)
                outs[j] = (out[0] if isinstance(out, tuple) else out).detach().cpu()

                del inp_j, out, kwargs

            torch.cuda.synchronize()

        for h in handles:
            h.remove()

        # -----------------------------------------------------------------
        # Move accumulated XXT matrices to CPU immediately.
        # -----------------------------------------------------------------

        pending_xxt = []

        for local_path, lstr in group:
            la = get_submodule(layer, local_path)

            if not (isinstance(la, nn.Linear) and hasattr(la, "raw_xxt_matrix")):
                continue

            raw_acc = la.raw_xxt_matrix
            raw_xxt_cpu = (raw_acc / n_tokens).detach().to(torch.float64).cpu()

            la.raw_xxt_matrix = None  # pyright: ignore[reportArgumentType]
            del la.raw_xxt_matrix
            del raw_acc

            pending_xxt.append((lstr, raw_xxt_cpu))

        # Move decoder layer itself back to CPU before eig/Cholesky work.
        decoder_layers[idx] = layer.cpu()
        del layer

        cuda_cleanup()
        vram_usage(f"After layer {idx} forward and CPU offload")

        # -----------------------------------------------------------------
        # Process whitening matrices one at a time.
        # -----------------------------------------------------------------

        for lstr, raw_xxt_cpu in pending_xxt:
            if is_v2:
                if raw_xxt_cpu.shape[0] > SOLVER_GPU_MAX_DIM:
                    print(
                        f"[WHITENING] Large matrix detected for {lstr}, "
                        f"routing eigh to in-process JAX GPU..."
                    )

                    L_s, U_s = eigh_jax_gpu_from_cpu(raw_xxt_cpu)

                else:
                    raw_xxt_gpu = raw_xxt_cpu.to(device)

                    L_s, U_s = torch.linalg.eigh(raw_xxt_gpu)
                    L_s = L_s.cpu()
                    U_s = U_s.cpu()

                    del raw_xxt_gpu
                    cuda_cleanup()

                L_s = L_s.flip(0).clamp(min=0.0)
                U_s = U_s.flip(1)

                wm = (U_s.to(torch.float32), L_s)

                del U_s, L_s

            else:
                if raw_xxt_cpu.shape[0] > SOLVER_GPU_MAX_DIM:
                    print(
                        f"[WHITENING] Large matrix ({raw_xxt_cpu.shape[0]}x{raw_xxt_cpu.shape[0]}) "
                        f"detected for {lstr}, routing Cholesky to CPU via scipy..."
                    )

                    raw_xxt_np = raw_xxt_cpu.numpy()

                    try:
                        wm = torch.from_numpy(
                            scipy.linalg.cholesky(
                                raw_xxt_np,
                                lower=True,
                                overwrite_a=False,
                            )
                        )
                    except scipy.linalg.LinAlgError:
                        print(f"[WARNING] Not positive-definite: {lstr}. Applying regularization.")

                        eigvals_np, _ = scipy.linalg.eigh(
                            raw_xxt_np,
                            driver="evd",
                            lower=True,
                        )

                        raw_xxt_np += (-eigvals_np[0] + 1e-6) * np.eye(
                            raw_xxt_np.shape[0],
                            dtype=raw_xxt_np.dtype,
                        )

                        wm = torch.from_numpy(
                            scipy.linalg.cholesky(
                                raw_xxt_np,
                                lower=True,
                                overwrite_a=True,
                            )
                        )

                        del eigvals_np

                    del raw_xxt_np

                else:
                    raw_xxt_gpu = raw_xxt_cpu.to(device)

                    try:
                        wm = torch.linalg.cholesky(raw_xxt_gpu).cpu()
                    except Exception:
                        print(f"[WARNING] Not positive-definite: {lstr}. Applying regularization.")

                        eigvals = torch.linalg.eigvalsh(raw_xxt_gpu)

                        raw_xxt_gpu += (-eigvals[0] + 1e-6) * torch.eye(
                            raw_xxt_gpu.shape[0],
                            dtype=raw_xxt_gpu.dtype,
                            device=device,
                        )

                        wm = torch.linalg.cholesky(raw_xxt_gpu).cpu()

                        del eigvals

                    del raw_xxt_gpu
                    cuda_cleanup()

            fname = lstr.replace(".", "_") + ".pt"
            fpath = os.path.join(wm_dir, fname)
            torch.save(wm, fpath)
            whitening_matrices_paths[lstr] = fpath

            print(f"[WHITENING] Saved {lstr} -> {fpath}")

            del wm
            del raw_xxt_cpu

            gc.collect()
            vram_usage(f"After saving whitening matrix for layer {lstr}")

        del pending_xxt

        cuda_cleanup()
        vram_usage(f"After decoder layer {idx} complete")

        # Thread activations forward.
        inps, outs = outs, inps  # pyright: ignore[reportAssignmentType]

    # ---------------------------------------------------------------------
    # Save checkpoint for inputs to end_layer, unless end_layer is final.
    # ---------------------------------------------------------------------

    if end_layer < num_decoder_layers:
        saved_ckpt_path = save_activation_checkpoint(act_ckpt_dir, model_name, version_str, n_tokens, end_layer, inps, captured)

        # The checkpoint we loaded for start_layer is no longer needed after
        # the next checkpoint has been safely written.
        if loaded_ckpt_path is not None and loaded_ckpt_path != saved_ckpt_path:
            try:
                if loaded_ckpt_path and os.path.exists(loaded_ckpt_path):
                    os.remove(loaded_ckpt_path)
                    print(f"[ACT-CKPT] Deleted {loaded_ckpt_path}")
            except Exception as e:
                print(f"[ACT-CKPT][WARNING] Could not delete {loaded_ckpt_path}: {e}")
    else:
        print("[ACT-CKPT] Reached final decoder layer. Deleting activation checkpoints.")
        try:
            for fname in os.listdir(act_ckpt_dir):
                if fname.startswith("inputs_to_layer_") and fname.endswith(".pt"):
                    fpath = os.path.join(act_ckpt_dir, fname)
                    try:
                        if os.path.exists(fpath):
                            os.remove(fpath)
                            print(f"[ACT-CKPT] Deleted {fpath}")
                    except Exception as e:
                        print(f"[ACT-CKPT][WARNING] Could not delete {fpath}: {e}")
        except FileNotFoundError:
            pass

    del inps, outs, captured

    cuda_cleanup()

    print(
        f"[WHITENING] Done for layer range [{start_layer}, {end_layer}). "
        f"{len(whitening_matrices_paths)} matrices saved to {wm_dir}"
    )

    return whitening_matrices_paths

def allocate_ratios(
        group_criterion: Union[GroupBy, Literal["global", "decoder", "type"]],
        score_map: Dict,
        layers_str: List[str],
        target_ratio: float,
        group_patterns: Dict[str, List[str]] | None = None,
        bypass_early_layers: int = 2,
        bypass_ratio: float = 0.0
) -> Dict[str, float]:
    """
    Redistributes compression budget within each weight group.
    Groups: MLP (gate, up, down), Q proj, K proj, V proj, Attention out proj.
    
    Within each group, matrices with higher score get a lower
    compression ratio and vice versa.

    Early layers (defined by bypass_early_layers) are mathematically isolated 
    from redistribution and strictly assigned the bypass_ratio. A bypass_ratio 
    of 0.0 means 0% parameter removal (no compression) for those layers.
    In case some layers are bypassed, it still preserves
    the global target_ratio across the entire model, 
    giving a higher compression ratio to allowed layers.
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
    print(f"[BUDGET] Bypassing first {bypass_early_layers} layers (assigning custom bypass_ratio: {bypass_ratio:.4f})")

    # Group weight matrices by desired criterion
    groups = defaultdict(list)
    unmatched_keys = []
    missing_score_keys = []
    ratio_map = {}

    # Filter and bypass early layers
    total_matrices = len(layers_str)
    active_layers = []
    bypassed_count = 0
    
    for key in layers_str:
        match = re.search(r'\.layers\.(\d+)\.', key)
        layer_idx = int(match.group(1)) if match else -1

        if 0 <= layer_idx < bypass_early_layers:
            ratio_map[key] = bypass_ratio
            bypassed_count += 1
        else:
            active_layers.append(key)

    active_count = len(active_layers)

    # Calculate dynamic adjusted compression ratio
    if active_count > 0:
        # Shift the target ratio for the remaining active layers to preserve the global target
        adjusted_target_ratio = ((total_matrices * target_ratio) - (bypassed_count * bypass_ratio)) / active_count
        
        # Guard against impossible mathematical boundaries
        if adjusted_target_ratio < 0.0 or adjusted_target_ratio > 1.0:
            print(f"[BUDGET][WARNING] Mathematical boundary exceeded! Adjusted target ratio for active layers is {adjusted_target_ratio:.4f}.")
            adjusted_target_ratio = max(1e-2, min(adjusted_target_ratio, 0.9))
            print(f"[BUDGET][WARNING] Clamped adjusted target ratio to {adjusted_target_ratio:.4f}")
    else:
        adjusted_target_ratio = target_ratio

    print(f"[BUDGET] Total Matrices: {total_matrices} | Global Target Ratio: {target_ratio:.4f}")
    print(f"[BUDGET] Bypassed Matrices: {bypassed_count} (Ratio: {bypass_ratio:.4f})")
    print(f"[BUDGET] Active Matrices: {active_count} | Adjusted Target Ratio: {adjusted_target_ratio:.4f}")

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
            ratio_map[keys[0]] = adjusted_target_ratio
            continue

        # Inverse-log normalization:
        #   high score  -> 1/log(scores) is small -> less compression (matrix is information-dense)
        #   low score   -> 1/log(scores) is large -> more compression (matrix is redundant)
        log_scores = torch.log(scores)
        inv_log_scores = 1.0 / log_scores
        normalized = inv_log_scores / inv_log_scores.sum()

        # Scale so that the mean ratio across the group equals `target_ratio`,
        # preserving the global memory budget
        ratios = inv_log_scores.shape[0] * adjusted_target_ratio * normalized

        print(f"  [GROUP: {group_name}] Redistributing over {len(keys)} layers (Mean target: {adjusted_target_ratio:.4f}):")
        for key, r in zip(keys, ratios.tolist()):
            # Clamp ratio to (0, 1) as a safety measure before refinement
            ratio_map[key] = max(1e-2, min(r, 0.9))
            print(f"    - {key:<50} | Ratio: {ratio_map[key]:.4f} | Score: {score_map[key]:.6f}")

    # Fallback for any unmatched layers
    final_fallbacks = []
    for key in active_layers:
        if key not in ratio_map:
            ratio_map[key] = adjusted_target_ratio
            # Identify the reason for the fallback
            reason = "Missing score data" if key in missing_score_keys else "No pattern match"
            final_fallbacks.append((key, reason))

    # --- Summary Report ---
    print(f"\n[BUDGET] Allocation Summary:")
    print(f"  - Protected Early Layers:           {bypassed_count} matrices (Fixed Ratio: {bypass_ratio:.4f})")
    print(f"  - Successfully Redistributed:       {len(ratio_map) - bypassed_count - len(final_fallbacks)} matrices (Mean Ratio: {adjusted_target_ratio:.4f})")
    print(f"  - Fallback (assigned adj_ratio):    {len(final_fallbacks)} matrices")
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
        hf_token: Optional[str] = None,
        whitening_only: bool = False,
        whitening_start_layer: int = 0,
        whitening_end_layer: Optional[int] = None,
        bypass_early_layers: int = 2,
        bypass_ratio: float = 0.0
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
        device_map=None,
        low_cpu_mem_usage=True,
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
            max(max_length * dataset["max_samples"], 1),
            device,
            is_v2,
            save_path or "./tmp",
            whitening_start_layer,
            whitening_end_layer
        )
    ram_usage("After loading whitening matrices")
    vram_usage("After loading whitening matrices")

    if whitening_only:
        print("[DEBUG] Whitening-only run complete. Exiting before compression/evaluation.")
        raise SystemExit(0)

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
            # Skip scoring phase for bypassed layers
            match = re.search(r'\.layers\.(\d+)\.', layers_str[i])
            layer_idx = int(match.group(1)) if match else -1
            if 0 <= layer_idx < bypass_early_layers:
                continue

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
        ratio_map = allocate_ratios(
            group_criterion, score_map, layers_str, ratio,
            group_patterns, bypass_early_layers, bypass_ratio
        )
        torch.cuda.empty_cache()
        steps_counter += 1
        ram_usage("After performing scores calculation")
        vram_usage("After performing scores calculation")
    else:
        # Replicate active ratio scaling for homogeneous path
        total_matrices = len(layers_str)
        bypassed_count = 0
        for k in layers_str:
            m = re.search(r'\.layers\.(\d+)\.', k)
            if m and 0 <= int(m.group(1)) < bypass_early_layers:
                bypassed_count += 1

        active_count = total_matrices - bypassed_count
        
        if active_count > 0:
            adjusted_target_ratio = ((total_matrices * ratio) - (bypassed_count * bypass_ratio)) / active_count
            adjusted_target_ratio = max(0.01, min(adjusted_target_ratio, 0.99))
        else:
            adjusted_target_ratio = ratio
            
        ratio_map = {}
        for k in layers_str:
            m = re.search(r'\.layers\.(\d+)\.', k)
            layer_idx = int(m.group(1)) if m else -1
            if 0 <= layer_idx < bypass_early_layers:
                ratio_map[k] = bypass_ratio
            else:
                ratio_map[k] = adjusted_target_ratio

    # Compress layers using the calculated compression ratios
    vram_usage("Before performing layer compression")
    for i, (layer, attr) in tqdm(
        enumerate(zip(layers_list, attributes)),
        total=len(layers_list),
        desc=f"Step {steps_counter}/{steps}: Compressing layers..."
    ):
        layer_ratio = ratio_map[layers_str[i]]
        # If the assigned ratio is exactly 0.0, skip SVD and preserve full-rank
        if layer_ratio == 0.0:
            continue

        # Get weight matrix
        layer_attr = getattr(layer, attr)
        W = layer_attr.weight.data.to(device, dtype=torch.float64)
        
        # Compute rank from compression ratio
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
        bypassed_layers_str = "_" + str(bypass_early_layers) if bypass_early_layers >= 0 else ""
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
           bypassed_layers_str +
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