import re
import gc
import math
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
from typing import Dict, Optional, List, Union, Literal, Tuple
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq # pyright: ignore[reportPrivateImportUsage]
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from torch.utils.data import DataLoader
from .utils import *
from .modules import *

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

# TODO pass random calibration sample to `check_layer_activation_error` (DIAGNOSTICS)

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

    # Set start and end decoder layers for whitening matrices generation
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

    # Try loading activation checkpoint for start_layer.
    loaded_ckpt, loaded_inps, loaded_captured, loaded_ckpt_path = try_load_activation_checkpoint(
        act_ckpt_dir, model_name, version_str, n_tokens, start_layer
    )

    if loaded_ckpt:
        inps: List[torch.Tensor] = loaded_inps # type: ignore[assignment]
        captured: List[Dict] = loaded_captured # type: ignore[assignment]
        loop_start_layer = start_layer
        print(f"[WHITENING] Starting directly from decoder layer {start_layer} using activation checkpoint.")
    else:
        # PHASE 1: Capture layer_0 inputs.
        print("[WHITENING] No usable activation checkpoint. Capturing layer_0 inputs...")

        # Move modules executed prior to the decoders on the device
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.to(device)

        captured: List[Dict] = []

        # Catch the layer_0 inputs
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
                captured.append({
                    "inp": inp.detach().cpu(),
                    "attention_mask": kwargs.get("attention_mask", None),
                    "position_ids": kwargs.get("position_ids", None),
                    "cache_position": kwargs.get("cache_position", None),
                    "position_embeddings": kwargs.get("position_embeddings", None),
                    "past_key_values": kwargs.get("past_key_values", None),
                })

                entry = captured[-1]

                if entry["attention_mask"] is not None:
                    entry["attention_mask"] = entry["attention_mask"].detach().cpu()

                if entry["position_ids"] is not None:
                    entry["position_ids"] = entry["position_ids"].detach().cpu()

                if entry["cache_position"] is not None:
                    entry["cache_position"] = entry["cache_position"].detach().cpu()

                if entry["position_embeddings"] is not None:
                    entry["position_embeddings"] = (
                        entry["position_embeddings"][0].detach().cpu(), 
                        entry["position_embeddings"][1].detach().cpu()
                    )

                if entry["past_key_values"] is not None:
                    entry["past_key_values"] = entry["past_key_values"].detach().cpu()

                raise CatcherExit(Exception)

        # Move layer_0 to device and replace it with catcher
        decoder_layers[0] = decoder_layers[0].to(device)
        original_layer0 = decoder_layers[0]
        decoder_layers[0] = Catcher(original_layer0)

        # Catch inputs
        with torch.no_grad():
            for batch in tqdm(loader, desc="Capturing layer_0 inputs"):
                try:
                    batch = {
                        k: v.to(device)
                        for k, v in batch.items()
                        if k in ("input_ids", "attention_mask")
                    }
                    model(**batch, use_cache=False)
                except CatcherExit as e:
                    pass
                finally:
                    del batch

        # Move layer_0 back to cpu
        decoder_layers[0] = original_layer0
        decoder_layers[0] = decoder_layers[0].cpu()

        # Move modules executed prior to the decoders back to cpu
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.cpu()

        cuda_cleanup()

        print(f"[WHITENING] Captured layer_0 inputs for {len(captured)} batches.")
        vram_usage("After layer 0 capture")

        # Move inputs to a dedicated list and empty the `entry["inp"]` tensors
        inps = [entry["inp"] for entry in captured]
        for entry in captured:
            entry["inp"] = None

        loop_start_layer = 0

    # Prepare outputs tensors list
    outs: List[Optional[torch.Tensor]] = [None] * len(inps)

    # Hook definition to accumulate XX^T for the desired sublayers
    def hook(module, input, output):
        inp = input[0].detach().to(dtype=module.raw_xxt_matrix.dtype)
        act = torch.einsum("bsi,bsj->ij", inp, inp)
        module.raw_xxt_matrix.add_(act)
        del inp, act

    whitening_matrices_paths: Dict[str, str] = {}

    # PHASE 2: Replay decoder layers.
    for idx in tqdm(range(loop_start_layer, end_layer), desc="Computing whitening matrices..."):
        should_save_this_layer = start_layer <= idx < end_layer

        print(f"[WHITENING] Processing decoder layer {idx} | save={should_save_this_layer}")
        
        # Move the decoder layer to device and get the desired sublayers
        layer = decoder_layers[idx].to(device)
        group = decoder_groups.get(idx, []) if should_save_this_layer else []

        # Initialize empty accumulation matrices and register forward hooks
        handles = []
        for local_path, lstr in group:
            la = get_submodule(layer, local_path)

            if isinstance(la, nn.Linear):
                la.raw_xxt_matrix = torch.zeros(
                    la.in_features,
                    la.in_features,
                    dtype=torch.float64, # accumulate in fp64 (lots of eigenvalues were negative with fp32 accumulation)
                    device=device,
                )
                handles.append(la.register_forward_hook(hook))

        # Replay every calibration batch through this decoder layer.
        with torch.no_grad():
            for j, entry in enumerate(captured):
                inp_j = inps[j].to(device)

                # Recover kwargs captured from layer_0
                kwargs = {}
                if entry["attention_mask"] is not None:
                    kwargs["attention_mask"] = entry["attention_mask"].to(device)

                if entry["position_ids"] is not None:
                    kwargs["position_ids"] = entry["position_ids"].to(device)

                if entry["cache_position"] is not None:
                    kwargs["cache_position"] = entry["cache_position"].to(device)

                if entry["position_embeddings"] is not None:
                    cos, sin = entry["position_embeddings"]
                    kwargs["position_embeddings"] = (
                        cos.to(device),
                        sin.to(device),
                    )

                if entry["past_key_values"] is not None:
                    kwargs["past_key_values"] = entry["past_key_values"].to(device)

                # Run sample through the decoder layer and save output
                out = layer(inp_j, use_cache=False, **kwargs)
                outs[j] = (out[0] if isinstance(out, tuple) else out).detach().cpu()

                del inp_j, out, kwargs

            torch.cuda.synchronize()

        # Remove registered forward hooks from decoder layer
        for h in handles:
            h.remove()

        # Move accumulated XXT matrices to CPU immediately.
        pending_xxt = []
        for local_path, lstr in group:
            la = get_submodule(layer, local_path)

            if not(isinstance(la, nn.Linear) and hasattr(la, "raw_xxt_matrix")):
                continue
            
            # Detach XXT matrix from graph, cast it to higher precision, compute covariance and move it to cpu
            raw_xxt_cpu = la.raw_xxt_matrix.detach().to(torch.float64).cpu() / n_tokens # pyright: ignore[reportCallIssue]

            # Check for unexpected asymmetries
            skew = (raw_xxt_cpu - raw_xxt_cpu.T).abs().max()
            scale = raw_xxt_cpu.abs().max().clamp_min(1e-12)
            rel_skew = skew / scale
            if rel_skew > 1e-5:
                print(f"[WARNING] XXT has unexpected asymmetry: rel_skew={rel_skew:.2e}")
            # Symmetrize to correct any eventual asymmetry
            raw_xxt_cpu = (raw_xxt_cpu + raw_xxt_cpu.transpose(0, 1)) * 0.5

            la.raw_xxt_matrix = None  # pyright: ignore[reportArgumentType]
            del la.raw_xxt_matrix

            pending_xxt.append((lstr, raw_xxt_cpu))

        # Move decoder layer itself back to CPU before eig/Cholesky work.
        decoder_layers[idx] = layer.cpu()
        del layer
        cuda_cleanup()

        vram_usage(f"After layer {idx} forward and CPU offload")

        # Process whitening matrices one at a time.
        for lstr, raw_xxt_cpu in pending_xxt:
            if is_v2:
                # Route to jax if the matrix is too large (pytorch fails due to a cuSolver index error)
                if raw_xxt_cpu.shape[0] > SOLVER_GPU_MAX_DIM:
                    print(
                        f"[WHITENING] Large matrix detected for {lstr}, "
                        f"routing eigh to in-process JAX GPU..."
                    )
                    L_s, U_s = eigh_jax_gpu_from_cpu(raw_xxt_cpu)
                else:
                    # Move XXT to device
                    raw_xxt_gpu = raw_xxt_cpu.to(device)

                    # Peform eigenvalue decomposition using `eigh` and move results to CPU
                    L_s, U_s = torch.linalg.eigh(raw_xxt_gpu)

                    del raw_xxt_gpu

                # Order eigenvalues/singular values in a descending order
                L_s = L_s.flip(0).cpu()
                U_s = U_s.flip(1).cpu()
                
                # Eigenvalues/singular values diagnostics
                neg = L_s[L_s < 0]
                if neg.numel() > 0:
                    rel_neg = neg.abs().max() / L_s.abs().max().clamp_min(1e-30)
                    print(
                        f"[V2][XXT] negative eigvals: {neg.numel()}/{L_s.numel()}, "
                        f"min={L_s.min().item():.3e}, rel_min={rel_neg.item():.3e}"
                    )

                # Put outputs into a tuple
                wm = (U_s, L_s)

                del U_s, L_s

            else:
                # Route to scipy on CPU if the matrix is too large (pytorch fails due to a cuSolver index error)
                if raw_xxt_cpu.shape[0] > SOLVER_GPU_MAX_DIM:
                    print(
                        f"[WHITENING] Large matrix ({raw_xxt_cpu.shape[0]}x{raw_xxt_cpu.shape[0]}) "
                        f"detected for {lstr}, routing Cholesky to CPU via scipy..."
                    )

                    raw_xxt_np = raw_xxt_cpu.numpy()

                    # Perform cholesky decomposition with safeguards (cholesky works only for PD matrices)
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
                    # Move XXT to device
                    raw_xxt_gpu = raw_xxt_cpu.to(device)

                    # Perform cholesky decomposition with safeguards (cholesky works only for PD matrices)
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

            fname = lstr.replace(".", "_") + ".pt"
            fpath = os.path.join(wm_dir, fname)
            torch.save(wm, fpath)
            whitening_matrices_paths[lstr] = fpath

            print(f"[WHITENING] Saved {lstr} -> {fpath}")

            del wm
            del raw_xxt_cpu
            cuda_cleanup()
            vram_usage(f"After saving whitening matrix for layer {lstr}")

        del pending_xxt

        cuda_cleanup()
        vram_usage(f"After decoder layer {idx} complete")

        # Thread activations forward.
        inps, outs = outs, inps  # pyright: ignore[reportAssignmentType]

    # Save checkpoint for inputs to end_layer, unless end_layer is final.
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


class FixedVLowRankUSolver:
    """
    Solves W_u in y ~= W_u(W_v x) while keeping W_v fixed.

    This mirrors the closed-form local_update helper present in the upstream
    SVDLLM.py file. The full paper pipeline updates both factors through the
    LoRA-based two-pass routine implemented below.
    """

    def __init__(
            self,
            layer_name: str,
            fixed_w_v: torch.Tensor,
            out_features: int,
            bias: Optional[torch.Tensor],
            device: str,
            ridge: float,
    ):
        self.layer_name = layer_name
        self.device = device
        self.ridge = ridge
        self.dtype = torch.float32
        self.fixed_w_v = fixed_w_v.detach().to(device=device, dtype=self.dtype).contiguous()
        self.rank = self.fixed_w_v.shape[0]
        self.gram = torch.zeros(self.rank, self.rank, dtype=self.dtype, device=device)
        self.cross = torch.zeros(self.rank, out_features, dtype=self.dtype, device=device)
        self.bias = bias.detach().to(device=device, dtype=self.dtype) if bias is not None else None

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        x = inp.detach().reshape(-1, inp.shape[-1]).to(device=self.device, dtype=self.dtype)
        y = out.detach().reshape(-1, out.shape[-1]).to(device=self.device, dtype=self.dtype)

        if self.bias is not None:
            y = y - self.bias

        z = torch.matmul(x, self.fixed_w_v.transpose(0, 1))
        self.gram.addmm_(z.transpose(0, 1), z)
        self.cross.addmm_(z.transpose(0, 1), y)

        del x, y, z

    @torch.no_grad()
    def solve(self, compressed_dtype: str) -> Tuple[torch.Tensor, torch.Tensor]:
        H = (self.gram + self.gram.transpose(0, 1)) * 0.5
        scale = torch.diag(H).abs().mean().clamp_min(1.0)
        eye = torch.eye(self.rank, dtype=self.dtype, device=self.device)

        W_u_t = None
        reg = self.ridge * scale
        for _ in range(5):
            try:
                W_u_t = torch.linalg.solve(H + reg * eye, self.cross)
                break
            except RuntimeError:
                reg = reg * 10.0

        if W_u_t is None:
            print(f"[SEQ-UPDATE][WARNING] Falling back to lstsq for {self.layer_name}")
            W_u_t = torch.linalg.lstsq(H + reg * eye, self.cross).solution

        out_dtype = DtypeMap.get_dtype(compressed_dtype)
        W_u = W_u_t.transpose(0, 1).detach().cpu().to(out_dtype).contiguous()
        W_v = self.fixed_w_v.detach().cpu().to(out_dtype).contiguous()

        del H, eye, W_u_t
        return W_u, W_v


class SequentialLocalUpdateRunner:
    """
    Owns the low-VRAM U-only sequential update workflow.

    This is useful when VRAM pressure is the primary constraint. For the
    paper/upstream-repo faithful U-then-V update, use run_sequential_lora_update.
    """

    def __init__(
            self,
            model: Qwen2ForCausalLM,
            loader: DataLoader,
            device: str,
            compressed_dtype: str,
            ridge: float = 1e-6,
            pin_cpu_offload: bool = False,
    ):
        self.model = model
        self.loader = loader
        self.device = device
        self.compressed_dtype = compressed_dtype
        self.ridge = ridge
        self.pin_cpu_offload = (
            pin_cpu_offload
            and torch.cuda.is_available()
            and str(device).startswith("cuda")
        )
        self.non_blocking = self.pin_cpu_offload

    def _cpu_offload_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.detach().cpu()
        if self.pin_cpu_offload:
            tensor = tensor.pin_memory()
        return tensor

    def _tree_to_cpu(self, value):
        if value is None:
            return None
        if torch.is_tensor(value):
            return self._cpu_offload_tensor(value)
        if isinstance(value, tuple):
            return tuple(self._tree_to_cpu(v) for v in value)
        if isinstance(value, list):
            return [self._tree_to_cpu(v) for v in value]
        return value

    def _tree_to_device(self, value):
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.to(self.device, non_blocking=self.non_blocking)
        if isinstance(value, tuple):
            return tuple(self._tree_to_device(v) for v in value)
        if isinstance(value, list):
            return [self._tree_to_device(v) for v in value]
        return value

    def _decoder_kwargs(self, entry: Dict) -> Dict:
        kwargs = {}
        for key in ("attention_mask", "position_ids", "cache_position", "position_embeddings", "past_key_values"):
            value = entry.get(key, None)
            if value is not None:
                kwargs[key] = self._tree_to_device(value)
        return kwargs

    def _capture_layer0_inputs(self) -> Tuple[List[torch.Tensor], List[Dict]]:
        decoder_layers = self.model.model.layers
        self.model.model.embed_tokens = self.model.model.embed_tokens.to(self.device)
        if hasattr(self.model.model, "rotary_emb"):
            self.model.model.rotary_emb = self.model.model.rotary_emb.to(self.device)

        captured: List[Dict] = []

        runner = self

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
                captured.append({
                    "inp": inp.detach().cpu(),
                    "attention_mask": runner._tree_to_cpu(kwargs.get("attention_mask", None)),
                    "position_ids": runner._tree_to_cpu(kwargs.get("position_ids", None)),
                    "cache_position": runner._tree_to_cpu(kwargs.get("cache_position", None)),
                    "position_embeddings": runner._tree_to_cpu(kwargs.get("position_embeddings", None)),
                    "past_key_values": runner._tree_to_cpu(kwargs.get("past_key_values", None)),
                })
                raise CatcherExit

        original_layer0 = decoder_layers[0].to(self.device)
        decoder_layers[0] = Catcher(original_layer0)

        try:
            for batch in tqdm(self.loader, desc="Capturing layer_0 inputs for local update"):
                try:
                    batch = {
                        k: v.to(self.device, non_blocking=self.non_blocking)
                        for k, v in batch.items()
                        if k in ("input_ids", "attention_mask")
                    }
                    self.model(**batch, use_cache=False)
                except CatcherExit:
                    pass
                finally:
                    del batch
        finally:
            decoder_layers[0] = original_layer0
            decoder_layers[0] = decoder_layers[0].cpu()
            self.model.model.embed_tokens = self.model.model.embed_tokens.cpu()
            if hasattr(self.model.model, "rotary_emb"):
                self.model.model.rotary_emb = self.model.model.rotary_emb.cpu()
            cuda_cleanup()

        inps = [entry["inp"] for entry in captured]
        for entry in captured:
            entry["inp"] = None

        print(f"[SEQ-UPDATE] Captured layer_0 inputs for {len(inps)} batches.")
        return inps, captured

    def _layer_hidden(self, layer: nn.Module, inp: torch.Tensor, entry: Dict) -> torch.Tensor:
        out = layer(inp.to(self.device, non_blocking=self.non_blocking), use_cache=False, **self._decoder_kwargs(entry))
        return out[0] if isinstance(out, tuple) else out

    @torch.no_grad()
    def _run_layer_for_hooks(self, layer: nn.Module, inps: List[torch.Tensor], captured: List[Dict]) -> None:
        for inp, entry in zip(inps, captured):
            hidden = self._layer_hidden(layer, inp, entry)
            del hidden

        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize()

    @torch.no_grad()
    def _replay_layer_to_cpu(self, layer: nn.Module, inps: List[torch.Tensor], captured: List[Dict]) -> List[torch.Tensor]:
        outs: List[Optional[torch.Tensor]] = [None] * len(inps)
        for idx, (inp, entry) in enumerate(zip(inps, captured)):
            hidden = self._layer_hidden(layer, inp, entry)
            outs[idx] = self._cpu_offload_tensor(hidden)
            del hidden

        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize()

        return outs  # type: ignore[return-value]

    def _groups(self, layers_str: List[str]) -> Dict[int, List[Tuple[str, str]]]:
        groups: Dict[int, List[Tuple[str, str]]] = defaultdict(list)
        for lstr in layers_str:
            match = re.search(r"model\.layers\.(\d+)\.(.*)", lstr)
            if match is not None:
                groups[int(match.group(1))].append((match.group(2), lstr))
        return groups

    @staticmethod
    def _parent_and_attr(layer: nn.Module, local_path: str):
        if "." not in local_path:
            return layer, local_path
        parent_path, attr = local_path.rsplit(".", 1)
        return get_submodule(layer, parent_path), attr

    def _register_solver_hook(self, module: nn.Linear, solver: FixedVLowRankUSolver):
        def add_batch(_, inp, out):
            solver.add_batch(inp[0], out)
        return module.register_forward_hook(add_batch)

    @torch.no_grad()
    def update_taw_only_checkpoint(
            self,
            dense_reference_model: Qwen2ForCausalLM,
            layers_str: List[str],
    ) -> Dict[str, int]:
        rank_map: Dict[str, int] = {}
        student_layers = self.model.model.layers
        dense_reference_layers = dense_reference_model.model.layers
        groups = self._groups(layers_str)

        print("[SEQ-UPDATE] Starting sequential update for TAW-only compressed model.")
        inps, captured = self._capture_layer0_inputs()
        vram_usage("After compressed-update layer_0 capture")

        for idx in tqdm(range(len(student_layers)), desc="Updating compressed decoder layers"):
            student_layer_cpu = student_layers[idx]
            group = groups.get(idx, [])

            if group:
                dense_reference_layer = dense_reference_layers[idx].to(self.device)

                # Lowest-VRAM mode: one selected matrix at a time. This repeats
                # the dense-reference layer replay, but avoids keeping solver
                # buffers for every projection in the decoder layer at once.
                for local_path, lstr in group:
                    student_parent, attr = self._parent_and_attr(student_layer_cpu, local_path)
                    dense_reference_parent, _ = self._parent_and_attr(dense_reference_layer, local_path)
                    lowrank = getattr(student_parent, attr)
                    dense_reference_linear = getattr(dense_reference_parent, attr)

                    if not isinstance(lowrank, LowRank) or not isinstance(dense_reference_linear, nn.Linear):
                        print(f"[SEQ-UPDATE][WARNING] Skipping {lstr}: expected LowRank student and Linear dense reference.")
                        continue

                    fixed_w_v = lowrank.W_v.weight.detach()
                    solver = FixedVLowRankUSolver(
                        layer_name=lstr,
                        fixed_w_v=fixed_w_v,
                        out_features=dense_reference_linear.out_features,
                        bias=dense_reference_linear.bias,
                        device=self.device,
                        ridge=self.ridge,
                    )
                    handle = self._register_solver_hook(dense_reference_linear, solver)
                    self._run_layer_for_hooks(dense_reference_layer, inps, captured)
                    handle.remove()

                    W_u, W_v = solver.solve(self.compressed_dtype)
                    lowrank.W_u.weight.copy_(W_u.to(lowrank.W_u.weight.device, dtype=lowrank.W_u.weight.dtype))
                    lowrank.W_v.weight.copy_(W_v.to(lowrank.W_v.weight.device, dtype=lowrank.W_v.weight.dtype))
                    if lowrank.W_u.bias is not None and dense_reference_linear.bias is not None:
                        lowrank.W_u.bias.copy_(dense_reference_linear.bias.detach().cpu().to(lowrank.W_u.bias.dtype))
                    if self.compressed_dtype in ("float16", "fp16"):
                        factor_range_report(lstr, W_u, W_v)

                    rank_map[lstr] = fixed_w_v.shape[0]
                    del W_u, W_v, solver, handle
                    cuda_cleanup()

                dense_reference_layers[idx] = dense_reference_layer.cpu()
                del dense_reference_layer
                cuda_cleanup()

            student_layer = student_layer_cpu.to(self.device)
            outs = self._replay_layer_to_cpu(student_layer, inps, captured)
            student_layers[idx] = student_layer.cpu()
            del student_layer, student_layer_cpu, inps
            inps = outs

            cuda_cleanup()
            vram_usage(f"After compressed update decoder layer {idx}")

        del inps, captured
        cuda_cleanup()
        print("[SEQ-UPDATE] TAW-only checkpoint sequential local update complete.")
        return rank_map


def run_sequential_lora_update(
        model: Qwen2ForCausalLM,
        loader: DataLoader,
        device: str,
        layers_str: Optional[List[str]] = None,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        epochs: int = 2,
        max_steps: Optional[int] = None,
        grad_accum_steps: int = 1,
        gradient_checkpointing: bool = False,
) -> Qwen2ForCausalLM:
    """
    Sequential low-rank update.

    First LoRA-tune the U-side compressed projections, merge the adapter,
    then LoRA-tune the V-side projections and merge again.
    Each compressed projection is a LowRank module containing W_u and W_v, so
    the two phases target those inner Linear modules directly.
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "The paper-faithful sequential update mode requires `peft`. "
            "Install it in this environment, or use "
            "`--sequential_update_method local_u` for the low-VRAM closed-form path."
        ) from exc

    if max_steps is not None and max_steps <= 0:
        max_steps = None
    if max_steps is None and epochs <= 0:
        raise ValueError("LoRA sequential update needs epochs > 0 or max_steps > 0.")

    grad_accum_steps = max(1, int(grad_accum_steps))
    selected_layers = set(layers_str or [])
    non_blocking = str(device).startswith("cuda") and torch.cuda.is_available()

    def factor_targets(factor_name: str) -> List[str]:
        suffix = f".{factor_name}"
        targets = []
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear) or not name.endswith(suffix):
                continue
            lowrank_name = name[:-len(suffix)]
            if selected_layers and lowrank_name not in selected_layers:
                continue
            targets.append(name)
        return targets

    original_use_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = False
    model = model.to(device) # pyright: ignore[reportArgumentType]

    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    phases = [
        ("first_half", "W_u"),
        ("second_half", "W_v"),
    ]

    for phase_name, factor_name in phases:
        target_modules = factor_targets(factor_name)
        if not target_modules:
            raise RuntimeError(
                f"No LoRA targets found for {factor_name}. "
                "Expected compressed LowRank modules with inner W_u/W_v linears."
            )

        print(
            f"[SEQ-UPDATE][LoRA] {phase_name}: tuning {factor_name} "
            f"for {len(target_modules)} compressed projections."
        )

        model.requires_grad_(False)
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config) # pyright: ignore[reportAssignmentType]
        model.train()

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in trainable_params)
        if trainable_count == 0:
            raise RuntimeError(f"LoRA phase {phase_name} produced no trainable parameters.")
        print(f"[SEQ-UPDATE][LoRA] Trainable parameters: {trainable_count:,}")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        optimizer.zero_grad(set_to_none=True)

        optimizer_steps = 0
        micro_steps = 0
        skipped_all_masked_batches = 0
        should_stop = False
        total_epochs = 10**12 if max_steps is not None else max(1, int(epochs))

        for epoch_idx in range(total_epochs):
            progress = tqdm(
                loader,
                desc=f"Sequential LoRA {phase_name} epoch {epoch_idx + 1}",
            )
            for batch in progress:
                batch = {
                    k: v.to(device, non_blocking=non_blocking)
                    for k, v in batch.items()
                    if k in ("input_ids", "attention_mask", "labels")
                }
                labels = batch.get("labels", None)
                if labels is None:
                    labels = batch["input_ids"].clone()
                    if "attention_mask" in batch:
                        labels = labels.masked_fill(batch["attention_mask"].eq(0), -100)

                if not torch.any(labels.ne(-100)):
                    skipped_all_masked_batches += 1
                    del batch, labels
                    continue

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask", None),
                    labels=labels,
                    use_cache=False,
                )
                loss = outputs.loss / grad_accum_steps
                loss.backward()
                micro_steps += 1

                if micro_steps % grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                    progress.set_postfix(
                        loss=f"{(loss.detach().item() * grad_accum_steps):.4f}",
                        step=optimizer_steps,
                    )

                    if max_steps is not None and optimizer_steps >= max_steps:
                        should_stop = True

                del batch, labels, outputs, loss

                if should_stop:
                    break

            if should_stop:
                break

        if micro_steps > 0 and micro_steps % grad_accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        if optimizer_steps == 0:
            raise RuntimeError(
                f"LoRA phase {phase_name} completed zero optimizer steps. "
                "Check that the finetuning dataset has non-masked labels."
            )

        if skipped_all_masked_batches > 0:
            print(
                f"[SEQ-UPDATE][LoRA][WARNING] Skipped {skipped_all_masked_batches} "
                "batches because every label was -100."
            )

        print(f"[SEQ-UPDATE][LoRA] {phase_name}: completed {optimizer_steps} optimizer steps.")
        model.eval()
        model = model.merge_and_unload() # pyright: ignore[reportCallIssue]
        model.requires_grad_(False)

        del optimizer, trainable_params
        cuda_cleanup()

    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    if original_use_cache is not None:
        model.config.use_cache = original_use_cache

    model = model.cpu()
    cuda_cleanup()
    print("[SEQ-UPDATE][LoRA] Sequential U-then-V update complete.")
    return model
 
def allocate_ratios(
        group_criterion: Union[GroupBy, Literal["global", "decoder", "type"]],
        score_map: Dict,
        layers_str: List[str],
        target_ratio: float,
        param_count_map: Dict[str, int],
        group_patterns: Dict[str, List[str]] | None = None,
        bypass_early_layers: int = 2,
        bypass_ratio: float = 0.0,
        max_ratio: float = 0.9,
        target_total_params: Optional[int] = None
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

    For same-shape TYPE groups, this reduces to the usual V2 behavior.
    For GLOBAL and DECODER groups, this preserves actual removed parameters.
    """
    if isinstance(group_criterion, str):
        try:
            group_criterion = GroupBy(group_criterion)
        except ValueError:
            raise ValueError(
                f"Invalid `group_criterion`: '{group_criterion}'. "
                f"Expected one of: {[e.value for e in GroupBy]}"
            )

    print(f"\n[BUDGET] Parameter-aware redistribution: {group_criterion.value.upper()}")
    print(f"[BUDGET] Global target ratio: {target_ratio:.6f}")
    print(f"[BUDGET] Bypassing first {bypass_early_layers} layers with ratio {bypass_ratio:.6f}")

    ratio_map: Dict[str, float] = {}

    selected_total_params = sum(param_count_map[k] for k in layers_str)

    if target_total_params is None:
        target_total_params = selected_total_params

    target_removed = target_ratio * target_total_params

    bypassed_keys = [
        k for k in layers_str
        if is_bypassed_key(k, bypass_early_layers)
    ]

    active_keys = [
        k for k in layers_str
        if not is_bypassed_key(k, bypass_early_layers)
    ]

    bypassed_removed = 0.0
    for k in bypassed_keys:
        ratio_map[k] = bypass_ratio
        bypassed_removed += param_count_map[k] * bypass_ratio

    active_params = sum(param_count_map[k] for k in active_keys)
    active_budget = target_removed - bypassed_removed
    active_capacity = active_params * max_ratio

    if active_budget < 0:
        print(
            f"[BUDGET][WARNING] Bypass budget exceeds target. "
            f"active_budget={active_budget:.2f}; setting active budget to 0."
        )
        active_budget = 0.0

    if active_budget > active_capacity:
        print(
            f"[BUDGET][WARNING] Active budget exceeds max capacity. "
            f"requested={active_budget:.2f}, capacity={active_capacity:.2f}. Clamping."
        )
        active_budget = active_capacity

    active_target_ratio = active_budget / active_params if active_params > 0 else 0.0

    print(f"[BUDGET] Selected params:           {selected_total_params:,}")
    print(f"[BUDGET] Target denominator params: {target_total_params:,}")
    print(f"[BUDGET] Target removed params:     {target_removed:,.0f}")
    print(f"[BUDGET] Bypassed matrices:         {len(bypassed_keys)}")
    print(f"[BUDGET] Bypassed removed params:   {bypassed_removed:,.0f}")
    print(f"[BUDGET] Active matrices:           {len(active_keys)}")
    print(f"[BUDGET] Active params:             {active_params:,}")
    print(f"[BUDGET] Active budget:             {active_budget:,.0f}")
    print(f"[BUDGET] Active target ratio:       {active_target_ratio:.6f}")

    groups: Dict[str, List[str]] = defaultdict(list)
    missing_score_keys = []
    unmatched_keys = []

    match group_criterion:
        case GroupBy.GLOBAL:
            for key in active_keys:
                if key in score_map:
                    groups["global"].append(key)
                else:
                    missing_score_keys.append(key)

        case GroupBy.DECODER:
            for key in active_keys:
                if key not in score_map:
                    missing_score_keys.append(key)
                    continue

                layer_idx = get_layer_idx_from_key(key)
                groups[f"layer_{layer_idx}"].append(key)

        case GroupBy.TYPE:
            if group_patterns is None:
                raise ValueError("`group_patterns` required for GroupBy.TYPE")

            for key in active_keys:
                if key not in score_map:
                    missing_score_keys.append(key)
                    continue

                group_name = None
                for name, patterns in group_patterns.items():
                    if any(p in key for p in patterns):
                        group_name = name
                        break

                if group_name is None:
                    unmatched_keys.append(key)
                else:
                    groups[group_name].append(key)

    grouped_keys = set()
    for keys in groups.values():
        grouped_keys.update(keys)

    fallback_keys = [k for k in active_keys if k not in grouped_keys]

    # Fallback keys get the active target ratio.
    fallback_removed = 0.0
    for k in fallback_keys:
        ratio_map[k] = active_target_ratio
        fallback_removed += param_count_map[k] * active_target_ratio

    remaining_budget = max(0.0, active_budget - fallback_removed)
    grouped_params = sum(param_count_map[k] for k in grouped_keys)

    print(f"[BUDGET] Grouped active params:    {grouped_params:,}")
    print(f"[BUDGET] Fallback keys:            {len(fallback_keys)}")
    print(f"[BUDGET] Remaining group budget:   {remaining_budget:,.0f}")

    for group_name, keys in groups.items():
        if not keys:
            print(f"  [GROUP: {group_name}] Empty")
            continue

        group_params = sum(param_count_map[k] for k in keys)

        # Allocate the global active budget to groups proportional to params.
        group_budget = (
            remaining_budget * group_params / grouped_params
            if grouped_params > 0
            else 0.0
        )

        # TODO understand well how it works
        group_ratio_map = allocate_param_weighted_group(
            keys=keys,
            score_map=score_map,
            param_count_map=param_count_map,
            group_budget=group_budget,
            max_ratio=max_ratio,
            offset=1.5
        )

        ratio_map.update(group_ratio_map)

        actual_group_removed = sum(
            param_count_map[k] * ratio_map[k]
            for k in keys
        )

        print(
            f"  [GROUP: {group_name}] "
            f"matrices={len(keys):>3} | "
            f"params={group_params:>14,} | "
            f"budget={group_budget:>14,.0f} | "
            f"actual_removed≈{actual_group_removed:>14,.0f}"
        )

        for k in keys:
            print(
                f"    - {k:<55} "
                f"| params={param_count_map[k]:>12,} "
                f"| ratio={ratio_map[k]:.6f} "
                f"| score={score_map[k]:.6f}"
            )

    actual_removed = sum(
        param_count_map[k] * ratio_map.get(k, 0.0)
        for k in layers_str
    )

    print("\n[BUDGET] Allocation Summary:")
    print(f"  - Target overall ratio:                 {target_ratio:.6f}")
    print(f"  - Actual selected ratio approx: {actual_removed / selected_total_params:.6f}")
    print(f"  - Actual overall ratio approx:  {actual_removed / target_total_params:.6f}")
    print(f"  - Target removed:               {target_removed:,.0f}")
    print(f"  - Actual removed:               {actual_removed:,.0f}")
    print(f"  - Missing score keys:           {len(missing_score_keys)}")
    print(f"  - Unmatched keys:               {len(unmatched_keys)}")
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
        compressed_dtype: str = "float16",
        batch_size: int = 32,
        seed: Optional[int] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_path: Optional[str] = None,
        whitening_mat_path: Optional[str] = None,
        compress_mlp: bool = False,
        compress_att_q: bool = False,
        compress_att_k: bool = False,
        compress_att_v: bool = False,
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
        bypass_ratio: float = 0.0,
        ratio_scope: Literal["selected", "all"] = "selected",
        eps: float = 1e-6,
        sequential_update: bool = False,
        sequential_update_ridge: float = 1e-6,
        sequential_update_method: Literal["lora", "local_u"] = "lora",
        sequential_lora_r: int = 8,
        sequential_lora_alpha: int = 16,
        sequential_lora_dropout: float = 0.05,
        sequential_lora_lr: float = 1e-4,
        sequential_lora_weight_decay: float = 0.0,
        sequential_lora_epochs: int = 2,
        sequential_lora_max_steps: Optional[int] = None,
        sequential_lora_grad_accum_steps: Optional[int] = None,
        sequential_lora_effective_batch_size: int = 64,
        sequential_lora_gradient_checkpointing: bool = False,
        finetune_dataset: str = "yahma/alpaca-cleaned",
        max_finetune_samples: int = 50000,
        finetune_cutoff_len: int = 256,
        finetune_train_on_inputs: bool = False,
        finetune_add_eos_token: bool = False,
        pin_cpu_offload: bool = False
):
    # Load model and tokenizer
    vram_usage("Before loading original model")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model: Qwen2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=None,
        max_position_embeddings=max_length, # Set desired max model length
        use_cache=False, # Disable kv cache (not needed here)
        low_cpu_mem_usage=True,
        use_safetensors=True,
        token=hf_token,
        trust_remote_code=True
    ) # type: ignore
    ram_usage("After loading original model")
    vram_usage("After loading original model")

    # Set model to evaluation mode
    model.eval()

    # Inspect uncompressed model logits
    logits_debug(model, tokenizer, "The responsibility of an AI assistant is", "cpu")

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
    calibration_dataloader = DataLoader(
        calibration_dataset, # pyright: ignore[reportArgumentType]
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_cpu_offload and str(device).startswith("cuda"),
    )
    ram_usage("After loading dataset")
    vram_usage("After loading dataset")
    print("=== FINAL DATASET STRUCTURE ===")
    print(calibration_dataset)

    # Get list of sublayers that we want to compress
    layers_str = generate_paths(
        compress_mlp, 
        compress_att_q,
        compress_att_k,
        compress_att_v,
        compress_att_out, 
        layers_number=model.config.num_hidden_layers
    )
    if len(layers_str) == 0:
        raise ValueError("No layers selected for compression.")
    layers_list, attributes = get_layers(model, layers_str, True)

    # Build parameters count map
    param_count_map = build_param_count_map(
        layers_str=layers_str,
        layers_list=layers_list,
        attributes=attributes,
        include_bias=False,
    )

    # Overall count of parameters of sublayers that we want to compress
    selected_total_params = sum(param_count_map[k] for k in layers_str)

    if ratio_scope == "all":
        # Denominator includes all projection matrices we conceptually care about:
        # MLP + q/k/v + o.
        # We repeat the same process used to get the count of parameters
        # of sublayers that we want to compress, but we select all sublayers
        budget_layers_str = generate_paths(
            mlp=True,
            q=True,
            k=True,
            v=True,
            attention_output=True,
            layers_number=model.config.num_hidden_layers,
        )
        budget_layers_list, budget_attributes = get_layers(model, budget_layers_str, True)

        budget_param_count_map = build_param_count_map(
            layers_str=budget_layers_str,
            layers_list=budget_layers_list,
            attributes=budget_attributes,
            include_bias=False,
        )

        # Count of parameters of all targetable sublayers
        target_total_params = sum(budget_param_count_map[k] for k in budget_layers_str)

        print("\n[BUDGET] Ratio scope: ALL")
        print(f"[BUDGET] Selected compressible params: {selected_total_params:,}")
        print(f"[BUDGET] Target denominator params:    {target_total_params:,}")
        print(f"[BUDGET] Selected fraction:           {selected_total_params / target_total_params:.6f}")
        print(
            f"[BUDGET] If homogeneous over selected only, needed selected ratio to reach the desired overall compression ratio of {ratio} would be "
            f"{(ratio * target_total_params / selected_total_params):.6f}"
        )
    else:
        target_total_params = selected_total_params

        print("\n[BUDGET] Ratio scope: SELECTED")
        print(f"[BUDGET] Selected compressible params: {selected_total_params:,}")
        print(f"[BUDGET] Target denominator params:    {target_total_params:,}")

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
            print(f"[WARNING] Whitening matrices for this model do not exist in this path: {whitening_mat_actual_path}. Generating in place...")
            print("=== WHITENING MATRICES GENERATION ===")
            whitening_matrices = get_whitening_matrices(
                model,
                model_name,
                calibration_dataloader,
                layers_str,
                max(max_length * dataset["max_samples"], 1), # overall count of tokens of the calibration dataset
                device,
                is_v2,
                save_path or "./tmp",
                whitening_start_layer,
                whitening_end_layer
            )
    else:
        print("=== WHITENING MATRICES GENERATION ===")
        whitening_matrices = get_whitening_matrices(
            model,
            model_name,
            calibration_dataloader,
            layers_str,
            max(max_length * dataset["max_samples"], 1), # overall count of tokens of the calibration dataset
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

    # TODO put all into one loop, exclude scoring pass for homogeneous. Perform only one svd of D, there's no need of doing one svdvals and one svd
    # TODO try with covariance matrix - done, to monitor
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
        score_probe_ratio = compute_active_target_ratio(
            layers_str=layers_str,
            param_count_map=param_count_map,
            target_ratio=ratio,
            bypass_early_layers=bypass_early_layers,
            bypass_ratio=bypass_ratio,
            max_ratio=0.9,
            target_total_params=target_total_params
        )
        print(f"[BUDGET] Score probe ratio for active layers: {score_probe_ratio:.6f}")
        with torch.no_grad():
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
                W = layer_attr.weight.detach().to(device, dtype=torch.float64)

                if is_v2:
                    # Perform SVD on whitening matrix (S)
                    U_s, L_s = load_whitening_data(whitening_matrices, layers_str[i], device, keep=True)

                    # Auxiliary matrix
                    L_s_sqrt = torch.sqrt(L_s.clamp_min(eps))


                    # Perform SVD on W x U_s x sqrt(L_s)
                    C_sqrt = U_s * L_s_sqrt.unsqueeze(0)
                    D = torch.matmul(W, C_sqrt)
                    # Calculate singular values only
                    L = torch.linalg.svdvals(D)
                else:
                    whitening_matrix = load_whitening_data(whitening_matrices, layers_str[i], device, keep=True)
                    # Perform SVD on W x Chol(XXT)
                    WS = torch.matmul(W, whitening_matrix) # pyright: ignore[reportArgumentType]
                    # Calculate singular values only
                    L = torch.linalg.svdvals(WS)

                # Compute a tentative rank under the uniform target ratio.
                rank = int(
                    (W.shape[0] * W.shape[1] * (1.0 - score_probe_ratio))
                    / (W.shape[0] + W.shape[1])
                )
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
                        # After whitening, entropy loss equals the sum of normalized singular values of the tail - TODO what if we don't normalize?
                        norm_spectrum = L/L.sum()
                        score_map[layers_str[i]] = -(norm_spectrum[rank:] * torch.log(norm_spectrum[rank:] + 1e-9)).sum().item()
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
            group_criterion=group_criterion,
            score_map=score_map,
            layers_str=layers_str,
            target_ratio=ratio,
            param_count_map=param_count_map,
            group_patterns=group_patterns,
            bypass_early_layers=bypass_early_layers,
            bypass_ratio=bypass_ratio,
            max_ratio=0.9,
            target_total_params=target_total_params
        )
        torch.cuda.empty_cache()
        steps_counter += 1
        ram_usage("After performing scores calculation")
        vram_usage("After performing scores calculation")
    else:
        active_target_ratio = compute_active_target_ratio(
            layers_str=layers_str,
            param_count_map=param_count_map,
            target_ratio=ratio,
            bypass_early_layers=bypass_early_layers,
            bypass_ratio=bypass_ratio,
            max_ratio=0.9,
            target_total_params=target_total_params
        )

        ratio_map = {}

        for k in layers_str:
            if is_bypassed_key(k, bypass_early_layers):
                ratio_map[k] = bypass_ratio
            else:
                ratio_map[k] = active_target_ratio

        selected_total_params = sum(param_count_map[k] for k in layers_str)
        actual_removed = sum(param_count_map[k] * ratio_map[k] for k in layers_str)

        print("\n[BUDGET] Homogeneous Allocation Summary:")
        print(f"  - Target overall ratio:          {ratio:.6f}")
        print(f"  - Active selected ratio:         {active_target_ratio:.6f}")
        print(f"  - Actual selected ratio approx:  {actual_removed / selected_total_params:.6f}")
        print(f"  - Actual overall ratio approx:   {actual_removed / target_total_params:.6f}")
        print(f"  - Target removed:                {ratio * target_total_params:,.0f}")
        print(f"  - Actual removed:                {actual_removed:,.0f}")
        print("-" * 80 + "\n")

    # Compress layers using the calculated compression ratios
    vram_usage("Before performing layer compression")
    with torch.no_grad():
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
            W = layer_attr.weight.detach().to(device, dtype=torch.float64)
            
            # Compute rank from compression ratio
            rank = int((W.shape[0] * W.shape[1] * (1 - layer_ratio)) / (W.shape[0] + W.shape[1])) # TODO restore rank compression
            #rank = min(W.shape[0], W.shape[1])
            
            if is_v2:
                # heterogeneous-v2 path - stream U_s and L_s calculated during the previous steps
                U_s, L_s = load_whitening_data(whitening_matrices, layers_str[i], device, keep=False)
                L_s_clean = L_s.clamp_min(eps)

                # Auxiliary matrix
                L_s_sqrt = torch.sqrt(L_s_clean)            

                # Perform SVD on W x U_s x sqrt(L_s)
                C_sqrt = U_s * L_s_sqrt.unsqueeze(0)
                D = torch.matmul(W, C_sqrt)
                # Free W as soon as D is ready
                W = None
                del W

                U_ws, L_ws, V_wsT = torch.linalg.svd(D, full_matrices=False)
                # Free D as soon as U_ws, L_ws and V_wsT are ready
                D = None
                del D

                # Calculate 1/sqrt(L_s)
                L_s_sqrt_inv = torch.rsqrt(L_s_clean)

                # Free U_s and L_s
                L_s = L_s_sqrt = None
                del L_s, L_s_sqrt

                # Calculate final rank and truncate matrices
                #rank = max(1, min(rank, L_ws.shape[0])) # TODO restore rank compression
                rank = max(1, min(rank, L_ws.shape[0] - 1))
                rank_map[layers_str[i]] = rank
                U_ws_r = U_ws[:, :rank].contiguous()
                L_ws_r = L_ws[:rank].clone()
                V_wsT_r = V_wsT[:rank, :].contiguous()
                L_ws_r_sqrt = torch.sqrt(L_ws_r)

                # Free full-rank matrices as soon as truncated slices are built
                U_ws = L_ws = V_wsT = None
                del U_ws, L_ws, V_wsT

                # Compute approximate weight matrix, split in two matrices
                W_u = (U_ws_r * L_ws_r_sqrt.unsqueeze(0)).cpu().to(DtypeMap.get_dtype(compressed_dtype)).contiguous()
                W_v = (L_ws_r_sqrt.unsqueeze(1) * torch.matmul((V_wsT_r * L_s_sqrt_inv.unsqueeze(0)), U_s.transpose(0, 1))).cpu().to(DtypeMap.get_dtype(compressed_dtype)).contiguous()
                # Free low-rank matrices, leave only W_u and W_v
                U_s = L_s_sqrt_inv = U_ws_r = L_ws_r = V_wsT_r = L_ws_r_sqrt = None
                del U_ws_r, L_ws_r, V_wsT_r, L_ws_r_sqrt
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
                U_r = U[:, :rank].contiguous()
                L_r_sqrt = torch.sqrt(L[:rank].clone())
                VT_r = torch.matmul(VT[:rank, :].contiguous(), whitening_matrix_inv)
                # Free full-rank matrices as soon as truncated slices are built
                U = L = VT = whitening_matrix_inv = None
                del U, L, VT, whitening_matrix_inv

                # Compute approximate weight matrix, split in two matrices
                W_u = (U_r * L_r_sqrt.unsqueeze(0)).cpu().to(DtypeMap.get_dtype(compressed_dtype))
                W_v = (VT_r * L_r_sqrt.unsqueeze(1)).cpu().to(DtypeMap.get_dtype(compressed_dtype))
                # Free low-rank matrices, leave only W_u and W_v
                U_r = VT_r = L_r_sqrt = None
                del U_r, VT_r, L_r_sqrt

            # Replace the original nn.Linear with the LowRank module, which implements the forward pass as W_u(W_v(x)).
            van = LowRank(
                layer_attr.in_features,
                layer_attr.out_features,
                rank,
                layer_attr.bias is not None
            ).to(device="cpu", dtype=DtypeMap.get_dtype(compressed_dtype))
            van.requires_grad_(False)

            van.W_u.weight.copy_(W_u)
            van.W_v.weight.copy_(W_v)
            if layer_attr.bias is not None:
                van.W_u.bias.copy_(layer_attr.bias.detach().to(DtypeMap.get_dtype(compressed_dtype)))

            # Overflow check for fp16 case only + throw error if any value is not finite
            if compressed_dtype == "float16" or compressed_dtype == "fp16":
                factor_range_report(layers_str[i], W_u, W_v)

            # Check relative diference between lowrank and original weight matrix
            check_weights_relative_difference(
                layers_str[i], 
                layer_attr, 
                van, 
                device="cuda"
            )

            # Check lowrank module equivalence to a single nn.Linear (uses the compressed matrices in both cases)
            check_lowrank_equivalence(
                layers_str[i], 
                layer_attr, 
                van, 
                device="cuda"
            )

            # Check activation relative error between compressed and original layer
            check_layer_activation_error(
                layers_str[i],
                layer_attr,
                van,
                device=device
            )

            setattr(layer, attr, van)

            # Free ram and vram from all leftover matrices
            W_u = W_v = None
            del W_u, W_v

    sequential_update_samples: Optional[int] = None

    if sequential_update:
        print("[SEQ-UPDATE] TAW compression complete. Starting sequential low-rank update.")
        update_layers = [layer_key for layer_key in layers_str if layer_key in rank_map]

        if not update_layers:
            raise RuntimeError(
                "Sequential update was requested, but no TAW-compressed LowRank "
                "layers were produced. Check compression ratio and bypass settings."
        )

        if sequential_update_method == "lora":
            if sequential_lora_grad_accum_steps is None:
                sequential_lora_grad_accum_steps = max(
                    1,
                    math.ceil(sequential_lora_effective_batch_size / batch_size),
                )
            actual_effective_batch_size = batch_size * sequential_lora_grad_accum_steps
            print(
                "[SEQ-UPDATE][LoRA] "
                f"micro_batch_size={batch_size}, "
                f"grad_accum_steps={sequential_lora_grad_accum_steps}, "
                f"effective_batch_size={actual_effective_batch_size}"
            )
            print("[SEQ-UPDATE][LoRA] Loading fine-tuning dataset.")
            finetune_dataset_tokenized, sequential_update_samples = tokenize_finetune_dataset(
                dataset_spec=finetune_dataset,
                tokenizer=tokenizer,
                max_samples=max_finetune_samples,
                cutoff_len=finetune_cutoff_len,
                seed=seed,
                train_on_inputs=finetune_train_on_inputs,
                add_eos_token=finetune_add_eos_token,
            )
            finetune_collator = DataCollatorForSeq2Seq(
                tokenizer,
                pad_to_multiple_of=8,
                return_tensors="pt",
                padding=True,
            )
            finetune_dataloader = DataLoader(
                finetune_dataset_tokenized, # pyright: ignore[reportArgumentType]
                batch_size=batch_size,
                shuffle=True,
                pin_memory=pin_cpu_offload and str(device).startswith("cuda"),
                collate_fn=finetune_collator,
            )
            model = run_sequential_lora_update(
                model=model,
                loader=finetune_dataloader,
                device=device,
                layers_str=update_layers,
                lora_r=sequential_lora_r,
                lora_alpha=sequential_lora_alpha,
                lora_dropout=sequential_lora_dropout,
                learning_rate=sequential_lora_lr,
                weight_decay=sequential_lora_weight_decay,
                epochs=sequential_lora_epochs,
                max_steps=sequential_lora_max_steps,
                grad_accum_steps=sequential_lora_grad_accum_steps,
                gradient_checkpointing=sequential_lora_gradient_checkpointing,
            )
            del finetune_dataset_tokenized, finetune_dataloader, finetune_collator
        elif sequential_update_method == "local_u":
            sequential_update_samples = dataset["max_samples"]
            model = model.cpu()
            cuda_cleanup()

            dense_reference_model: Qwen2ForCausalLM = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                device_map=None,
                max_position_embeddings=max_length,
                use_cache=False,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                token=hf_token,
                trust_remote_code=True,
            ) # type: ignore[assignment]
            dense_reference_model.eval()
            dense_reference_model.requires_grad_(False)

            runner = SequentialLocalUpdateRunner(
                model=model,
                loader=calibration_dataloader,
                device=device,
                compressed_dtype=compressed_dtype,
                ridge=sequential_update_ridge,
                pin_cpu_offload=pin_cpu_offload,
            )
            updated_rank_map = runner.update_taw_only_checkpoint(
                dense_reference_model=dense_reference_model,
                layers_str=update_layers,
            )
            rank_map.update(updated_rank_map)

            del runner, dense_reference_model, updated_rank_map
        else:
            raise ValueError(f"Unknown sequential_update_method: {sequential_update_method}")

        cuda_cleanup()
        ram_usage("After performing sequential low-rank update")
        vram_usage("After performing sequential low-rank update")

    # Inspect lowrank matrices
    for name, p in model.named_parameters():
        if "W_u" in name or "W_v" in name:
            print(name, p.dtype, p.device, torch.isfinite(p).all().item(), p.norm().item())
            if not torch.isfinite(p).all():
                raise RuntimeError(f"{name} has NaN/Inf")
    
    # Inspect compressed model logits
    logits_debug(model, tokenizer, "The responsibility of an AI assistant is", "cpu")

    ram_usage("After performing layer compression")
    vram_usage("After performing layer compression")

    model.requires_grad_(False) # No fine tuning is needed
    model.eval()

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
        compress_att_q_str = "_q" if compress_att_q else ""
        compress_att_k_str = "_k" if compress_att_k else ""
        compress_att_v_str = "_v" if compress_att_v else ""
        compress_att_out_str = "_out" if compress_att_out else ""
        compress_mlp_str = "_mlp" if compress_mlp else ""
        ratio_scope_str = ratio_scope_str = "_all" if compress_att_q and compress_att_k and compress_att_v and compress_att_out and compress_mlp else "_" + str(ratio_scope)
        heterogeneous_str = "_het" if heterogeneous else "_hom"
        group_criterion_str = ("_" + group_criterion) if heterogeneous else ""
        score_metric_substr = score_metric.replace("|", "") if len(score_metric.split("|")) > 1 else score_metric
        score_metric_str = ("_" + score_metric_substr) if heterogeneous else ""
        v2_str = "_v2" if is_v2 else ""
        bypassed_layers_str = "_" + str(bypass_early_layers) if bypass_early_layers >= 0 else ""
        sequential_update_str = f"_upd_{sequential_update_method}" if sequential_update else ""

        payload = {
            "state_dict": model.state_dict(),
            "rank_map": rank_map,

            # General fix for meta + to_empty loading.
            "non_persistent_buffers": collect_non_persistent_buffers(model),

            # Useful metadata.
            "config": model.config.to_dict(),
            "generation_config": (
                model.generation_config.to_dict() # pyright: ignore[reportOptionalMemberAccess]
                if getattr(model, "generation_config", None) is not None
                else None
            ),
            "svd_llm_metadata": {
                "format_version": 2 if is_v2 else 1,
                "base_model_name": model_name,
                "parameter_dtypes": dtype_summary(model),
                "lowrank_parameter_dtypes": dtype_summary(model, only_lowrank=True),
                "num_lowrank_modules": len(rank_map),
                "rank_map_preview": list(rank_map.items())[:10],
                "sequential_update": sequential_update,
                "sequential_update_method": sequential_update_method if sequential_update else None,
                "sequential_update_dataset": finetune_dataset if sequential_update and sequential_update_method == "lora" else None,
                "sequential_update_samples": sequential_update_samples,
                "finetune_cutoff_len": finetune_cutoff_len if sequential_update and sequential_update_method == "lora" else None,
                "finetune_train_on_inputs": finetune_train_on_inputs if sequential_update and sequential_update_method == "lora" else None,
                "sequential_lora_r": sequential_lora_r if sequential_update_method == "lora" and sequential_update else None,
                "sequential_lora_alpha": sequential_lora_alpha if sequential_update_method == "lora" and sequential_update else None,
                "sequential_lora_dropout": sequential_lora_dropout if sequential_update_method == "lora" and sequential_update else None,
                "sequential_lora_lr": sequential_lora_lr if sequential_update_method == "lora" and sequential_update else None,
                "sequential_lora_epochs": sequential_lora_epochs if sequential_update_method == "lora" and sequential_update else None,
                "sequential_lora_max_steps": sequential_lora_max_steps if sequential_update_method == "lora" and sequential_update else None,
                "sequential_lora_grad_accum_steps": sequential_lora_grad_accum_steps if sequential_update_method == "lora" and sequential_update else None,
                "sequential_lora_effective_batch_size": batch_size * sequential_lora_grad_accum_steps if sequential_update_method == "lora" and sequential_update and sequential_lora_grad_accum_steps is not None else None,
            },
        }

        torch.save(payload, save_path_model + 
           model_name.replace("/", "_").replace("-", "_") + 
           compress_att_q_str +
           compress_att_k_str +
           compress_att_v_str + 
           compress_att_out_str + 
           compress_mlp_str + 
           ratio_scope_str + "_" +
           str(round(ratio, 2)) +
           heterogeneous_str + 
           group_criterion_str +
           score_metric_str +
           bypassed_layers_str +
           sequential_update_str +
           v2_str + 
           ".pt")
        print("[DEBUG] Compressed model saved succesfully")

    cuda_cleanup()
    return model, tokenizer
