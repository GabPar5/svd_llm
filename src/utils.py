import os
import gc
import psutil
import torch.nn as nn
import torch.nn.functional as F
import torch
import random
import sys
import re
import resource
from typing import Dict, Optional, List
from tqdm import tqdm
from enum import Enum
from datasets import load_dataset, load_from_disk, Dataset
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from .modules import *

# Threshold above which cuSOLVER 32-bit indexing overflows
SOLVER_GPU_MAX_DIM = 32000

class CatcherExit(Exception):
    pass

class GroupBy(str, Enum):
    GLOBAL="global"
    DECODER="decoder"
    TYPE="type"

class ScoreMetric(str, Enum):
    TRUNCATION="truncation"
    ENTROPY="entropy"
    
    @classmethod
    def _missing_(cls, value):
        if not isinstance(value, str):
            return None
            
        if re.fullmatch(r"norm\|(\d+|inf|-inf)", value):
            obj = str.__new__(cls, value)
            obj._value_ = value
            # Standardize the name (e.g., "norm|-inf" becomes "NORM_INF_NEG")
            name = value.upper().replace("|", "_").replace("-", "NEG_")
            obj._name_ = name
            
            # Cache it to ensure ScoreMetric("norm|2") is ScoreMetric("norm|2")
            cls._value2member_map_[value] = obj
            return obj
            
        return super()._missing_(value)

class DtypeMap(Enum):
    float32= torch.float32
    fp32= float32
    float16= torch.float16
    fp16= float16
    bfloat16= torch.bfloat16
    bf16= bfloat16

    @classmethod
    def get_dtype(cls, _v) -> torch.dtype:
        if isinstance(_v, str):
            return cls[_v].value
        elif isinstance(_v, torch.dtype):
            return _v
        else:
            raise TypeError(f"{type(_v).__name__}")
        
def cuda_cleanup():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()

def vram_usage(msg=""):
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2
    torch.cuda.reset_peak_memory_stats()
    print(f"[VRAM] {msg} | allocated={alloc:.1f} MiB | reserved={reserved:.1f} MiB | peak={peak:.1f} MiB")

def ram_usage(msg=""):
    # Get current Process RAM
    process = psutil.Process(os.getpid())
    process_ram = process.memory_info().rss / 1024**2 
    
    # Get Peak Process RAM
    # ru_maxrss gives the maximum resident set size used by the process.
    peak_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_ram = peak_usage / 1024
        
    # Overall System RAM
    sys_mem = psutil.virtual_memory()
    sys_used = sys_mem.used / 1024**2
    sys_total = sys_mem.total / 1024**2
    
    print(f"[RAM] {msg} | process={process_ram:.1f} MiB | peak={peak_ram:.1f} MiB | system={sys_used:.1f}/{sys_total:.1f} MiB")

def concatenate_text(batch):
        if "instruction" in batch:
            texts = [
                f"{instr}\n{inp}" if inp.strip() else instr
                for instr, inp in zip(batch["instruction"], batch["input"])
            ]
            return {"concatenated": ["\n\n".join(texts)]}
        elif "text" in batch:
            return {"concatenated": ["\n\n".join(batch["text"])]}
        elif "page" in batch:
            return {"concatenated": ["\n\n".join(batch["page"])]}
        else:
            raise ValueError(f"Unrecognized dataset format. Available columns: {list(batch.keys())}")
        
def tokenize_concatenated(batch, tokenizer: Qwen2TokenizerFast):
        return tokenizer(
            batch["concatenated"],
            truncation=False, # we want the full token stream
            padding=False, # no padding
            return_attention_mask=False # we'll create all-ones masks later
        )

def sample_chunks(batch, max_length: int, max_samples: int, seed: Optional[int]):
    rng = random.Random(seed)
    token_stream = batch["token_stream"][0]
    total_tokens = len(token_stream)

    input_ids = []
    attention_mask = []
    for _ in range(max_samples):
        i = rng.randint(0, total_tokens - max_length - 1)
        j = i + max_length
        input_ids.append(token_stream[i:j])
        attention_mask.append([1] * max_length)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }

def tokenize_dataset(
        name: str,
        subset: str,
        split: str,
        tokenizer,
        max_samples: int = 256,
        batch_size: int = 32,
        max_length: int = 2048,
        seed: Optional[int] = None,
        save_path: Optional[str] = None
):
    # Step 1: Load dataset
    print(f"[DEBUG] Dataset name/path: {name}")
    if os.path.isdir(name):
        print("[DEBUG] Loading dataset from disk...")
        df: Dataset = load_from_disk(name + "/" + split) # pyright: ignore[reportAssignmentType]
    else:
        print("[DEBUG] Loading dataset from hub...")
        if subset is not None:
            df: Dataset = load_dataset(path=name, name=subset, split=split, num_proc=8) # pyright: ignore[reportAssignmentType]
        else:
            df: Dataset = load_dataset(path=name, split=split, num_proc=8)
        if save_path and not os.path.exists(save_path + "/calibration_datasets/" + name + "/" + split):
            print("[DEBUG] Saving dataset to disk...")
            df.save_to_disk(save_path + "/calibration_datasets/" + name + "/" + subset + "/" + split)

    # Step 2: Concatenate all text into one long string
    concatenated = df.map(
        concatenate_text,
        batched=True,
        batch_size=len(df), # process entire dataset in one batch
        remove_columns=df.column_names,
        load_from_cache_file=False,
        desc="Concatenating text..."
    )

    # Step 3: Tokenize the single concatenated string
    tokenized = concatenated.map(
        tokenize_concatenated,
        batched=True,
        batch_size=1,
        remove_columns=["concatenated"],
        load_from_cache_file=False,
        fn_kwargs={"tokenizer": tokenizer},
        desc="Tokenizing concatenated text..."
    )

    # Step 4: Flatten into a single 1D list of token IDs
    # After the map above, tokenized["input_ids"] is a list containing
    # one element (since batch_size=1 above): a very long list of token IDs.
    # We flatten it into a plain Python list.
    all_token_ids = [tid for chunk in tokenized["input_ids"] for tid in chunk]
    total_tokens = len(all_token_ids)
    print(f"[DEBUG] Total tokens in concatenated stream: {total_tokens}")
    print(f"[DEBUG] Requested samples: {max_samples} x {max_length} = {max_samples * max_length} tokens")

    if total_tokens < max_length + 1:
        raise ValueError(f"Not enough tokens ({total_tokens}) to sample even one chunk of length {max_length}.")

    if total_tokens < max_samples * max_length:
        actual_samples = total_tokens // max_length
        print(f"[WARNING] Not enough tokens for {max_samples} samples. Reducing to {actual_samples}.")
        max_samples = actual_samples

    # Step 5: Randomly sample overlapping fixed-length chunks
    # Wrap the flat token list into a temporary Dataset so we can use .map()
    chunk_input = Dataset.from_dict({"token_stream": [all_token_ids]})
    chunked = chunk_input.map(
        sample_chunks,
        batched=True,
        batch_size=1,
        remove_columns=["token_stream"],
        load_from_cache_file=False,
        fn_kwargs={"max_length": max_length, "max_samples": max_samples, "seed": seed},
        desc="Sampling random chunks..."
    )

    return chunked.with_format("torch"), max_samples

def tokenize_finetune_dataset(
        dataset_spec: str,
        tokenizer,
        max_samples: int = 50000,
        cutoff_len: int = 256,
        seed: Optional[int] = None,
        train_on_inputs: bool = False,
        add_eos_token: bool = False,
):
    """
    Tokenize the dataset used by the LoRA sequential update.

    Default usage mirrors upstream SVD-LLM/Alpaca-LoRA:
    `dataset_spec="yahma/alpaca-cleaned"`, split=train, Alpaca prompt format,
    and labels masked on the instruction/input part unless train_on_inputs=True.
    """
    parts = dataset_spec.split(":")
    dataset_name = parts[0]
    dataset_subset = parts[1] if len(parts) > 1 and parts[1] else None
    dataset_split = parts[2] if len(parts) > 2 and parts[2] else "train"

    print(f"[FINETUNE] Dataset: {dataset_name} | subset={dataset_subset} | split={dataset_split}")

    if os.path.isdir(dataset_name):
        loaded = load_from_disk(dataset_name)
        df = loaded[dataset_split] if hasattr(loaded, "keys") and dataset_split in loaded else loaded
    elif dataset_subset is not None:
        df = load_dataset(dataset_name, dataset_subset, split=dataset_split)
    else:
        df = load_dataset(dataset_name, split=dataset_split)

    if max_samples is not None and max_samples > 0 and len(df) > max_samples:
        df = df.shuffle(seed=seed).select(range(max_samples)) # pyright: ignore[reportAttributeAccessIssue]
    else:
        df = df.shuffle(seed=seed) # pyright: ignore[reportAttributeAccessIssue]

    actual_samples = len(df)
    print(f"[FINETUNE] Using {actual_samples} samples, cutoff_len={cutoff_len}")

    def alpaca_prompt(instruction: str, input_text: str = "", output_text: Optional[str] = None) -> str:
        if input_text and input_text.strip():
            prompt = (
                "Below is an instruction that describes a task, paired with an input "
                "that provides further context. Write a response that appropriately "
                "completes the request.\n\n"
                "### Instruction:\n"
                f"{instruction}\n\n"
                "### Input:\n"
                f"{input_text}\n\n"
                "### Response:\n"
            )
        else:
            prompt = (
                "Below is an instruction that describes a task. Write a response "
                "that appropriately completes the request.\n\n"
                "### Instruction:\n"
                f"{instruction}\n\n"
                "### Response:\n"
            )

        if output_text is not None:
            prompt += str(output_text)
        return prompt

    def tokenize_prompt(prompt: str, should_add_eos: bool = True) -> Dict:
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            should_add_eos
            and tokenizer.eos_token_id is not None
            and len(result["input_ids"]) > 0
            and result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize(data_point: Dict) -> Dict:
        if "instruction" in data_point and "output" in data_point:
            instruction = str(data_point.get("instruction") or "")
            input_text = str(data_point.get("input") or "")
            output_text = str(data_point.get("output") or "")

            tokenized = tokenize_prompt(
                alpaca_prompt(instruction, input_text, output_text),
                should_add_eos=True,
            )
            if not train_on_inputs:
                user_prompt = alpaca_prompt(instruction, input_text, None)
                tokenized_user = tokenize_prompt(
                    user_prompt,
                    should_add_eos=add_eos_token,
                )
                user_prompt_len = len(tokenized_user["input_ids"])
                if add_eos_token and user_prompt_len > 0:
                    user_prompt_len -= 1
                tokenized["labels"] = (
                    [-100] * user_prompt_len
                    + tokenized["labels"][user_prompt_len:]
                )
            return tokenized

        text = None
        for field in ("text", "sentence", "page", "content"):
            if field in data_point and data_point[field] is not None:
                text = str(data_point[field])
                break
        if text is None:
            raise ValueError(
                "Unsupported finetune dataset format. Expected Alpaca-style "
                "`instruction`/`input`/`output` fields or a text-like field."
            )
        return tokenize_prompt(text, should_add_eos=True)

    tokenized = df.map(
        generate_and_tokenize,
        remove_columns=df.column_names, # pyright: ignore[reportArgumentType]
        load_from_cache_file=False,
        desc="Tokenizing finetune dataset...",
    )

    return tokenized, actual_samples

def generate_paths(mlp: bool, q: bool, k: bool, v: bool, attention_output: bool, layers_number: int) -> list[str]:
    list_paths=[]
    if layers_number >= 0:
        if mlp:
            list_paths += [f'model.layers.{layers_number - 1 - i}.mlp.gate_proj' for i in range(layers_number)]
            list_paths += [f'model.layers.{layers_number - 1 - i}.mlp.up_proj' for i in range(layers_number)]
            list_paths += [f'model.layers.{layers_number - 1 - i}.mlp.down_proj' for i in range(layers_number)]
        if q:
            list_paths += [f'model.layers.{layers_number - 1 - i}.self_attn.q_proj' for i in range(layers_number)]
        if k:
            list_paths += [f'model.layers.{layers_number - 1 - i}.self_attn.k_proj' for i in range(layers_number)]
        if v:
            list_paths += [f'model.layers.{layers_number - 1 - i}.self_attn.v_proj' for i in range(layers_number)]
        if attention_output:
            list_paths += [f'model.layers.{layers_number - 1 - i}.self_attn.o_proj' for i in range(layers_number)]
    return list_paths

def get_layers(model: nn.Module, layers_str: list[str], split_attributes=False):
    paths = [layer.split('.') for layer in layers_str]
    if split_attributes:
        attributes = [layer[-1] for layer in paths]
        paths = [layer[:-1] for layer in paths]

    layers_list = []
    for layer in paths:
        tmp_layer = model
        for sub_layer in layer:
            tmp_layer = getattr(tmp_layer, sub_layer)
        layers_list.append(tmp_layer)
    if split_attributes:
        return layers_list, attributes
    else:
        return layers_list
    
def get_group(
        layer_path: str, 
        group_patterns: Dict[str, List[str]]
    ) -> Optional[str]:
    for group_name, patterns in group_patterns.items():
        if any(layer_path.endswith(p) for p in patterns):
            return group_name
    return None

def get_submodule(root, path):
    obj = root
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj

def make_captured_meta(captured: List[Dict]) -> List[Dict]:
        """
        Store only reusable metadata, not the layer input tensor itself.
        """
        meta = []

        for entry in captured:
            meta.append({
                "inp": None,
                "attention_mask": entry.get("attention_mask", None),
                "position_ids": entry.get("position_ids", None),
                "cache_position": entry.get("cache_position", None),
                "position_embeddings": entry.get("position_embeddings", None),
                "past_key_values": entry.get("past_key_values", None),
            })

        return meta

def save_activation_checkpoint(act_ckpt_dir: str, model_name: str, version_str: str, n_tokens: int, layer_idx: int, inps: List[torch.Tensor], captured: List[Dict]) -> str:
        """
        Saves CPU activations that are inputs to decoder layer `layer_idx`.
        """
        path_tmp = os.path.join(act_ckpt_dir, f"inputs_to_layer_{layer_idx}.pt") + ".tmp"
        path_final = os.path.join(act_ckpt_dir, f"inputs_to_layer_{layer_idx}.pt")

        cpu_inps = []
        for x in inps:
            if x is None:
                raise RuntimeError(f"Cannot save activation checkpoint for layer {layer_idx}: found None input.")
            cpu_inps.append(x.detach().cpu())

        payload = {
            "model_name": model_name,
            "version": version_str,
            "activation_layer": layer_idx,
            "num_batches": len(cpu_inps),
            "n_tokens": n_tokens,
            "inps": cpu_inps,
            "captured_meta": make_captured_meta(captured),
        }

        torch.save(payload, path_tmp)
        os.replace(path_tmp, path_final)

        print(f"[ACT-CKPT] Saved inputs to layer {layer_idx}: {path_final}")

        del payload, cpu_inps
        gc.collect()

        return path_final

def try_load_activation_checkpoint(act_ckpt_dir: str, model_name: str, version_str: str, n_tokens: int, layer_idx: int):
        """
        Returns:
            loaded: bool
            inps: List[Tensor] or None
            captured: List[Dict] or None
            path: str or None
        """
        path = os.path.join(act_ckpt_dir, f"inputs_to_layer_{layer_idx}.pt")

        if layer_idx == 0:
            return False, None, None, None

        if not os.path.exists(path):
            print(f"[ACT-CKPT] No checkpoint found for inputs to layer {layer_idx}: {path}")
            return False, None, None, None

        print(f"[ACT-CKPT] Loading inputs to layer {layer_idx}: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)

        if payload.get("model_name") != model_name:
            print("[ACT-CKPT][WARNING] Checkpoint model_name mismatch. Ignoring checkpoint.")
            del payload
            gc.collect()
            return False, None, None, None

        if payload.get("version") != version_str:
            print("[ACT-CKPT][WARNING] Checkpoint version mismatch. Ignoring checkpoint.")
            del payload
            gc.collect()
            return False, None, None, None
        
        if int(payload.get("n_tokens", -1)) != int(n_tokens):
            print("[ACT-CKPT][WARNING] Checkpoint n_tokens mismatch. Ignoring checkpoint.")
            del payload
            gc.collect()
            return False, None, None, None

        if int(payload.get("activation_layer", -1)) != layer_idx:
            print("[ACT-CKPT][WARNING] Checkpoint activation_layer mismatch. Ignoring checkpoint.")
            del payload
            gc.collect()
            return False, None, None, None

        inps: List[torch.Tensor] = payload["inps"]
        captured_meta: List[Dict] = payload["captured_meta"]

        if len(inps) != len(captured_meta):
            print("[ACT-CKPT][WARNING] Checkpoint batch count mismatch. Ignoring checkpoint.")
            del payload
            gc.collect()
            return False, None, None, None

        print(f"[ACT-CKPT] Loaded {len(inps)} activation batches for layer {layer_idx}")

        return True, inps, captured_meta, path

def load_whitening_data(
        whitening_matrix_paths: Dict[str, str],
        key: str,
        device: str,
        keep: bool = False
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """
    Loads whitening data (either wm tensor or (U_s, L_s) tuple) from disk.
    If keep=False, removes the path so it can't be loaded again.
    """
    fpath = whitening_matrix_paths.get(key) if keep else whitening_matrix_paths.pop(key)
    data = torch.load(fpath, map_location="cpu", weights_only=True) # pyright: ignore[reportArgumentType]
    
    if isinstance(data, tuple):
        # V2 path
        U_s, L_s = data
        return U_s.to(device, dtype=torch.float64), L_s.to(device, dtype=torch.float64)
    else:
        # V1 path
        return data.to(device, dtype=torch.float64)

def get_layer_idx_from_key(key: str) -> int:
    m = re.search(r"\.layers\.(\d+)\.", key)
    return int(m.group(1)) if m else -1


def is_bypassed_key(key: str, bypass_early_layers: int) -> bool:
    idx = get_layer_idx_from_key(key)
    return 0 <= idx < bypass_early_layers


def build_param_count_map(
    layers_str: List[str],
    layers_list: List[nn.Module],
    attributes: List[str],
    include_bias: bool = False,
) -> Dict[str, int]:
    """
    Counts parameters affected by compression.

    Default include_bias=False because our low-rank ratio formula compresses
    only weight params. Bias is preserved unchanged.
    """
    param_count_map = {}

    for key, layer, attr in zip(layers_str, layers_list, attributes):
        linear = getattr(layer, attr)
        n = linear.weight.numel()

        if include_bias and linear.bias is not None:
            n += linear.bias.numel()

        param_count_map[key] = int(n)

    return param_count_map


def compute_active_target_ratio(
    layers_str: List[str],
    param_count_map: Dict[str, int],
    target_ratio: float,
    bypass_early_layers: int,
    bypass_ratio: float,
    max_ratio: float = 0.9,
    target_total_params: Optional[int] = None
) -> float:
    selected_total_params = sum(param_count_map[k] for k in layers_str)

    if target_total_params is None:
        target_total_params = selected_total_params

    target_removed = target_ratio * target_total_params

    bypassed_removed = 0.0
    active_params = 0

    for k in layers_str:
        p = param_count_map[k]

        if is_bypassed_key(k, bypass_early_layers):
            bypassed_removed += p * bypass_ratio
        else:
            active_params += p

    if active_params <= 0:
        return target_ratio

    active_budget = target_removed - bypassed_removed
    active_capacity = active_params * max_ratio

    if active_budget < 0:
        print(
            f"[BUDGET][WARNING] Bypassed layers already remove more than target. "
            f"active_budget={active_budget:.2f}; clamping to 0."
        )
        active_budget = 0.0

    if active_budget > active_capacity:
        print(
            f"[BUDGET][WARNING] Requested active budget exceeds selected active capacity. "
            f"requested={active_budget:,.0f}, capacity={active_capacity:,.0f}. "
            f"Clamping. Actual overall compression will be lower than target."
        )
        active_budget = active_capacity

    return active_budget / active_params

def _redundancy_from_scores(scores: torch.Tensor, offset: float = 1.5) -> torch.Tensor:
    """
    High truncation score = important / less redundant = lower compression.
    Low truncation score = redundant = higher compression.

    So redundancy weight is 1 / log(score).
    """
    # TODO - print a warning when fallback is triggered
    scores = scores.to(torch.float64)
    # Handle nan and infinite values (fallback)
    scores = torch.nan_to_num(
        scores,
        nan=1.0 + 1e-6,
        posinf=1e30,
        neginf=1.0 + 1e-6,
    )
    # Handle negative values (fallback)
    scores = torch.clamp(scores, min=0.0)

    if offset <= 1.0:
        raise ValueError("`offset` must be > 1.0")

    # Shift the score to guarantee it is strictly > 1.0 before ratio allocation.
    weights = 1.0 / torch.log(scores + offset)
    # Handle nan and infinite values (fallback)
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

    if weights.sum() <= 0:
        weights = torch.ones_like(weights)

    return weights


def allocate_param_weighted_group(
    keys: List[str],
    score_map: Dict[str, float],
    param_count_map: Dict[str, int],
    group_budget: float,
    max_ratio: float = 0.9,
    offset: float = 1.5
) -> Dict[str, float]:
    """
    Allocates removal budget inside one group.

    This preserves:
        sum_i param_i * ratio_i = group_budget

    instead of:
        mean_i ratio_i = target_ratio
    """
    if not keys:
        return {}

    scores = torch.tensor([score_map[k] for k in keys], dtype=torch.float64)
    weights = _redundancy_from_scores(scores, offset)

    params = torch.tensor([param_count_map[k] for k in keys], dtype=torch.float64)

    group_capacity = float((params * max_ratio).sum().item())
    remaining_budget = max(0.0, min(float(group_budget), group_capacity))

    ratios = torch.zeros(len(keys), dtype=torch.float64)
    active = torch.ones(len(keys), dtype=torch.bool)

    # Water-fill with per-matrix max_ratio cap.
    while remaining_budget > 1e-6 and bool(active.any()):
        idx = torch.nonzero(active, as_tuple=False).flatten()

        p = params[idx]
        w = weights[idx]

        denom = (p * w).sum()

        # TODO - log when fallback is used
        if denom <= 0:
            w = torch.ones_like(w)
            denom = (p * w).sum()

        proposed = remaining_budget * w / denom
        cap_left = max_ratio - ratios[idx]

        clipped = proposed >= cap_left

        if not bool(clipped.any()):
            ratios[idx] += proposed
            remaining_budget = 0.0
            break

        clipped_idx = idx[clipped]
        spend = float((params[clipped_idx] * cap_left[clipped]).sum().item())

        ratios[clipped_idx] = max_ratio
        remaining_budget -= spend
        active[clipped_idx] = False

    return {k: float(r) for k, r in zip(keys, ratios.tolist())}

def factor_range_report(name, W_u, W_v):
    if not torch.isfinite(W_u).all() or not torch.isfinite(W_v).all():
        raise RuntimeError(f"{name}: fp16 cast produced inf/nan")
    for label, W in [("W_u", W_u), ("W_v", W_v)]:
        amax = W.float().abs().max().item()
        rms = W.float().pow(2).mean().sqrt().item()
        print(f"[FACTOR-RANGE] {name}.{label}: absmax={amax:.6e}, rms={rms:.6e}")

        if amax > 60000:
            print(f"[WARNING] {name}.{label} is near fp16 overflow range")

@torch.no_grad()
def check_weights_relative_difference(layer_name, layer_attr, van, device="cuda"):
    W_hat = (
        van.W_u.weight.to(device).float()
        @ van.W_v.weight.to(device).float()
    )
    W_orig = layer_attr.weight.to(device).float()

    rel_w = (W_orig - W_hat).norm() / W_orig.norm().clamp_min(1e-12)
    print(f"[CHECK] {layer_name}: Relative error of reduced rank reconstruction: {rel_w:.3e}")

    van.cpu()

@torch.no_grad()
def check_lowrank_equivalence(layer_name, layer_attr, van, device="cuda"):
    input_dtype = layer_attr.weight.dtype
    factor_dtype = van.W_v.weight.dtype

    x = torch.randn(
        2, 8, layer_attr.in_features,
        device=device,
        dtype=input_dtype,
    )

    van = van.to(device).eval()

    # Match LowRank.forward(): input is cast to factor dtype internally.
    x_factor = x.to(factor_dtype)

    W_hat = van.W_u.weight @ van.W_v.weight
    b = van.W_u.bias

    y_dense_hat = F.linear(x_factor, W_hat, b).to(input_dtype)
    y_lowrank = van(x)

    rel = (y_dense_hat.float() - y_lowrank.float()).norm() / y_dense_hat.float().norm().clamp_min(1e-12)
    print(f"[CHECK] {layer_name}: lowrank-vs-dense relerr={rel.item():.3e}")

    if not torch.isfinite(y_lowrank).all():
        raise RuntimeError(f"{layer_name}: LowRank output has NaN/Inf")

    van.cpu()

@torch.no_grad()
def check_layer_activation_error(name, old_linear, van, device="cuda"):
    x = torch.randn(
        2, 128, old_linear.in_features,
        device=device, 
        dtype=old_linear.weight.dtype
    )

    y0 = old_linear.to(device).eval()(x)
    y1 = van.to(device).eval()(x)

    rel = (y0.float() - y1.float()).norm() / y0.float().norm().clamp_min(1e-12)
    max_abs = (y0.float() - y1.float()).abs().max()

    print(
        f"[APPROX-ACT] {name}: "
        f"rel_act_err={rel.item():.6e}, "
        f"max_abs={max_abs.item():.6e}, "
        f"y0_norm={y0.float().norm().item():.6e}, "
        f"y1_norm={y1.float().norm().item():.6e}"
    )

    van.cpu()
    old_linear.cpu()

    W_hat = b = y_dense_hat = y_lowrank = None
    del W_hat, b, y_dense_hat, y_lowrank

@torch.no_grad()
def logits_debug(model, tokenizer, text, device="cuda"):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt").to(device)

    out = model(**inputs, use_cache=False)
    logits = out.logits[:, -1, :].float()

    print("finite logits:", torch.isfinite(logits).all().item())
    print("logits norm:", logits.norm().item())
    print("logits min/max:", logits.min().item(), logits.max().item())

    probs = torch.softmax(logits, dim=-1)
    vals, ids = torch.topk(probs, k=10, dim=-1)

    for p, tid in zip(vals[0].tolist(), ids[0].tolist()):
        print(f"{p:.5f}", repr(tokenizer.decode([tid])))

def collect_non_persistent_buffers(model: torch.nn.Module):
    """
    Save buffers that are registered on the model but absent from state_dict().
    These include things like RoPE inv_freq in some HF models.

    Returns CPU tensors so the checkpoint is device-independent.
    """
    state_keys = set(model.state_dict().keys())

    extra_buffers = {}
    for name, buf in model.named_buffers():
        if name not in state_keys:
            extra_buffers[name] = buf.detach().cpu().clone()

    return extra_buffers

def dtype_summary(model: torch.nn.Module, only_lowrank: bool = False):
    counts = {}
    for name, p in model.named_parameters():
        if only_lowrank and ".W_u." not in name and ".W_v." not in name:
            continue
        counts[str(p.dtype)] = counts.get(str(p.dtype), 0) + p.numel()
    return counts

def get_parent_module(root: torch.nn.Module, tensor_name: str):
    parts = tensor_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]

@torch.no_grad()
def restore_non_persistent_buffers(
    model: torch.nn.Module,
    saved_buffers: dict[str, torch.Tensor],
    device: str | torch.device,
    strict: bool = True,
):
    if not saved_buffers:
        print("[LOAD] No non-persistent buffers found in checkpoint.")
        return

    restored = []
    missing = []
    shape_mismatch = []

    for name, saved in saved_buffers.items():
        try:
            parent, attr = get_parent_module(model, name)
            current = getattr(parent, attr)
        except AttributeError:
            missing.append(name)
            continue

        if not torch.is_tensor(current):
            missing.append(name)
            continue

        if tuple(current.shape) != tuple(saved.shape):
            shape_mismatch.append(
                (name, tuple(current.shape), tuple(saved.shape))
            )
            continue

        current.copy_(
            saved.to(
                device=current.device if current.device.type != "meta" else device,
                dtype=current.dtype,
            )
        )
        restored.append(name)

    print(f"[LOAD] Restored {len(restored)} non-persistent buffers.")
    for name in restored[:20]:
        print(f"[LOAD]   restored buffer: {name}")

    if missing:
        msg = f"Missing non-persistent buffers in model: {missing[:20]}"
        if strict:
            raise RuntimeError(msg)
        print("[LOAD][WARNING]", msg)

    if shape_mismatch:
        msg = f"Shape-mismatched non-persistent buffers: {shape_mismatch[:20]}"
        if strict:
            raise RuntimeError(msg)
        print("[LOAD][WARNING]", msg)

# TODO - define transformers model and (possibly) save to huggingface. This would reduce exposure to bugs during compressed model loading
def apply_lowrank(model, rank_map, state_dict=None):
    """
    Replace MLP linear layers with LowRank modules.
    rank_map: dict with keys like 'model.layers.0.mlp.down_proj', 'model.layers.0.mlp.gate_proj', etc.
    """
    for layer_name, rank in rank_map.items():
        layer_path = layer_name.split(".")[:-1]
        attr_name = layer_name.split(".")[-1]

        parent = model
        for sub_layer in layer_path:
            parent = getattr(parent, sub_layer)

        old = getattr(parent, attr_name)

        if state_dict is not None:
            factor_key = f"{layer_name}.W_v.weight"
            factor_dtype = state_dict[factor_key].dtype
        else:
            factor_dtype = old.weight.dtype

        lowrank = LowRank(
            in_features=old.in_features,
            out_features=old.out_features,
            rank=rank,
            bias=old.bias is not None,
        ).to(device=old.weight.device, dtype=factor_dtype)

        lowrank.requires_grad_(False)
        setattr(parent, attr_name, lowrank)

@torch.no_grad()
def ppl_eval(
        model,
        tokenizer,
        dataset_name: str = "wikitext",
        subset: str = "wikitext-2-raw-v1",
        split: str = "test",
        eval_max_length: int = 2048,
        batch_size: int | str = "auto",
        device: str = "cuda"
) -> float:
    """
    Evaluates perplexity using the exact same methodology as the SVD-LLM paper.

    The key design choices that make this directly comparable to the paper are:
      1. All test documents are concatenated into a single token stream with
         double-newline separators before tokenization, so there are no
         artificial document boundaries that would give the model a "cold start"
         at the beginning of each document.
      2. The stream is sliced into non-overlapping fixed-length chunks of
         exactly model_seq_len tokens. The final incomplete chunk is discarded
         via integer division.
      3. Perplexity is computed as exp(mean NLL) where the mean is taken
         uniformly over every token position across all chunks.
      4. Batches containing non-finite logits (NaN or inf) are skipped.
    """
    # Concatenate all samples with "\n\n"
    data = load_dataset(path=dataset_name, name=subset, split=split, num_proc=8)
    text = "\n\n".join(data["text"]) # pyright: ignore
    encodings = tokenizer(text, truncation=False, padding=False, return_tensors="pt")

    # input_ids has shape [1, total_tokens]; we take [0] to get a 1D tensor
    # just like the original's `input_ids[0]`, then work with it as a 2D
    # [num_chunks, model_seq_len] tensor after slicing.
    total_tokens = encodings.input_ids.shape[1]
    print(f"[PPL EVAL] Total tokens in test stream: {total_tokens}")

    # --- Step 2: slice into non-overlapping fixed-length chunks ---
    # Integer division naturally drops the final incomplete chunk,
    # exactly as `nsamples = test_ids.numel() // seq_len` does in the original.
    num_chunks = total_tokens // eval_max_length
    input_ids = encodings.input_ids[:, :num_chunks * eval_max_length]
    input_ids = input_ids.reshape(num_chunks, eval_max_length)
    print(f"[PPL EVAL] Evaluating on {num_chunks} complete chunks of {eval_max_length} tokens "
          f"({total_tokens - num_chunks * eval_max_length} tokens discarded from the tail)")

    # --- Step 3: compute NLL for each chunk ---
    batch_size_ppl = batch_size
    if not isinstance(batch_size, int):
        batch_size_ppl = 2 # Fallback if batch size was set to auto

    nlls = []
    for i in tqdm(range(0, num_chunks, batch_size_ppl), desc="Evaluating perplexity..."): # pyright: ignore[reportArgumentType]
        batch = input_ids[i : i + batch_size_ppl].to(device)  # pyright: ignore[reportOperatorIssue] # [B, model_seq_len]
        output = model(batch, use_cache=False)
        lm_logits = output.logits  # [B, model_seq_len, vocab_size]
        output = None
        del output

        # Skip batches with non-finite logits — this matches the original's
        # `if torch.isfinite(lm_logits).all()` guard and protects against
        # a single degenerate batch corrupting the entire perplexity estimate.
        if not torch.isfinite(lm_logits).all():
            print(f"[PPL EVAL] Warning: non-finite logits in batch starting at chunk {i}, skipping.")
            continue

        # Standard next-token-prediction loss: token i predicts token i+1,
        # so we shift logits and labels by one position.
        shift_logits = lm_logits[:, :-1, :].contiguous()   # [B, seq_len-1, vocab]
        shift_labels = batch[:, 1:].contiguous()            # [B, seq_len-1]
        lm_logits = batch = None
        del lm_logits, batch

        # reduction="none" gives us one loss value per token, which we
        # accumulate across batches before taking the mean — this ensures
        # the mean is computed over all tokens equally, not as a mean of
        # per-batch means (which would weight shorter final batches differently).
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1)
        )
        nlls.append(loss.cpu())
        loss = shift_logits = shift_labels = None
        del loss, shift_logits, shift_labels

    # --- Step 4: compute final perplexity ---
    # exp(mean NLL over all tokens) matches the original's
    # np.exp(torch.cat(nlls, dim=-1).mean().item())
    ppl = torch.exp(torch.cat(nlls).mean()).item()
    print(f"[PPL EVAL] Perplexity: {ppl:.4f}")
    return ppl

class Logger:
    def __init__(self, filename="compression_run.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Force to save immediately so we don't lose data on a crash

    def flush(self):
        self.terminal.flush()
