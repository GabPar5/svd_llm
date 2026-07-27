import gc
import math
import os
import psutil
import random
import re
import resource
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from datasets import Dataset, load_dataset, load_from_disk
from enum import Enum
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig # pyright: ignore[reportPrivateImportUsage]
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union
from .modules import *

# Threshold above which cuSOLVER 32-bit indexing overflows
SOLVER_GPU_MAX_DIM = 32000

# Decoder kwargs captured at layer 0 and replayed for every other decoder layer
CAPTURED_DECODER_KWARGS = (
    "attention_mask",
    "position_ids",
    "cache_position",
    "position_embeddings",
    "past_key_values",
)

class CatcherExit(Exception):
    pass

class GroupBy(str, Enum):
    GLOBAL="global"
    DECODER="decoder"
    TYPE="type"

class ScoreMetric(str, Enum):
    TRUNCATION="truncation"
    TRUNCATION_SQ="truncation_sq"
    ENTROPY="entropy"
    ENTROPY_SQ="entropy_sq"
    EFF_RANK="eff_rank"
    EFF_RANK_SQ="eff_rank_sq"
    FULL_NORM_TAIL_ENTROPY="full_norm_tail_entropy"
    FULL_NORM_SQ_TAIL_ENTROPY="full_norm_sq_tail_entropy"
    FULL_NORM_TAIL_EFF_RANK="full_norm_tail_eff_rank"
    FULL_NORM_SQ_TAIL_EFF_RANK="full_norm_sq_tail_eff_rank"

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

def cuda_cleanup() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()

def vram_usage(msg: str = "") -> None:
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2
    torch.cuda.reset_peak_memory_stats()
    print(f"[VRAM] {msg} | allocated={alloc:.1f} MiB | reserved={reserved:.1f} MiB | peak={peak:.1f} MiB")

def ram_usage(msg: str = "") -> None:
    # Get current Process RAM
    process = psutil.Process(os.getpid())
    process_ram = process.memory_info().rss / 1024**2

    # Get Peak Process RAM
    # ru_maxrss gives the maximum resident set size used by the process
    peak_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_ram = peak_usage / 1024

    # Overall System RAM
    sys_mem = psutil.virtual_memory()
    sys_used = sys_mem.used / 1024**2
    sys_total = sys_mem.total / 1024**2

    print(f"[RAM] {msg} | process={process_ram:.1f} MiB | peak={peak_ram:.1f} MiB | system={sys_used:.1f}/{sys_total:.1f} MiB")

def sanitize_model_name(model_name: str) -> str:
    """Turn a HF model id or local path into a single filesystem-safe token"""
    return model_name.replace("/", "_").replace("-", "_")

def parse_dataset_spec(spec: str) -> Tuple[str, Optional[str], str]:
    """
    Split a "datasetNameOrPath[:subset[:split]]" specification.

    Subset is None when omitted and split defaults to "train".
    """
    parts = spec.split(":")
    name = parts[0]
    subset = parts[1] if len(parts) > 1 and parts[1] else None
    split = parts[2] if len(parts) > 2 and parts[2] else "train"
    return name, subset, split

def pin_memory_enabled(pin_cpu_offload: bool, device: Union[str, torch.device]) -> bool:
    """Pinned host memory only pays off for CUDA transfers"""
    is_cuda_device = (
        torch.cuda.is_available()
        and str(device).startswith("cuda")
    )
    return pin_cpu_offload and is_cuda_device

def synchronize_device(device: Union[str, torch.device]) -> None:
    """Synchronize only when the work actually runs on CUDA"""
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()

def offload_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Default calibration offload policy: detach and move to CPU"""
    return tensor.detach().cpu()

def tree_map_tensors(value: Any, fn: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    """Apply `fn` to every tensor inside a None/tensor/tuple/list structure"""
    if value is None:
        return None
    if torch.is_tensor(value):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(tree_map_tensors(item, fn) for item in value)
    if isinstance(value, list):
        return [tree_map_tensors(item, fn) for item in value]
    return value

def concatenate_text(batch: Dict[str, List[str]]) -> Dict[str, List[str]]:
    if "instruction" in batch:
        texts = [
            f"{instr}\n{inp}" if inp.strip() else instr
            for instr, inp in zip(batch["instruction"], batch["input"])
        ]
        return {"concatenated": [ "\n\n".join(texts) ]}
    elif "text" in batch:
        return {"concatenated": [ "\n\n".join(batch["text"]) ]}
    elif "page" in batch:
        return {"concatenated": [ "\n\n".join(batch["page"]) ]}
    else:
        raise ValueError(f"Unrecognized dataset format. Available columns: {list(batch.keys())}")

def tokenize_concatenated(batch: Dict[str, List[str]], tokenizer: Qwen2TokenizerFast) -> Any:
    return tokenizer(
        batch["concatenated"],
        truncation=False, # we want the full token stream
        padding=False, # no padding
        return_attention_mask=False, # we'll create all-ones masks later
    )

def sample_chunks(
        batch: Dict[str, List[List[int]]],
        max_length: int,
        max_samples: int,
        seed: Optional[int]
) -> Dict[str, List[List[int]]]:
    rng = random.Random(seed)
    token_stream = batch["token_stream"][0]
    total_tokens = len(token_stream)

    input_ids = []
    attention_mask = []
    for _ in range(max_samples):
        i = rng.randint(0, total_tokens - max_length - 1)
        j = i + max_length
        input_ids.append(token_stream[i:j])
        attention_mask.append([ 1 ] * max_length)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

def tokenize_dataset(
        name: str,
        subset: Optional[str],
        split: str,
        tokenizer,
        max_samples: int = 256,
        batch_size: int = 32,
        max_length: int = 2048,
        seed: Optional[int] = None,
        save_path: Optional[str] = None
) -> Tuple[Dataset, int]:
    # Step 1: Load dataset
    print(f"[DEBUG] Dataset name/path: {name}")
    if os.path.isdir(name):
        print("[DEBUG] Loading dataset from disk...")
        df: Dataset = load_from_disk(os.path.join(name, split)) # pyright: ignore[reportAssignmentType]
    else:
        print("[DEBUG] Loading dataset from hub...")
        if subset is not None:
            df: Dataset = load_dataset(path=name, name=subset, split=split, num_proc=8) # pyright: ignore[reportAssignmentType]
        else:
            df: Dataset = load_dataset(path=name, split=split, num_proc=8) # pyright: ignore[reportAssignmentType]

        if save_path:
            dataset_cache_path = os.path.join(save_path, "calibration_datasets", name, subset or "", split)
            if not os.path.exists(dataset_cache_path):
                print("[DEBUG] Saving dataset to disk...")
                df.save_to_disk(dataset_cache_path)

    # Step 2: Concatenate all text into one long string
    concatenated = df.map(
        concatenate_text,
        batched=True,
        batch_size=len(df), # process entire dataset in one batch
        remove_columns=df.column_names,
        load_from_cache_file=False,
        desc="Concatenating text...",
    )

    # Step 3: Tokenize the single concatenated string
    tokenized = concatenated.map(
        tokenize_concatenated,
        batched=True,
        batch_size=1,
        remove_columns=[ "concatenated" ],
        load_from_cache_file=False,
        fn_kwargs={"tokenizer": tokenizer},
        desc="Tokenizing concatenated text...",
    )

    # Step 4: Flatten into a single 1D list of token IDs
    # After the map above, tokenized["input_ids"] is a list containing
    # one element (since batch_size=1 above): a very long list of token IDs
    # We flatten it into a plain Python list
    all_token_ids = [tid for chunk in tokenized["input_ids"] for tid in chunk]
    total_tokens = len(all_token_ids)
    print(f"[DEBUG] Total tokens in concatenated stream: {total_tokens}")
    print(f"[DEBUG] Requested samples: {max_samples} x {max_length} = {max_samples * max_length} tokens")

    if total_tokens < max_length + 1:
        raise ValueError(f"Not enough tokens ({total_tokens}) to sample even one chunk of length {max_length}")

    if total_tokens < max_samples * max_length:
        actual_samples = total_tokens // max_length
        print(f"[WARNING] Not enough tokens for {max_samples} samples. Reducing to {actual_samples}")
        max_samples = actual_samples

    # Step 5: Randomly sample overlapping fixed-length chunks
    # Wrap the flat token list into a temporary Dataset so we can use .map()
    chunk_input = Dataset.from_dict({"token_stream": [ all_token_ids ]})
    chunked = chunk_input.map(
        sample_chunks,
        batched=True,
        batch_size=1,
        remove_columns=[ "token_stream" ],
        load_from_cache_file=False,
        fn_kwargs={"max_length": max_length, "max_samples": max_samples, "seed": seed},
        desc="Sampling random chunks...",
    )

    return chunked.with_format("torch"), max_samples

def alpaca_prompt(instruction: str, input_text: str = "", output_text: Optional[str] = None) -> str:
    """Render the Alpaca instruction template used by the upstream SVD-LLM LoRA script"""
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

def tokenize_finetune_dataset(
        dataset_spec: str,
        tokenizer,
        max_samples: int = 50000,
        cutoff_len: int = 256,
        seed: Optional[int] = None,
        train_on_inputs: bool = False,
        add_eos_token: bool = False,
        val_set_size: int = 2000,
        val_split_seed: int = 42,
) -> Tuple[Dataset, Optional[Dataset], Dict[str, Any]]:
    """
    Tokenize the dataset used by the LoRA sequential update.

    Default usage mirrors upstream SVD-LLM/Alpaca-LoRA:
    `dataset_spec="yahma/alpaca-cleaned"`, split=train, Alpaca prompt format,
    and labels masked on the instruction/input part unless train_on_inputs=True.
    """
    dataset_name, dataset_subset, dataset_split = parse_dataset_spec(dataset_spec)

    print(f"[FINETUNE] Dataset: {dataset_name} | subset={dataset_subset} | split={dataset_split}")

    if os.path.isdir(dataset_name):
        loaded = load_from_disk(dataset_name)
        df = loaded[dataset_split] if hasattr(loaded, "keys") and dataset_split in loaded else loaded
    elif dataset_subset is not None:
        df = load_dataset(dataset_name, dataset_subset, split=dataset_split)
    else:
        df = load_dataset(dataset_name, split=dataset_split)

    sample_limit = max_samples
    if max_samples is not None and max_samples > 0 and val_set_size > 0:
        # Keep max_samples as the intended training-set budget; add validation
        # examples before splitting so default 50k + 2k mirrors upstream better
        sample_limit = max_samples + val_set_size

    if sample_limit is not None and sample_limit > 0 and len(df) > sample_limit:
        df = df.shuffle(seed=seed).select(range(sample_limit)) # pyright: ignore[reportAttributeAccessIssue]
    else:
        df = df.shuffle(seed=seed) # pyright: ignore[reportAttributeAccessIssue]

    requested_samples = len(df)
    print(f"[FINETUNE] Tokenizing up to {requested_samples} samples, cutoff_len={cutoff_len}")

    eval_raw = None
    actual_val_set_size = 0
    if val_set_size > 0 and len(df) > 1:
        actual_val_set_size = min(int(val_set_size), len(df) - 1)
        if actual_val_set_size != val_set_size:
            print(
                f"[FINETUNE][WARNING] Requested val_set_size={val_set_size}, "
                f"using {actual_val_set_size} to keep a non-empty train split",
            )

        split = df.train_test_split( # pyright: ignore[reportAttributeAccessIssue]
            test_size=actual_val_set_size,
            shuffle=True,
            seed=val_split_seed,
        )
        train_raw = split["train"]
        eval_raw = split["test"]
    else:
        train_raw = df

    def tokenize_prompt(prompt: str, should_add_eos: bool = True) -> Dict:
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        needs_eos = (
            should_add_eos
            and tokenizer.eos_token_id is not None
            and len(result["input_ids"]) > 0
            and result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
        )
        if needs_eos:
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
                    [ -100 ] * user_prompt_len
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
                "`instruction`/`input`/`output` fields or a text-like field",
            )
        return tokenize_prompt(text, should_add_eos=True)

    def tokenize_and_filter_split(raw_split, split_name: str, allow_empty: bool = False) -> Tuple[Dataset, int]:
        tokenized = raw_split.map(
            generate_and_tokenize,
            remove_columns=raw_split.column_names, # pyright: ignore[reportArgumentType]
            load_from_cache_file=False,
            desc=f"Tokenizing {split_name} finetune dataset...",
        )

        before_filter = len(tokenized)
        tokenized = tokenized.filter(
            lambda example: any(label != -100 for label in example["labels"]),
            load_from_cache_file=False,
            desc=f"Filtering all-masked {split_name} finetune samples...",
        )
        dropped = before_filter - len(tokenized)

        if dropped > 0:
            print(
                f"[FINETUNE][WARNING] Dropped {dropped} {split_name} samples "
                "with all labels masked. This usually means the instruction/input "
                "alone reached cutoff_len; consider increasing --finetune_cutoff_len",
            )

        if len(tokenized) == 0 and not allow_empty:
            raise ValueError(
                f"All {split_name} finetune samples were filtered because every "
                "label was -100. Increase --finetune_cutoff_len, use a shorter "
                "dataset, or set --finetune_train_on_inputs",
            )

        return tokenized, dropped

    train_dataset, dropped_train = tokenize_and_filter_split(train_raw, "train")
    eval_dataset = None
    dropped_eval = 0
    if eval_raw is not None:
        eval_dataset, dropped_eval = tokenize_and_filter_split(eval_raw, "validation", allow_empty=True)
        if len(eval_dataset) == 0:
            print("[FINETUNE][WARNING] Validation split is empty after filtering; disabling evaluation")
            eval_dataset = None

    stats: Dict[str, Any] = {
        "requested_samples": requested_samples,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset) if eval_dataset is not None else 0,
        "val_set_size": actual_val_set_size,
        "dropped_train_all_masked": dropped_train,
        "dropped_eval_all_masked": dropped_eval,
    }

    print(
        "[FINETUNE] Supervised samples after filtering: "
        f"train={stats['train_samples']} | validation={stats['eval_samples']}",
    )

    return train_dataset, eval_dataset, stats

def generate_paths(mlp: bool, q: bool, k: bool, v: bool, attention_output: bool, layers_number: int) -> List[str]:
    list_paths = []
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

def get_layer_parents(model: nn.Module, layers_str: List[str]) -> Tuple[List[nn.Module], List[str]]:
    """
    Resolve target keys into their parent module and attribute name.

    `model.layers.0.mlp.down_proj` resolves to the `mlp` module and `down_proj`,
    which is what layer replacement needs.
    """
    parents = []
    attributes = []
    for key in layers_str:
        parent_path, _, attribute = key.rpartition(".")
        parents.append(get_submodule(model, parent_path))
        attributes.append(attribute)
    return parents, attributes

def get_submodule(root: nn.Module, path: str) -> Any:
    """Resolve a dotted module path, an empty path meaning the root itself"""
    obj = root
    if not path:
        return obj
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj

def get_parent_module(root: nn.Module, tensor_name: str) -> Tuple[nn.Module, str]:
    """Resolve the module holding `tensor_name` and the attribute name under it"""
    parent_path, _, attribute = tensor_name.rpartition(".")
    return get_submodule(root, parent_path), attribute

def group_keys_by_decoder_layer(layers_str: List[str]) -> Dict[int, List[Tuple[str, str]]]:
    """Map a decoder layer index to its [ (path relative to the layer, full key) ] targets"""
    groups: Dict[int, List[Tuple[str, str]]] = defaultdict(list)
    for key in layers_str:
        match = re.search(r"model\.layers\.(\d+)\.(.*)", key)
        if match is None:
            continue
        groups[int(match.group(1))].append((match.group(2), key))
    return groups

def get_layer_idx_from_key(key: str) -> int:
    match = re.search(r"\.layers\.(\d+)\.", key)
    return int(match.group(1)) if match else -1

def is_bypassed_key(key: str, bypass_early_layers: int) -> bool:
    idx = get_layer_idx_from_key(key)
    return 0 <= idx < bypass_early_layers

def make_captured_meta(captured: List[Dict]) -> List[Dict]:
    """Store only reusable metadata, not the layer input tensor itself"""
    meta = []

    for entry in captured:
        meta.append({
            "inp": None,
            **{key: entry.get(key, None) for key in CAPTURED_DECODER_KWARGS},
        })

    return meta

def save_activation_checkpoint(
        act_ckpt_dir: str,
        model_name: str,
        version_str: str,
        n_tokens: int,
        layer_idx: int,
        inps: List[torch.Tensor],
        captured: List[Dict]
) -> str:
    """Saves CPU activations that are inputs to decoder layer `layer_idx`"""
    path_final = os.path.join(act_ckpt_dir, f"inputs_to_layer_{layer_idx}.pt")
    path_tmp = path_final + ".tmp"

    cpu_inps = []
    for x in inps:
        if x is None:
            raise RuntimeError(f"Cannot save activation checkpoint for layer {layer_idx}: found None input")
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

def try_load_activation_checkpoint(
        act_ckpt_dir: str,
        model_name: str,
        version_str: str,
        n_tokens: int,
        layer_idx: int
) -> Tuple[bool, Optional[List[torch.Tensor]], Optional[List[Dict]], Optional[str]]:
    """
    Load the activations that are inputs to decoder layer `layer_idx`, if usable.

    Returns (loaded, inps, captured, path).
    """
    path = os.path.join(act_ckpt_dir, f"inputs_to_layer_{layer_idx}.pt")

    if layer_idx == 0:
        return False, None, None, None

    if not os.path.exists(path):
        print(f"[ACT-CKPT] No checkpoint found for inputs to layer {layer_idx}: {path}")
        return False, None, None, None

    print(f"[ACT-CKPT] Loading inputs to layer {layer_idx}: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)

    def reject(reason: str) -> Tuple[bool, None, None, None]:
        print(f"[ACT-CKPT][WARNING] Checkpoint {reason} mismatch. Ignoring checkpoint")
        gc.collect()
        return False, None, None, None

    expected_fields = (
        ("model_name", payload.get("model_name"), model_name),
        ("version", payload.get("version"), version_str),
        ("n_tokens", int(payload.get("n_tokens", -1)), int(n_tokens)),
        ("activation_layer", int(payload.get("activation_layer", -1)), layer_idx),
    )

    for field, found, expected in expected_fields:
        if found != expected:
            del payload
            return reject(field)

    inps: List[torch.Tensor] = payload["inps"]
    captured_meta: List[Dict] = payload["captured_meta"]

    if len(inps) != len(captured_meta):
        del payload
        return reject("batch count")

    print(f"[ACT-CKPT] Loaded {len(inps)} activation batches for layer {layer_idx}")

    return True, inps, captured_meta, path

def capture_layer0_inputs(
        model: Qwen2ForCausalLM,
        loader: DataLoader,
        device: str,
        offload: Callable[[torch.Tensor], torch.Tensor] = offload_to_cpu,
        non_blocking: bool = False,
        desc: str = "Capturing layer_0 inputs"
) -> Tuple[List[torch.Tensor], List[Dict]]:
    """
    Run the calibration loader up to decoder layer 0 and capture its inputs.

    Only the modules executed before the decoders are moved to `device`, and
    every captured tensor goes through `offload` so the calibration set can be
    replayed layer by layer afterwards.

    Returns the per-batch layer inputs and the per-batch decoder kwargs, the
    latter with the input tensor stripped out.
    """
    decoder_layers = model.model.layers
    captured: List[Dict] = []

    model.model.embed_tokens = model.model.embed_tokens.to(device)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(device)

    class Catcher(nn.Module):
        def __init__(self, module: nn.Module):
            super().__init__()
            self.module = module

        def __getattr__(self, name: str):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

        def forward(self, inp: torch.Tensor, **kwargs):
            captured.append({
                "inp": offload(inp),
                **{
                    key: tree_map_tensors(kwargs.get(key, None), offload)
                    for key in CAPTURED_DECODER_KWARGS
                },
            })
            raise CatcherExit

    original_layer0 = decoder_layers[0].to(device)
    decoder_layers[0] = Catcher(original_layer0)

    try:
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc):
                try:
                    batch = {
                        key: value.to(device, non_blocking=non_blocking)
                        for key, value in batch.items()
                        if key in ("input_ids", "attention_mask")
                    }
                    model(**batch, use_cache=False)
                except CatcherExit:
                    pass
                finally:
                    del batch
    finally:
        decoder_layers[0] = original_layer0.cpu()
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.cpu()
        cuda_cleanup()

    # Move inputs to a dedicated list and empty the `entry["inp"]` tensors
    inps = [entry["inp"] for entry in captured]
    for entry in captured:
        entry["inp"] = None

    print(f"[CAPTURE] Captured layer_0 inputs for {len(inps)} batches")

    return inps, captured

def decoder_kwargs_to_device(
        entry: Dict,
        device: str,
        non_blocking: bool = False
) -> Dict[str, Any]:
    """Rebuild the decoder kwargs of one captured batch on `device`"""
    kwargs = {}
    for key in CAPTURED_DECODER_KWARGS:
        value = entry.get(key, None)
        if value is not None:
            kwargs[key] = tree_map_tensors(
                value,
                lambda tensor: tensor.to(device, non_blocking=non_blocking),
            )
    return kwargs

def decoder_layer_output(out: Any) -> torch.Tensor:
    """Decoder layers return either the hidden states or a tuple starting with them"""
    if isinstance(out, tuple):
        return out[0]
    return out

def load_whitening_data(
        whitening_matrix_paths: Dict[str, str],
        key: str,
        device: str,
        keep: bool = False
) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
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

class ActiveBudget(NamedTuple):
    """Removal budget left to the matrices that are not bypassed"""
    selected_params: int
    target_total_params: int
    target_removed: float
    bypassed_keys: List[str]
    active_keys: List[str]
    bypassed_removed: float
    active_params: int
    active_budget: float
    active_ratio: float

def compute_active_budget(
        layers_str: List[str],
        param_count_map: Dict[str, int],
        target_ratio: float,
        bypass_early_layers: int,
        bypass_ratio: float,
        max_ratio: float = 0.9,
        target_total_params: Optional[int] = None
) -> ActiveBudget:
    """
    Split the global removal target between bypassed and active matrices.

    Bypassed matrices are charged at `bypass_ratio` and removed from the
    redistribution, so the remaining budget has to be absorbed by the active
    ones. The budget is clamped to what those can physically give up.
    """
    selected_params = sum(param_count_map[k] for k in layers_str)

    if target_total_params is None:
        target_total_params = selected_params

    target_removed = target_ratio * target_total_params

    bypassed_keys = [k for k in layers_str if is_bypassed_key(k, bypass_early_layers)]
    active_keys = [k for k in layers_str if not is_bypassed_key(k, bypass_early_layers)]

    bypassed_removed = sum(param_count_map[k] * bypass_ratio for k in bypassed_keys)
    active_params = sum(param_count_map[k] for k in active_keys)

    active_budget = target_removed - bypassed_removed
    active_capacity = active_params * max_ratio

    if active_budget < 0:
        print(
            f"[BUDGET][WARNING] Bypassed layers already remove more than target. "
            f"active_budget={active_budget:.2f}; clamping to 0",
        )
        active_budget = 0.0

    if active_budget > active_capacity:
        print(
            f"[BUDGET][WARNING] Requested active budget exceeds selected active capacity. "
            f"requested={active_budget:,.0f}, capacity={active_capacity:,.0f}. "
            f"Clamping. Actual overall compression will be lower than target",
        )
        active_budget = active_capacity

    active_ratio = active_budget / active_params if active_params > 0 else target_ratio

    return ActiveBudget(
        selected_params=selected_params,
        target_total_params=target_total_params,
        target_removed=target_removed,
        bypassed_keys=bypassed_keys,
        active_keys=active_keys,
        bypassed_removed=bypassed_removed,
        active_params=active_params,
        active_budget=active_budget,
        active_ratio=active_ratio,
    )

def redundancy_from_scores(scores: torch.Tensor, offset: float = 1.5) -> torch.Tensor:
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

    if offset < 0.0:
        raise ValueError("`offset` must be >= 0.0")

    # Shift the score to guarantee it is strictly > 1.0 before ratio allocation
    weights = 1.0 / torch.log(scores + offset)
    # Handle nan and infinite values (fallback)
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

    return weights

def normalized_spectrum(singular_values: torch.Tensor, squared: bool = False) -> torch.Tensor:
    """Turn singular values into a probability distribution over the spectrum"""
    spectrum = singular_values.pow(2) if squared else singular_values
    return spectrum / spectrum.sum()

def spectrum_entropy(spectrum: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Shannon entropy of a (possibly truncated) normalized spectrum"""
    return -(spectrum * torch.log(spectrum.clamp_min(eps))).sum()

def parse_norm_order(score_metric: str) -> float:
    """Read the p of a "norm|p" score metric, supporting the signed infinity forms"""
    _, _, order = score_metric.partition("|")
    if order.startswith("-"):
        return -float(order[1:])
    return float(order)

def compute_spectrum_score(
        score_metric: ScoreMetric,
        singular_values: torch.Tensor,
        rank: int,
        eps: float = 1e-6
) -> float:
    """
    Importance score of one whitened weight matrix, read from its spectrum.

    Truncation-style metrics measure the tail that the target rank would drop,
    while entropy and effective-rank metrics measure how flat the spectrum is.
    The `full_norm_*` variants normalize over the full spectrum but score the
    tail only. A high score means an important, less redundant matrix.
    """
    tail = singular_values[rank:]

    match score_metric:
        case ScoreMetric.TRUNCATION:
            # After whitening, theoretical truncation loss equals to the L2 norm of truncated singular values = 2-schatten norm = Frobenius norm
            return torch.linalg.norm(tail, ord=2).item()
        case ScoreMetric.TRUNCATION_SQ:
            return torch.sum(tail.pow(2)).item()
        case ScoreMetric.FULL_NORM_TAIL_ENTROPY:
            # After whitening, entropy loss equals the sum of normalized singular values of the tail
            return spectrum_entropy(normalized_spectrum(singular_values)[rank:], eps).item()
        case ScoreMetric.FULL_NORM_SQ_TAIL_ENTROPY:
            # Same of entropy loss but with squared singular values
            return spectrum_entropy(normalized_spectrum(singular_values, squared=True)[rank:], eps).item()
        case ScoreMetric.FULL_NORM_TAIL_EFF_RANK:
            # Effective rank is the exponential of the entropy loss
            return torch.exp(spectrum_entropy(normalized_spectrum(singular_values)[rank:], eps)).item()
        case ScoreMetric.FULL_NORM_SQ_TAIL_EFF_RANK:
            # Same of effective rank but with squared singular values (like D-Rank paper)
            return torch.exp(spectrum_entropy(normalized_spectrum(singular_values, squared=True)[rank:], eps)).item()
        case ScoreMetric.ENTROPY:
            # After whitening, entropy equals the sum of normalized singular values
            return spectrum_entropy(normalized_spectrum(singular_values), eps).item()
        case ScoreMetric.ENTROPY_SQ:
            # Same of entropy but with squared singular values
            return spectrum_entropy(normalized_spectrum(singular_values, squared=True), eps).item()
        case ScoreMetric.EFF_RANK:
            # Effective rank is the exponential of the entropy
            return torch.exp(spectrum_entropy(normalized_spectrum(singular_values), eps)).item()
        case ScoreMetric.EFF_RANK_SQ:
            # Same of effective rank but with squared singular values (like D-Rank paper)
            return torch.exp(spectrum_entropy(normalized_spectrum(singular_values, squared=True), eps)).item()
        case metric if metric.startswith("norm"):
            # After whitening, norm loss equals to the Lp norm of truncated singular values = p-schatten norm
            # WARNING: with p=2 this is the same of the truncation metric
            return torch.linalg.norm(tail, ord=parse_norm_order(metric)).item()

    raise ValueError(f"Unsupported `score_metric`: {score_metric}")

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
    weights = redundancy_from_scores(scores, offset)

    params = torch.tensor([param_count_map[k] for k in keys], dtype=torch.float64)

    group_capacity = float((params * max_ratio).sum().item())
    remaining_budget = max(0.0, min(float(group_budget), group_capacity))

    ratios = torch.zeros(len(keys), dtype=torch.float64)
    active = torch.ones(len(keys), dtype=torch.bool)

    # Water-fill with per-matrix max_ratio cap
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

def resolve_grad_accum_steps(effective_batch_size: int, micro_batch_size: int, backend: str) -> int:
    """The HF Trainer needs an exact split, the custom loop rounds up"""
    if backend == "trainer":
        return max(1, effective_batch_size // micro_batch_size)
    return max(1, math.ceil(effective_batch_size / micro_batch_size))

def validate_lora_batching(
        effective_batch_size: int,
        micro_batch_size: int,
        backend: str,
        require_exact_split: bool = True
) -> None:
    """
    Guard the LoRA batching arguments before any model work is done.

    The exact-split rule only applies when the accumulation steps are derived
    from the two batch sizes; an explicit --sequential_lora_grad_accum_steps
    overrides them.
    """
    if micro_batch_size <= 0:
        raise ValueError("--sequential_lora_micro_batch_size must be > 0")

    if effective_batch_size <= 0:
        raise ValueError("--sequential_lora_effective_batch_size must be > 0")

    needs_exact_split = (
        require_exact_split
        and backend == "trainer"
        and effective_batch_size % micro_batch_size != 0
    )
    if needs_exact_split:
        raise ValueError(
            "--sequential_lora_effective_batch_size must be divisible by "
            "--sequential_lora_micro_batch_size when --sequential_lora_backend trainer",
        )

def build_lora_update_metadata(
        is_lora: bool,
        backend: str,
        finetune_dataset: str,
        finetune_cutoff_len: int,
        finetune_train_on_inputs: bool,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_lr: float,
        lora_epochs: int,
        lora_max_steps: Optional[int],
        grad_accum_steps: Optional[int],
        micro_batch_size: int,
        val_set_size: int,
        fallback_dataset: Optional[str] = None
) -> Dict[str, Any]:
    """
    Checkpoint metadata describing a LoRA sequential update.

    Every LoRA-specific entry is None when the update did not use LoRA, so the
    saved metadata keys stay identical across update methods.
    """
    if not is_lora:
        return {
            "sequential_lora_backend": None,
            "sequential_update_dataset": fallback_dataset,
            "finetune_cutoff_len": None,
            "finetune_train_on_inputs": None,
            "sequential_lora_r": None,
            "sequential_lora_alpha": None,
            "sequential_lora_dropout": None,
            "sequential_lora_lr": None,
            "sequential_lora_epochs": None,
            "sequential_lora_max_steps": None,
            "sequential_lora_grad_accum_steps": None,
            "sequential_lora_micro_batch_size": None,
            "sequential_lora_effective_batch_size": None,
            "sequential_lora_val_set_size": None,
        }

    effective_batch_size = None
    if grad_accum_steps is not None:
        effective_batch_size = micro_batch_size * grad_accum_steps

    return {
        "sequential_lora_backend": backend,
        "sequential_update_dataset": finetune_dataset,
        "finetune_cutoff_len": finetune_cutoff_len,
        "finetune_train_on_inputs": finetune_train_on_inputs,
        "sequential_lora_r": lora_r,
        "sequential_lora_alpha": lora_alpha,
        "sequential_lora_dropout": lora_dropout,
        "sequential_lora_lr": lora_lr,
        "sequential_lora_epochs": lora_epochs,
        "sequential_lora_max_steps": lora_max_steps,
        "sequential_lora_grad_accum_steps": grad_accum_steps,
        "sequential_lora_micro_batch_size": micro_batch_size,
        "sequential_lora_effective_batch_size": effective_batch_size,
        "sequential_lora_val_set_size": val_set_size,
    }

def factor_range_report(name: str, W_u: torch.Tensor, W_v: torch.Tensor) -> None:
    if not torch.isfinite(W_u).all() or not torch.isfinite(W_v).all():
        raise RuntimeError(f"{name}: fp16 cast produced inf/nan")
    for label, W in (("W_u", W_u), ("W_v", W_v)):
        amax = W.float().abs().max().item()
        rms = W.float().pow(2).mean().sqrt().item()
        print(f"[FACTOR-RANGE] {name}.{label}: absmax={amax:.6e}, rms={rms:.6e}")

        if amax > 60000:
            print(f"[WARNING] {name}.{label} is near fp16 overflow range")

@torch.no_grad()
def check_weights_relative_difference(
        layer_name: str,
        layer_attr: nn.Linear,
        van: LowRank,
        device: str = "cuda"
) -> None:
    W_hat = (
        van.W_u.weight.to(device).float()
        @ van.W_v.weight.to(device).float()
    )
    W_orig = layer_attr.weight.to(device).float()

    rel_w = (W_orig - W_hat).norm() / W_orig.norm().clamp_min(1e-12)
    print(f"[CHECK] {layer_name}: Relative error of reduced rank reconstruction: {rel_w:.3e}")

    van.cpu()

@torch.no_grad()
def check_lowrank_equivalence(
        layer_name: str,
        layer_attr: nn.Linear,
        van: LowRank,
        device: str = "cuda"
) -> None:
    input_dtype = layer_attr.weight.dtype
    factor_dtype = van.W_v.weight.dtype

    x = torch.randn(
        2, 8, layer_attr.in_features,
        device=device,
        dtype=input_dtype,
    )

    van = van.to(device).eval()

    # Match LowRank.forward(): input is cast to factor dtype internally
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
def check_layer_activation_error(
        name: str,
        old_linear: nn.Linear,
        van: LowRank,
        device: str = "cuda"
) -> None:
    x = torch.randn(
        2, 128, old_linear.in_features,
        device=device,
        dtype=old_linear.weight.dtype,
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
        f"y1_norm={y1.float().norm().item():.6e}",
    )

    van.cpu()
    old_linear.cpu()

@torch.no_grad()
def logits_debug(model: nn.Module, tokenizer, text: str, device: str = "cuda") -> None:
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
        print(f"{p:.5f}", repr(tokenizer.decode([ tid ])))

def dtype_summary(model: nn.Module, only_lowrank: bool = False) -> Dict[str, int]:
    counts = {}
    for name, p in model.named_parameters():
        if only_lowrank and ".W_u." not in name and ".W_v." not in name:
            continue
        counts[str(p.dtype)] = counts.get(str(p.dtype), 0) + p.numel()
    return counts

def infer_lowrank_dtype(state_dict: Dict[str, torch.Tensor]) -> Optional[torch.dtype]:
    """Read the dtype the LowRank factors were saved with"""
    for name, tensor in state_dict.items():
        if ".W_u." in name or ".W_v." in name:
            return tensor.dtype
    return None

def assert_mixed_dtype(
        model: nn.Module,
        expected_base_dtype: Optional[torch.dtype] = None,
        expected_lowrank_dtype: Optional[torch.dtype] = None
) -> None:
    lowrank_dtypes = set()
    non_lowrank_dtypes = set()

    for name, p in model.named_parameters():
        if ".W_u." in name or ".W_v." in name:
            lowrank_dtypes.add(p.dtype)
        else:
            non_lowrank_dtypes.add(p.dtype)

    print("[DTYPE] non-lowrank dtypes:", non_lowrank_dtypes)
    print("[DTYPE] lowrank dtypes:", lowrank_dtypes)

    if expected_base_dtype is not None and expected_base_dtype not in non_lowrank_dtypes:
        raise RuntimeError(f"Expected base dtype {expected_base_dtype}, got {non_lowrank_dtypes}")

    if expected_lowrank_dtype is not None and lowrank_dtypes != {expected_lowrank_dtype}:
        raise RuntimeError(f"Expected LowRank dtype {expected_lowrank_dtype}, got {lowrank_dtypes}")

def audit_buffers(
        model: nn.Module,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
        label: str = "[BUFFER-AUDIT]"
) -> None:
    state_keys = set(state_dict.keys()) if state_dict is not None else set()

    print(label, "non-state buffers:")
    for name, buf in model.named_buffers():
        if name in state_keys:
            continue

        if buf.device.type == "meta":
            print(f"{label} {name}: META dtype={buf.dtype} shape={tuple(buf.shape)}")
            continue

        is_float = buf.is_floating_point()
        finite = torch.isfinite(buf).all().item() if is_float else "N/A"
        mn = buf.float().min().item() if buf.numel() and is_float else "N/A"
        mx = buf.float().max().item() if buf.numel() and is_float else "N/A"
        sum_abs = buf.float().abs().sum().item() if buf.numel() and is_float else "N/A"
        print(
            f"{label} {name}: device={buf.device} dtype={buf.dtype} "
            f"shape={tuple(buf.shape)} finite={finite} min={mn} max={mx} sum_abs={sum_abs}",
        )

def collect_non_persistent_buffers(model: nn.Module) -> Dict[str, torch.Tensor]:
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

@torch.no_grad()
def restore_non_persistent_buffers(
        model: nn.Module,
        saved_buffers: Dict[str, torch.Tensor],
        device: Union[str, torch.device],
        strict: bool = True,
) -> None:
    if not saved_buffers:
        print("[LOAD] No non-persistent buffers found in checkpoint")
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
                (name, tuple(current.shape), tuple(saved.shape)),
            )
            continue

        current.copy_(
            saved.to(
                device=current.device if current.device.type != "meta" else device,
                dtype=current.dtype,
            ),
        )
        restored.append(name)

    print(f"[LOAD] Restored {len(restored)} non-persistent buffers")
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
def apply_lowrank(
        model: nn.Module,
        rank_map: Dict[str, int],
        state_dict: Optional[Dict[str, torch.Tensor]] = None
) -> None:
    """
    Replace the linear layers listed in `rank_map` with LowRank modules.

    rank_map keys are full module paths, e.g. 'model.layers.0.mlp.down_proj'.
    When a state dict is given, the factor dtype is taken from it so mixed
    precision checkpoints keep their compressed dtype.
    """
    for layer_name, rank in rank_map.items():
        parent, attr_name = get_parent_module(model, layer_name)
        old = getattr(parent, attr_name)

        if state_dict is not None:
            factor_dtype = state_dict[f"{layer_name}.W_v.weight"].dtype
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

def set_attn_implementation(model: nn.Module, attn_implementation: str) -> None:
    """Newer transformers releases expose a setter, older ones only the config field"""
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation(attn_implementation)
    else:
        model.config._attn_implementation = attn_implementation

def build_run_name(
        model_name: str,
        ratio: float,
        compress_mlp: bool,
        compress_att_q: bool,
        compress_att_k: bool,
        compress_att_v: bool,
        compress_att_out: bool,
        ratio_scope: str,
        heterogeneous: bool,
        group_criterion: str,
        score_metric: str,
        bypass_early_layers: int,
        sequential_update: bool,
        sequential_update_method: str,
        is_v2: bool
) -> str:
    """
    Encode a whole compression configuration into one filename token.

    This name identifies the checkpoint, the evaluation JSON and the log file of
    a run, and `generate_tables.py` parses it back to label the result tables,
    so it has to stay stable.
    """
    # Enum members hold their value as a plain string, keep the token stable
    group_criterion = str(getattr(group_criterion, "value", group_criterion))
    score_metric = str(getattr(score_metric, "value", score_metric))

    compresses_everything = (
        compress_att_q
        and compress_att_k
        and compress_att_v
        and compress_att_out
        and compress_mlp
    )
    ratio_scope_str = "_all" if compresses_everything else f"_{ratio_scope}"

    score_metric_str = ""
    group_criterion_str = ""
    if heterogeneous:
        # Score metrics such as "norm|2" carry a separator that filenames cannot hold
        score_metric_str = "_" + score_metric.replace("|", "")
        group_criterion_str = f"_{group_criterion}"

    parts = [
        sanitize_model_name(model_name),
        "_q" if compress_att_q else "",
        "_k" if compress_att_k else "",
        "_v" if compress_att_v else "",
        "_out" if compress_att_out else "",
        "_mlp" if compress_mlp else "",
        ratio_scope_str,
        f"_{round(ratio, 2)}",
        "_het" if heterogeneous else "_hom",
        group_criterion_str,
        score_metric_str,
        f"_{bypass_early_layers}" if bypass_early_layers >= 0 else "",
        f"_upd_{sequential_update_method}" if sequential_update else "",
        "_v2" if is_v2 else "",
    ]

    return "".join(parts)

def save_compressed_checkpoint(
        model: nn.Module,
        checkpoint_path: str,
        rank_map: Dict[str, int],
        metadata: Dict[str, Any]
) -> None:
    """
    Write a compressed model to disk.

    The payload keeps the low-rank state dict together with everything needed to
    rebuild the module structure later: the rank map, the buffers missing from
    state_dict(), and the model/generation configs.
    """
    generation_config = None
    if getattr(model, "generation_config", None) is not None:
        generation_config = model.generation_config.to_dict() # pyright: ignore[reportAttributeAccessIssue]

    payload = {
        "state_dict": model.state_dict(),
        "rank_map": rank_map,

        # General fix for meta + to_empty loading
        "non_persistent_buffers": collect_non_persistent_buffers(model),

        # Useful metadata
        "config": model.config.to_dict(), # pyright: ignore[reportAttributeAccessIssue]
        "generation_config": generation_config,
        "svd_llm_metadata": metadata,
    }

    torch.save(payload, checkpoint_path)
    del payload
    print(f"[DEBUG] Compressed checkpoint saved to: {checkpoint_path}")

def load_compressed_model(
        base_model_name: str,
        checkpoint_path: str,
        model_dtype: Union[str, torch.dtype],
        device: str,
        hf_token: Optional[str] = None,
        attn_implementation: Optional[str] = None,
        audit: bool = False
) -> Tuple[nn.Module, Dict[str, int], Dict[str, Any]]:
    """
    Rebuild a compressed model from a checkpoint written by `save_compressed_checkpoint`.

    The base HF config provides the architecture, `apply_lowrank` restores the
    LowRank modules at the saved ranks, and only then can the state dict be
    loaded strictly. `audit=True` additionally dumps the non-state buffers and
    asserts the base/LowRank dtypes.

    Returns the model, its rank map and the checkpoint metadata.
    """
    print(f"[LOAD] Loading compressed checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    rank_map = checkpoint["rank_map"]
    state_dict = checkpoint["state_dict"]
    extra_buffers = checkpoint.get("non_persistent_buffers", {})
    metadata = checkpoint.get("svd_llm_metadata", {})

    torch_dtype = DtypeMap.get_dtype(model_dtype)
    compressed_dtype = infer_lowrank_dtype(state_dict)

    print(f"[LOAD] Loading base config from: {base_model_name}")
    config = AutoConfig.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        token=hf_token,
        dtype=torch_dtype,
    )

    print("[LOAD] Instantiating base model architecture...")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
            dtype=torch_dtype,
        )

    try:
        model.generation_config = GenerationConfig.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            token=hf_token,
        )
    except Exception as e:
        print(f"[WARNING] Could not load generation_config for {base_model_name}: {e}")
        model.generation_config = GenerationConfig.from_model_config(config)

    print("[LOAD] Applying LowRank module structure...")
    apply_lowrank(model, rank_map, state_dict)
    model.to_empty(device=device)

    if audit:
        audit_buffers(model, state_dict, "[AFTER-TO-EMPTY]")

    print("[LOAD] Loading compressed state dict...")
    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=True,
        assign=True,
    )

    restore_non_persistent_buffers(
        model=model,
        saved_buffers=extra_buffers,
        device=device,
        strict=True,
    )

    if audit:
        audit_buffers(model, state_dict, "[AFTER-LOAD]")
        assert_mixed_dtype(
            model,
            expected_base_dtype=torch_dtype,
            expected_lowrank_dtype=compressed_dtype,
        )

    if missing:
        print(f"[WARNING] Missing keys: {len(missing)}")
        print(missing[:20])

    if unexpected:
        print(f"[WARNING] Unexpected keys: {len(unexpected)}")
        print(unexpected[:20])

    if missing or unexpected:
        raise RuntimeError(
            "State dict mismatch. The compressed checkpoint may not match "
            "the base model or rank_map",
        )

    if attn_implementation is not None:
        set_attn_implementation(model, attn_implementation)

    del checkpoint, state_dict, extra_buffers
    cuda_cleanup()

    return model, rank_map, metadata

@torch.no_grad()
def ppl_eval(
        model: nn.Module,
        tokenizer,
        dataset_name: str = "wikitext",
        subset: str = "wikitext-2-raw-v1",
        split: str = "test",
        eval_max_length: int = 2048,
        batch_size: Union[int, str] = "auto",
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
    # [num_chunks, model_seq_len] tensor after slicing
    total_tokens = encodings.input_ids.shape[1]
    print(f"[PPL EVAL] Total tokens in test stream: {total_tokens}")

    # --- Step 2: slice into non-overlapping fixed-length chunks ---
    # Integer division naturally drops the final incomplete chunk,
    # exactly as `nsamples = test_ids.numel() // seq_len` does in the original
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
        # a single degenerate batch corrupting the entire perplexity estimate
        if not torch.isfinite(lm_logits).all():
            print(f"[PPL EVAL] Warning: non-finite logits in batch starting at chunk {i}, skipping")
            continue

        # Standard next-token-prediction loss: token i predicts token i+1,
        # so we shift logits and labels by one position
        shift_logits = lm_logits[:, :-1, :].contiguous()   # [B, seq_len-1, vocab]
        shift_labels = batch[:, 1:].contiguous()            # [B, seq_len-1]
        lm_logits = batch = None
        del lm_logits, batch

        # reduction="none" gives us one loss value per token, which we
        # accumulate across batches before taking the mean — this ensures
        # the mean is computed over all tokens equally, not as a mean of
        # per-batch means (which would weight shorter final batches differently)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
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
    """Tee stdout to a log file so long runs survive a lost terminal"""

    def __init__(self, filename: str = "compression_run.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Force to save immediately so we don't lose data on a crash

    def flush(self) -> None:
        self.terminal.flush()
