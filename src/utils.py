import gc
import inspect
import json
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
from typing import Any, Callable, Dict, List, Literal, NamedTuple, Optional, Tuple, Type, Union
from .modules import *

# Threshold above which cuSOLVER 32-bit indexing overflows
SOLVER_GPU_MAX_DIM = 32000

# Peak device memory a dense solver holds, as a multiple of the n x n input.
# `eigh` keeps an input copy and the eigenvector output and asks cuSOLVER for a
# `syevd` workspace of ~3n^2 on top; `cholesky` writes a second n x n factor and
# needs next to no workspace. Both are within a few percent on Qwen2.5-32B's
# 27648-wide down-projection, where 5n^2 in fp64 is 28.5 GiB
CUDA_EIGH_MEMORY_FACTOR = 5
CUDA_CHOLESKY_MEMORY_FACTOR = 2

# Free VRAM a solver must leave behind rather than spend
CUDA_SOLVER_HEADROOM = 1.10

# Rows of X the whitening hook casts to fp64 at a time. Casting a whole
# calibration batch at once costs seq_len * batch * in_features * 8 bytes, which
# on a 27648-wide input is 7.25 GiB of pure transient
XXT_ROW_CHUNK = 4096

# Enough to bracket and then resolve a scale to double precision
MAX_BISECTION_STEPS = 128

# A score metric of the form `composite|<local>|<end_to_end>` fuses a per-matrix
# spectral score with a per-block one, instead of reading the spectrum alone
COMPOSITE_PREFIX = "composite|"
END_TO_END_SCORES = ( "block_influence", )

# Relative score spread inside a group below which the allocation is homogeneous
# in all but name, whatever policy is asked for
DEGENERATE_SCORE_SPREAD = 1e-3

# Sidecar recording how a run was configured, written next to every artifact
RUN_CONFIG_SUFFIX = ".config.json"
RUN_CONFIG_SCHEMA_VERSION = 1

# Policy knobs a run name has to distinguish, as (knob, filename prefix, default).
# Sweeping one of these leaves every other token untouched, so without a token of
# its own the whole sweep would collapse onto a single checkpoint
KNOB_FILENAME_TOKENS = (
    ( "offset", "off", 1.5 ),
    ( "softmax_temp", "temp", 1.0 ),
    ( "outer_offset", "ooff", 1.5 ),
)

# Defaults of the remaining swept knobs, kept beside the ones above so that
# "emit only when non-default" reads from one place
DEFAULT_FUSION_ALPHA = 0.5
DEFAULT_BYPASS_RATIO = 0.0
DEFAULT_MIN_RANK_FRACTION = 0.0

# Marks a `rank_map` entry as block diagonal in head space rather than a plain
# joint rank, which is what keeps checkpoints written before it readable
HEAD_BLOCK_KIND = "head_block"
DEFAULT_SEED = 6363

# Never persisted to disk, the sidecar is committed alongside results
REDACTED_ARG_KEYS = ( "hf_token", )

# Cached beside the whitening artifacts they were derived from
LAYER_IMPORTANCE_FILENAME = "layer_importance.pt"
SPECTRA_DIRNAME = "spectra"

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
    HIERARCHICAL="hierarchical"

class InnerAllocation(str, Enum):
    """
    How a removal budget is split across the matrices sitting inside one group.

    `WATERFILL` allocates in ratio space and is the V2-derived baseline the rest
    are measured against; the others allocate in rank space or from an explicit
    optimization objective, so they carry a different implicit bias.
    """
    WATERFILL="waterfill"
    DRANK_LAGRANGIAN="drank_lagrangian"
    SWIFT_POOL="swift_pool"
    SOFTMAX_TEMP="softmax_temp"

class OuterAllocation(str, Enum):
    """
    How a removal budget is split across groups before any group is filled.

    `PARAM_SHARE` gives every group the same average ratio, so it expresses no
    preference of its own and leaves the whole decision to the inner policy.
    """
    PARAM_SHARE="param_share"
    WATERFILL="waterfill"

class ScoreMetric(str, Enum):
    TRUNCATION="truncation"
    TRUNCATION_SQ="truncation_sq"
    ENTROPY="entropy"
    ENTROPY_SQ="entropy_sq"
    EFF_RANK="eff_rank"
    EFF_RANK_SQ="eff_rank_sq"
    # Scale-free counterparts of the four above, each divided by the ceiling its
    # own spectrum length imposes. The raw forms are only comparable between two
    # matrices of the same min(out, in), which holds for every family of an MHA
    # model and fails under GQA, where k and v carry a spectrum `heads /
    # kv_heads` times shorter and so score low for a purely dimensional reason
    TRUNCATION_REL="truncation_rel"
    TRUNCATION_SQ_REL="truncation_sq_rel"
    ENTROPY_REL="entropy_rel"
    ENTROPY_SQ_REL="entropy_sq_rel"
    EFF_RANK_REL="eff_rank_rel"
    EFF_RANK_SQ_REL="eff_rank_sq_rel"
    FULL_NORM_TAIL_ENTROPY="full_norm_tail_entropy"
    FULL_NORM_SQ_TAIL_ENTROPY="full_norm_sq_tail_entropy"
    FULL_NORM_TAIL_EFF_RANK="full_norm_tail_eff_rank"
    FULL_NORM_SQ_TAIL_EFF_RANK="full_norm_sq_tail_eff_rank"

    @classmethod
    def _missing_(cls, value):
        if not isinstance(value, str):
            return None

        def register(name: str):
            obj = str.__new__(cls, value)
            obj._value_ = value
            obj._name_ = name

            # Cache it to ensure ScoreMetric("norm|2") is ScoreMetric("norm|2")
            cls._value2member_map_[value] = obj
            return obj

        if re.fullmatch(r"norm\|(\d+|inf|-inf)", value):
            # Standardize the name (e.g., "norm|-inf" becomes "NORM_INF_NEG")
            return register(value.upper().replace("|", "_").replace("-", "NEG_"))

        if value.startswith(COMPOSITE_PREFIX):
            # Split from the right, so a local metric carrying its own separator
            # ("composite|norm|2|block_influence") still reads unambiguously
            local, _, end_to_end = value[len(COMPOSITE_PREFIX):].rpartition("|")

            is_valid = (
                bool(local)
                and not local.startswith(COMPOSITE_PREFIX)
                and end_to_end in END_TO_END_SCORES
            )
            if not is_valid:
                return None

            try:
                cls(local)
            except ValueError:
                return None

            return register(value.upper().replace("|", "_"))

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

def cuda_solver_fits(
        n: int,
        itemsize: int,
        device: str,
        factor: int
) -> Tuple[bool, float, float]:
    """
    Whether a dense n x n solve fits in the VRAM that is free right now.

    `SOLVER_GPU_MAX_DIM` answers a different question -- cuSOLVER's 32-bit
    indexing is a correctness bound and does not move -- so a matrix can clear
    it and still not fit. The budget is read against `mem_get_info` rather than
    the device total because the whitening loop is not necessarily alone on the
    GPU, and a matrix that fitted for fifty layers stops fitting the moment
    something else takes the room.

    Returns (fits, needed_gib, free_gib) so the caller can report the shortfall
    it is routing around.
    """
    needed = factor * n * n * itemsize
    needed_gib = needed / 1024**3

    if not device.startswith("cuda") or not torch.cuda.is_available():
        return False, needed_gib, 0.0

    free_bytes, _ = torch.cuda.mem_get_info(device)
    free_gib = free_bytes / 1024**3

    return free_bytes > needed * CUDA_SOLVER_HEADROOM, needed_gib, free_gib

def vram_usage(msg: str = "") -> None:
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2
    torch.cuda.reset_peak_memory_stats()

    # The device's own view, which is the one a solver's allocation is judged
    # against. Anything holding VRAM outside the caching allocator is invisible
    # to `memory_allocated` -- another process, a second framework's allocator,
    # or on a coherent-memory host the pages the driver has pulled into HBM --
    # so `untracked` is what explains a solve that fitted for fifty layers and
    # then stopped fitting
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    device_used = (total_bytes - free_bytes) / 1024**2
    untracked = device_used - reserved

    print(
        f"[VRAM] {msg} | allocated={alloc:.1f} MiB | reserved={reserved:.1f} MiB | peak={peak:.1f} MiB "
        f"| device_used={device_used:.1f} MiB | untracked={untracked:.1f} MiB",
    )

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

def matrix_shapes_from_config(config: Any) -> Dict[str, Tuple[int, int]]:
    """
    (out_features, in_features) of every compressible matrix, read from a config.

    The pipeline itself takes shapes from the loaded weights; this exists for
    offline analysis, where the point is not to load a 7B model to find out how
    many parameters a projection has. Kept next to `generate_paths` because the
    two have to name the same seven matrices.
    """
    hidden = int(config.hidden_size)
    intermediate = int(config.intermediate_size)
    heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", None) or heads)
    head_dim = int(getattr(config, "head_dim", None) or hidden // heads)

    q_features = heads * head_dim
    kv_features = kv_heads * head_dim

    return {
        "self_attn.q_proj": ( q_features, hidden ),
        "self_attn.k_proj": ( kv_features, hidden ),
        "self_attn.v_proj": ( kv_features, hidden ),
        "self_attn.o_proj": ( hidden, q_features ),
        "mlp.gate_proj": ( intermediate, hidden ),
        "mlp.up_proj": ( intermediate, hidden ),
        "mlp.down_proj": ( hidden, intermediate ),
    }

def head_partition_from_config(config: Any) -> Dict[str, int]:
    """
    Number of attention heads the output of each head-partitioned matrix holds.

    Only the key and value projections are listed: they are the ones a grouped
    query attention shares between query heads, so a rank collapse in one of
    their heads reaches `num_attention_heads / num_key_value_heads` of them.
    `generate_paths` names the same matrices, and `head_partition_map` turns
    this into the fully qualified keys everything downstream is keyed by.
    """
    heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", None) or heads)

    return {
        "self_attn.k_proj": kv_heads,
        "self_attn.v_proj": kv_heads,
    }

def kv_sharing_from_config(config: Any) -> int:
    """
    How many query heads read each key/value head, 1 under multi-head attention.

    This is the amplifier that makes a rank cut on `k_proj` or `v_proj` cost
    more under grouped-query attention than under MHA, where each head owns its
    own key and value and the damage stays local to it.
    """
    heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", None) or heads)

    return max(1, heads // max(1, kv_heads))

def head_partition_map(layers_str: List[str], config: Any) -> Dict[str, int]:
    """Head count per target key, empty for the targets that are not partitioned"""
    by_type = head_partition_from_config(config)

    return {
        key: heads
        for key in layers_str
        for suffix, heads in by_type.items()
        if key.endswith(suffix)
    }

def head_block_rank(out_features: int, in_features: int, heads: int, ratio: float) -> int:
    """
    Per-head rank realizing `ratio` under a block-diagonal factorization.

    `heads * rank * (in + head_dim)` parameters must equal `(1 - ratio)` of the
    dense `heads * head_dim * in`, so the head count cancels and the rank is
    the same one a single head-sized matrix would get.
    """
    head_dim = out_features // heads
    rank = int((1.0 - ratio) * head_dim * in_features / (head_dim + in_features))

    return max(1, min(rank, head_dim))

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

def is_bypassed_key(
        key: str,
        bypass_early_layers: int,
        bypass_late_layers: int = -1,
        num_layers: Optional[int] = None
) -> bool:
    """
    Whether a matrix lives in a decoder layer excluded from redistribution.

    Both ends can be bypassed in the same run. Resolving the tail needs the
    decoder depth, so the late test is inert when `num_layers` is unknown.
    """
    idx = get_layer_idx_from_key(key)

    if idx < 0:
        return False

    if idx < bypass_early_layers:
        return True

    is_late_bypassed = (
        bypass_late_layers > 0
        and num_layers is not None
        and idx >= num_layers - bypass_late_layers
    )

    return is_late_bypassed

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

def scratch_root(save_path: Optional[str], scratch_path: Optional[str] = None) -> str:
    """
    Root of the regenerable intermediates: whitening artifacts, activation
    checkpoints and the LoRA trainer state. They dwarf the results they produce,
    so a run can park them on a scratch disk and keep `--save_path` for the
    checkpoints, logs and evaluations. Unset, everything stays where it was
    """
    return scratch_path or save_path or "./tmp"

def whitening_dir(base_path: str, model_name: str, version_str: str) -> str:
    """The one place the whitening artifact layout is spelled out"""
    return os.path.join(base_path, "whitening_matrices", sanitize_model_name(model_name), version_str)

def layer_importance_path(wm_dir: str) -> str:
    return os.path.join(wm_dir, LAYER_IMPORTANCE_FILENAME)

def block_influence_from_sums(cos_sum: float, tokens: int) -> float:
    """
    Block Influence of one decoder block: 1 - E[cos(x_in, x_out)].

    High values mean the block rotates the residual stream a lot, so it is doing
    more work and is a worse compression target. Sums are kept rather than the
    mean so chunked whitening runs can merge exactly.
    """
    if tokens <= 0:
        return 0.0
    return 1.0 - (cos_sum / tokens)

def save_layer_importance(
        wm_dir: str,
        model_name: str,
        version_str: str,
        n_tokens: int,
        per_layer: Dict[int, Dict[str, float]]
) -> str:
    """
    Persist per-block Block Influence, merging with whatever an earlier chunk left.

    Chunked whitening runs cover disjoint layer ranges, so entries accumulate
    across runs instead of replacing one another.
    """
    os.makedirs(wm_dir, exist_ok=True)
    path = layer_importance_path(wm_dir)

    merged: Dict[int, Dict[str, float]] = {}
    existing = None

    if os.path.exists(path):
        try:
            existing = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:
            print(f"[IMPORTANCE][WARNING] Overwriting unreadable {path}: {error}")
            existing = None

    is_reusable = (
        isinstance(existing, dict)
        and existing.get("model_name") == model_name
        and existing.get("version") == version_str
        and int(existing.get("n_tokens", -1)) == int(n_tokens)
    )
    if is_reusable:
        merged.update(existing.get("per_layer", {})) # pyright: ignore[reportOptionalMemberAccess]
    elif existing is not None:
        print(f"[IMPORTANCE][WARNING] Discarding {path}, it was built for another configuration")

    merged.update(per_layer)

    torch.save(
        {
            "model_name": model_name,
            "version": version_str,
            "n_tokens": int(n_tokens),
            "metric": "block_influence",
            "per_layer": merged,
        },
        path,
    )

    print(f"[IMPORTANCE] Saved Block Influence for {len(merged)} decoder blocks -> {path}")

    return path

def load_layer_importance(
        wm_dir: str,
        model_name: str,
        version_str: str,
        n_tokens: int,
        num_layers: Optional[int] = None
) -> Optional[Dict[int, float]]:
    """
    Read cached Block Influence, validated against the run that produced it.

    Returns None when absent, stale, or incomplete, so callers can fall back
    rather than silently allocating on another configuration's signal.
    """
    path = layer_importance_path(wm_dir)

    if not os.path.exists(path):
        return None

    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        print(f"[IMPORTANCE][WARNING] Could not read {path}: {error}")
        return None

    def reject(reason: str) -> None:
        print(f"[IMPORTANCE][WARNING] Ignoring {path}: {reason}")

    if not isinstance(blob, dict):
        reject("unexpected file structure")
        return None

    if blob.get("model_name") != model_name:
        reject(f"built for model {blob.get('model_name')}")
        return None

    if blob.get("version") != version_str:
        reject(f"built for whitening version {blob.get('version')}")
        return None

    if int(blob.get("n_tokens", -1)) != int(n_tokens):
        reject(f"built for n_tokens={blob.get('n_tokens')}, expected {n_tokens}")
        return None

    per_layer = blob.get("per_layer", {})
    importance = {
        int(idx): block_influence_from_sums(float(entry["cos_sum"]), int(entry["tokens"]))
        for idx, entry in per_layer.items()
    }

    if num_layers is not None and len(importance) < num_layers:
        reject(f"only {len(importance)}/{num_layers} decoder blocks recorded")
        return None

    return importance

def spectra_dir(wm_dir: str) -> str:
    return os.path.join(wm_dir, SPECTRA_DIRNAME)

def spectrum_path(wm_dir: str, key: str) -> str:
    return os.path.join(spectra_dir(wm_dir), key.replace(".", "_") + ".pt")

def save_spectrum(wm_dir: str, key: str, singular_values: torch.Tensor, n_tokens: int) -> str:
    """
    Cache the raw singular values of one whitened matrix.

    Raw rather than rescaled: every score metric is derivable from the spectrum,
    so caching it once makes re-scoring under any metric free, and it keeps the
    cache valid independently of how scores are normalized later on.
    """
    directory = spectra_dir(wm_dir)
    os.makedirs(directory, exist_ok=True)
    path = spectrum_path(wm_dir, key)

    torch.save(
        {
            "key": key,
            "n_tokens": int(n_tokens),
            "singular_values": singular_values.detach().to(torch.float64).cpu(),
        },
        path,
    )

    return path

def load_spectrum(wm_dir: str, key: str, n_tokens: int) -> Optional[torch.Tensor]:
    """Read a cached raw spectrum, or None when it is absent or built elsewhere"""
    path = spectrum_path(wm_dir, key)

    if not os.path.exists(path):
        return None

    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        print(f"[SPECTRA][WARNING] Could not read {path}: {error}")
        return None

    if not isinstance(blob, dict) or blob.get("key") != key:
        print(f"[SPECTRA][WARNING] Ignoring {path}: it describes another matrix")
        return None

    if int(blob.get("n_tokens", -1)) != int(n_tokens):
        print(
            f"[SPECTRA][WARNING] Ignoring {path}: built for "
            f"n_tokens={blob.get('n_tokens')}, expected {n_tokens}",
        )
        return None

    return blob["singular_values"]

def load_spectra_cache(wm_dir: str) -> Tuple[Dict[str, torch.Tensor], Optional[int]]:
    """
    Read every cached spectrum in a whitening directory, keyed by matrix path.

    Each record carries its own key and calibration size, so the layout on disk
    is never re-derived from filenames. Returns the shared token count alongside
    the spectra; it is None on an empty cache, and a cache mixing two
    calibration sizes is a hard error rather than a silent comparison of scores
    that were never on the same scale.
    """
    directory = spectra_dir(wm_dir)

    if not os.path.isdir(directory):
        return {}, None

    spectra: Dict[str, torch.Tensor] = {}
    token_counts = set()

    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".pt"):
            continue

        path = os.path.join(directory, filename)

        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:
            print(f"[SPECTRA][WARNING] Could not read {path}: {error}")
            continue

        if not isinstance(blob, dict) or "key" not in blob:
            print(f"[SPECTRA][WARNING] Ignoring {path}: unexpected file structure")
            continue

        spectra[str(blob["key"])] = blob["singular_values"]
        token_counts.add(int(blob.get("n_tokens", -1)))

    if len(token_counts) > 1:
        raise ValueError(
            f"Spectra in {directory} were built with different calibration sizes "
            f"({sorted(token_counts)}). Truncation-style scores are not comparable "
            f"across them, so rebuild the cache with one `--max_whitening_samples`",
        )

    return spectra, token_counts.pop() if token_counts else None

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

def build_shape_map(
        layers_str: List[str],
        layers_list: List[nn.Module],
        attributes: List[str]
) -> Dict[str, Tuple[int, int]]:
    """
    (out_features, in_features) of every target, read off the loaded weights.

    Rank-space policies need more than the parameter count: one rank of the
    factorization costs `out + in` parameters, and that cost is not recoverable
    from `out * in` alone.
    """
    shape_map: Dict[str, Tuple[int, int]] = {}

    for key, layer, attr in zip(layers_str, layers_list, attributes):
        out_features, in_features = getattr(layer, attr).weight.shape
        shape_map[key] = ( int(out_features), int(in_features) )

    return shape_map

def rank_cost(shape_map: Dict[str, Tuple[int, int]], keys: List[str]) -> torch.Tensor:
    """Parameters one extra rank costs per matrix, `omega = out + in` in D-Rank"""
    return torch.tensor(
        [shape_map[key][0] + shape_map[key][1] for key in keys],
        dtype=torch.float64,
    )

def matrix_ratio_cap(
        shape: Tuple[int, int],
        max_ratio: float,
        min_rank_fraction: float
) -> float:
    """
    Tightest removal ratio that still leaves `min_rank_fraction` of full rank.

    A ratio is a share of parameters, and the rank it buys depends on the
    shape: `rank = out * in * (1 - ratio) / (out + in)`, so demanding
    `rank >= f * min(out, in)` reads

        ratio <= 1 - f * (out + in) / max(out, in)

    which is `1 - 2f` on a square matrix. A scalar `--max_ratio` therefore
    means different things to different shapes -- 0.9 leaves a square matrix
    5% of its rank and a 512x3584 projection 8.75% -- and this is what makes
    the guard rail say the same thing everywhere. `min_rank_fraction` of 0.0
    leaves `max_ratio` alone.
    """
    out_features, in_features = shape
    ceiling = 1.0

    if min_rank_fraction > 0.0:
        ceiling = 1.0 - min_rank_fraction * (out_features + in_features) / max(out_features, in_features)

    return max(0.0, min(max_ratio, ceiling))

def build_cap_map(
        keys: List[str],
        max_ratio: float,
        shape_map: Optional[Dict[str, Tuple[int, int]]] = None,
        min_rank_fraction: float = 0.0
) -> Dict[str, float]:
    """
    Per-matrix ratio ceiling, the single source every policy clamps against.

    Without shapes, or without a rank floor, this is `max_ratio` everywhere and
    the allocation is exactly what a scalar cap produced.
    """
    if shape_map is None or min_rank_fraction <= 0.0:
        return {key: max_ratio for key in keys}

    return {
        key: matrix_ratio_cap(shape_map[key], max_ratio, min_rank_fraction)
        for key in keys
    }

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
        target_total_params: Optional[int] = None,
        bypass_late_layers: int = -1,
        num_layers: Optional[int] = None,
        cap_map: Optional[Dict[str, float]] = None
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

    bypassed_keys = [
        k for k in layers_str
        if is_bypassed_key(k, bypass_early_layers, bypass_late_layers, num_layers)
    ]
    active_keys = [
        k for k in layers_str
        if not is_bypassed_key(k, bypass_early_layers, bypass_late_layers, num_layers)
    ]

    bypassed_removed = sum(param_count_map[k] * bypass_ratio for k in bypassed_keys)
    active_params = sum(param_count_map[k] for k in active_keys)

    active_budget = target_removed - bypassed_removed
    active_capacity = sum(
        param_count_map[k] * (cap_map or {}).get(k, max_ratio) for k in active_keys
    )

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

def entropy_ceiling(length: int) -> float:
    """
    Largest entropy a spectrum of this length can carry, `log N`.

    Dividing by it turns entropy into a scale-free flatness in [0, 1], and its
    exponential turns effective rank into a retained fraction of the spectrum.
    A length of one carries no ceiling, so the caller reports the degenerate
    value directly rather than dividing by zero.
    """
    return math.log(length) if length > 1 else 0.0

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

    The `*_rel` variants divide by what the spectrum's own length allows, which
    is what makes two matrices of different shape comparable: every allocation
    group that mixes shapes reads these scores against each other, and the raw
    forms carry `min(out, in)` in their units.
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
        case ScoreMetric.TRUNCATION_REL:
            # Tail energy as a fraction of the whole, so the scale of ||W||_F drops out
            return (torch.linalg.norm(tail, ord=2) / torch.linalg.norm(singular_values, ord=2)).item()
        case ScoreMetric.TRUNCATION_SQ_REL:
            return (torch.sum(tail.pow(2)) / torch.sum(singular_values.pow(2))).item()
        case ScoreMetric.ENTROPY_REL:
            ceiling = entropy_ceiling(len(singular_values))
            if ceiling == 0.0:
                return 0.0
            return spectrum_entropy(normalized_spectrum(singular_values), eps).item() / ceiling
        case ScoreMetric.ENTROPY_SQ_REL:
            ceiling = entropy_ceiling(len(singular_values))
            if ceiling == 0.0:
                return 0.0
            return spectrum_entropy(normalized_spectrum(singular_values, squared=True), eps).item() / ceiling
        case ScoreMetric.EFF_RANK_REL:
            # Effective rank as a fraction of the full rank, in (0, 1]
            entropy = spectrum_entropy(normalized_spectrum(singular_values), eps)
            return torch.exp(entropy).item() / len(singular_values)
        case ScoreMetric.EFF_RANK_SQ_REL:
            entropy = spectrum_entropy(normalized_spectrum(singular_values, squared=True), eps)
            return torch.exp(entropy).item() / len(singular_values)
        case metric if metric.startswith("norm"):
            # After whitening, norm loss equals to the Lp norm of truncated singular values = p-schatten norm
            # WARNING: with p=2 this is the same of the truncation metric
            return torch.linalg.norm(tail, ord=parse_norm_order(metric)).item()

    raise ValueError(f"Unsupported `score_metric`: {score_metric}")

class CompositeScore(NamedTuple):
    """The two halves of a `composite|<local>|<end_to_end>` score metric"""
    local: ScoreMetric
    end_to_end: str

def parse_composite_metric(score_metric: Union[ScoreMetric, str]) -> Optional[CompositeScore]:
    """
    Split a composite metric into the spectral half and the end-to-end half.

    Returns None for an ordinary metric, so callers can branch on whether a
    second signal has to be fused in after the score pass.
    """
    value = str(getattr(score_metric, "value", score_metric))

    if not value.startswith(COMPOSITE_PREFIX):
        return None

    local, _, end_to_end = value[len(COMPOSITE_PREFIX):].rpartition("|")

    if not local or end_to_end not in END_TO_END_SCORES:
        raise ValueError(
            f"Invalid composite score metric '{value}'. Expected "
            f"composite|<local>|<end_to_end> with end_to_end one of {list(END_TO_END_SCORES)}",
        )

    return CompositeScore(local=ScoreMetric(local), end_to_end=end_to_end)

def normalize_block_influence(importance_map: Dict[int, float]) -> Dict[int, float]:
    """
    Min-max normalize Block Influence across blocks and shift it into [1, 2].

    Swift-SVD Eq. 12 raises this factor to a power, so it has to stay at or
    above 1: a block can then only ever raise a matrix's score, never send it to
    zero because its neighbours happened to move the residual stream more.
    """
    values = list(importance_map.values())
    lowest = min(values)
    span = max(values) - lowest

    # A constant influence carries no preference, so every block weighs the same
    return {
        layer: (value - lowest) / span + 1.0 if span > 0 else 1.0
        for layer, value in importance_map.items()
    }

def compose_scores(
        score_map: Dict[str, float],
        importance_map: Optional[Dict[int, float]],
        fusion_alpha: float = 0.5
) -> Dict[str, float]:
    """
    Fuse each matrix's spectral score with the influence of the block it sits in.

    Geometric fusion, following Swift-SVD Eq. 12 and ROCKET Eq. 5:

        s = beta^alpha * log(e + local)^(1 - alpha)

    The local score goes through `log(e + .)` so that both factors are at least
    1 whatever metric produced it, which is also what keeps the geometric mean
    monotone in both. `alpha = 0` leaves the local score alone, in log form,
    which is Swift-SVD's local-only candidate; `alpha = 1` leaves the block
    importance alone.

    Watch the dynamic range: `beta^0.5` spans [1, 1.41] while `log(e + local)`
    is typically 5 to 8 for a truncation score, so the fused score is far
    flatter than the raw one. Under the same `--offset` a composite allocation
    therefore sits closer to homogeneous, which is a property of the fusion and
    not evidence that the second signal adds nothing.
    """
    if not 0.0 <= fusion_alpha <= 1.0:
        raise ValueError("`fusion_alpha` must be in [0.0, 1.0]")

    if not importance_map:
        raise ValueError(
            "A composite score metric needs the cached Block Influence. Run the "
            "whitening pass so that `layer_importance.pt` sits next to the "
            "whitening matrices, then compress against the same calibration set",
        )

    normalized = normalize_block_influence(importance_map)

    fused: Dict[str, float] = {}
    missing: List[str] = []

    for key, local in score_map.items():
        layer_idx = get_layer_idx_from_key(key)

        if layer_idx not in normalized:
            missing.append(key)
            continue

        local_factor = math.log(math.e + max(0.0, float(local)))
        fused[key] = (normalized[layer_idx] ** fusion_alpha) * (local_factor ** (1.0 - fusion_alpha))

    if missing:
        raise ValueError(
            f"No Block Influence for {len(missing)} of {len(score_map)} scored matrices, "
            f"e.g. '{missing[0]}'. The cache does not cover every compressed block",
        )

    return fused

def group_params_tensor(param_count_map: Dict[str, int], keys: List[str]) -> torch.Tensor:
    return torch.tensor([param_count_map[key] for key in keys], dtype=torch.float64)

def group_scores_tensor(score_map: Dict[str, float], keys: List[str]) -> torch.Tensor:
    return torch.tensor([score_map[key] for key in keys], dtype=torch.float64)

def group_caps_tensor(
        cap_map: Optional[Dict[str, float]],
        keys: List[str],
        max_ratio: float
) -> torch.Tensor:
    """
    Ratio ceiling per matrix of one group, falling back to the scalar cap.

    Policies take `cap_map` as optional so that calling one directly, as the
    offline tooling and the tests do, still works with `max_ratio` alone.
    """
    if cap_map is None:
        return torch.full((len(keys),), float(max_ratio), dtype=torch.float64)

    return torch.tensor([cap_map.get(key, max_ratio) for key in keys], dtype=torch.float64)

def clamp_group_budget(params: torch.Tensor, group_budget: float, caps: torch.Tensor) -> float:
    """
    What a group can actually be asked to give up.

    The shell clamps the global budget to the global capacity, which does not
    bound each group individually once an outer policy has divided it, so every
    policy re-clamps to its own capacity before allocating.
    """
    capacity = float((params * caps).sum().item())
    return max(0.0, min(float(group_budget), capacity))

def waterfill_ratios(
        params: torch.Tensor,
        weights: torch.Tensor,
        group_budget: float,
        caps: torch.Tensor
) -> torch.Tensor:
    """
    Spread a removal budget over matrices in proportion to `weights`.

    Ratio-space allocation: what is divided is the removed parameters, so
    `sum_i param_i * ratio_i = group_budget` holds by construction. Matrices
    that reach their own ceiling are frozen and their unspent share flows to the
    rest, which is what makes this a water-fill rather than a single division.
    """
    remaining_budget = clamp_group_budget(params, group_budget, caps)

    ratios = torch.zeros_like(params)
    active = torch.ones_like(params, dtype=torch.bool)

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
        cap_left = caps[idx] - ratios[idx]

        clipped = proposed >= cap_left

        if not bool(clipped.any()):
            ratios[idx] += proposed
            remaining_budget = 0.0
            break

        clipped_idx = idx[clipped]
        spend = float((params[clipped_idx] * cap_left[clipped]).sum().item())

        ratios[clipped_idx] = caps[clipped_idx]
        remaining_budget -= spend
        active[clipped_idx] = False

    return ratios

def bounded_proportional_split(
        share: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        total: float
) -> torch.Tensor:
    """
    Divide `total` in proportion to `share`, keeping every entry within bounds.

    Solves for the scale `c` with `sum_i clamp(c * share_i, lower_i, upper_i) =
    total`. That sum is non-decreasing in `c`, so a bisection brackets the
    answer and the pinned set it settles on then gives `c` in closed form.

    Pinning greedily instead — clamping the one-shot division and re-dividing
    the remainder — is not equivalent: an entry pinned to its lower bound early
    can turn out to be the only one able to absorb what the others cannot take,
    and the budget it never receives is silently lost.
    """
    share = torch.clamp(share, min=0.0)

    lower_sum = float(lower.sum().item())
    upper_sum = float(upper.sum().item())

    if total <= lower_sum:
        return lower.clone()

    if total >= upper_sum:
        return upper.clone()

    def filled(scale: float) -> torch.Tensor:
        return torch.clamp(share * scale, min=lower, max=upper)

    low = 0.0
    high = 1.0

    for _ in range(MAX_BISECTION_STEPS):
        if float(filled(high).sum().item()) >= total:
            break
        high *= 2.0

    for _ in range(MAX_BISECTION_STEPS):
        middle = 0.5 * (low + high)

        if float(filled(middle).sum().item()) < total:
            low = middle
        else:
            high = middle

    values = filled(high)
    interior = (values > lower) & (values < upper)
    denom = float(share[interior].sum().item())

    # With the pinned entries known, the scale is exact instead of bisected
    if denom > 0:
        pinned_total = float(values[~interior].sum().item())
        values = torch.where(interior, share * ((total - pinned_total) / denom), values)

    return torch.clamp(values, min=lower, max=upper)

def allocate_param_weighted_group(
        keys: List[str],
        score_map: Dict[str, float],
        param_count_map: Dict[str, int],
        group_budget: float,
        max_ratio: float = 0.9,
        cap_map: Optional[Dict[str, float]] = None,
        offset: float = 1.5
) -> Dict[str, float]:
    """
    Allocates removal budget inside one group, the `waterfill` inner policy.

    This preserves:
        sum_i param_i * ratio_i = group_budget

    instead of:
        mean_i ratio_i = target_ratio

    The parameter weighting is where this departs from SVD-LLM V2's Algorithm 1,
    which preserves the mean ratio instead and so does not hit a global
    parameter target once a group mixes matrix shapes.
    """
    # TODO understand well how it works
    if not keys:
        return {}

    params = group_params_tensor(param_count_map, keys)
    weights = redundancy_from_scores(group_scores_tensor(score_map, keys), offset)
    ratios = waterfill_ratios(params, weights, group_budget, group_caps_tensor(cap_map, keys, max_ratio))

    return {k: float(r) for k, r in zip(keys, ratios.tolist())}

def allocate_softmax_temp(
        keys: List[str],
        score_map: Dict[str, float],
        param_count_map: Dict[str, int],
        group_budget: float,
        max_ratio: float = 0.9,
        cap_map: Optional[Dict[str, float]] = None,
        softmax_temp: float = 1.0
) -> Dict[str, float]:
    """
    MoDeGPT's entropic allocation (Eq. 10-11), the `softmax_temp` inner policy.

    Solves `max sum_i s_i (1 - r_i) + eps H(r)` under a fixed budget, whose
    optimum for large `eps` is `r proportional to softmax(-s / eps)`. The
    entropic term is what separates this from `waterfill`: `1 / log(s + offset)`
    reads as the same trade-off without a regularizer to name.

    Two deliberate departures from the paper. MoDeGPT preserves the mean ratio
    (`phi = L phi_avg softmax(-s/eps)`) while the budget here is in parameters,
    which is the same adaptation `waterfill` makes and reduces to the paper when
    every matrix has the same size. And scores are min-max normalized within the
    group before the softmax, because MoDeGPT's Block Influence already lives in
    [0, 1] while ours span orders of magnitude by metric; that also fixes the
    largest-to-smallest weight ratio at `exp(1 / softmax_temp)`, whatever the
    metric.
    """
    if not keys:
        return {}

    if softmax_temp <= 0.0:
        raise ValueError("`softmax_temp` must be > 0.0")

    scores = group_scores_tensor(score_map, keys)
    scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    span = float((scores.max() - scores.min()).item())
    # A constant score carries no preference, so every matrix weighs the same
    normalized = (scores - scores.min()) / span if span > 0 else torch.zeros_like(scores)

    weights = torch.softmax(-normalized / softmax_temp, dim=0)
    params = group_params_tensor(param_count_map, keys)
    ratios = waterfill_ratios(params, weights, group_budget, group_caps_tensor(cap_map, keys, max_ratio))

    return {k: float(r) for k, r in zip(keys, ratios.tolist())}

def allocate_swift_pool(
        keys: List[str],
        score_map: Dict[str, float],
        param_count_map: Dict[str, int],
        group_budget: float,
        max_ratio: float = 0.9,
        cap_map: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Swift-SVD's floor-plus-pool allocation (Alg. 2), the `swift_pool` policy.

    Every matrix starts at the most aggressive ratio allowed and then buys ratio
    back in proportion to its score, out of a flexible pool:

        ratio_i = max_ratio - pool * score_i / sum_j (score_j * param_j)

    which is linear in the score, where `waterfill` is logarithmic and
    `drank_lagrangian` is a square root. That difference in shape is the whole
    point of comparing them.

    Two departures. Swift-SVD's guaranteed minimal rank `k_bar * delta` is
    dropped in favour of the shared `--max_ratio` floor, which at a 0.2 target
    is a far lower floor than the paper's `delta = 0.5` (an effective cap of 0.6
    against 0.9), so this runs more aggressively than the original. And the
    paper's pool is in raw rank units, which only preserves a parameter budget
    when every matrix in the group has the same shape; the reduction is applied
    to the ratio instead, so the two agree exactly under `--group_criterion
    type` and stay budget-exact when a group mixes shapes.

    The paper's `s_i = beta^alpha * log(e + eps)^(1-alpha)` is a *score*, not
    part of this policy: reproducing Swift-SVD means pairing it with the
    composite score metric, not with a raw truncation score.
    """
    if not keys:
        return {}

    params = group_params_tensor(param_count_map, keys)
    scores = torch.nan_to_num(group_scores_tensor(score_map, keys), nan=0.0, posinf=0.0, neginf=0.0)
    scores = torch.clamp(scores, min=0.0)

    caps = group_caps_tensor(cap_map, keys, max_ratio)
    budget = clamp_group_budget(params, group_budget, caps)

    ratios = caps.clone()
    active = torch.ones_like(params, dtype=torch.bool)

    # Matrices whose score would buy back more than the whole ratio are pinned
    # dense and the rest re-solve for the pool they are left with
    for _ in range(params.numel() + 1):
        idx = torch.nonzero(active, as_tuple=False).flatten()

        if idx.numel() == 0:
            break

        p = params[idx]
        s = scores[idx]

        # What the active matrices could give up beyond the target, i.e. the pool
        pool = float((p * caps[idx]).sum().item()) - budget
        denom = float((p * s).sum().item())

        if pool <= 0.0 or denom <= 0.0:
            # No pool to hand out, or no signal to hand it out by: split flat.
            # Capped because pinning matrices dense can leave the rest without
            # the capacity to cover the budget, and the shell reports the drift
            active_params = float(p.sum().item())
            flat = budget / active_params if active_params > 0 else 0.0
            ratios[idx] = torch.clamp(torch.full_like(p, flat), max=caps[idx])
            break

        proposed = caps[idx] - pool * s / denom
        dense = proposed < 0.0

        if not bool(dense.any()):
            ratios[idx] = proposed
            break

        pinned = idx[dense]
        ratios[pinned] = 0.0
        active[pinned] = False

    return {k: float(r) for k, r in zip(keys, ratios.tolist())}

def allocate_drank_lagrangian(
        keys: List[str],
        score_map: Dict[str, float],
        param_count_map: Dict[str, int],
        group_budget: float,
        max_ratio: float = 0.9,
        cap_map: Optional[Dict[str, float]] = None,
        shape_map: Optional[Dict[str, Tuple[int, int]]] = None
) -> Dict[str, float]:
    """
    D-Rank's Lagrangian allocation (Eq. 3-7), the `drank_lagrangian` policy.

    Minimizes `sum_i score_i / k_i` under `sum_i k_i * omega_i = retained`, whose
    stationary point is `k_i proportional to sqrt(score_i / omega_i)` with
    `omega_i = out_i + in_i` the parameter cost of one rank. In retained
    parameters that reads `omega_i k_i proportional to sqrt(score_i * omega_i)`,
    which is Eq. 7 rewritten in the units the budget is measured in.

    This allocates rank rather than ratio, so unlike the ratio-space policies a
    uniform score does *not* leave every matrix at the flat ratio: cheap-per-rank
    matrices still get more rank. That bias is a property of the family, not a
    bug, and it is what makes the comparison worth running.

    D-Rank's `R_eff` is exactly `--score_metric eff_rank_sq`. What is not
    reproduced here is its grouped horizontal concatenation under a shared basis
    and its Q/K-to-V rebalancing, both of which change the decomposition itself
    rather than the allocation.
    """
    if not keys:
        return {}

    if shape_map is None:
        raise ValueError(
            "`drank_lagrangian` allocates in rank space and needs `shape_map`: "
            "the parameter cost of one rank is out + in, which the parameter "
            "count alone does not carry",
        )

    params = group_params_tensor(param_count_map, keys)
    omega = rank_cost(shape_map, keys)
    scores = torch.nan_to_num(group_scores_tensor(score_map, keys), nan=0.0, posinf=0.0, neginf=0.0)
    scores = torch.clamp(scores, min=0.0)

    caps = group_caps_tensor(cap_map, keys, max_ratio)
    budget = clamp_group_budget(params, group_budget, caps)
    retained_total = float(params.sum().item()) - budget

    share = torch.sqrt(scores * omega)

    if float(share.sum().item()) <= 0.0:
        # No signal at all: fall back to the flat ratio rather than to a rank
        # rule nobody asked for
        print("[BUDGET][WARNING] drank_lagrangian got no usable score, falling back to a flat ratio")
        share = params.clone()

    retained = bounded_proportional_split(
        share=share,
        lower=params * (1.0 - caps),
        upper=params.clone(),
        total=retained_total,
    )
    ratios = 1.0 - retained / params

    return {k: float(r) for k, r in zip(keys, ratios.tolist())}

def allocate_group_budgets_param_share(
        groups: Dict[str, List[str]],
        param_count_map: Dict[str, int],
        remaining_budget: float
) -> Dict[str, float]:
    """
    Split the removal budget across groups in proportion to their parameters.

    Every group is then asked for the same average ratio, which is why this
    outer policy leaves the allocation decision entirely to the inner one, and
    why it is the controlled baseline the hierarchical outer level is measured
    against.
    """
    grouped_params = sum(param_count_map[k] for keys in groups.values() for k in keys)

    if grouped_params <= 0:
        return {name: 0.0 for name in groups}

    return {
        name: remaining_budget * sum(param_count_map[k] for k in keys) / grouped_params
        for name, keys in groups.items()
    }

def allocate_group_budgets_waterfill(
        groups: Dict[str, List[str]],
        param_count_map: Dict[str, int],
        remaining_budget: float,
        max_ratio: float = 0.9,
        cap_map: Optional[Dict[str, float]] = None,
        group_scores: Optional[Dict[str, float]] = None,
        outer_offset: float = 1.5
) -> Dict[str, float]:
    """
    Water-fill the budget across groups by their end-to-end importance.

    This is `waterfill` one level up: a decoder block that transforms the
    residual stream more is more important, so it is asked for less removal, and
    the block-level budget it gives up flows to the rest. Its own knob
    (`outer_offset`) is deliberately separate from the inner one, so that tuning
    how matrices compete inside a block cannot silently change how blocks
    compete against each other.
    """
    if group_scores is None:
        raise ValueError(
            "The `waterfill` outer allocation needs one score per group. Use "
            "`--group_criterion hierarchical`, which scores each decoder block by "
            "its Block Influence, and make sure the whitening cache holds it",
        )

    names = [name for name, keys in groups.items() if keys]
    missing = [name for name in names if name not in group_scores]

    if missing:
        raise ValueError(
            f"No score for {len(missing)} of {len(names)} groups, e.g. '{missing[0]}'. "
            f"The cached Block Influence does not cover every decoder block being compressed",
        )

    params = torch.tensor(
        [sum(param_count_map[key] for key in groups[name]) for name in names],
        dtype=torch.float64,
    )
    weights = redundancy_from_scores(
        torch.tensor([group_scores[name] for name in names], dtype=torch.float64),
        outer_offset,
    )
    # A block can give up no more than its own matrices can, which is their
    # ceilings weighted by what each one contributes to the block
    caps = torch.tensor(
        [
            sum(param_count_map[key] * (cap_map or {}).get(key, max_ratio) for key in groups[name])
            / max(1, sum(param_count_map[key] for key in groups[name]))
            for name in names
        ],
        dtype=torch.float64,
    )
    ratios = waterfill_ratios(params, weights, remaining_budget, caps)

    budgets = {name: 0.0 for name in groups}
    budgets.update({
        name: float(group_params * ratio)
        for name, group_params, ratio in zip(names, params.tolist(), ratios.tolist())
    })

    return budgets

# An allocation policy maps its targets to a removal figure per target: ratios per
# matrix for an inner policy, budgets per group for an outer one
AllocationPolicy = Callable[..., Dict[str, float]]

# Data every policy of that level may ask for. Anything else it declares in its
# signature is a knob, which is how a run records which flags actually applied.
# A policy only ever receives what it names, so none has to carry a parameter it
# does not use
INNER_POLICY_ARGS = ( "keys", "score_map", "param_count_map", "group_budget", "max_ratio", "cap_map", "shape_map" )
OUTER_POLICY_ARGS = ( "groups", "param_count_map", "remaining_budget", "max_ratio", "cap_map", "group_scores" )

# Only wired policies live here, so the CLI can advertise exactly what runs
INNER_POLICIES: Dict[InnerAllocation, AllocationPolicy] = {
    InnerAllocation.WATERFILL: allocate_param_weighted_group,
    InnerAllocation.DRANK_LAGRANGIAN: allocate_drank_lagrangian,
    InnerAllocation.SWIFT_POOL: allocate_swift_pool,
    InnerAllocation.SOFTMAX_TEMP: allocate_softmax_temp,
}

OUTER_POLICIES: Dict[OuterAllocation, AllocationPolicy] = {
    OuterAllocation.PARAM_SHARE: allocate_group_budgets_param_share,
    OuterAllocation.WATERFILL: allocate_group_budgets_waterfill,
}

# Ratio-space policies decide a removal fraction, so a score that carries no
# preference leaves every matrix at the flat ratio. Rank-space policies decide a
# retained rank instead, and a matrix that costs fewer parameters per rank still
# gets more of them even under a constant score. That difference in implicit
# bias is the reason both families are in the comparison
RATIO_SPACE_POLICIES = frozenset({
    InnerAllocation.WATERFILL,
    InnerAllocation.SWIFT_POOL,
    InnerAllocation.SOFTMAX_TEMP,
})
RANK_SPACE_POLICIES = frozenset({ InnerAllocation.DRANK_LAGRANGIAN })

def resolve_policy(
        value: Union[Enum, str],
        enum_cls: Type[Enum],
        registry: Dict[Any, AllocationPolicy],
        label: str
) -> Tuple[Any, AllocationPolicy]:
    """
    Coerce a policy name to its enum member and look up its implementation.

    The enum is the agreed vocabulary while the registry is what is actually
    wired, so a name that is spelled right but not implemented yet fails with a
    different, clearer error than a typo does.
    """
    if isinstance(value, str):
        try:
            value = enum_cls(value)
        except ValueError:
            raise ValueError(
                f"Invalid `{label}`: '{value}'. "
                f"Expected one of: {[e.value for e in enum_cls]}",
            )

    policy = registry.get(value)

    if policy is None:
        raise NotImplementedError(
            f"`{label}` '{value.value}' is not implemented yet. " # pyright: ignore[reportAttributeAccessIssue]
            f"Available: {[e.value for e in registry]}",
        )

    return value, policy

def resolve_inner_policy(inner_allocation: Union[InnerAllocation, str]) -> Tuple[InnerAllocation, AllocationPolicy]:
    return resolve_policy(inner_allocation, InnerAllocation, INNER_POLICIES, "inner_allocation") # pyright: ignore[reportArgumentType]

def resolve_outer_policy(outer_allocation: Union[OuterAllocation, str]) -> Tuple[OuterAllocation, AllocationPolicy]:
    return resolve_policy(outer_allocation, OuterAllocation, OUTER_POLICIES, "outer_allocation") # pyright: ignore[reportArgumentType]

def policy_knob_names(policy: AllocationPolicy, contract_args: Tuple[str, ...]) -> List[str]:
    """
    Names of the tunables a policy reads, beyond what every policy receives.

    Knobs are declared as plain named parameters, so the signature is the single
    source of truth for which flags apply to a run: no policy has to repeat its
    own knob list, and none can silently ignore one it claims to read.
    """
    parameters = inspect.signature(policy).parameters

    return [
        name for name, parameter in parameters.items()
        if name not in contract_args and parameter.kind is parameter.POSITIONAL_OR_KEYWORD
    ]

def select_policy_knobs(
        policy: AllocationPolicy,
        contract_args: Tuple[str, ...],
        knobs: Dict[str, Any]
) -> Dict[str, Any]:
    """Narrow the run's knobs down to the ones a policy declares, ignoring the rest"""
    names = policy_knob_names(policy, contract_args)
    return {name: knobs[name] for name in names if name in knobs}

def select_policy_arguments(policy: AllocationPolicy, available: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hand a policy exactly the arguments it names, and nothing else.

    Levels of the contract differ in what they need — only the rank-space
    policies care about shapes, only the outer one about per-group scores — so
    filtering by signature keeps every policy free of parameters it ignores.
    """
    parameters = inspect.signature(policy).parameters
    return {name: value for name, value in available.items() if name in parameters}

class AllocationGroups(NamedTuple):
    """Active matrices bucketed for allocation, plus the ones that fell through"""
    groups: Dict[str, List[str]]
    missing_score_keys: List[str]
    unmatched_keys: List[str]

def build_allocation_groups(
        group_criterion: GroupBy,
        active_keys: List[str],
        score_map: Dict[str, float],
        group_patterns: Optional[Dict[str, List[str]]] = None
) -> AllocationGroups:
    """
    Bucket the matrices a budget will be spread over, one bucket per group.

    A matrix without a score cannot be ranked against its peers, so it is
    reported rather than grouped and the caller falls it back to the flat ratio.

    `HIERARCHICAL` buckets exactly like `DECODER`; what separates them is that
    only the hierarchical criterion lets an outer policy see a per-block score,
    so `hierarchical` with the `param_share` outer policy reproduces `decoder`
    to the digit and is the controlled baseline for the outer level.
    """
    groups: Dict[str, List[str]] = defaultdict(list)
    missing_score_keys: List[str] = []
    unmatched_keys: List[str] = []

    match group_criterion:
        case GroupBy.GLOBAL:
            for key in active_keys:
                if key in score_map:
                    groups["global"].append(key)
                else:
                    missing_score_keys.append(key)

        case GroupBy.DECODER | GroupBy.HIERARCHICAL:
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

    return AllocationGroups(
        groups=groups,
        missing_score_keys=missing_score_keys,
        unmatched_keys=unmatched_keys,
    )

def warn_on_degenerate_scores(groups: Dict[str, List[str]], score_map: Dict[str, float]) -> bool:
    """
    Report a score that cannot rank anything within its group.

    A policy only ever compares matrices inside a group, so a score that is
    constant there produces the flat ratio no matter which policy runs. Fusing a
    per-block signal into a per-matrix score does exactly this under per-block
    grouping, and the run would otherwise look heterogeneous while being
    homogeneous to the digit.
    """
    spreads = []

    for keys in groups.values():
        if len(keys) < 2:
            continue

        values = [score_map[key] for key in keys]
        largest = max(abs(value) for value in values)
        spreads.append((max(values) - min(values)) / largest if largest > 0 else 0.0)

    if not spreads or max(spreads) >= DEGENERATE_SCORE_SPREAD:
        return False

    print(
        f"[BUDGET][WARNING] Scores vary by at most {max(spreads):.2e} inside every group, "
        f"so this allocation is homogeneous whatever the policy. A per-block score fused "
        f"into a per-matrix one behaves this way under per-block grouping: it can only "
        f"differentiate through the outer level",
    )

    return True

def allocation_knobs(
        offset: float = 1.5,
        softmax_temp: float = 1.0,
        outer_offset: float = 1.5
) -> Dict[str, Any]:
    """
    Every tunable an allocation policy may read, gathered in one place.

    Policies declare the subset they use, so this is the only list that has to
    grow when a knob is added and no policy can be handed one it never reads.
    """
    return {
        "offset": offset,
        "softmax_temp": softmax_temp,
        "outer_offset": outer_offset,
    }

class AllocationPolicies(NamedTuple):
    """The resolved policy pair of a run, together with the knobs each one reads"""
    inner_allocation: InnerAllocation
    outer_allocation: OuterAllocation
    inner_policy: AllocationPolicy
    outer_policy: AllocationPolicy
    inner_knobs: Dict[str, Any]
    outer_knobs: Dict[str, Any]

    def describe(self) -> Dict[str, Any]:
        """The part worth recording next to a run's results"""
        return {
            "inner_allocation": self.inner_allocation.value,
            "outer_allocation": self.outer_allocation.value,
            "inner_knobs": self.inner_knobs,
            "outer_knobs": self.outer_knobs,
        }

def resolve_allocation_policies(
        inner_allocation: Union[InnerAllocation, str],
        outer_allocation: Union[OuterAllocation, str],
        knobs: Dict[str, Any]
) -> AllocationPolicies:
    """
    Resolve both levels at once and hand each the subset of knobs it declares.

    Callers that only need to report a configuration use the same entry point as
    the allocator itself, so what a run records is what it actually ran.
    """
    inner_allocation, inner_policy = resolve_inner_policy(inner_allocation)
    outer_allocation, outer_policy = resolve_outer_policy(outer_allocation)

    return AllocationPolicies(
        inner_allocation=inner_allocation,
        outer_allocation=outer_allocation,
        inner_policy=inner_policy,
        outer_policy=outer_policy,
        inner_knobs=select_policy_knobs(inner_policy, INNER_POLICY_ARGS, knobs),
        outer_knobs=select_policy_knobs(outer_policy, OUTER_POLICY_ARGS, knobs),
    )

def build_group_scores(
        group_criterion: GroupBy,
        groups: Dict[str, List[str]],
        importance_map: Optional[Dict[int, float]]
) -> Optional[Dict[str, float]]:
    """
    Score each group as a whole, for the outer level to allocate across.

    Only the hierarchical criterion produces these: its groups are decoder
    blocks, which is the one granularity end-to-end Block Influence is measured
    at. Every other criterion returns None, so an outer policy that needs group
    scores fails loudly instead of silently receiving something meaningless.
    """
    if group_criterion is not GroupBy.HIERARCHICAL or importance_map is None:
        return None

    group_scores: Dict[str, float] = {}

    for name, keys in groups.items():
        if not keys:
            continue

        layer_idx = get_layer_idx_from_key(keys[0])

        if layer_idx in importance_map:
            group_scores[name] = float(importance_map[layer_idx])

    return group_scores

def allocate_ratios(
        group_criterion: Union[GroupBy, Literal["global", "decoder", "type", "hierarchical"]],
        score_map: Dict,
        layers_str: List[str],
        target_ratio: float,
        param_count_map: Dict[str, int],
        offset: float = 1.5,
        group_patterns: Dict[str, List[str]] | None = None,
        bypass_early_layers: int = 2,
        bypass_ratio: float = 0.0,
        max_ratio: float = 0.9,
        target_total_params: Optional[int] = None,
        bypass_late_layers: int = -1,
        num_layers: Optional[int] = None,
        inner_allocation: Union[InnerAllocation, str] = InnerAllocation.WATERFILL,
        outer_allocation: Union[OuterAllocation, str] = OuterAllocation.PARAM_SHARE,
        shape_map: Optional[Dict[str, Tuple[int, int]]] = None,
        importance_map: Optional[Dict[int, float]] = None,
        softmax_temp: float = 1.0,
        outer_offset: float = 1.5,
        min_rank_fraction: float = 0.0
) -> Dict[str, float]:
    """
    Redistributes compression budget within each weight group.
    Groups: MLP (gate, up, down), Q proj, K proj, V proj, Attention out proj.

    Within each group, matrices with higher score get a lower
    compression ratio and vice versa.

    Bypassed layers (the first `bypass_early_layers` and the last
    `bypass_late_layers`, either or both) are mathematically isolated from
    redistribution and strictly assigned the bypass_ratio. A bypass_ratio
    of 0.0 means 0% parameter removal (no compression) for those layers.
    In case some layers are bypassed, it still preserves
    the global target_ratio across the entire model,
    giving a higher compression ratio to allowed layers.

    For same-shape TYPE groups, this reduces to the usual V2 behavior.
    For GLOBAL and DECODER groups, this preserves actual removed parameters.

    This function is the shell: it owns grouping, the budget split and all the
    instrumentation, while the two pluggable policies own the arithmetic that
    turns scores into removal figures. Under HIERARCHICAL the two levels use
    different signals: the outer one splits the budget across decoder blocks by
    their end-to-end Block Influence, the inner one splits each block's share by
    the local spectral scores.
    """
    if isinstance(group_criterion, str):
        try:
            group_criterion = GroupBy(group_criterion)
        except ValueError:
            raise ValueError(
                f"Invalid `group_criterion`: '{group_criterion}'. "
                f"Expected one of: {[e.value for e in GroupBy]}",
            )

    knobs = allocation_knobs(offset=offset, softmax_temp=softmax_temp, outer_offset=outer_offset)
    policies = resolve_allocation_policies(inner_allocation, outer_allocation, knobs)
    inner_knobs = policies.inner_knobs
    outer_knobs = policies.outer_knobs

    print(f"\n[BUDGET] Parameter-aware redistribution: {group_criterion.value.upper()}")
    print(f"[BUDGET] Global target ratio: {target_ratio:.6f}")
    print(
        f"[BUDGET] Bypassing first {bypass_early_layers} and last "
        f"{bypass_late_layers} layers with ratio {bypass_ratio:.6f}",
    )
    cap_map = build_cap_map(layers_str, max_ratio, shape_map, min_rank_fraction)
    distinct_caps = sorted(set(round(cap, 6) for cap in cap_map.values()))

    print(f"[BUDGET] Per-matrix max ratio: {max_ratio:.6f}")
    if min_rank_fraction > 0.0:
        print(
            f"[BUDGET] Min retained rank fraction: {min_rank_fraction:.6f} "
            f"-> {len(distinct_caps)} distinct ceiling(s) in "
            f"[{distinct_caps[0]:.6f}, {distinct_caps[-1]:.6f}]",
        )
    print(f"[BUDGET] Outer policy: {policies.outer_allocation.value} | knobs: {outer_knobs}")
    print(f"[BUDGET] Inner policy: {policies.inner_allocation.value} | knobs: {inner_knobs}")

    budget = compute_active_budget(
        layers_str=layers_str,
        param_count_map=param_count_map,
        target_ratio=target_ratio,
        bypass_early_layers=bypass_early_layers,
        bypass_ratio=bypass_ratio,
        max_ratio=max_ratio,
        target_total_params=target_total_params,
        bypass_late_layers=bypass_late_layers,
        num_layers=num_layers,
        cap_map=cap_map,
    )

    selected_total_params = budget.selected_params
    target_total_params = budget.target_total_params
    active_keys = budget.active_keys
    active_budget = budget.active_budget
    active_target_ratio = budget.active_ratio

    # Bypassed matrices are pinned before any redistribution happens
    ratio_map: Dict[str, float] = {k: bypass_ratio for k in budget.bypassed_keys}

    print(f"[BUDGET] Selected params:           {selected_total_params:,}")
    print(f"[BUDGET] Target denominator params: {target_total_params:,}")
    print(f"[BUDGET] Target removed params:     {budget.target_removed:,.0f}")
    print(f"[BUDGET] Bypassed matrices:         {len(budget.bypassed_keys)}")
    print(f"[BUDGET] Bypassed removed params:   {budget.bypassed_removed:,.0f}")
    print(f"[BUDGET] Active matrices:           {len(active_keys)}")
    print(f"[BUDGET] Active params:             {budget.active_params:,}")
    print(f"[BUDGET] Active budget:             {active_budget:,.0f}")
    print(f"[BUDGET] Active target ratio:       {active_target_ratio:.6f}")

    grouping = build_allocation_groups(
        group_criterion=group_criterion,
        active_keys=active_keys,
        score_map=score_map,
        group_patterns=group_patterns,
    )
    groups = grouping.groups
    missing_score_keys = grouping.missing_score_keys
    unmatched_keys = grouping.unmatched_keys

    grouped_keys = set()
    for keys in groups.values():
        grouped_keys.update(keys)

    fallback_keys = [k for k in active_keys if k not in grouped_keys]

    # Fallback keys get the active target ratio
    fallback_removed = 0.0
    for k in fallback_keys:
        ratio_map[k] = active_target_ratio
        fallback_removed += param_count_map[k] * active_target_ratio

    remaining_budget = max(0.0, active_budget - fallback_removed)
    grouped_params = sum(param_count_map[k] for k in grouped_keys)

    print(f"[BUDGET] Grouped active params:    {grouped_params:,}")
    print(f"[BUDGET] Fallback keys:            {len(fallback_keys)}")
    print(f"[BUDGET] Remaining group budget:   {remaining_budget:,.0f}")

    warn_on_degenerate_scores(groups, score_map)

    group_scores = build_group_scores(group_criterion, groups, importance_map)

    if group_scores is not None:
        print(f"[BUDGET] Group scores available for {len(group_scores)}/{len(groups)} groups")

    group_budgets = policies.outer_policy(
        **select_policy_arguments(
            policies.outer_policy,
            {
                "groups": groups,
                "param_count_map": param_count_map,
                "remaining_budget": remaining_budget,
                "max_ratio": max_ratio,
                "cap_map": cap_map,
                "group_scores": group_scores,
                **outer_knobs,
            },
        ),
    )

    for group_name, keys in groups.items():
        if not keys:
            print(f"  [GROUP: {group_name}] Empty")
            continue

        group_params = sum(param_count_map[k] for k in keys)
        group_budget = group_budgets.get(group_name, 0.0)

        group_ratio_map = policies.inner_policy(
            **select_policy_arguments(
                policies.inner_policy,
                {
                    "keys": keys,
                    "score_map": score_map,
                    "param_count_map": param_count_map,
                    "group_budget": group_budget,
                    "max_ratio": max_ratio,
                    "cap_map": cap_map,
                    "shape_map": shape_map,
                    **inner_knobs,
                },
            ),
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
            f"actual_removed~{actual_group_removed:>14,.0f}",
        )

        for k in keys:
            # The offset only shapes the allocation of the policies that read it
            offset_note = ""
            if "offset" in inner_knobs:
                offset_note = f" (+ offset = {(score_map[k] + offset):.6f})"

            print(
                f"    - {k:<55} "
                f"| params={param_count_map[k]:>12,} "
                f"| ratio={ratio_map[k]:.6f} "
                f"| score={score_map[k]:.6f}{offset_note}",
            )

    actual_removed = sum(
        param_count_map[k] * ratio_map.get(k, 0.0)
        for k in layers_str
    )

    print("\n[BUDGET] Allocation Summary:")
    print(f"  - Target overall ratio:                 {target_ratio:.6f}")
    print(f"  - Actual selected ratio approx: {actual_removed / selected_total_params:.6f}")
    print(f"  - Actual overall ratio approx:  {actual_removed / target_total_params:.6f}")
    print(f"  - Target removed:               {budget.target_removed:,.0f}")
    print(f"  - Actual removed:               {actual_removed:,.0f}")
    print(f"  - Missing score keys:           {len(missing_score_keys)}")
    print(f"  - Unmatched keys:               {len(unmatched_keys)}")
    print("-" * 80 + "\n")

    return ratio_map

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
        van: Union[LowRank, HeadBlockLowRank],
        device: str = "cuda"
) -> None:
    W_hat = van.to(device).dense_weight().float()
    W_orig = layer_attr.weight.to(device).float()

    rel_w = (W_orig - W_hat).norm() / W_orig.norm().clamp_min(1e-12)
    print(f"[CHECK] {layer_name}: Relative error of reduced rank reconstruction: {rel_w:.3e}")

    van.cpu()

@torch.no_grad()
def check_lowrank_equivalence(
        layer_name: str,
        layer_attr: nn.Linear,
        van: Union[LowRank, HeadBlockLowRank],
        device: str = "cuda"
) -> None:
    input_dtype = layer_attr.weight.dtype
    factor_dtype = van.factor_dtype

    x = torch.randn(
        2, 8, layer_attr.in_features,
        device=device,
        dtype=input_dtype,
    )

    van = van.to(device).eval()

    # Match LowRank.forward(): input is cast to factor dtype internally
    x_factor = x.to(factor_dtype)

    W_hat = van.dense_weight()
    b = van.bias if isinstance(van, HeadBlockLowRank) else van.W_u.bias

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
        van: Union[LowRank, HeadBlockLowRank],
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

def is_lowrank_parameter(name: str) -> bool:
    """
    Whether a state-dict key belongs to a low-rank factor.

    `HeadBlockLowRank` holds `W_u` as a bare parameter rather than a linear, so
    the key ends there and a trailing-dot test would miss it
    """
    return ".W_u" in name or ".W_v" in name

def infer_lowrank_dtype(state_dict: Dict[str, torch.Tensor]) -> Optional[torch.dtype]:
    """Read the dtype the low-rank factors were saved with"""
    for name, tensor in state_dict.items():
        if is_lowrank_parameter(name):
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
        if is_lowrank_parameter(name):
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
def head_block_entry(heads: int, rank: int) -> Dict[str, int]:
    """
    A `rank_map` value describing a block-diagonal factorization.

    Plain integers stay the joint `LowRank` form, so every checkpoint written
    before head-block truncation existed reads back unchanged.
    """
    return {"kind": HEAD_BLOCK_KIND, "heads": int(heads), "rank": int(rank)}

def is_head_block_entry(entry: Any) -> bool:
    return isinstance(entry, dict) and entry.get("kind") == HEAD_BLOCK_KIND

def apply_lowrank(
        model: nn.Module,
        rank_map: Dict[str, Any],
        state_dict: Optional[Dict[str, torch.Tensor]] = None
) -> None:
    """
    Replace the linear layers listed in `rank_map` with low-rank modules.

    rank_map keys are full module paths, e.g. 'model.layers.0.mlp.down_proj'.
    A plain integer value installs a `LowRank`; a `head_block_entry` installs a
    `HeadBlockLowRank` at that per-head rank. When a state dict is given, the
    factor dtype is taken from it so mixed precision checkpoints keep their
    compressed dtype.
    """
    for layer_name, entry in rank_map.items():
        parent, attr_name = get_parent_module(model, layer_name)
        old = getattr(parent, attr_name)

        if state_dict is not None:
            factor_dtype = state_dict[f"{layer_name}.W_v.weight"].dtype
        else:
            factor_dtype = old.weight.dtype

        if is_head_block_entry(entry):
            lowrank = HeadBlockLowRank(
                in_features=old.in_features,
                out_features=old.out_features,
                heads=entry["heads"],
                rank=entry["rank"],
                bias=old.bias is not None,
            ).to(device=old.weight.device, dtype=factor_dtype)
        else:
            lowrank = LowRank(
                in_features=old.in_features,
                out_features=old.out_features,
                rank=entry,
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
        is_v2: bool,
        bypass_late_layers: int = -1,
        max_ratio: float = 0.9,
        inner_allocation: str = InnerAllocation.WATERFILL.value,
        outer_allocation: str = OuterAllocation.PARAM_SHARE.value,
        bypass_ratio: float = DEFAULT_BYPASS_RATIO,
        fusion_alpha: float = DEFAULT_FUSION_ALPHA,
        seed: Optional[int] = DEFAULT_SEED,
        offset: float = 1.5,
        softmax_temp: float = 1.0,
        outer_offset: float = 1.5,
        min_rank_fraction: float = DEFAULT_MIN_RANK_FRACTION,
        head_block_svd: bool = False
) -> str:
    """
    Encode a whole compression configuration into one filename token.

    This name identifies the checkpoint, the evaluation JSON and the log file of
    a run, and `generate_tables.py` parses it back to label the result tables,
    so it has to stay stable.

    Dimensions added after the original layout only emit a token when they leave
    their default, which keeps every pre-existing run name byte-identical.
    """
    # Enum members hold their value as a plain string, keep the token stable
    group_criterion = str(getattr(group_criterion, "value", group_criterion))
    score_metric = str(getattr(score_metric, "value", score_metric))
    inner_allocation = str(getattr(inner_allocation, "value", inner_allocation))
    outer_allocation = str(getattr(outer_allocation, "value", outer_allocation))

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
        # Score metrics carry a separator that filenames cannot hold. "norm|2"
        # drops it, since names in that form are already on disk; a composite
        # metric turns it into "_", which stays readable across three parts
        if score_metric.startswith(COMPOSITE_PREFIX):
            score_metric_str = "_" + score_metric.replace("|", "_")
        else:
            score_metric_str = "_" + score_metric.replace("|", "")

        group_criterion_str = f"_{group_criterion}"

    # The bare-integer bypass token is kept whenever only the early end is used,
    # so names already on disk keep parsing; both ends need a prefixed form
    bypass_str = ""
    if bypass_late_layers > 0:
        bypass_str = f"_byp{max(0, bypass_early_layers)}-{bypass_late_layers}"
    elif bypass_early_layers >= 0:
        bypass_str = f"_{bypass_early_layers}"

    max_ratio_str = "" if max_ratio == 0.9 else f"_cap{round(max_ratio, 2)}"

    min_rank_str = ""
    if min_rank_fraction != DEFAULT_MIN_RANK_FRACTION:
        min_rank_str = f"_mrf{round(min_rank_fraction, 3)}"

    head_block_str = "_hb" if head_block_svd else ""

    # A knob earns a token only when it leaves its default *and* the run actually
    # reads it, so passing --offset to a policy that ignores it cannot fork the
    # name into two entries for what is the same run
    knob_tokens: List[str] = []

    if seed is not None and seed != DEFAULT_SEED:
        knob_tokens.append(f"_seed{seed}")

    bypasses_any_layer = bypass_early_layers > 0 or bypass_late_layers > 0
    if bypasses_any_layer and bypass_ratio != DEFAULT_BYPASS_RATIO:
        knob_tokens.append(f"_bypr{round(bypass_ratio, 3)}")

    if heterogeneous:
        if score_metric.startswith(COMPOSITE_PREFIX) and fusion_alpha != DEFAULT_FUSION_ALPHA:
            knob_tokens.append(f"_fa{round(fusion_alpha, 3)}")

        # The policies declare the knobs they read as named parameters, so this
        # is the same relevance test the sidecar uses, from the same signatures
        policies = resolve_allocation_policies(
            inner_allocation,
            outer_allocation,
            allocation_knobs(offset=offset, softmax_temp=softmax_temp, outer_offset=outer_offset),
        )
        live_knobs = {**policies.inner_knobs, **policies.outer_knobs}

        for knob, prefix, default in KNOB_FILENAME_TOKENS:
            value = live_knobs.get(knob)
            if value is not None and value != default:
                knob_tokens.append(f"_{prefix}{round(value, 3)}")

    knob_str = "".join(knob_tokens)

    # Placed after the bypass token so `parse_filename` still finds that one at
    # the position it expects, and simply ignores these trailing tokens
    policy_str = ""
    if heterogeneous and inner_allocation != InnerAllocation.WATERFILL.value:
        policy_str = f"_{inner_allocation}"

    outer_policy_str = ""
    if heterogeneous and outer_allocation != OuterAllocation.PARAM_SHARE.value:
        outer_policy_str = f"_out{outer_allocation}"

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
        bypass_str,
        policy_str,
        outer_policy_str,
        max_ratio_str,
        min_rank_str,
        head_block_str,
        knob_str,
        f"_upd_{sequential_update_method}" if sequential_update else "",
        "_v2" if is_v2 else "",
    ]

    return "".join(parts)

def sanitize_run_args(args_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Drop credentials before a run configuration is written next to results"""
    return {k: v for k, v in args_dict.items() if k not in REDACTED_ARG_KEYS}

def run_config_path(directory: str, run_name: str) -> str:
    return os.path.join(directory, f"{run_name}{RUN_CONFIG_SUFFIX}")

def save_run_config(directory: str, run_name: str, config: Dict[str, Any]) -> str:
    """
    Write, or merge into, the sidecar describing how a run was configured.

    The filename can only carry a handful of tokens and is parsed positionally,
    so it cannot express every dimension of a run. This sidecar is the
    authoritative record instead, and `generate_tables.py` prefers it over
    `parse_filename`. Writers are additive: the compression step records the
    realized allocation, the entry point records the resolved arguments. Only
    keys the caller passes are replaced, which is what lets an evaluation-only
    run add its results without overwriting the compression that produced them.
    """
    os.makedirs(directory, exist_ok=True)
    path = run_config_path(directory, run_name)

    merged: Dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as config_file:
                merged = json.load(config_file)
        except (OSError, ValueError) as error:
            print(f"[CONFIG][WARNING] Overwriting unreadable sidecar {path}: {error}")
            merged = {}

    merged.update(config)
    merged["run_name"] = run_name
    merged["schema_version"] = RUN_CONFIG_SCHEMA_VERSION

    with open(path, "w", encoding="utf-8") as config_file:
        json.dump(merged, config_file, indent=2, default=str)

    print(f"[CONFIG] Wrote run config: {path}")

    return path

def merge_eval_results(path: str, results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge freshly evaluated tasks into the results file of the same run.

    A run measures whichever tasks it was asked for, so writing that set over
    the file would drop every task an earlier invocation measured: re-running
    one task to add it would silently delete the rest. Only the `results`
    entries are merged, because everything else in the payload describes the
    invocation that produced it and the fresh copy is the accurate one.
    """
    stored: Dict[str, Any] = {}

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as results_file:
                stored = json.load(results_file)
        except (OSError, ValueError) as error:
            print(f"[EVAL][WARNING] Overwriting unreadable results {path}: {error}")
            stored = {}

    evaluated = results.get("results") or {}
    kept = {task: entry for task, entry in (stored.get("results") or {}).items() if task not in evaluated}

    if kept:
        print(f"[EVAL] Keeping {len(kept)} task(s) measured by an earlier run: {', '.join(sorted(kept))}")

    merged = { **stored, **results }
    merged["results"] = { **kept, **evaluated }

    return merged

def summarize_allocation(
        ratio_map: Dict[str, float],
        param_count_map: Dict[str, int],
        layers_str: List[str],
        target_ratio: float,
        selected_total_params: int,
        target_total_params: int,
        bypassed_keys: List[str],
        active_keys: List[str],
        policies: Optional[AllocationPolicies] = None
) -> Dict[str, Any]:
    """
    Realized allocation facts, recorded so results can be checked against target.

    Allocation policies are compared at a fixed budget, so a run whose realized
    ratio drifted from its target is not comparable to one that hit it. Keeping
    both numbers next to the results makes that visible instead of implicit.

    `policies` is absent on homogeneous runs, which allocate nothing.
    """
    actual_removed = sum(param_count_map[k] * ratio_map.get(k, 0.0) for k in layers_str)
    assigned_ratios = [ratio_map.get(k, 0.0) for k in layers_str]

    policy_record = policies.describe() if policies is not None else {}

    return {
        **policy_record,
        "target_ratio": target_ratio,
        "selected_params": selected_total_params,
        "target_total_params": target_total_params,
        "target_removed_params": target_ratio * target_total_params,
        "actual_removed_params": actual_removed,
        "realized_selected_ratio": actual_removed / selected_total_params if selected_total_params else 0.0,
        "realized_overall_ratio": actual_removed / target_total_params if target_total_params else 0.0,
        "num_matrices": len(layers_str),
        "num_bypassed_matrices": len(bypassed_keys),
        "num_active_matrices": len(active_keys),
        "min_assigned_ratio": min(assigned_ratios) if assigned_ratios else 0.0,
        "max_assigned_ratio": max(assigned_ratios) if assigned_ratios else 0.0,
        "ratio_map": ratio_map,
    }

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
        subset: Optional[str] = "wikitext-2-raw-v1",
        split: str = "test",
        eval_max_length: int = 2048,
        batch_size: Union[int, str] = "auto",
        device: str = "cuda",
        data_files: Optional[Dict[str, str]] = None
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
    # `data_files` pins a specific shard for datasets whose split is too large to
    # join whole, so `num_proc` is dropped there to keep the single-file read simple
    if data_files is None:
        data = load_dataset(path=dataset_name, name=subset, split=split, num_proc=8)
    else:
        data = load_dataset(path=dataset_name, name=subset, split=split, data_files=data_files)

    print(f"[PPL] {dataset_name} | subset={subset} | split={split} | documents={len(data):,}") # pyright: ignore
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
    #
    # The cast to fp64 is not cosmetic. With a fp16 model the per-token losses come
    # out of the criterion in fp16, so both the mean and its exponential land on the
    # fp16 grid: near perplexity 7.8 that grid is spaced 0.0039, and a screening
    # ratio whose whole field of configurations spans 0.24 was being reported at
    # 1.6% granularity, with distinct allocations colliding on one value. The
    # methodology is untouched — still the uniform mean over every token position
    ppl = torch.exp(torch.cat(nlls).to(torch.float64).mean()).item()
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
