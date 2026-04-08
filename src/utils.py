import os
import torch.nn as nn
import torch
import random
import sys
from typing import Dict, Optional, List
from tqdm import tqdm
from enum import Enum
from datasets import load_dataset, load_from_disk, Dataset
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast


class GroupBy(str, Enum):
    GLOBAL="global"
    DECODER="decoder"
    TYPE="type"

class ScoreMetric(str, Enum):
    TRUNCATION="truncation"
    ENTROPY="entropy"

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
        
def vram_usage(msg=""):
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2
    torch.cuda.reset_peak_memory_stats()
    print(f"[VRAM] {msg} | allocated={alloc:.1f} MiB | reserved={reserved:.1f} MiB | peak={peak:.1f} MiB")

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
            truncation=False,           # we want the full token stream
            padding=False,              # absolutely no padding
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
        df = load_from_disk(name + "/" + split)
    else:
        print("[DEBUG] Loading dataset from hub...")
        if subset is not None:
            df = load_dataset(path=name, name=subset, split=split, num_proc=8)
        else:
            df = load_dataset(path=name, split=split, num_proc=8)
        if save_path and not os.path.exists(save_path + "/calibration_datasets/" + name + "/" + split):
            print("[DEBUG] Saving dataset to disk...")
            df.save_to_disk(save_path + "/calibration_datasets/" + name + "/" + subset + "/" + split)

    # Step 2: Concatenate all text into one long string
    concatenated = df.map(
        concatenate_text,
        batched=True,
        batch_size=len(df), # process entire dataset in one batch
        remove_columns=df.column_names, # pyright: ignore[reportArgumentType]
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

def generate_paths(mlp: bool, qkv: bool, attention_output: bool, layers_number: int) -> list[str]:
    list_paths=[]
    if layers_number >= 0:
        if mlp:
            list_paths += [f'model.layers.{layers_number - 1 - i}.mlp.gate_proj' for i in range(layers_number)]
            list_paths += [f'model.layers.{layers_number - 1 - i}.mlp.up_proj' for i in range(layers_number)]
            list_paths += [f'model.layers.{layers_number - 1 - i}.mlp.down_proj' for i in range(layers_number)]
        if qkv:
            list_paths += [f'model.layers.{layers_number - 1 - i}.self_attn.q_proj' for i in range(layers_number)]
            list_paths += [f'model.layers.{layers_number - 1 - i}.self_attn.k_proj' for i in range(layers_number)]
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

@torch.no_grad()
def ppl_eval(
        model,
        tokenizer,
        dataset_name: str = "wikitext",
        subset: str = "wikitext-2-raw-v1",
        split: str = "test",
        eval_max_length: int = 2048,
        batch_size: int = 8,
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
    text = "\n\n".join(data["text"])
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
    nlls = []
    for i in tqdm(range(0, num_chunks, batch_size), desc="Evaluating perplexity..."):
        batch = input_ids[i : i + batch_size].to(device)  # [B, model_seq_len]
        output = model(batch, use_cache=False)
        lm_logits = output.logits  # [B, model_seq_len, vocab_size]

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