from typing import Dict, Optional, List
from enum import Enum
import os
from datasets import load_dataset, load_from_disk
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
import torch.nn as nn
import torch

class DtypeMap(Enum):
    float32: torch.dtype = torch.float32
    fp32: torch.dtype = float32
    float16: torch.dtype = torch.float16
    fp16: torch.dtype = float16
    bfloat16: torch.dtype = torch.bfloat16
    bf16: torch.dtype = bfloat16

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

def build_example(batch: Dict[str, List[str]]):
    instructions = batch["instruction"]
    inputs = batch["input"]
    
    # Combine instruction and input (if available)
    out_examples = [
        f"{instr}\n{inp}" if inp.strip() else instr
        for instr, inp in zip(instructions, inputs)
    ]

    return {"final_input": out_examples}

def tokenize_example(
        batch: Dict[str, List[str]], 
        tokenizer: Qwen2TokenizerFast,
        seq_len: int
):
    # Build message with system prompt
    messages = [
        [{"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": example}]
    for example in batch["final_input"]]

    # Apply the chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    tokenized_example = tokenizer(
        text, 
        padding = True,
        return_tensors = "pt"
    )

    return tokenized_example

def tokenize_dataset(
        name: str,
        split: str,
        tokenizer: Qwen2TokenizerFast,
        max_samples: int = 512,
        seq_len: int = 512,
        batch_size: int = 32,
        seed: Optional[int] = None,
        save_path: Optional[str] = None
):
    # Load train split, shuffle it and take samples
    if os.path.isdir(name):
        print("DEBUG: Loading dataset from disk...")
        df = load_from_disk(name + "/" + split)
    else:
        print("DEBUG: Loading dataset from hub...")
        df = load_dataset(name, split=split, num_proc=8)
        if save_path and not os.path.exists(save_path + "/calibration_datasets/" + name + "/" + split):
            print("DEBUG: Saving dataset to disk...")
            df.save_to_disk(save_path + "/calibration_datasets/" + name + "/" + split)
    shuffled_df = df.shuffle(seed)
    df = shuffled_df.select(range(max_samples)).flatten_indices()

    # Preprocess examples
    preprocessed_df = df.map(build_example, batched = True, batch_size = batch_size, remove_columns = ["instruction", "input", "output", "text"], load_from_cache_file=False)

    # Tokenize preprocessed examples
    tokenized_df = preprocessed_df.map(tokenize_example, batched = True, batch_size = batch_size, fn_kwargs = {"tokenizer": tokenizer, "seq_len": seq_len}, load_from_cache_file=False)

    return tokenized_df.with_format("torch")

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