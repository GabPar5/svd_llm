import argparse
import gc
import os
import re
import sys
import torch

from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from src.svd_llm import apply_lowrank
from src.utils import DtypeMap


def cuda_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

def safe_filename(name: str) -> str:
    name = os.path.basename(name)
    name = re.sub(r"\.pt$", "", name)
    name = re.sub(r"[^A-Za-z0-9._+-]+", "_", name)
    return name

def safe_pad_token_setup(tokenizer, model):
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    eos = model.generation_config.eos_token_id
    pad_id = eos[0] if isinstance(eos, (list, tuple)) else eos

    model.generation_config.pad_token_id = pad_id
    tokenizer.padding_side = "left"


def load_original_model(
    model_name: str,
    dtype: str,
    device: str,
    hf_token: str | None = None,
):
    torch_dtype = DtypeMap.get_dtype(dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=hf_token,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,
        device_map=device,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        trust_remote_code=True,
        token=hf_token,
    )

    model.eval()
    model.config.use_cache = True

    safe_pad_token_setup(tokenizer, model)

    return model, tokenizer


def load_compressed_model(
    base_model_name: str,
    checkpoint_path: str,
    dtype: str,
    device: str,
    tokenizer_path: str | None = None,
    hf_token: str | None = None,
):
    """
    Loads a compressed checkpoint saved like:

        torch.save({
            "state_dict": model.state_dict(),
            "rank_map": rank_map,
        }, checkpoint_path)

    The base HF config is required because the checkpoint only stores weights,
    not the full model architecture/config.
    """
    torch_dtype = DtypeMap.get_dtype(dtype)

    if tokenizer_path is None:
        tokenizer_path = os.path.dirname(checkpoint_path)

        # Fallback to base model tokenizer if the checkpoint directory does not contain one.
        if not os.path.exists(os.path.join(tokenizer_path, "tokenizer_config.json")):
            tokenizer_path = base_model_name

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        token=hf_token,
    )

    print(f"[LOAD] Loading compressed checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    rank_map = checkpoint["rank_map"]
    state_dict = checkpoint["state_dict"]

    print(f"[LOAD] Loading base config from: {base_model_name}")
    config = AutoConfig.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        token=hf_token,
    )

    print("[LOAD] Instantiating base model architecture...")
    model = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=True,
        dtype=torch_dtype,
    )

    print("[LOAD] Applying LowRank module structure...")
    apply_lowrank(model, rank_map)

    print("[LOAD] Loading compressed state dict...")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"[WARNING] Missing keys: {len(missing)}")
        print(missing[:20])

    if unexpected:
        print(f"[WARNING] Unexpected keys: {len(unexpected)}")
        print(unexpected[:20])

    if missing or unexpected:
        raise RuntimeError(
            "State dict mismatch. The compressed checkpoint may not match "
            "the base model or rank_map."
        )

    del checkpoint, state_dict, rank_map
    cuda_cleanup()

    model.to(device=device, dtype=torch_dtype)
    model.eval()
    model.config.use_cache = True

    safe_pad_token_setup(tokenizer, model)

    return model, tokenizer


def build_prompt(
    tokenizer,
    prompt: str,
    use_chat_template: bool = False,
    system_prompt: str | None = None,
):
    if not use_chat_template:
        return prompt

    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": prompt})

    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Tokenizer does not support apply_chat_template().")

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.inference_mode()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=False,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }

    if do_sample:
        generation_kwargs.update(
            {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }
        )

    output_ids = model.generate(
        **inputs,
        **generation_kwargs,
    )

    prompt_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0, prompt_len:]

    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return text


def find_checkpoints(folder: str, recursive: bool = False) -> List[str]:
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Not a folder: {folder}")

    checkpoints = []

    if recursive:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(".pt"):
                    checkpoints.append(os.path.join(root, file))
    else:
        for file in os.listdir(folder):
            if file.endswith(".pt"):
                checkpoints.append(os.path.join(folder, file))

    checkpoints.sort()
    return checkpoints


def run_one_model(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
):
    print("[GEN] Greedy decoding...")
    greedy_output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=repetition_penalty,
    )

    print("[GEN] Conditional sampling...")
    sampling_output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    return greedy_output, sampling_output


def save_result(
    output_dir: str,
    model_label: str,
    prompt: str,
    greedy_output: str,
    sampling_output: str,
):
    os.makedirs(output_dir, exist_ok=True)

    filename = safe_filename(model_label) + ".txt"
    path = os.path.join(output_dir, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_label}\n")
        f.write("=" * 100 + "\n\n")

        f.write("Prompt:\n")
        f.write(prompt)
        f.write("\n\n")

        f.write("Output (Greedy Decoding):\n")
        f.write(greedy_output)
        f.write("\n\n")

        f.write("Output (Conditional Sampling):\n")
        f.write(sampling_output)
        f.write("\n")

    print(f"[SAVE] Saved result to: {path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="HF model name/path, e.g. Qwen/Qwen2.5-32B",
    )

    parser.add_argument(
        "--compressed_checkpoint",
        type=str,
        default=None,
        help="Path to one compressed .pt checkpoint."
    )

    parser.add_argument(
        "--compressed_folder",
        type=str,
        default=None,
        help="Folder containing compressed .pt checkpoints to run sequentially.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When using --compressed_folder, recursively find .pt files.",
    )

    parser.add_argument(
        "--include_original",
        action="store_true",
        help="Also run the original HF model before compressed checkpoints.",
    )

    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Optional tokenizer path. For compressed models, defaults to checkpoint directory.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"],
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/text_generation/",
        help="Folder where .txt generations are saved.",
    )

    parser.add_argument(
        "--use_chat_template",
        action="store_true",
        help="Use tokenizer.apply_chat_template(). Useful for Qwen2.5-Instruct.",
    )

    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            raw_prompt = f.read()
    elif args.prompt:
        raw_prompt = args.prompt
    else:
        raise ValueError("Provide --prompt or --prompt_file.")
    
    checkpoints = []

    if args.compressed_checkpoint:
        checkpoints.append(args.compressed_checkpoint)

    if args.compressed_folder:
        checkpoints.extend(find_checkpoints(args.compressed_folder, recursive=args.recursive))

    if not args.include_original and not checkpoints:
        raise ValueError(
            "Nothing to run. Provide --compressed_checkpoint, --compressed_folder, "
            "or use --include_original."
        )
    
    if args.include_original:
        print("[RUN] Loading original model...")
        model, tokenizer = load_original_model(
            model_name=args.base_model,
            dtype=args.dtype,
            device=args.device,
            hf_token=args.hf_token,
        )

        final_prompt = build_prompt(
            tokenizer=tokenizer,
            prompt=raw_prompt,
            use_chat_template=args.use_chat_template,
            system_prompt=args.system_prompt,
        )

        greedy_output, sampling_output = run_one_model(
            model=model,
            tokenizer=tokenizer,
            prompt=final_prompt,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )

        save_result(
            output_dir=args.output_dir,
            model_label=safe_filename(args.base_model) + "_original",
            prompt=raw_prompt,
            greedy_output=greedy_output,
            sampling_output=sampling_output,
        )

        del model, tokenizer
        cuda_cleanup()

    for idx, checkpoint_path in enumerate(checkpoints, start=1):
        print("=" * 100)
        print(f"[RUN] {idx}/{len(checkpoints)}: {checkpoint_path}")
        print("=" * 100)

        model, tokenizer = load_compressed_model(
            base_model_name=args.base_model,
            checkpoint_path=checkpoint_path,
            dtype=args.dtype,
            device=args.device,
            tokenizer_path=args.tokenizer_path,
            hf_token=args.hf_token,
        )

        final_prompt = build_prompt(
            tokenizer=tokenizer,
            prompt=raw_prompt,
            use_chat_template=args.use_chat_template,
            system_prompt=args.system_prompt,
        )

        greedy_output, sampling_output = run_one_model(
            model=model,
            tokenizer=tokenizer,
            prompt=final_prompt,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )

        save_result(
            output_dir=args.output_dir,
            model_label=checkpoint_path,
            prompt=raw_prompt,
            greedy_output=greedy_output,
            sampling_output=sampling_output,
        )

        del model, tokenizer
        cuda_cleanup()

    print("[DONE] All generations completed.")


if __name__ == "__main__":
    main()