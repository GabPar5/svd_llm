from src.utils import *
from src.svd_llm import *
import argparse
import gc
import json
from transformers import AutoModelForCausalLM, AutoConfig
import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import setup_logging, handle_non_serializable


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-1.5B', help='LLM to load from huggingface')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='Weights dtype (original and compressed)')
    parser.add_argument('--compression_ratio', type=float, default=0.2, help='Target compression ratio,(0,1), default=0.2, means only keeping about 20%% of the params.')
    parser.add_argument('--calibration_dataset', type=str, default='tatsu-lab/alpaca:train',help='Calibration dataset, format is "datasetNameOrPath:split"')
    parser.add_argument('--max_whitening_samples', type=int, default=1024, help='Number of calibration data samples for whitening.')
    parser.add_argument('--seq_len', type=int, default=512, help='Fixed length for calibration samples.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for data preprocessing and forward pass')
    parser.add_argument('--seed',type=int, default=6363, help='Seed for sampling the calibration data')
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the whitening matrices and the compressed model checkpoints.')
    parser.add_argument('--whitening_mat_path', type=str, default=None, help='Local path to load the whitening matrices')
    parser.add_argument('--use_compressed', action='store_true', help='Use compressed model for evaluation')
    parser.add_argument('--compressed_model_path', type=str, default=None, help='Local path to load the compressed model - if you need to do evaluation only')
    parser.add_argument('--compress_mlp', action='store_true', help='Compress MLP weights')
    parser.add_argument('--compress_att_qkv', action='store_true', help='Compress attention qkv matrices')
    parser.add_argument('--compress_att_out', action='store_true', help='Compress attention output projection matrix')
    parser.add_argument('--hf_token', type=str, default=None, help='Huggingface token to download/upload models')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model on a set of tasks')
    parser.add_argument('--eval_batch_size', type=int, default=1, help='Evaluation batch size')
    parser.add_argument('--eval_tasks', type=str, default='wikitext|0',help='Evaluation tsks, the pattern is "taskName1,taskName2,...,taskNameK|numShots"')
    parser.add_argument('--eval_temperature', type=float, default=0.7,help='Evaluation temperature (conditional sampling)')
    parser.add_argument('--max_eval_tokens', type=int, default=1024,help='Max number of tokens to generate during evaluation')

    args = parser.parse_args()

    if args.use_compressed:
        if args.compressed_model_path:
            print("DEBUG: Loading compressed model from disk...")
            vram_usage("Before loading compressed model")
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.compressed_model_path, padding_side="left")

            # Load model config from HF and instantiate base model
            config = AutoConfig.from_pretrained(args.model)
            model = AutoModelForCausalLM.from_config(config)
            # Avoid warning
            model.generation_config.pad_token_id = model.generation_config.eos_token_id

            # Load checkpoint
            checkpoint = torch.load(args.compressed_model_path, map_location="cpu")
            rank_map = checkpoint["rank_map"]

            # Replace compressed layers with LowRank modules
            apply_lowrank(model, rank_map)
            print(model)

            # Load weights
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model=model.to(args.device, dtype=DtypeMap.get_dtype(args.dtype))
            del checkpoint, rank_map
            gc.collect()
            torch.cuda.empty_cache()
            vram_usage("After loading compressed model")
        else:
            dataset_name = args.calibration_dataset.split(":")[0]
            dataset_split = args.calibration_dataset.split(":")[1]
            model, tokenizer = compress_svd_llm(
                model_name = args.model,
                ratio = round(1-args.compression_ratio, 2),
                dataset = {"dataset_name": dataset_name, 
                        "split": dataset_split, 
                        "max_samples": args.max_whitening_samples, 
                        "seq_len": args.seq_len},
                dtype = args.dtype,
                batch_size = args.batch_size,
                seed = args.seed,
                device = args.device,
                save_path = args.save_path,
                whitening_mat_path = args.whitening_mat_path,
                compress_mlp = args.compress_mlp,
                compress_att_qkv = args.compress_att_qkv,
                compress_att_out = args.compress_att_out,
                hf_token = args.hf_token
            )
            model=model.to(args.device, dtype=DtypeMap.get_dtype(args.dtype))
            vram_usage("After loading compressed model")
    else:
        print("DEBUG: Loading original model from the hub...")
        vram_usage("Before loading original model")
        tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=args.dtype,
            device_map=args.device,
            use_safetensors=True,
            token=args.hf_token
        )
        # Avoid warning
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        vram_usage("After loading original model")
    
    if args.evaluate:
        # Don't return last KV cache
        model.config.use_cache = False

        # Setup logging level
        setup_logging("DEBUG")

        # Preprocess tasks
        tasks_shots = args.eval_tasks.split("|")
        if len(tasks_shots) > 2:
            raise ValueError('The argument `eval_tasks_split` must be a string following this format: "taskName1,taskName2,...,taskNameK|numShots"')
        elif  len(tasks_shots) == 1:
            # Default zero-shot
            num_fewshot = 0
        else:
            num_fewshot = int(tasks_shots[1])
        tasks_list = tasks_shots[0].split(",")
        print(f"DEBUG: Num few-shots: {num_fewshot}")
        print(f"DEBUG: List of evaluation tasks: {tasks_list}")

        # Load model
        eval_model = HFLM(
            pretrained=model,
            tokenizer = tokenizer,
            batch_size=args.eval_batch_size,
            device = args.device,
            dtype = args.dtype,
            max_length = args.max_eval_tokens
        )


        vram_usage("Before evaluation")

        results = lm_eval.simple_evaluate(
            model=eval_model,
            tasks=tasks_list,
            num_fewshot=num_fewshot,
            batch_size=args.eval_batch_size,
            device=args.device,
            use_cache=None,
            gen_kwargs={
                "temperature": args.eval_temperature,
                "do_sample": True,
                "max_gen_toks": args.max_eval_tokens,
            },
            random_seed=args.seed,
            numpy_random_seed=args.seed,
            torch_random_seed=args.seed,
            fewshot_random_seed=args.seed
        )

        # Save results
        if args.compressed_model_path:
            model_path = "/".join(args.compressed_model_path.split("/")[:-1])
        else:
            model_path = args.save_path + \
                         "/models/" + \
                         args.model.replace("/", "_").replace("-", "_") + \
                         "/"
        with open(model_path + "/evaluation_results.json", "w") as f:
            json.dump(results, f, default=handle_non_serializable, indent=2)

        vram_usage("After evaluation")
