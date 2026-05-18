from src.utils import *
from src.svd_llm import *
import argparse
import gc
import json
import torch
import lm_eval
import multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GenerationConfig
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import setup_logging, handle_non_serializable


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', 
        type=str, 
        default='Qwen/Qwen2.5-7B', 
        help='LLM to load from huggingface'
    )
    parser.add_argument(
        '--run_v2', 
        action='store_true', 
        help='Run SVD-LLM V2'
    )
    parser.add_argument(
        '--dtype', 
        type=str, 
        default='float32', 
        help='Weights dtype (original and compressed)'
    )
    parser.add_argument(
        '--compression_ratio', 
        type=float,
        default=0.2, 
        help='Target compression ratio,(0,1), default=0.2, means removing about 20%% of the params.'
    )
    parser.add_argument(
        "--ratio_scope",
        type=str,
        default="selected",
        choices=["selected", "all"],
        help=(
            "selected: compression_ratio applies only to selected matrices. "
            "all: compression_ratio applies to all targetable projection matrices "
            "(MLP + q/k/v/o), even if only a subset is selected for compression."
        ),
    )
    parser.add_argument(
        '--calibration_dataset', 
        type=str, 
        default='EleutherAI/wikitext_document_level:wikitext-2-raw-v1:train',
        help='Calibration dataset, format is "datasetNameOrPath:subset:split"'
    )
    parser.add_argument(
        '--max_length', 
        type=int, 
        default=2048, 
        help='Maximum context length for the LLM during compression'
    )
    parser.add_argument(
        '--max_whitening_samples', 
        type=int, 
        default=256, 
        help='Number of calibration data samples for whitening.'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=2, 
        help='Batch size for data preprocessing and calibration forward pass'
    )
    parser.add_argument(
        '--seed',
        type=int, 
        default=6363, 
        help='Seed for sampling the calibration data'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default="cuda", 
        help='device'
    )
    parser.add_argument(
        '--save_path', 
        type=str, 
        default=None, 
        help='Base path to save the whitening matrices and the compressed model checkpoints'
    )
    parser.add_argument(
        '--whitening_mat_path', 
        type=str, 
        default=None, 
        help='Local path to load the whitening matrices'
    )
    parser.add_argument(
        "--whitening_only", 
        action="store_true"
    )
    parser.add_argument(
        "--whitening_start_layer", 
        type=int, 
        default=0
    )
    parser.add_argument(
        "--whitening_end_layer", 
        type=int, 
        default=None
    )
    parser.add_argument(
        '--use_compressed', 
        action='store_true', 
        help='Use compressed model for evaluation'
    )
    parser.add_argument(
        '--compressed_model_path', 
        type=str, 
        default=None, 
        help='Local path to load the compressed model - if you need to do evaluation only'
    )
    parser.add_argument(
        '--compress_mlp', 
        action='store_true', 
        help='Compress MLP weights'
    )
    parser.add_argument(
        '--compress_att_q', 
        action='store_true', 
        help='Compress attention query projection matrices'
    )
    parser.add_argument(
        '--compress_att_k', 
        action='store_true', 
        help='Compress attention key projection matrices'
    )
    parser.add_argument(
        '--compress_att_v', 
        action='store_true', 
        help='Compress attention value projection matrices'
    )
    parser.add_argument(
        '--compress_att_out', 
        action='store_true', 
        help='Compress attention output projection matrix'
    )
    parser.add_argument(
        '--het', 
        action='store_true', 
        help='Assign heterogeneous compression ratio'
    )
    parser.add_argument(
        '--bypass_early_layers', 
        type=int, 
        default=2, 
        help='Number of starting layers which bypass heterogeneous compression (or compression at all)'
    )
    parser.add_argument(
        '--bypass_ratio', 
        type=float, 
        default=0.0, 
        help='Compression ratio for the bypassed layers'
    )
    parser.add_argument(
        '--group_criterion', 
        type=str, 
        default="type", 
        help='Criterion used to group weight matrices in heterogeneous setting. Possible values are "type", "global" and "decoder"'
    )
    parser.add_argument(
        '--group_patterns', 
        type=str, 
        default="q_proj:self_attn.q_proj;k_proj:self_attn.k_proj;v_proj:self_attn.v_proj;o_proj:self_attn.o_proj;gate_proj:mlp.gate_proj;up_proj:mlp.up_proj;down_proj:mlp.down_proj", 
        help='Group patterns used when grouping weight matrices by type, the pattern is "groupName1:weightType1,weightType2;groupName2:weightType1,weightType2;..."'
    )
    parser.add_argument(
        '--score_metric', 
        type=str, 
        default="truncation", 
        help='Score metric to use for weight importance during heterogeneous ratio allocation. Possible values are "truncation" and "entropy"'
    )
    parser.add_argument(
        '--hf_token', 
        type=str, 
        default=None, 
        help='Huggingface token to download/upload models'
    )
    parser.add_argument(
        '--evaluate', 
        action='store_true', 
        help='Evaluate the model on a set of tasks'
    )
    parser.add_argument(
        '--eval_sampling', 
        action='store_true', 
        help='Use conditional sampling during evaluation'
    )
    parser.add_argument(
        '--eval_batch_size', 
        type=str, 
        default="auto", 
        help='Evaluation batch size'
    )
    parser.add_argument(
        '--eval_tasks', 
        type=str, 
        default='wikitext|0',
        help='Evaluation tasks, the pattern is "taskName1,taskName2,...,taskNameK|numShots" or "taskName1,taskName2,...,taskNameK|numShots1,numShots2,...,numShotsK"'
    )
    parser.add_argument(
        '--eval_max_length', 
        type=int, 
        default=4096, 
        help='Maximum context length for the LLM during evaluation'
    )
    parser.add_argument(
        '--max_eval_tokens', 
        type=int, 
        default=256,
        help='Maximum number of tokens to generate during evaluation'
    )

    args = parser.parse_args()

    if not args.use_compressed:
        print("DEBUG: Loading original model from the hub...")
        vram_usage("Before loading original model")
        model_eval_path = args.save_path + \
                     "/eval/" + \
                     args.model.replace("/", "_").replace("-", "_") + \
                     "/"
        model_name = args.model.replace("/", "_").replace("-", "_")
        
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=args.dtype,
            device_map=args.device,
            use_safetensors=True,
            token=args.hf_token, 
            trust_remote_code=True
        )
        # Avoid warning
        eos = model.generation_config.eos_token_id # pyright: ignore[reportOptionalMemberAccess]
        model.generation_config.pad_token_id = eos[0] if isinstance(eos, list) else eos # pyright: ignore[reportOptionalMemberAccess]
        vram_usage("After loading original model")
    elif args.compressed_model_path:
        print("DEBUG: Loading compressed model from disk...")
        vram_usage("Before loading compressed model")
        model_eval_path = args.save_path + \
                     "/eval/" + \
                     args.model.replace("/", "_").replace("-", "_") + \
                     "/"
        model_name = args.compressed_model_path.split("/")[-1][:-3]

        # Load tokenizer
        tokenizer_path = "/".join(args.compressed_model_path.split("/")[:-1])
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Load checkpoint
        checkpoint = torch.load(
            args.compressed_model_path,
            map_location=args.device,
            weights_only=False,
        )

        rank_map = checkpoint["rank_map"]
        state_dict = checkpoint["state_dict"]

        # Load model config from HF and instantiate base model
        config = AutoConfig.from_pretrained(
            args.model,
            trust_remote_code=True,
            token=args.hf_token,
            dtype=DtypeMap.get_dtype(args.dtype)
        )

        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=True,
                dtype=DtypeMap.get_dtype(args.dtype),
            )
        # Avoid warning
        eos = model.generation_config.eos_token_id # pyright: ignore[reportOptionalMemberAccess]
        model.generation_config.pad_token_id = eos[0] if isinstance(eos, list) else eos # pyright: ignore[reportOptionalMemberAccess]
        try:
            model.generation_config = GenerationConfig.from_pretrained(
                args.model,
                trust_remote_code=True,
                token=args.hf_token,
            )
        except Exception as e:
            print(f"[WARNING] Could not load generation_config for {args.model}: {e}")
            model.generation_config = GenerationConfig.from_model_config(config)
        eos = model.generation_config.eos_token_id # pyright: ignore[reportOptionalMemberAccess]
        if eos is None:
            eos = tokenizer.eos_token_id
            model.generation_config.eos_token_id = eos # pyright: ignore[reportOptionalMemberAccess]
        model.generation_config.pad_token_id = eos[0] if isinstance(eos, list) else eos # pyright: ignore[reportOptionalMemberAccess]

        # Replace compressed layers with LowRank modules
        apply_lowrank(model, rank_map)
        model.to_empty(device=args.device)

        missing, unexpected = model.load_state_dict(
            state_dict, 
            strict=False, 
            assign=True
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
                "the base model or rank_map."
            )

        # Clean memory
        del checkpoint, state_dict, rank_map
        cuda_cleanup()
        gc.collect()
        vram_usage("After loading compressed model")
    else:
        model_eval_path = args.save_path + \
                     "/eval/" + \
                     args.model.replace("/", "_").replace("-", "_") + \
                     "/"
        compress_att_q_str = "_q" if args.compress_att_q else ""
        compress_att_k_str = "_k" if args.compress_att_k else ""
        compress_att_v_str = "_v" if args.compress_att_v else ""
        compress_att_out_str = "_out" if args.compress_att_out else ""
        compress_mlp_str = "_mlp" if args.compress_mlp else ""
        ratio_scope_str = "_all" if args.compress_att_q and args.compress_att_k and args.compress_att_v and args.compress_att_out and args.compress_mlp else "_" + str(args.ratio_scope)
        heterogeneous_str = "_het" if args.het else "_hom"
        group_criterion_str = ("_" + args.group_criterion) if args.het else ""
        score_metric_substr = args.score_metric.replace("|", "") if len(args.score_metric.split("|")) > 1 else args.score_metric
        score_metric_str = ("_" + score_metric_substr) if args.het else ""
        bypassed_layers_str = "_" + str(args.bypass_early_layers) if args.bypass_early_layers >= 0 else ""
        v2_str = "_v2" if args.run_v2 else ""
        model_name = args.model.replace("/", "_").replace("-", "_") + \
                     compress_att_q_str + \
                     compress_att_k_str + \
                     compress_att_v_str + \
                     compress_att_out_str + \
                     compress_mlp_str + \
                     ratio_scope_str + \
                     "_" + \
                     str(round(args.compression_ratio, 2)) + \
                     heterogeneous_str + \
                     group_criterion_str + \
                     score_metric_str + \
                     bypassed_layers_str + \
                     v2_str

        dataset_name = args.calibration_dataset.split(":")[0]
        dataset_subset = args.calibration_dataset.split(":")[1]
        dataset_split = args.calibration_dataset.split(":")[2]

        group_patterns_list = list(map(lambda x: x.split(":"), args.group_patterns.split(";")))
        group_patterns_dict = {}
        for group in group_patterns_list:
            group_patterns_dict[group[0]] = group[1].split(",")

        # Initialize logger
        model_log_path = args.save_path + "/logs/" + args.model.replace("/", "_").replace("-", "_") + "/"
        if not os.path.isdir(model_log_path):
            os.mkdir(model_log_path)
        sys.stdout = Logger(
            filename= model_log_path + model_name + ".log"
        )

        model, tokenizer = compress_svd_llm(
            model_name = args.model,
            ratio = round(args.compression_ratio, 2),
            dataset = {
                "name": dataset_name, 
                "subset": dataset_subset,
                "split": dataset_split, 
                "max_samples": args.max_whitening_samples
            },
            max_length = args.max_length,
            is_v2 = args.run_v2,
            dtype = args.dtype,
            batch_size = args.batch_size,
            seed = args.seed,
            device = args.device,
            save_path = args.save_path,
            whitening_mat_path = args.whitening_mat_path,
            compress_mlp = args.compress_mlp,
            compress_att_q = args.compress_att_q,
            compress_att_k = args.compress_att_k,
            compress_att_v = args.compress_att_v,
            compress_att_out = args.compress_att_out,
            score_metric=args.score_metric,
            heterogeneous = args.het,
            group_criterion = args.group_criterion,
            group_patterns = group_patterns_dict,
            hf_token = args.hf_token,
            whitening_only = args.whitening_only,
            whitening_start_layer = args.whitening_start_layer,
            whitening_end_layer = args.whitening_end_layer,
            bypass_early_layers = args.bypass_early_layers,
            bypass_ratio = args.bypass_ratio,
            ratio_scope=args.ratio_scope
        )
        model=model.to(args.device, dtype=DtypeMap.get_dtype(args.dtype))
        print(model)

        gc.collect()
        torch.cuda.empty_cache()
        vram_usage("After loading compressed model")
        
    
    if args.evaluate:
        # Set model into evaluation mode
        model.eval()
        model.config.use_cache = False
        # Set tokenizer padding
        tokenizer.padding_side = "left"

        # Setup logging level
        setup_logging("DEBUG") # pyright: ignore[reportArgumentType]

        # Preprocess tasks
        tasks_shots = args.eval_tasks.split("|")
        tasks_list = tasks_shots[0].split(",")
        if len(tasks_shots) > 2:
            raise ValueError('The argument `eval_tasks_split` must be a string following these formats: "taskName1,taskName2,...,taskNameK|numShots" or "taskName1,taskName2,...,taskNameK|numShots1,numShots2,...,numShotsK"')
        elif len(tasks_shots) == 1:
            # Default to zero-shot
            num_fewshot = 0
        else:
            if len(tasks_shots[1].split(",")) > 1:
                num_fewshot = tasks_shots[1].split(",")
            else:
                num_fewshot = int(tasks_shots[1])
        
        if isinstance(num_fewshot, list):
            tasks_dict = [{"task": tasks_list[i], "num_fewshot": int(num_fewshot[i])} for i in range(len(tasks_list)) if tasks_list[i] != "wikitext" and tasks_list[i] != "c4"]
        else:
            tasks_dict = [{"task": tasks_list[i], "num_fewshot": int(num_fewshot)} for i in range(len(tasks_list)) if tasks_list[i] != "wikitext" and tasks_list[i] != "c4"]
        print(f"[DEBUG] Num few-shots: {num_fewshot}")
        print(f"[DEBUG] List of evaluation tasks: {tasks_list}")
        print(f"[DEBUG] Tasks dictionaries: {tasks_dict}")
        print(f"[DEBUG] HF model context length: {model.config.max_position_embeddings}")

        # Clamp max model context
        max_length = min(
            args.eval_max_length,
            model.config.max_position_embeddings - args.max_eval_tokens
        )
        print(f"[DEBUG] Evaluation context length: {max_length}")

        results = {}
        wikitext_ppl = None
        c4_ppl = None
        if "wikitext" in tasks_list:
            wikitext_ppl = ppl_eval(
                model,
                tokenizer,
                dataset_name="wikitext",
                subset="wikitext-2-raw-v1",
                split="test",
                eval_max_length=max_length,
                batch_size=args.eval_batch_size,
                device=args.device
            )
            torch.cuda.empty_cache()
            gc.collect()
        if "c4" in tasks_list:
            # TODO c4 task
            c4_ppl = ppl_eval(
                model,
                tokenizer,
                dataset_name="wikitext",
                subset="wikitext-2-raw-v1",
                split="test",
                eval_max_length=max_length,
                batch_size=args.eval_batch_size,
                device=args.device
            )
            torch.cuda.empty_cache()
            gc.collect()

        if tasks_dict is not None and len(tasks_dict) > 0:
            model.config.use_cache = True
            model.generation_config.max_new_tokens = args.max_eval_tokens # pyright: ignore[reportOptionalMemberAccess]
            # WARNING - PyRight reports lots of issues when dealing with lm-eval-harness 
            eval_model = HFLM(
                pretrained=model, # pyright: ignore[reportCallIssue]
                tokenizer = tokenizer, # pyright: ignore[reportCallIssue]
                batch_size=args.eval_batch_size, # pyright: ignore[reportCallIssue]
                max_batch_size=128, # pyright: ignore[reportCallIssue]
                device = args.device, # pyright: ignore[reportCallIssue]
                dtype = args.dtype, # pyright: ignore[reportCallIssue]
                max_length = max_length # pyright: ignore[reportCallIssue]
            )
            print(f"[DEBUG] HFLM model context length: {eval_model.max_length}") # pyright: ignore[reportAttributeAccessIssue]


            vram_usage("Before evaluation")

            # Run evaluation 
            results = lm_eval.simple_evaluate(
                model=eval_model, # pyright: ignore[reportCallIssue]
                tasks=tasks_dict,  # type: ignore
                batch_size=args.eval_batch_size, # pyright: ignore[reportCallIssue]
                max_batch_size=128, # pyright: ignore[reportCallIssue]
                device=args.device, # pyright: ignore[reportCallIssue]
                use_cache=None, # pyright: ignore[reportCallIssue]
                log_samples=False, # pyright: ignore[reportCallIssue]
                fewshot_as_multiturn=False, # pyright: ignore[reportCallIssue]
                gen_kwargs={ # pyright: ignore[reportCallIssue]
                    "do_sample": args.eval_sampling,
                    "max_gen_toks": args.max_eval_tokens
                },
                apply_chat_template=False,
                random_seed=args.seed, # pyright: ignore[reportCallIssue]
                numpy_random_seed=args.seed, # pyright: ignore[reportCallIssue]
                torch_random_seed=args.seed, # pyright: ignore[reportCallIssue]
                fewshot_random_seed=args.seed # pyright: ignore[reportCallIssue]
            ) # pyright: ignore[reportCallIssue]

        # SAVE RESULTS
        if not os.path.isdir(model_eval_path):
            os.mkdir(model_eval_path)

        if "results" not in results:  # pyright: ignore[reportOperatorIssue]
            results["results"] = {} # pyright: ignore[reportOptionalSubscript]

        if wikitext_ppl is not None:
            results["results"]["wikitext"] = { # pyright: ignore
                "alias": "wikitext",
                "token_perplexity,none": wikitext_ppl,
                "token_perplexity_stderr,none": "N/A"
            }

        if c4_ppl is not None:
            results["results"]["c4"] = { # pyright: ignore
                "alias": "c4",
                "token_perplexity,none": c4_ppl,
                "token_perplexity_stderr,none": "N/A"
            }
            
        with open(model_eval_path + model_name + ".json", "w") as f:
            json.dump(results, f, default=handle_non_serializable, indent=2)

        vram_usage("After evaluation")
