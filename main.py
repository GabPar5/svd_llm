from src.utils import *
from src.svd_llm import *
import argparse
import gc
import importlib.util
import json
import math
import torch
import lm_eval
import multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GenerationConfig, DataCollatorForSeq2Seq # pyright: ignore[reportPrivateImportUsage]
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from lm_eval.utils import setup_logging, handle_non_serializable

# TODO fix loading compressed model path

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
        '--model_dtype', 
        type=str, 
        default='float32', 
        help='Weights dtype for the model'
    )
    parser.add_argument(
        '--compressed_dtype', 
        type=str, 
        default='float32', 
        help='Weights dtype for the compressed modules (if it\'s different that `model_dtype` it generates a mixed precision model)'
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
        "--sequential_update",
        "--finetune_on_the_fly",
        dest="sequential_update",
        action="store_true",
        help=(
            "Enable SVD-LLM sequential update after whitening/truncation. "
            "Use --sequential_update_method to choose the paper-faithful LoRA "
            "U-then-V path or the low-VRAM U-only closed-form path."
        )
    )
    parser.add_argument(
        "--update_taw_only",
        "--finetune_compressed",
        dest="update_taw_only",
        action="store_true",
        help=(
            "Load a checkpoint compressed by truncation-aware data whitening only "
            "and run SVD-LLM's sequential low-rank approximation step on it. "
            "--finetune_compressed is kept as a backwards-compatible alias."
        )
    )
    parser.add_argument(
        "--sequential_update_ridge",
        type=float,
        default=1e-6,
        help="Ridge regularization used only by --sequential_update_method local_u."
    )
    parser.add_argument(
        "--sequential_update_method",
        type=str,
        default="lora",
        choices=["lora", "local_u"],
        help=(
            "lora: upstream-repo/paper-style update, first W_u then W_v with LoRA. "
            "local_u: low-VRAM closed-form update of W_u only, matching the helper "
            "inside upstream SVDLLM.py but not the full paper procedure."
        )
    )
    parser.add_argument(
        "--sequential_lora_r",
        type=int,
        default=8,
        help="LoRA rank for --sequential_update_method lora."
    )
    parser.add_argument(
        "--sequential_lora_backend",
        type=str,
        default="trainer",
        choices=["trainer", "custom"],
        help=(
            "trainer: upstream-faithful HF Trainer LoRA update. "
            "custom: the previous lightweight local training loop."
        )
    )
    parser.add_argument(
        "--sequential_lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha for --sequential_update_method lora."
    )
    parser.add_argument(
        "--sequential_lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout for --sequential_update_method lora."
    )
    parser.add_argument(
        "--sequential_lora_lr",
        type=float,
        default=1e-4,
        help="Learning rate for --sequential_update_method lora."
    )
    parser.add_argument(
        "--sequential_lora_weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for --sequential_update_method lora."
    )
    parser.add_argument(
        "--sequential_lora_epochs",
        type=int,
        default=2,
        help="Number of epochs for each LoRA phase."
    )
    parser.add_argument(
        "--sequential_lora_max_steps",
        type=int,
        default=None,
        help="Optional max optimizer steps per LoRA phase. Overrides epochs when set."
    )
    parser.add_argument(
        "--sequential_lora_grad_accum_steps",
        type=int,
        default=None,
        help=(
            "Gradient accumulation steps for each LoRA phase. If omitted, it is "
            "computed from --sequential_lora_effective_batch_size / "
            "--sequential_lora_micro_batch_size."
        )
    )
    parser.add_argument(
        "--sequential_lora_effective_batch_size",
        type=int,
        default=64,
        help=(
            "Target effective batch size for LoRA sequential update. The upstream "
            "SVD-LLM example uses 64."
        )
    )
    parser.add_argument(
        "--sequential_lora_micro_batch_size",
        type=int,
        default=4,
        help=(
            "Per-device microbatch size used only by the LoRA sequential update. "
            "The upstream SVD-LLM script defaults to 4."
        )
    )
    parser.add_argument(
        "--sequential_lora_val_set_size",
        type=int,
        default=2000,
        help="Validation split size for the LoRA sequential update."
    )
    parser.add_argument(
        "--sequential_lora_gradient_checkpointing",
        action="store_true",
        help="Enable model gradient checkpointing during the LoRA sequential update."
    )
    parser.add_argument(
        "--finetune_dataset",
        type=str,
        default="yahma/alpaca-cleaned",
        help=(
            "Dataset used by --sequential_update_method lora. Format: "
            "dataset_name[:subset[:split]]. Defaults to the Alpaca dataset used "
            "by the upstream SVD-LLM LoRA script."
        )
    )
    parser.add_argument(
        "--max_finetune_samples",
        type=int,
        default=50000,
        help=(
            "Maximum training samples used by the LoRA sequential update. "
            "Validation samples are added on top when --sequential_lora_val_set_size > 0."
        )
    )
    parser.add_argument(
        "--finetune_cutoff_len",
        type=int,
        default=256,
        help="Prompt cutoff length for the LoRA sequential update."
    )
    parser.add_argument(
        "--finetune_train_on_inputs",
        action="store_true",
        help="If set, include instruction/input tokens in the LoRA loss."
    )
    parser.add_argument(
        "--finetune_add_eos_token",
        action="store_true",
        help="Match Alpaca-LoRA's optional EOS handling for the user prompt mask."
    )
    parser.add_argument(
        "--pin_cpu_offload",
        action="store_true",
        help=(
            "Pin CPU-offloaded calibration activations and use non-blocking CPU/GPU "
            "transfers in the sequential update path. Useful on GH200/Grace-Hopper "
            "systems with large RAM, but can pin many GB of host memory."
        )
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["eager", "sdpa", "flash_attention_2", "flash_attention_3"],
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
        default=-1, 
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

    if args.update_taw_only and (not args.use_compressed or not args.compressed_model_path):
        raise ValueError("--update_taw_only requires --use_compressed and --compressed_model_path.")

    if args.sequential_lora_micro_batch_size <= 0:
        raise ValueError("--sequential_lora_micro_batch_size must be > 0.")

    if (
        args.sequential_update_method == "lora"
        and args.sequential_lora_backend == "trainer"
        and args.sequential_lora_effective_batch_size % args.sequential_lora_micro_batch_size != 0
    ):
        raise ValueError(
            "--sequential_lora_effective_batch_size must be divisible by "
            "--sequential_lora_micro_batch_size when --sequential_lora_backend trainer."
        )

    if args.sequential_lora_grad_accum_steps is None:
        args.sequential_lora_grad_accum_steps = max(
            1,
            (
                args.sequential_lora_effective_batch_size // args.sequential_lora_micro_batch_size
                if args.sequential_lora_backend == "trainer"
                else math.ceil(args.sequential_lora_effective_batch_size / args.sequential_lora_micro_batch_size)
            ),
        )

    if (
        (args.sequential_update or args.update_taw_only)
        and args.sequential_update_method == "lora"
        and importlib.util.find_spec("peft") is None
    ):
        raise ImportError(
            "--sequential_update_method lora follows the upstream SVD-LLM LoRA "
            "pipeline and requires `peft`. Install peft, or use "
            "--sequential_update_method local_u for the low-VRAM U-only update."
        )

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
            dtype=args.model_dtype,
            device_map=args.device,
            attn_implementation=args.attn_implementation,
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
        checkpoint_device = "cpu" if args.update_taw_only else args.device
        checkpoint = torch.load(
            args.compressed_model_path,
            map_location=checkpoint_device,
            weights_only=False,
        )

        rank_map = checkpoint["rank_map"]
        state_dict = checkpoint["state_dict"]
        extra_buffers = checkpoint.get("non_persistent_buffers", {})
        checkpoint_metadata = checkpoint.get("svd_llm_metadata", {})

        # Load model config from HF and instantiate base model
        config = AutoConfig.from_pretrained(
            args.model,
            trust_remote_code=True,
            token=args.hf_token,
            dtype=DtypeMap.get_dtype(args.model_dtype)
        )

        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=True,
                dtype=DtypeMap.get_dtype(args.model_dtype),
            )
        
        try:
            model.generation_config = GenerationConfig.from_pretrained(
                args.model,
                trust_remote_code=True,
                token=args.hf_token,
            )
        except Exception as e:
            print(f"[WARNING] Could not load generation_config for {args.model}: {e}")
            model.generation_config = GenerationConfig.from_model_config(config)

        # Replace compressed layers with LowRank modules
        apply_lowrank(model, rank_map, state_dict)
        model.to_empty(device=checkpoint_device)

        missing, unexpected = model.load_state_dict(
            state_dict, 
            strict=True, 
            assign=True
        )

        restore_non_persistent_buffers(
            model=model,
            saved_buffers=extra_buffers,
            device=checkpoint_device,
            strict=True,
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
        
        if hasattr(model, "set_attn_implementation"):
            model.set_attn_implementation(args.attn_implementation)
        else:
            model.config._attn_implementation = args.attn_implementation

        # Clean memory
        del checkpoint, state_dict, extra_buffers
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
        sequential_update_str = f"_upd_{args.sequential_update_method}" if args.sequential_update else ""
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
                     sequential_update_str + \
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
            dtype = args.model_dtype,
            compressed_dtype = args.compressed_dtype,
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
            ratio_scope=args.ratio_scope,
            sequential_update=args.sequential_update,
            sequential_update_ridge=args.sequential_update_ridge,
            sequential_update_method=args.sequential_update_method,
            sequential_lora_backend=args.sequential_lora_backend,
            sequential_lora_r=args.sequential_lora_r,
            sequential_lora_alpha=args.sequential_lora_alpha,
            sequential_lora_dropout=args.sequential_lora_dropout,
            sequential_lora_lr=args.sequential_lora_lr,
            sequential_lora_weight_decay=args.sequential_lora_weight_decay,
            sequential_lora_epochs=args.sequential_lora_epochs,
            sequential_lora_max_steps=args.sequential_lora_max_steps,
            sequential_lora_grad_accum_steps=args.sequential_lora_grad_accum_steps,
            sequential_lora_effective_batch_size=args.sequential_lora_effective_batch_size,
            sequential_lora_micro_batch_size=args.sequential_lora_micro_batch_size,
            sequential_lora_val_set_size=args.sequential_lora_val_set_size,
            sequential_lora_gradient_checkpointing=args.sequential_lora_gradient_checkpointing,
            finetune_dataset=args.finetune_dataset,
            max_finetune_samples=args.max_finetune_samples,
            finetune_cutoff_len=args.finetune_cutoff_len,
            finetune_train_on_inputs=args.finetune_train_on_inputs,
            finetune_add_eos_token=args.finetune_add_eos_token,
            pin_cpu_offload=args.pin_cpu_offload
        )
        model=model.to(args.device)
        print(model)

        gc.collect()
        torch.cuda.empty_cache()
        vram_usage("After loading compressed model")
        
    if args.update_taw_only:
        print("DEBUG: Running sequential low-rank update on TAW-only compressed checkpoint...")

        if checkpoint_metadata.get("sequential_update", False) is True:
            raise ValueError(
                "--update_taw_only expects a checkpoint compressed by truncation-aware "
                "data whitening only, but checkpoint metadata says sequential_update=True."
            )

        if args.sequential_update_method == "lora":
            train_dataset, eval_dataset, finetune_stats = tokenize_finetune_dataset(
                dataset_spec=args.finetune_dataset,
                tokenizer=tokenizer,
                max_samples=args.max_finetune_samples,
                cutoff_len=args.finetune_cutoff_len,
                seed=args.seed,
                train_on_inputs=args.finetune_train_on_inputs,
                add_eos_token=args.finetune_add_eos_token,
                val_set_size=args.sequential_lora_val_set_size,
            )
            update_samples = finetune_stats["train_samples"]
            finetune_collator = DataCollatorForSeq2Seq(
                tokenizer,
                pad_to_multiple_of=8,
                return_tensors="pt",
                padding=True,
            )
            update_dataset = train_dataset
            update_dataloader = None
            if args.sequential_lora_backend == "custom":
                update_dataloader = DataLoader(
                    train_dataset, # pyright: ignore[reportArgumentType]
                    batch_size=args.sequential_lora_micro_batch_size,
                    shuffle=True,
                    pin_memory=args.pin_cpu_offload and str(args.device).startswith("cuda"),
                    collate_fn=finetune_collator,
                )
        else:
            dataset_name = args.calibration_dataset.split(":")[0]
            dataset_subset = args.calibration_dataset.split(":")[1]
            dataset_split = args.calibration_dataset.split(":")[2]

            update_dataset, update_samples = tokenize_dataset(
                dataset_name,
                dataset_subset,
                dataset_split,
                tokenizer,
                args.max_whitening_samples,
                args.batch_size,
                args.max_length,
                args.seed,
                args.save_path
            )
            finetune_collator = None
            update_dataloader = DataLoader(
                update_dataset, # pyright: ignore[reportArgumentType]
                batch_size=args.batch_size,
                shuffle=False,
                pin_memory=args.pin_cpu_offload and str(args.device).startswith("cuda"),
            )
            train_dataset = update_dataset
            eval_dataset = None
            finetune_stats = None

        selected_by_flags = any([
            args.compress_mlp,
            args.compress_att_q,
            args.compress_att_k,
            args.compress_att_v,
            args.compress_att_out,
        ])

        if selected_by_flags:
            requested_layers = generate_paths(
                args.compress_mlp,
                args.compress_att_q,
                args.compress_att_k,
                args.compress_att_v,
                args.compress_att_out,
                layers_number=model.config.num_hidden_layers
            )
            update_layers = [key for key in requested_layers if key in rank_map]
        else:
            update_layers = list(rank_map.keys())

        if not update_layers:
            raise ValueError("No LowRank layers selected for --update_taw_only.")

        print(
            f"[SEQ-UPDATE] Updating {len(update_layers)} compressed matrices "
            f"with method={args.sequential_update_method}."
        )

        model = model.cpu()
        cuda_cleanup()

        if args.sequential_update_method == "lora":
            print(
                "[SEQ-UPDATE][LoRA] "
                f"backend={args.sequential_lora_backend}, "
                f"micro_batch_size={args.sequential_lora_micro_batch_size}, "
                f"grad_accum_steps={args.sequential_lora_grad_accum_steps}, "
                f"effective_batch_size={args.sequential_lora_micro_batch_size * args.sequential_lora_grad_accum_steps}"
            )
            model = run_sequential_lora_update(
                model=model, # pyright: ignore[reportArgumentType]
                device=args.device,
                layers_str=update_layers,
                backend=args.sequential_lora_backend,
                loader=update_dataloader,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=finetune_collator,
                tokenizer=tokenizer,
                output_dir=os.path.join(args.save_path or "./tmp", "sequential_lora_trainer", model_name),
                model_dtype=args.model_dtype,
                lora_r=args.sequential_lora_r,
                lora_alpha=args.sequential_lora_alpha,
                lora_dropout=args.sequential_lora_dropout,
                learning_rate=args.sequential_lora_lr,
                weight_decay=args.sequential_lora_weight_decay,
                epochs=args.sequential_lora_epochs,
                max_steps=args.sequential_lora_max_steps,
                micro_batch_size=args.sequential_lora_micro_batch_size,
                effective_batch_size=args.sequential_lora_effective_batch_size,
                grad_accum_steps=args.sequential_lora_grad_accum_steps,
                gradient_checkpointing=args.sequential_lora_gradient_checkpointing,
            )
            updated_rank_map = {key: rank_map[key] for key in update_layers}
        elif args.sequential_update_method == "local_u":
            dense_reference_model = AutoModelForCausalLM.from_pretrained(
                args.model,
                dtype=args.model_dtype,
                device_map=None,
                max_position_embeddings=args.max_length,
                use_cache=False,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                token=args.hf_token,
                trust_remote_code=True
            )
            dense_reference_model.eval()
            dense_reference_model.requires_grad_(False)

            runner = SequentialLocalUpdateRunner(
                model=model, # pyright: ignore[reportArgumentType]
                loader=update_dataloader, # pyright: ignore[reportArgumentType]
                device=args.device,
                compressed_dtype=args.compressed_dtype,
                ridge=args.sequential_update_ridge,
                pin_cpu_offload=args.pin_cpu_offload,
            )
            updated_rank_map = runner.update_taw_only_checkpoint(
                dense_reference_model=dense_reference_model, # type: ignore[arg-type]
                layers_str=update_layers,
            )
            rank_map.update(updated_rank_map)
            del runner, dense_reference_model
        else:
            raise ValueError(f"Unknown sequential_update_method: {args.sequential_update_method}")

        del update_dataset, update_dataloader, finetune_collator, train_dataset, eval_dataset
        cuda_cleanup()

        model.requires_grad_(False)
        model.eval()

        if args.compressed_model_path.endswith(".pt"):
            updated_model_path = args.compressed_model_path[:-3] + f"_sequpd_{args.sequential_update_method}.pt"
        else:
            updated_model_path = args.compressed_model_path + f"_sequpd_{args.sequential_update_method}.pt"

        metadata = dict(checkpoint_metadata)
        metadata.update({
            "sequential_update": True,
            "sequential_update_method": args.sequential_update_method,
            "sequential_lora_backend": args.sequential_lora_backend if args.sequential_update_method == "lora" else None,
            "sequential_update_source": "taw_only_checkpoint",
            "taw_only_checkpoint_path": args.compressed_model_path,
            "update_samples": update_samples,
            "sequential_update_finetune_stats": finetune_stats,
            "sequential_update_dataset": args.finetune_dataset if args.sequential_update_method == "lora" else args.calibration_dataset,
            "finetune_cutoff_len": args.finetune_cutoff_len if args.sequential_update_method == "lora" else None,
            "finetune_train_on_inputs": args.finetune_train_on_inputs if args.sequential_update_method == "lora" else None,
            "num_updated_lowrank_modules": len(updated_rank_map),
            "sequential_lora_r": args.sequential_lora_r if args.sequential_update_method == "lora" else None,
            "sequential_lora_alpha": args.sequential_lora_alpha if args.sequential_update_method == "lora" else None,
            "sequential_lora_dropout": args.sequential_lora_dropout if args.sequential_update_method == "lora" else None,
            "sequential_lora_lr": args.sequential_lora_lr if args.sequential_update_method == "lora" else None,
            "sequential_lora_epochs": args.sequential_lora_epochs if args.sequential_update_method == "lora" else None,
            "sequential_lora_max_steps": args.sequential_lora_max_steps if args.sequential_update_method == "lora" else None,
            "sequential_lora_grad_accum_steps": args.sequential_lora_grad_accum_steps if args.sequential_update_method == "lora" else None,
            "sequential_lora_micro_batch_size": args.sequential_lora_micro_batch_size if args.sequential_update_method == "lora" else None,
            "sequential_lora_effective_batch_size": args.sequential_lora_micro_batch_size * args.sequential_lora_grad_accum_steps if args.sequential_update_method == "lora" else None,
            "sequential_lora_val_set_size": args.sequential_lora_val_set_size if args.sequential_update_method == "lora" else None,
        })

        payload = {
            "state_dict": model.state_dict(),
            "rank_map": rank_map,
            "non_persistent_buffers": collect_non_persistent_buffers(model),
            "config": model.config.to_dict(),
            "generation_config": (
                model.generation_config.to_dict() # pyright: ignore[reportOptionalMemberAccess]
                if getattr(model, "generation_config", None) is not None
                else None
            ),
            "svd_llm_metadata": metadata,
        }

        torch.save(payload, updated_model_path)
        del payload
        print(f"[DEBUG] Sequentially updated checkpoint saved to: {updated_model_path}")

        # Evaluation/result filenames must follow the newly saved checkpoint,
        # otherwise --update_taw_only overwrites the TAW-only JSON/log labels.
        model_name = os.path.basename(updated_model_path)
        if model_name.endswith(".pt"):
            model_name = model_name[:-3]
        print(f"[DEBUG] Evaluation/result basename: {model_name}")

        if args.evaluate:
            model = model.to(args.device)
        vram_usage("After finetuning compressed model")

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
        
        lm_eval_task_names = [t for t in tasks_list if t not in {"wikitext", "c4"}]
        if isinstance(num_fewshot, list):
            shot_by_task = {
                tasks_list[i]: int(num_fewshot[i])
                for i in range(len(tasks_list))
            }
        else:
            shot_by_task = {t: int(num_fewshot) for t in tasks_list}

        task_manager = TaskManager()
        loaded = task_manager.load(lm_eval_task_names)
        task_objects = []

        for task_name, task_obj in loaded["tasks"].items():
            task_obj.set_config("num_fewshot", shot_by_task[task_name])
            task_objects.append(task_obj)

        print(f"[DEBUG] Num few-shots: {num_fewshot}")
        print(f"[DEBUG] List of evaluation tasks: {tasks_list}")
        print(f"[DEBUG] Tasks dictionaries: {task_objects}")
        print(f"[DEBUG] HF model context length: {model.config.max_position_embeddings}")

        # Clamp max model context
        max_gen_task_context_length = min(
            args.eval_max_length,
            model.config.max_position_embeddings - args.max_eval_tokens
        )
        max_length = min(
            args.eval_max_length,
            model.config.max_position_embeddings
        )
        print(f"[DEBUG] Evaluation context length for generation tasks: {max_gen_task_context_length}")

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

        if task_objects is not None and len(task_objects) > 0:
            model.config.use_cache = True
            # Avoid HF generate() warning from a stale global GenerationConfig.
            if getattr(model, "generation_config", None) is not None:
                model.generation_config.max_new_tokens = None # pyright: ignore[reportOptionalMemberAccess]
            # WARNING - PyRight reports lots of issues when dealing with lm-eval-harness 
            eval_model = HFLM(
                pretrained=model, # pyright: ignore[reportCallIssue]
                tokenizer = tokenizer, # pyright: ignore[reportCallIssue]
                batch_size=args.eval_batch_size, # pyright: ignore[reportCallIssue]
                max_batch_size=128, # pyright: ignore[reportCallIssue]
                device = args.device, # pyright: ignore[reportCallIssue]
                max_length = max_length # pyright: ignore[reportCallIssue] # TODO revert to max generation tokens if necessary
            )
            print(f"[DEBUG] HFLM model context length: {eval_model.max_length}") # pyright: ignore[reportAttributeAccessIssue]


            vram_usage("Before evaluation")

            # Run evaluation 
            results = lm_eval.simple_evaluate(
                model=eval_model, # pyright: ignore[reportCallIssue]
                tasks=task_objects,  # type: ignore
                num_fewshot=None, # pyright: ignore[reportCallIssue]
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
