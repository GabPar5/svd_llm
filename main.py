from src.utils import *
from src.svd_llm import *
import argparse
import importlib.util
import json
import lm_eval
import multiprocessing as mp
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from lm_eval.utils import setup_logging, handle_non_serializable
from transformers import AutoModelForCausalLM, AutoTokenizer # pyright: ignore[reportPrivateImportUsage]

# TODO fix loading compressed model path

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen2.5-7B',
        help='LLM to load from huggingface',
    )
    parser.add_argument(
        '--run_v2',
        action='store_true',
        help='Run SVD-LLM V2',
    )
    parser.add_argument(
        '--model_dtype',
        type=str,
        default='float32',
        help='Weights dtype for the model',
    )
    parser.add_argument(
        '--compressed_dtype',
        type=str,
        default='float32',
        help='Weights dtype for the compressed modules (if it\'s different that `model_dtype` it generates a mixed precision model)',
    )
    parser.add_argument(
        '--compression_ratio',
        type=float,
        default=0.2,
        help='Target compression ratio,(0,1), default=0.2, means removing about 20%% of the params',
    )
    parser.add_argument(
        "--ratio_scope",
        type=str,
        default="selected",
        choices=[ "selected", "all" ],
        help=(
            "selected: compression_ratio applies only to selected matrices. "
            "all: compression_ratio applies to all targetable projection matrices "
            "(MLP + q/k/v/o), even if only a subset is selected for compression"
        ),
    )
    parser.add_argument(
        '--calibration_dataset',
        type=str,
        default='EleutherAI/wikitext_document_level:wikitext-2-raw-v1:train',
        help='Calibration dataset, format is "datasetNameOrPath:subset:split"',
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=2048,
        help='Maximum context length for the LLM during compression',
    )
    parser.add_argument(
        '--max_whitening_samples',
        type=int,
        default=256,
        help='Number of calibration data samples for whitening',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='Batch size for data preprocessing and calibration forward pass',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=6363,
        help='Seed for sampling the calibration data',
    )
    parser.add_argument(
        '--device',
        type=str,
        default="cuda",
        help='device',
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default=None,
        help='Base path to save the whitening matrices and the compressed model checkpoints',
    )
    parser.add_argument(
        '--scratch_path',
        type=str,
        default=None,
        help=(
            'Base path for the regenerable intermediates (whitening matrices, '
            'activation checkpoints, sequential LoRA trainer state). Defaults to '
            '--save_path, so setting it keeps the bulky artifacts off a small or '
            'synced save directory'
        ),
    )
    parser.add_argument(
        '--no_save_checkpoint',
        action='store_true',
        help=(
            'Do not write the compressed .pt checkpoint (nor the tokenizer beside it). '
            'Logs, evaluation results and the run sidecar are still written under --save_path'
        ),
    )
    parser.add_argument(
        '--whitening_mat_path',
        type=str,
        default=None,
        help='Local path to load the whitening matrices',
    )
    parser.add_argument(
        "--whitening_only",
        action="store_true",
    )
    parser.add_argument(
        "--whitening_start_layer",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--whitening_end_layer",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--sequential_update",
        "--finetune_on_the_fly",
        dest="sequential_update",
        action="store_true",
        help=(
            "Enable SVD-LLM sequential update after whitening/truncation. "
            "Use --sequential_update_method to choose the paper-faithful LoRA "
            "U-then-V path or the low-VRAM U-only closed-form path"
        ),
    )
    parser.add_argument(
        "--update_taw_only",
        "--finetune_compressed",
        dest="update_taw_only",
        action="store_true",
        help=(
            "Load a checkpoint compressed by truncation-aware data whitening only "
            "and run SVD-LLM's sequential low-rank approximation step on it. "
            "--finetune_compressed is kept as a backwards-compatible alias"
        ),
    )
    parser.add_argument(
        "--sequential_update_ridge",
        type=float,
        default=1e-6,
        help="Ridge regularization used only by --sequential_update_method local_u",
    )
    parser.add_argument(
        "--sequential_update_method",
        type=str,
        default="lora",
        choices=[ "lora", "local_u" ],
        help=(
            "lora: upstream-repo/paper-style update, first W_u then W_v with LoRA. "
            "local_u: low-VRAM closed-form update of W_u only, matching the helper "
            "inside upstream SVDLLM.py but not the full paper procedure"
        ),
    )
    parser.add_argument(
        "--sequential_lora_r",
        type=int,
        default=8,
        help="LoRA rank for --sequential_update_method lora",
    )
    parser.add_argument(
        "--sequential_lora_backend",
        type=str,
        default="trainer",
        choices=[ "trainer", "custom" ],
        help=(
            "trainer: upstream-faithful HF Trainer LoRA update. "
            "custom: the previous lightweight local training loop"
        ),
    )
    parser.add_argument(
        "--sequential_lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha for --sequential_update_method lora",
    )
    parser.add_argument(
        "--sequential_lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout for --sequential_update_method lora",
    )
    parser.add_argument(
        "--sequential_lora_lr",
        type=float,
        default=1e-4,
        help="Learning rate for --sequential_update_method lora",
    )
    parser.add_argument(
        "--sequential_lora_weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for --sequential_update_method lora",
    )
    parser.add_argument(
        "--sequential_lora_epochs",
        type=int,
        default=2,
        help="Number of epochs for each LoRA phase",
    )
    parser.add_argument(
        "--sequential_lora_max_steps",
        type=int,
        default=None,
        help="Optional max optimizer steps per LoRA phase. Overrides epochs when set",
    )
    parser.add_argument(
        "--sequential_lora_grad_accum_steps",
        type=int,
        default=None,
        help=(
            "Gradient accumulation steps for each LoRA phase. If omitted, it is "
            "computed from --sequential_lora_effective_batch_size / "
            "--sequential_lora_micro_batch_size"
        ),
    )
    parser.add_argument(
        "--sequential_lora_effective_batch_size",
        type=int,
        default=64,
        help=(
            "Target effective batch size for LoRA sequential update. The upstream "
            "SVD-LLM example uses 64"
        ),
    )
    parser.add_argument(
        "--sequential_lora_micro_batch_size",
        type=int,
        default=4,
        help=(
            "Per-device microbatch size used only by the LoRA sequential update. "
            "The upstream SVD-LLM script defaults to 4"
        ),
    )
    parser.add_argument(
        "--sequential_lora_val_set_size",
        type=int,
        default=2000,
        help="Validation split size for the LoRA sequential update",
    )
    parser.add_argument(
        "--sequential_lora_gradient_checkpointing",
        action="store_true",
        help="Enable model gradient checkpointing during the LoRA sequential update",
    )
    parser.add_argument(
        "--finetune_dataset",
        type=str,
        default="yahma/alpaca-cleaned",
        help=(
            "Dataset used by --sequential_update_method lora. Format: "
            "dataset_name[:subset[:split]]. Defaults to the Alpaca dataset used "
            "by the upstream SVD-LLM LoRA script"
        ),
    )
    parser.add_argument(
        "--max_finetune_samples",
        type=int,
        default=50000,
        help=(
            "Maximum training samples used by the LoRA sequential update. "
            "Validation samples are added on top when --sequential_lora_val_set_size > 0"
        ),
    )
    parser.add_argument(
        "--finetune_cutoff_len",
        type=int,
        default=256,
        help="Prompt cutoff length for the LoRA sequential update",
    )
    parser.add_argument(
        "--finetune_train_on_inputs",
        action="store_true",
        help="If set, include instruction/input tokens in the LoRA loss",
    )
    parser.add_argument(
        "--finetune_add_eos_token",
        action="store_true",
        help="Match Alpaca-LoRA's optional EOS handling for the user prompt mask",
    )
    parser.add_argument(
        "--pin_cpu_offload",
        action="store_true",
        help=(
            "Pin CPU-offloaded calibration activations and use non-blocking CPU/GPU "
            "transfers in the sequential update path. Useful on GH200/Grace-Hopper "
            "systems with large RAM, but can pin many GB of host memory"
        ),
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=[ "eager", "sdpa", "flash_attention_2", "flash_attention_3" ],
    )
    parser.add_argument(
        '--use_compressed',
        action='store_true',
        help='Use compressed model for evaluation',
    )
    parser.add_argument(
        '--compressed_model_path',
        type=str,
        default=None,
        help='Local path to load the compressed model - if you need to do evaluation only',
    )
    parser.add_argument(
        '--compress_mlp',
        action='store_true',
        help='Compress MLP weights',
    )
    parser.add_argument(
        '--compress_att_q',
        action='store_true',
        help='Compress attention query projection matrices',
    )
    parser.add_argument(
        '--compress_att_k',
        action='store_true',
        help='Compress attention key projection matrices',
    )
    parser.add_argument(
        '--compress_att_v',
        action='store_true',
        help='Compress attention value projection matrices',
    )
    parser.add_argument(
        '--compress_att_out',
        action='store_true',
        help='Compress attention output projection matrix',
    )
    parser.add_argument(
        '--het',
        action='store_true',
        help='Assign heterogeneous compression ratio',
    )
    parser.add_argument(
        '--bypass_early_layers',
        type=int,
        default=-1,
        help='Number of starting layers which bypass heterogeneous compression (or compression at all)',
    )
    parser.add_argument(
        '--bypass_late_layers',
        type=int,
        default=-1,
        help='Number of ending layers which bypass heterogeneous compression (or compression at all). Can be combined with --bypass_early_layers',
    )
    parser.add_argument(
        '--bypass_ratio',
        type=float,
        default=0.0,
        help='Compression ratio for the bypassed layers, applied to both ends',
    )
    parser.add_argument(
        '--max_ratio',
        type=float,
        default=0.9,
        help='Upper bound on the compression ratio any single matrix may receive, shared by every allocation policy',
    )
    parser.add_argument(
        '--group_criterion',
        type=str,
        default="type",
        choices=[criterion.value for criterion in GroupBy],
        help=(
            'Criterion used to group weight matrices in heterogeneous setting. "hierarchical" groups by decoder '
            'block like "decoder" and additionally lets --outer_allocation score whole blocks by Block Influence'
        ),
    )
    parser.add_argument(
        '--inner_allocation',
        type=str,
        default=InnerAllocation.WATERFILL.value,
        choices=[policy.value for policy in INNER_POLICIES],
        help='Policy that splits a group budget across the matrices inside it. Only used in heterogeneous setting',
    )
    parser.add_argument(
        '--outer_allocation',
        type=str,
        default=OuterAllocation.PARAM_SHARE.value,
        choices=[policy.value for policy in OUTER_POLICIES],
        help=(
            'Policy that splits the budget across groups. "waterfill" needs --group_criterion hierarchical, since '
            'only that criterion scores whole decoder blocks'
        ),
    )
    parser.add_argument(
        '--group_patterns',
        type=str,
        default="q_proj:self_attn.q_proj;k_proj:self_attn.k_proj;v_proj:self_attn.v_proj;o_proj:self_attn.o_proj;gate_proj:mlp.gate_proj;up_proj:mlp.up_proj;down_proj:mlp.down_proj",
        help='Group patterns used when grouping weight matrices by type, the pattern is "groupName1:weightType1,weightType2;groupName2:weightType1,weightType2;..."',
    )
    parser.add_argument(
        '--score_metric',
        type=str,
        default="truncation",
        help=(
            "Score metric used for weight importance during heterogeneous ratio "
            "allocation. Possible values are \"truncation\", \"entropy\" and \"eff_rank\", "
            "each with a squared-spectrum variant (appending \"_sq\"), the tail "
            "variants \"full_norm_tail_entropy\", \"full_norm_sq_tail_entropy\", "
            "\"full_norm_tail_eff_rank\" and \"full_norm_sq_tail_eff_rank\", plus "
            "\"norm|p\" for the p-Schatten norm of the truncated tail, where p is a "
            "number, \"inf\" or \"-inf\". Prefixing any of them as "
            "\"composite|<local>|block_influence\" fuses that spectral score with the "
            "end-to-end influence of the decoder block, weighted by --fusion_alpha"
        ),
    )
    parser.add_argument(
        '--fusion_alpha',
        type=float,
        default=0.5,
        help=(
            'Weight of the end-to-end half of a composite score metric, in [0, 1]. 0 keeps the local score alone '
            '(in log form), 1 the block influence alone. Ignored by non-composite metrics'
        ),
    )
    parser.add_argument(
        '--offset',
        type=float,
        default=1.5,
        help='Offset added to scores to avoid log(x) with x <= 1. Read by the "waterfill" inner policy',
    )
    parser.add_argument(
        '--outer_offset',
        type=float,
        default=1.5,
        help=(
            'Same offset, applied to the per-block Block Influence by the "waterfill" outer policy. Kept separate '
            'from --offset so that tuning how matrices compete inside a block cannot change how blocks compete'
        ),
    )
    parser.add_argument(
        '--softmax_temp',
        type=float,
        default=1.0,
        help=(
            'Temperature of the "softmax_temp" inner policy. Scores are min-max normalized to [0, 1] within each '
            'group first, so the largest allocation weight exceeds the smallest by exp(1 / softmax_temp): 1.0 is '
            'nearly uniform, 0.2 spreads moderately, 0.05 strongly'
        ),
    )
    parser.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help='Huggingface token to download/upload models',
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate the model on a set of tasks',
    )
    parser.add_argument(
        '--eval_sampling',
        action='store_true',
        help='Use conditional sampling during evaluation',
    )
    parser.add_argument(
        '--eval_batch_size',
        type=str,
        default="auto",
        help='Evaluation batch size',
    )
    parser.add_argument(
        '--eval_tasks',
        type=str,
        default='wikitext|0',
        help='Evaluation tasks, the pattern is "taskName1,taskName2,...,taskNameK|numShots" or "taskName1,taskName2,...,taskNameK|numShots1,numShots2,...,numShotsK"',
    )
    parser.add_argument(
        '--eval_max_length',
        type=int,
        default=4096,
        help='Maximum context length for the LLM during evaluation',
    )
    parser.add_argument(
        '--max_eval_tokens',
        type=int,
        default=256,
        help='Maximum number of tokens to generate during evaluation',
    )

    args = parser.parse_args()

    if args.update_taw_only and (not args.use_compressed or not args.compressed_model_path):
        raise ValueError("--update_taw_only requires --use_compressed and --compressed_model_path")

    is_lora_update_requested = (
        (args.sequential_update or args.update_taw_only)
        and args.sequential_update_method == "lora"
    )

    validate_lora_batching(
        args.sequential_lora_effective_batch_size,
        args.sequential_lora_micro_batch_size,
        args.sequential_lora_backend,
        require_exact_split=args.sequential_update_method == "lora",
    )

    if args.sequential_lora_grad_accum_steps is None:
        args.sequential_lora_grad_accum_steps = resolve_grad_accum_steps(
            args.sequential_lora_effective_batch_size,
            args.sequential_lora_micro_batch_size,
            args.sequential_lora_backend,
        )

    if is_lora_update_requested and importlib.util.find_spec("peft") is None:
        raise ImportError(
            "--sequential_update_method lora follows the upstream SVD-LLM LoRA "
            "pipeline and requires `peft`. Install peft, or use "
            "--sequential_update_method local_u for the low-VRAM U-only update",
        )

    model_eval_path = os.path.join(args.save_path, "eval", sanitize_model_name(args.model))

    if not args.use_compressed:
        print("DEBUG: Loading original model from the hub...")
        vram_usage("Before loading original model")
        model_name = sanitize_model_name(args.model)

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
            trust_remote_code=True,
        )
        # Avoid warning
        eos = model.generation_config.eos_token_id # pyright: ignore[reportOptionalMemberAccess]
        if isinstance(eos, list):
            eos = eos[0]
        model.generation_config.pad_token_id = eos # pyright: ignore[reportOptionalMemberAccess]
        vram_usage("After loading original model")
    elif args.compressed_model_path:
        print("DEBUG: Loading compressed model from disk...")
        vram_usage("Before loading compressed model")
        model_name = os.path.splitext(os.path.basename(args.compressed_model_path))[0]

        # The tokenizer was saved next to the checkpoint
        tokenizer_path = os.path.dirname(args.compressed_model_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # The sequential update runs layer by layer from CPU
        checkpoint_device = "cpu" if args.update_taw_only else args.device

        model, rank_map, checkpoint_metadata = load_compressed_model(
            base_model_name=args.model,
            checkpoint_path=args.compressed_model_path,
            model_dtype=args.model_dtype,
            device=checkpoint_device,
            hf_token=args.hf_token,
            attn_implementation=args.attn_implementation,
        )

        vram_usage("After loading compressed model")
    else:
        model_name = build_run_name(
            model_name=args.model,
            ratio=args.compression_ratio,
            compress_mlp=args.compress_mlp,
            compress_att_q=args.compress_att_q,
            compress_att_k=args.compress_att_k,
            compress_att_v=args.compress_att_v,
            compress_att_out=args.compress_att_out,
            ratio_scope=args.ratio_scope,
            heterogeneous=args.het,
            group_criterion=args.group_criterion,
            score_metric=args.score_metric,
            bypass_early_layers=args.bypass_early_layers,
            sequential_update=args.sequential_update,
            sequential_update_method=args.sequential_update_method,
            is_v2=args.run_v2,
            bypass_late_layers=args.bypass_late_layers,
            max_ratio=args.max_ratio,
            inner_allocation=args.inner_allocation,
            outer_allocation=args.outer_allocation,
            bypass_ratio=args.bypass_ratio,
            fusion_alpha=args.fusion_alpha,
            seed=args.seed,
            offset=args.offset,
            softmax_temp=args.softmax_temp,
            outer_offset=args.outer_offset,
        )

        dataset_name, dataset_subset, dataset_split = parse_dataset_spec(args.calibration_dataset)

        group_patterns_dict = {}
        for group in args.group_patterns.split(";"):
            group_name, _, group_types = group.partition(":")
            group_patterns_dict[group_name] = group_types.split(",")

        # Initialize logger
        model_log_path = os.path.join(args.save_path, "logs", sanitize_model_name(args.model))
        os.makedirs(model_log_path, exist_ok=True)
        sys.stdout = Logger(
            filename=os.path.join(model_log_path, f"{model_name}.log"),
        )

        model, tokenizer = compress_svd_llm(
            model_name = args.model,
            ratio = round(args.compression_ratio, 2),
            dataset = {
                "name": dataset_name,
                "subset": dataset_subset,
                "split": dataset_split,
                "max_samples": args.max_whitening_samples,
            },
            max_length = args.max_length,
            is_v2 = args.run_v2,
            dtype = args.model_dtype,
            compressed_dtype = args.compressed_dtype,
            batch_size = args.batch_size,
            seed = args.seed,
            device = args.device,
            save_path = args.save_path,
            scratch_path = args.scratch_path,
            save_checkpoint = not args.no_save_checkpoint,
            whitening_mat_path = args.whitening_mat_path,
            compress_mlp = args.compress_mlp,
            compress_att_q = args.compress_att_q,
            compress_att_k = args.compress_att_k,
            compress_att_v = args.compress_att_v,
            compress_att_out = args.compress_att_out,
            score_metric=args.score_metric,
            heterogeneous = args.het,
            group_criterion = args.group_criterion,
            inner_allocation = args.inner_allocation,
            outer_allocation = args.outer_allocation,
            group_patterns = group_patterns_dict,
            hf_token = args.hf_token,
            whitening_only = args.whitening_only,
            whitening_start_layer = args.whitening_start_layer,
            whitening_end_layer = args.whitening_end_layer,
            bypass_early_layers = args.bypass_early_layers,
            bypass_late_layers = args.bypass_late_layers,
            bypass_ratio = args.bypass_ratio,
            max_ratio = args.max_ratio,
            ratio_scope=args.ratio_scope,
            offset=args.offset,
            softmax_temp=args.softmax_temp,
            outer_offset=args.outer_offset,
            fusion_alpha=args.fusion_alpha,
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
            pin_cpu_offload=args.pin_cpu_offload,
        )
        model = model.to(args.device)
        print(model)

        # Record the arguments beside the checkpoint the compression just wrote,
        # so a compress-only run is self-describing without an evaluation
        if args.save_path:
            save_run_config(
                directory=os.path.join(args.save_path, "models", sanitize_model_name(args.model)),
                run_name=model_name,
                config={ "args": sanitize_run_args(vars(args)) },
            )

        cuda_cleanup()
        vram_usage("After loading compressed model")

    if args.update_taw_only:
        print("DEBUG: Running sequential low-rank update on TAW-only compressed checkpoint...")

        if checkpoint_metadata.get("sequential_update", False) is True:
            raise ValueError(
                "--update_taw_only expects a checkpoint compressed by truncation-aware "
                "data whitening only, but checkpoint metadata says sequential_update=True",
            )

        if args.sequential_update_method == "lora":
            train_dataset, eval_dataset, finetune_stats, finetune_collator, update_dataloader = prepare_lora_update_data(
                tokenizer=tokenizer,
                finetune_dataset=args.finetune_dataset,
                max_finetune_samples=args.max_finetune_samples,
                finetune_cutoff_len=args.finetune_cutoff_len,
                finetune_train_on_inputs=args.finetune_train_on_inputs,
                finetune_add_eos_token=args.finetune_add_eos_token,
                val_set_size=args.sequential_lora_val_set_size,
                micro_batch_size=args.sequential_lora_micro_batch_size,
                backend=args.sequential_lora_backend,
                device=args.device,
                seed=args.seed,
                pin_cpu_offload=args.pin_cpu_offload,
            )
            update_dataset = train_dataset
            update_samples = finetune_stats["train_samples"]
        else:
            dataset_name, dataset_subset, dataset_split = parse_dataset_spec(args.calibration_dataset)

            update_dataset, update_samples = tokenize_dataset(
                dataset_name,
                dataset_subset,
                dataset_split,
                tokenizer,
                args.max_whitening_samples,
                args.batch_size,
                args.max_length,
                args.seed,
                args.save_path,
            )
            finetune_collator = None
            update_dataloader = DataLoader(
                update_dataset, # pyright: ignore[reportArgumentType]
                batch_size=args.batch_size,
                shuffle=False,
                pin_memory=pin_memory_enabled(args.pin_cpu_offload, args.device),
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
                layers_number=model.config.num_hidden_layers,
            )
            update_layers = [key for key in requested_layers if key in rank_map]
        else:
            update_layers = list(rank_map.keys())

        if not update_layers:
            raise ValueError("No LowRank layers selected for --update_taw_only")

        print(
            f"[SEQ-UPDATE] Updating {len(update_layers)} compressed matrices "
            f"with method={args.sequential_update_method}",
        )

        model = model.cpu()
        cuda_cleanup()

        if args.sequential_update_method == "lora":
            print(
                "[SEQ-UPDATE][LoRA] "
                f"backend={args.sequential_lora_backend}, "
                f"micro_batch_size={args.sequential_lora_micro_batch_size}, "
                f"grad_accum_steps={args.sequential_lora_grad_accum_steps}, "
                f"effective_batch_size={args.sequential_lora_micro_batch_size * args.sequential_lora_grad_accum_steps}",
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
                output_dir=os.path.join(
                    scratch_root(args.save_path, args.scratch_path),
                    "sequential_lora_trainer",
                    model_name,
                ),
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
            model, updated_rank_map = run_local_u_update(
                model=model, # pyright: ignore[reportArgumentType]
                base_model_name=args.model,
                loader=update_dataloader, # pyright: ignore[reportArgumentType]
                update_layers=update_layers,
                device=args.device,
                dtype=args.model_dtype,
                compressed_dtype=args.compressed_dtype,
                max_length=args.max_length,
                ridge=args.sequential_update_ridge,
                hf_token=args.hf_token,
                pin_cpu_offload=args.pin_cpu_offload,
            )
            rank_map.update(updated_rank_map)
        else:
            raise ValueError(f"Unknown sequential_update_method: {args.sequential_update_method}")

        del update_dataset, update_dataloader, finetune_collator, train_dataset, eval_dataset
        cuda_cleanup()

        model.requires_grad_(False)
        model.eval()

        checkpoint_stem = args.compressed_model_path
        if checkpoint_stem.endswith(".pt"):
            checkpoint_stem = checkpoint_stem[:-3]
        updated_model_path = f"{checkpoint_stem}_sequpd_{args.sequential_update_method}.pt"

        is_lora_update = args.sequential_update_method == "lora"
        metadata = dict(checkpoint_metadata)
        metadata.update({
            "sequential_update": True,
            "sequential_update_method": args.sequential_update_method,
            "sequential_update_source": "taw_only_checkpoint",
            "taw_only_checkpoint_path": args.compressed_model_path,
            "update_samples": update_samples,
            "sequential_update_finetune_stats": finetune_stats,
            "num_updated_lowrank_modules": len(updated_rank_map),
            **build_lora_update_metadata(
                is_lora=is_lora_update,
                backend=args.sequential_lora_backend,
                finetune_dataset=args.finetune_dataset,
                finetune_cutoff_len=args.finetune_cutoff_len,
                finetune_train_on_inputs=args.finetune_train_on_inputs,
                lora_r=args.sequential_lora_r,
                lora_alpha=args.sequential_lora_alpha,
                lora_dropout=args.sequential_lora_dropout,
                lora_lr=args.sequential_lora_lr,
                lora_epochs=args.sequential_lora_epochs,
                lora_max_steps=args.sequential_lora_max_steps,
                grad_accum_steps=args.sequential_lora_grad_accum_steps,
                micro_batch_size=args.sequential_lora_micro_batch_size,
                val_set_size=args.sequential_lora_val_set_size,
                fallback_dataset=args.calibration_dataset,
            ),
        })

        if args.no_save_checkpoint:
            print("[DEBUG] Checkpoint saving disabled, the updated model stays in memory only")
        else:
            save_compressed_checkpoint(
                model=model,
                checkpoint_path=updated_model_path,
                rank_map=rank_map,
                metadata=metadata,
            )

        # Evaluation/result filenames must follow the newly saved checkpoint,
        # otherwise --update_taw_only overwrites the TAW-only JSON/log labels
        model_name = os.path.splitext(os.path.basename(updated_model_path))[0]
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
            raise ValueError(
                'The argument `eval_tasks` must be a string following these formats: '
                '"taskName1,taskName2,...,taskNameK|numShots" or '
                '"taskName1,taskName2,...,taskNameK|numShots1,numShots2,...,numShotsK"'
            )
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
            model.config.max_position_embeddings - args.max_eval_tokens,
        )
        max_length = min(
            args.eval_max_length,
            model.config.max_position_embeddings,
        )
        print(f"[DEBUG] Evaluation context length for generation tasks: {max_gen_task_context_length}")

        results = {}

        # Perplexity tasks bypass lm-eval to stay comparable with the SVD-LLM paper
        ppl_tasks = {
            "wikitext": {
                "dataset_name": "wikitext",
                "subset": "wikitext-2-raw-v1",
                "split": "test",
            },
            # A single validation shard, which is what upstream SVD-LLM loads. The
            # full en/validation split is ~364k documents and cannot be joined
            "c4": {
                "dataset_name": "allenai/c4",
                "subset": None,
                "split": "validation",
                "data_files": { "validation": "en/c4-validation.00000-of-00008.json.gz" },
            },
        }
        ppl_results = {}

        for task_name, ppl_kwargs in ppl_tasks.items():
            if task_name not in tasks_list:
                continue

            ppl_results[task_name] = ppl_eval(
                model,
                tokenizer,
                eval_max_length=max_length,
                batch_size=args.eval_batch_size,
                device=args.device,
                **ppl_kwargs,
            )
            cuda_cleanup()

        if task_objects:
            model.config.use_cache = True
            # Avoid HF generate() warning from a stale global GenerationConfig
            if getattr(model, "generation_config", None) is not None:
                model.generation_config.max_new_tokens = None # pyright: ignore[reportOptionalMemberAccess]
            # WARNING - PyRight reports lots of issues when dealing with lm-eval-harness
            eval_model = HFLM(
                pretrained=model, # pyright: ignore[reportCallIssue]
                tokenizer = tokenizer, # pyright: ignore[reportCallIssue]
                batch_size=args.eval_batch_size, # pyright: ignore[reportCallIssue]
                max_batch_size=128, # pyright: ignore[reportCallIssue]
                device = args.device, # pyright: ignore[reportCallIssue]
                max_length = max_length, # pyright: ignore[reportCallIssue] # TODO revert to max generation tokens if necessary
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
                    "max_gen_toks": args.max_eval_tokens,
                },
                apply_chat_template=False,
                random_seed=args.seed, # pyright: ignore[reportCallIssue]
                numpy_random_seed=args.seed, # pyright: ignore[reportCallIssue]
                torch_random_seed=args.seed, # pyright: ignore[reportCallIssue]
                fewshot_random_seed=args.seed, # pyright: ignore[reportCallIssue]
            ) # pyright: ignore[reportCallIssue]

        # SAVE RESULTS
        os.makedirs(model_eval_path, exist_ok=True)

        if "results" not in results:  # pyright: ignore[reportOperatorIssue]
            results["results"] = {} # pyright: ignore[reportOptionalSubscript]

        for task_name, ppl in ppl_results.items():
            results["results"][task_name] = { # pyright: ignore
                "alias": task_name,
                "token_perplexity,none": ppl,
                "token_perplexity_stderr,none": "N/A",
            }

        # Tasks this run did not evaluate stay as an earlier run measured them,
        # so adding a task later cannot delete the ones already collected
        results_path = os.path.join(model_eval_path, f"{model_name}.json")
        results = merge_eval_results(results_path, results) # pyright: ignore[reportArgumentType]

        with open(results_path, "w") as f:
            json.dump(results, f, default=handle_non_serializable, indent=2)

        # Whatever either side already recorded about this run name, so a second
        # evaluation adds to the description rather than replacing it
        recorded: Dict[str, Any] = {}
        for recorded_path in (
            run_config_path(os.path.join(args.save_path, "models", sanitize_model_name(args.model)), model_name),
            run_config_path(model_eval_path, model_name),
        ):
            if os.path.exists(recorded_path):
                with open(recorded_path, "r", encoding="utf-8") as config_file:
                    recorded = { **recorded, **json.load(config_file) }

        run_args = sanitize_run_args(vars(args))
        eval_config = {
            **recorded,
            "eval_args": run_args,
            # `args.eval_tasks` only records the latest invocation, so the union
            # of what the file now holds is recorded beside it
            "evaluated_tasks": sorted(results.get("results", {})),
        }

        # generate_tables.py reads `args` in preference to parsing the filename,
        # so a run that only evaluated somebody else's checkpoint must not
        # overwrite it: it was invoked without the compression flags, and their
        # defaults would read back as a configuration nobody ran. The update
        # writes a checkpoint of its own, under a run name of its own
        wrote_this_checkpoint = not args.compressed_model_path or args.update_taw_only

        if wrote_this_checkpoint or "args" not in recorded:
            eval_config["args"] = run_args

        save_run_config(
            directory=model_eval_path,
            run_name=model_name,
            config=eval_config,
        )

        vram_usage("After evaluation")
