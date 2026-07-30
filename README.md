# svd_llm

Implementation of **SVD-LLM** (V1 and V2) for Qwen-like models (tested primarily on Qwen 2.5 and LLaMA). This tool performs post-training compression of Large Language Models using Singular Value Decomposition with truncation-aware data whitening, and extends it with **heterogeneous compression ratios**, where each weight matrix receives its own ratio based on a spectral importance score.

---

## Requirements

Everything runs as scripts from the repository root; there is no package to install. The environment is expected to provide:

| Dependency | Needed for |
| --- | --- |
| `torch`, `transformers`, `datasets` | everything (transformers must be recent enough to accept `dtype=` on `from_pretrained`/`from_config`) |
| `lm-eval` (lm-evaluation-harness) | `--evaluate` |
| `scipy`, `psutil`, `tqdm` | whitening fallbacks and run instrumentation |
| `jax` (GPU build) | V2 whitening of matrices larger than 32000 (`--run_v2`) |
| `peft` | `--sequential_update_method lora` |

---

## Quick Start

### Compress and Evaluate

Compress a model (e.g. Qwen 2.5 7B), save the results, and run an immediate evaluation:

```bash
python main.py \
    --model "Qwen/Qwen2.5-7B" \
    --save_path "./output" \
    --compress_mlp \
    --compress_att_q \
    --compress_att_k \
    --compress_att_v \
    --compress_att_out \
    --compression_ratio 0.2 \
    --batch_size 16 \
    --max_whitening_samples 2048 \
    --evaluate \
    --eval_batch_size "auto" \
    --eval_tasks "wikitext|0"
```

### Compress with Heterogeneous Ratios

Let every matrix get its own ratio, grouped per decoder layer, keeping the first two layers uncompressed:

```bash
python main.py \
    --model "Qwen/Qwen2.5-7B" \
    --save_path "./output" \
    --compress_mlp --compress_att_q --compress_att_k --compress_att_v --compress_att_out \
    --compression_ratio 0.2 \
    --het \
    --group_criterion decoder \
    --score_metric truncation \
    --bypass_early_layers 2 \
    --evaluate \
    --eval_tasks "wikitext,arc_easy,piqa|0"
```

### Evaluate a Compressed Model

If you have already compressed a model and saved the `.pt` checkpoint, you can bypass the compression stage:

```bash
python main.py \
    --model "Qwen/Qwen2.5-7B" \
    --use_compressed \
    --compressed_model_path "./output/models/Qwen_Qwen2.5_7B/<run_name>.pt" \
    --evaluate \
    --eval_batch_size "auto" \
    --eval_tasks "wikitext|0"
```

### Compute Whitening Matrices Only

Whitening dominates the runtime and its result is reusable, so it can be computed separately (and in layer chunks, resuming from saved activations):

```bash
# First chunk: decoder layers [0, 16)
python main.py --model "Qwen/Qwen2.5-7B" --save_path "./output" \
    --compress_mlp --compress_att_q --compress_att_k --compress_att_v --compress_att_out \
    --whitening_only --whitening_start_layer 0 --whitening_end_layer 16

# Later, compress reusing the cached matrices
python main.py --model "Qwen/Qwen2.5-7B" --save_path "./output" \
    --compress_mlp --compress_att_q --compress_att_k --compress_att_v --compress_att_out \
    --compression_ratio 0.2 --whitening_mat_path "./output/whitening_matrices"
```

### Sequential Low-Rank Update

The update can run right after truncation (`--sequential_update`), or later on a checkpoint that was compressed by truncation-aware whitening only:

```bash
python main.py \
    --model "Qwen/Qwen2.5-7B" \
    --save_path "./output" \
    --use_compressed \
    --compressed_model_path "./output/models/Qwen_Qwen2.5_7B/<run_name>.pt" \
    --update_taw_only \
    --sequential_update_method lora \
    --sequential_lora_backend trainer \
    --evaluate \
    --eval_tasks "wikitext|0"
```

The updated checkpoint is saved next to the original one with a `_sequpd_<method>` suffix, and evaluation results follow that new name.

### Run a Grid of Experiments

`run_experiments.py` runs `main.py` once per configuration, sequentially, continuing after a failed run. It reads two files (both gitignored):

- `args/base_args.json` — arguments shared by every run
- `args/experiments.json` — a list of dictionaries, each overriding the base arguments for one run

`args/base_args.json`:

```json
{
    "--model": "Qwen/Qwen2.5-7B",
    "--save_path": "./output",
    "--compress_mlp": true,
    "--compression_ratio": 0.2,
    "--evaluate": true,
    "--eval_tasks": "wikitext|0"
}
```

`args/experiments.json`:

```json
[
    { "--het": true, "--group_criterion": "decoder", "--score_metric": "truncation" },
    { "--het": true, "--group_criterion": "type", "--score_metric": "eff_rank_sq" }
]
```

Boolean `true` emits the bare flag, `null` drops the argument, anything else is passed as `--key value`.

```bash
python run_experiments.py
```

---

## Arguments Reference

### Core Configuration

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--model` | `str` | `Qwen/Qwen2.5-7B` | HF model identifier. |
| `--run_v2` | `flag` | `False` | Enable SVD-LLM V2 (eigendecomposition-based whitening). |
| `--model_dtype` | `str` | `float32` | Weights dtype for the model. |
| `--compressed_dtype` | `str` | `float32` | Weights dtype for the compressed modules; differing from `--model_dtype` yields a mixed precision model. |
| `--device` | `str` | `cuda` | Computing device. |
| `--attn_implementation` | `str` | `flash_attention_2` | One of `eager`, `sdpa`, `flash_attention_2`, `flash_attention_3`. |
| `--seed` | `int` | `6363` | Seed for calibration sampling and evaluation. |
| `--save_path` | `str` | `None` | Base output path for checkpoints, whitening matrices, logs and results. |
| `--hf_token` | `str` | `None` | Hugging Face token for restricted models. |

### Compression Targets

At least one target must be selected. `--use_compressed` without `--compressed_model_path` still goes through compression.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--compression_ratio` | `float` | `0.2` | Target ratio (e.g. `0.2` removes ~20% of the params). |
| `--ratio_scope` | `str` | `selected` | `selected`: ratio applies to chosen modules; `all`: ratio applies to all targetable matrices (MLP + q/k/v/o), so a chosen subset absorbs a higher average ratio. |
| `--compress_mlp` | `flag` | `False` | Compress MLP weights (`gate_proj`, `up_proj`, `down_proj`). |
| `--compress_att_q` | `flag` | `False` | Compress attention query projection. |
| `--compress_att_k` | `flag` | `False` | Compress attention key projection. |
| `--compress_att_v` | `flag` | `False` | Compress attention value projection. |
| `--compress_att_out` | `flag` | `False` | Compress attention output projection. |

### Heterogeneous Ratio Allocation

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--het` | `flag` | `False` | Enable heterogeneous compression ratio allocation. |
| `--score_metric` | `str` | `truncation` | Weight importance metric: `truncation`, `entropy`, `eff_rank`, each with a squared-spectrum variant (`_sq`); the tail variants `full_norm_tail_entropy`, `full_norm_sq_tail_entropy`, `full_norm_tail_eff_rank`, `full_norm_sq_tail_eff_rank`; or `norm\|p` for the p-Schatten norm of the truncated tail, with `p` a number, `inf` or `-inf`. |
| `--group_criterion` | `str` | `type` | Grouping used for redistribution: `type`, `global` or `decoder`. |
| `--group_patterns` | `str` | see `--help` | Group definitions for `--group_criterion type`, as `groupName:weightType1,weightType2;...`. |
| `--offset` | `float` | `1.5` | Offset added to scores so that `log(score + offset)` stays defined. |
| `--bypass_early_layers` | `int` | `-1` | Number of initial decoder layers exempted from redistribution (`-1` disables the exemption). |
| `--bypass_late_layers` | `int` | `-1` | Number of final decoder layers exempted from redistribution (`-1` disables the exemption). Can be combined with `--bypass_early_layers` to protect both ends in the same run. |
| `--bypass_ratio` | `float` | `0.0` | Ratio assigned to the bypassed layers at either end; `0.0` leaves them uncompressed. |
| `--max_ratio` | `float` | `0.9` | Upper bound on the ratio any single matrix may receive, shared by every allocation policy. |

The removal budget is preserved in parameters, not in average ratio: bypassed layers are charged at `--bypass_ratio` and the remaining budget is redistributed over the active matrices, capped at `--max_ratio` per matrix.

### Whitening & Calibration

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--calibration_dataset` | `str` | `EleutherAI/wikitext_document_level:wikitext-2-raw-v1:train` | Format: `dataset[:subset[:split]]`; a local path saved with `save_to_disk` also works. |
| `--max_length` | `int` | `2048` | Max context length during compression. |
| `--max_whitening_samples` | `int` | `256` | Number of samples for whitening matrix calculation. |
| `--batch_size` | `int` | `2` | Batch size for calibration forward pass. |
| `--whitening_mat_path` | `str` | `None` | Reuse whitening matrices from this directory; missing ones are generated in place. |
| `--whitening_only` | `flag` | `False` | Only calculate and save whitening matrices, then exit. |
| `--whitening_start_layer` | `int` | `0` | First decoder layer of the chunk (inclusive). |
| `--whitening_end_layer` | `int` | `None` | Last decoder layer of the chunk (exclusive); activations are checkpointed so the next chunk resumes. |
| `--pin_cpu_offload` | `flag` | `False` | Pin CPU-offloaded activations and use non-blocking transfers. Useful on large-RAM systems, but pins many GB of host memory. |

Two derived caches are written beside the whitening matrices they come from:

- **`layer_importance.pt`** — Block Influence per decoder block, `1 - E[cos(x_in, x_out)]`. It is accumulated inside the whitening replay, which already holds each block's input and output, so it costs no extra forward pass and is collected on every run regardless of the score metric. Chunked runs merge by layer index because raw sums are stored rather than means.
- **`spectra/`** — the raw singular values of each whitened matrix, cached before the `sqrt(n_tokens)` rescale. Every score metric is derivable from the spectrum, so one cache serves all of them and a repeated score pass skips the decomposition entirely (watch the `[SPECTRA]` line for the hit rate).

Both are validated on load against model name, whitening version and `n_tokens`, and are ignored rather than trusted when they were built for a different configuration.

### Sequential Low-Rank Update

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--sequential_update` | `flag` | `False` | Run the update right after truncation. Alias: `--finetune_on_the_fly`. |
| `--update_taw_only` | `flag` | `False` | Run the update on an existing truncation-aware-whitening-only checkpoint; requires `--use_compressed` and `--compressed_model_path`. Alias: `--finetune_compressed`. |
| `--sequential_update_method` | `str` | `lora` | `lora`: paper/upstream-style LoRA update of `W_u` then `W_v`. `local_u`: low-VRAM closed-form update of `W_u` only. |
| `--sequential_lora_backend` | `str` | `trainer` | `trainer`: HF Trainer loop. `custom`: lightweight local loop. |
| `--sequential_update_ridge` | `float` | `1e-6` | Ridge regularization, `local_u` only. |
| `--sequential_lora_r` | `int` | `8` | LoRA rank. |
| `--sequential_lora_alpha` | `int` | `16` | LoRA alpha. |
| `--sequential_lora_dropout` | `float` | `0.05` | LoRA dropout. |
| `--sequential_lora_lr` | `float` | `1e-4` | Learning rate. |
| `--sequential_lora_weight_decay` | `float` | `0.0` | Weight decay. |
| `--sequential_lora_epochs` | `int` | `2` | Epochs per LoRA phase. |
| `--sequential_lora_max_steps` | `int` | `None` | Max optimizer steps per phase; overrides epochs when set. |
| `--sequential_lora_micro_batch_size` | `int` | `4` | Per-device microbatch size. |
| `--sequential_lora_effective_batch_size` | `int` | `64` | Target effective batch size; must be divisible by the microbatch size with the `trainer` backend. |
| `--sequential_lora_grad_accum_steps` | `int` | `None` | Accumulation steps; derived from the two batch sizes when omitted. |
| `--sequential_lora_val_set_size` | `int` | `2000` | Validation split size. |
| `--sequential_lora_gradient_checkpointing` | `flag` | `False` | Enable gradient checkpointing during the update. |
| `--finetune_dataset` | `str` | `yahma/alpaca-cleaned` | Dataset for the LoRA update, as `dataset[:subset[:split]]`. |
| `--max_finetune_samples` | `int` | `50000` | Max training samples; validation samples are added on top. |
| `--finetune_cutoff_len` | `int` | `256` | Prompt cutoff length. |
| `--finetune_train_on_inputs` | `flag` | `False` | Include instruction/input tokens in the loss. |
| `--finetune_add_eos_token` | `flag` | `False` | Match Alpaca-LoRA's optional EOS handling for the prompt mask. |

### Loading & Evaluation

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--use_compressed` | `flag` | `False` | Skip loading the dense model from the hub. |
| `--compressed_model_path` | `str` | `None` | Load this compressed `.pt` checkpoint instead of compressing. |
| `--evaluate` | `flag` | `False` | Run evaluation tasks after loading/compressing. |
| `--eval_tasks` | `str` | `wikitext\|0` | `task1,task2\|shots` or `task1,task2\|shots1,shots2`. |
| `--eval_batch_size` | `str` | `auto` | Batch size for evaluation. |
| `--eval_max_length` | `int` | `4096` | Max context length during evaluation. |
| `--max_eval_tokens` | `int` | `256` | Max tokens to generate during evaluation. |
| `--eval_sampling` | `flag` | `False` | Use conditional sampling during generation tasks. |

`wikitext` and `c4` are measured with the repository's own perplexity routine, which reproduces the SVD-LLM paper methodology (documents concatenated into one stream, non-overlapping chunks, `exp(mean NLL)` over all tokens). All other tasks go through lm-evaluation-harness, and both result sets are merged into a single JSON.

`wikitext` reads `wikitext-2-raw-v1:test` (4358 documents). `c4` reads a single validation shard of `allenai/c4` (`en/c4-validation.00000-of-00008.json.gz`, 45576 documents), which is what upstream SVD-LLM loads; the full `en` validation split is ~364k documents and cannot be concatenated whole. Because the shard is ~10x wikitext, a `c4` evaluation costs noticeably more wall-clock than a `wikitext` one.

---

## Output Layout

Everything lands under `--save_path`:

```
output/
├── models/<model>/                 # compressed .pt checkpoints + tokenizer
│   └── <run_name>.config.json      # run configuration + realized allocation
├── eval/<model>/<run_name>.json    # merged evaluation results
│   └── <run_name>.config.json      # same sidecar, read by generate_tables.py
├── logs/<model>/<run_name>.log     # full stdout of the compression run
├── whitening_matrices/<model>/<v1|v2>/
│   ├── layer_importance.pt          # Block Influence per decoder block
│   └── spectra/                     # cached raw singular values, one per matrix
├── activation_checkpoints/<model>/<v1|v2>/
├── calibration_datasets/           # tokenized calibration data cache
└── sequential_lora_trainer/        # HF Trainer checkpoints of the LoRA update
```

### Run Names

Checkpoint, log and result filenames encode the whole configuration:

```
<model>[_q][_k][_v][_out][_mlp]_<ratio_scope>_<ratio>_<het|hom>[_<grouping>][_<score>][_<bypassed>][_cap<max_ratio>][_upd_<method>][_v2]
```

For example `Qwen_Qwen2.5_7B_q_k_v_out_mlp_all_0.2_het_decoder_truncation_2_v2`. `generate_tables.py` parses this convention back into table columns, so keep the two in sync when changing it.

The `<bypassed>` token is a bare integer when only `--bypass_early_layers` is used, and becomes `byp<early>-<late>` once `--bypass_late_layers` is set. `_cap<max_ratio>` appears only when `--max_ratio` leaves its `0.9` default. Both rules exist so that run names predating these options stay byte-identical.

### Run Configuration Sidecar

The filename is parsed positionally and cannot carry every dimension of a run, so each run also writes `<run_name>.config.json` next to its checkpoint and next to its evaluation JSON:

- `args` — the resolved command line (`--hf_token` is never persisted).
- `allocation` — target vs **realized** removal, per-matrix `ratio_map`, and the bypassed/active matrix counts. Written by the compression step, so it exists even for runs that never evaluate.
- `checkpoint_metadata` — the same metadata embedded in the `.pt`.

`generate_tables.py` prefers this sidecar over `parse_filename` and falls back to filename parsing for runs that predate it, so old results keep tabulating unchanged. Sidecars are skipped when the input directory is globbed for results.

A run whose realized ratio drifts from its target by more than 0.1% prints a `[BUDGET][WARNING]`; allocation policies are only comparable at equal realized compression, so use `allocation.realized_overall_ratio` rather than the requested `--compression_ratio` when comparing.

---

## Helper Scripts

### `generate_tables.py`

Turns a directory of evaluation JSONs into markdown or LaTeX tables, grouped by bypassed layers and compression ratio, with the best value per ratio highlighted and the uncompressed baseline shown as a faded row.

```bash
python generate_tables.py ./output/eval/Qwen_Qwen2.5_7B -f latex -o tables.tex
```

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `input_dir` | `Path` | — | Directory containing lm-eval JSON files. |
| `-p`, `--pattern` | `str` | `*.json` | Glob pattern for result files. |
| `-f`, `--format` | `str` | `markdown` | `markdown` or `latex`. |
| `-o`, `--output` | `Path` | `lm_eval_report.md` | Output file. |
| `-w`, `--table_width` | `float` | `1.6` | Table width passed to `adjustbox` (LaTeX only). |
| `--prefer-lm-eval-model-name` | `flag` | `False` | Take the model name from the JSON config instead of the filename. |

### `generate_text.py`

Generates text with one checkpoint, a whole folder of checkpoints (sequentially), and optionally the original model, writing one markdown file per model with both greedy and sampled output.

```bash
python generate_text.py \
    --base_model "Qwen/Qwen2.5-7B" \
    --compressed_folder ./output/models/Qwen_Qwen2.5_7B \
    --include_original \
    --prompt "The responsibility of an AI assistant is"
```

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--base_model` | `str` | required | HF model name/path providing the architecture and config. |
| `--compressed_checkpoint` | `str` | `None` | A single compressed `.pt` checkpoint. |
| `--compressed_folder` | `str` | `None` | Folder of `.pt` checkpoints to run one after another. |
| `--recursive` | `flag` | `False` | Search `--compressed_folder` recursively. |
| `--include_original` | `flag` | `False` | Also run the dense model from the hub. |
| `--tokenizer_path` | `str` | `None` | Defaults to the checkpoint directory, then to the base model. |
| `--prompt` / `--prompt_file` | `str` | `None` | Prompt text, or a file containing it. |
| `--use_chat_template` | `flag` | `False` | Wrap the prompt with the tokenizer chat template. |
| `--use_alpaca_prompt` | `flag` | `False` | Wrap the prompt with the Alpaca template, matching the default LoRA finetuning data. |
| `--system_prompt` | `str` | `None` | System message, chat template only. |
| `--dtype` | `str` | `bfloat16` | Weights dtype. |
| `--device` | `str` | `cuda` | Computing device. |
| `--output_dir` | `str` | `./output/text_generation/` | Where the markdown files are written. |
| `--max_new_tokens` | `int` | `256` | Generation length. |
| `--temperature` / `--top_p` / `--top_k` | `float` / `float` / `int` | `0.7` / `0.95` / `50` | Sampling parameters. |
| `--repetition_penalty` | `float` | `1.0` | Repetition penalty for both decoding modes. |

---

> [!TIP]
> Compression runs are long and print heavily tagged progress (`[WHITENING]`, `[BUDGET]`, `[SEQ-UPDATE]`, `[VRAM]`, `[RAM]`). The whole stdout of a compression run is also written to `output/logs/<model>/<run_name>.log`.

> [!NOTE]
> Compressed checkpoints are plain `torch.save` payloads (state dict, rank map, non-persistent buffers, configs and run metadata), not Hugging Face models, so they must be loaded through this repository.
