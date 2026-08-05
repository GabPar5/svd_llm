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

### Compress with the Hierarchical Allocator

Split the budget across decoder blocks by their end-to-end Block Influence, then across each block's matrices by their local spectral scores:

```bash
python main.py \
    --model "Qwen/Qwen2.5-7B" \
    --save_path "./output" \
    --compress_mlp --compress_att_q --compress_att_k --compress_att_v --compress_att_out \
    --compression_ratio 0.2 \
    --het \
    --group_criterion hierarchical \
    --outer_allocation waterfill \
    --inner_allocation waterfill \
    --score_metric truncation \
    --evaluate \
    --eval_tasks "wikitext|0"
```

The Block Influence comes from the whitening cache, so this needs a whitening run to have happened first. Explore the allocation offline with `allocation_report.py` before spending GPU time on it.

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

- `args/base_args.json` — arguments shared by every run, overridable with `--base`
- a stage file — a list of dictionaries, each overriding the base arguments for one run, passed as the positional argument and defaulting to `args/experiments.json`

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
python run_experiments.py                                          # args/experiments.json
python run_experiments.py args/experiments_stage2_score_grouping.json
python run_experiments.py args/experiments_stage3_policies.json --dry_run
```

A stage file may carry placeholders such as `__BEST_GROUPING__` for values that only the preceding stage's results can supply. Any string argument containing `__` aborts the whole stage before the first run, so an unfilled placeholder cannot quietly compress the wrong configuration. `EXPERIMENTS.md` describes the staged grid itself: what each stage answers, what to inspect, and which placeholder its results resolve.

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
| `--score_metric` | `str` | `truncation` | Weight importance metric: `truncation`, `entropy`, `eff_rank`, each with a squared-spectrum variant (`_sq`); the tail variants `full_norm_tail_entropy`, `full_norm_sq_tail_entropy`, `full_norm_tail_eff_rank`, `full_norm_sq_tail_eff_rank`; or `norm\|p` for the p-Schatten norm of the truncated tail, with `p` a number, `inf` or `-inf`. Any of them can be wrapped as `composite\|<local>\|block_influence`. |
| `--fusion_alpha` | `float` | `0.5` | Weight of the end-to-end half of a composite metric, in `[0,1]`. Ignored by non-composite metrics. |
| `--group_criterion` | `str` | `type` | Grouping used for redistribution: `type`, `global`, `decoder` or `hierarchical`. |
| `--inner_allocation` | `str` | `waterfill` | Policy that splits a group's budget across the matrices inside it: `waterfill`, `drank_lagrangian`, `swift_pool` or `softmax_temp`. |
| `--outer_allocation` | `str` | `param_share` | Policy that splits the budget across groups: `param_share` or `waterfill`. The latter requires `--group_criterion hierarchical`. |
| `--group_patterns` | `str` | see `--help` | Group definitions for `--group_criterion type`, as `groupName:weightType1,weightType2;...`. |
| `--offset` | `float` | `1.5` | Offset added to scores so that `log(score + offset)` stays defined. Read by the `waterfill` inner policy. |
| `--outer_offset` | `float` | `1.5` | The same offset applied to Block Influence by the `waterfill` outer policy. Kept separate so tuning how matrices compete inside a block cannot change how blocks compete. |
| `--softmax_temp` | `float` | `1.0` | Temperature of `softmax_temp`. Scores are min-max normalized to `[0,1]` per group first, so the largest allocation weight exceeds the smallest by `exp(1 / softmax_temp)`. |
| `--bypass_early_layers` | `int` | `-1` | Number of initial decoder layers exempted from redistribution (`-1` disables the exemption). |
| `--bypass_late_layers` | `int` | `-1` | Number of final decoder layers exempted from redistribution (`-1` disables the exemption). Can be combined with `--bypass_early_layers` to protect both ends in the same run. |
| `--bypass_ratio` | `float` | `0.0` | Ratio assigned to the bypassed layers at either end; `0.0` leaves them uncompressed. |
| `--max_ratio` | `float` | `0.9` | Upper bound on the ratio any single matrix may receive, shared by every allocation policy. |

The removal budget is preserved in parameters, not in average ratio: bypassed layers are charged at `--bypass_ratio` and the remaining budget is redistributed over the active matrices, capped at `--max_ratio` per matrix.

Allocation is split into a shell and two pluggable policies. `allocate_ratios` owns grouping, the budget split and every `[BUDGET]` line; an **outer** policy then divides the budget across groups and an **inner** policy divides each group's share across its matrices. A policy declares what it needs — data and knobs alike — as ordinary named parameters, and the shell hands each one only what it declares, which is also how the run configuration sidecar records the knobs that actually shaped an allocation. Adding a policy therefore means writing the function and registering it in `INNER_POLICIES` / `OUTER_POLICIES`; the CLI advertises the registry, so no unimplemented choice is ever offered.

#### Inner policies

| Policy | Origin | Rule | Family |
| --- | --- | --- | --- |
| `waterfill` | SVD-LLM V2 Alg. 1, parameter-weighted | `ratio ∝ 1 / log(score + offset)` | ratio space |
| `softmax_temp` | MoDeGPT Eq. 10-11 | `ratio ∝ softmax(−score / temp)`, the entropy-regularized optimum | ratio space |
| `swift_pool` | Swift-SVD Alg. 2 | `ratio = max_ratio − pool · score / Σ(score · params)`, linear buy-back from the cap | ratio space |
| `drank_lagrangian` | D-Rank Eq. 3-7 | `rank ∝ sqrt(score / ω)` with `ω = out + in`, the Lagrangian optimum of `Σ score/rank` | rank space |

Every policy preserves the removal budget in parameters, `Σ paramsᵢ · ratioᵢ = budget`. Where a source preserves the *mean ratio* instead (SVD-LLM V2 and MoDeGPT both do), the parameter-weighted form here reduces to the original whenever the matrices in a group are the same size.

The two families differ in what a score that carries no information implies. A ratio-space policy leaves every matrix at the flat ratio; `drank_lagrangian` does not, because one rank of a small matrix buys the same accuracy for fewer parameters, so it keeps favouring cheap-per-rank matrices regardless. On a group of identical shapes — which is what `--group_criterion type` produces, and what D-Rank itself assumes — the two families coincide. On a mixed group the bias is strong: at a 0.2 target with `--group_criterion decoder`, `drank_lagrangian` drives the attention projections to nearly dense and loads the budget onto the MLP.

`swift_pool` drops Swift-SVD's `δ` rank floor in favour of the shared `--max_ratio`, which is a far lower floor than the paper's `δ = 0.5` (an effective cap of 0.6 at a 0.2 target, against 0.9), so it runs more aggressively than the original. Its `s = β^α · log(e + ε)^(1−α)` is a *score*, not part of the policy; reproducing Swift-SVD means pairing it with a composite score metric rather than a raw truncation score. The paper's 11-candidate grid search over `α` is not implemented — at about an hour per compression and evaluation it is not affordable.

#### Composite scores

`--score_metric composite|<local>|block_influence` fuses a per-matrix spectral score with the per-block Block Influence, following Swift-SVD Eq. 12:

```
s = β^α · log(e + local)^(1 − α)
```

`β` is min-max normalized across decoder blocks and shifted into `[1,2]`, and the local score passes through `log(e + ·)` so that both factors stay at or above 1 whatever metric produced it — a block can only ever raise a matrix's score, never zero it out. `--fusion_alpha 0` leaves the local score alone in log form, which is Swift-SVD's local-only candidate; `1` leaves the block importance alone. The composite fuses *after* the score pass, so `compute_spectrum_score` stays purely spectral and the pass itself is unchanged. The local half may carry its own separator (`composite|norm|3|block_influence`) — the grammar splits from the right.

This is the scalar counterpart of `--group_criterion hierarchical`, and comparing the two is the point: the hierarchical allocator keeps the two signals separate at two granularities, while the composite collapses them into one number for a flat allocator.

Two effects to expect rather than be surprised by:

- **The fused score is much flatter than the raw one.** `β^0.5` spans `[1, 1.41]` while `log(e + local)` is typically 5 to 8 for a truncation score. On a small fixture the per-matrix ratio spread fell from a standard deviation of `0.111` to `0.021`. Under the same `--offset` a composite allocation therefore sits closer to homogeneous, which is a property of the fusion and not evidence that the second signal adds nothing. Pick an `--offset` for composite runs with `allocation_report.py` before spending GPU time.
- **A per-block score cannot differentiate inside a block.** With `--group_criterion decoder` or `hierarchical`, `--fusion_alpha 1` makes every matrix in a block share a score and the allocation becomes exactly homogeneous. That is precisely the argument for the hierarchical allocator: an end-to-end signal can only do work at the outer level. `allocate_ratios` prints a `[BUDGET][WARNING]` whenever scores vary by less than 0.1% inside every group, so this cannot go unnoticed mid-run.

#### The hierarchical outer level

`--group_criterion hierarchical` groups by decoder block, exactly like `decoder`, and additionally exposes a per-block score to the outer policy: the **Block Influence** cached during whitening. With `--outer_allocation waterfill`, blocks that transform the residual stream more are asked for less removal and the budget they give up flows to the rest; each block's share is then divided among its matrices by the inner policy on local spectral scores. Two signals, two granularities.

With the default `--outer_allocation param_share` the outer level expresses no preference and `hierarchical` reproduces `decoder` to the digit, which is the controlled baseline the outer level is measured against.

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
├── allocation_reports/<model>/      # CSV written by allocation_report.py
├── calibration_datasets/           # tokenized calibration data cache
└── sequential_lora_trainer/        # HF Trainer checkpoints of the LoRA update
```

### Run Names

Checkpoint, log and result filenames encode the whole configuration:

```
<model>[_q][_k][_v][_out][_mlp]_<ratio_scope>_<ratio>_<het|hom>[_<grouping>][_<score>][_<bypassed>][_<inner_allocation>][_out<outer_allocation>][_cap<max_ratio>][_<knobs>][_upd_<method>][_v2]
```

For example `Qwen_Qwen2.5_7B_q_k_v_out_mlp_all_0.2_het_decoder_truncation_2_v2`. `generate_tables.py` parses this convention back into table columns, so keep the two in sync when changing it.

The `<bypassed>` token is a bare integer when only `--bypass_early_layers` is used, and becomes `byp<early>-<late>` once `--bypass_late_layers` is set. `_<inner_allocation>` and `_out<outer_allocation>` appear only for a heterogeneous run that leaves the `waterfill` / `param_share` defaults, and both sit after `<bypassed>` so that token stays where `parse_filename` looks for it. `_cap<max_ratio>` appears only when `--max_ratio` leaves its `0.9` default. Every one of these rules exists so that run names predating the option stay byte-identical.

`<knobs>` is the remaining set of swept tunables, each emitted only when it leaves its default **and** the run actually reads it:

| Token | Flag | Emitted when |
|---|---|---|
| `_seed<n>` | `--seed` | always, since the calibration sample changes every downstream artifact |
| `_bypr<r>` | `--bypass_ratio` | at least one layer is bypassed |
| `_fa<a>` | `--fusion_alpha` | the score metric is a `composite\|...` |
| `_off<v>` | `--offset` | the resolved inner policy declares it |
| `_temp<v>` | `--softmax_temp` | the resolved inner policy declares it |
| `_ooff<v>` | `--outer_offset` | the resolved outer policy declares it |

The last three read the same policy signatures the sidecar reads, so `--offset` handed to a policy that ignores it does not fork the name into two entries for what is one run. Without these tokens a sweep over any single knob would leave every other token untouched and collapse the whole sweep onto one checkpoint.

### Run Configuration Sidecar

The filename is parsed positionally and cannot carry every dimension of a run, so each run also writes `<run_name>.config.json` next to its checkpoint and next to its evaluation JSON:

- `args` — the resolved command line (`--hf_token` is never persisted).
- `allocation` — target vs **realized** removal, per-matrix `ratio_map`, the bypassed/active matrix counts, and the two policies together with the knobs that actually applied to them. Written by the compression step, so it exists even for runs that never evaluate.
- `checkpoint_metadata` — the same metadata embedded in the `.pt`.

`generate_tables.py` prefers this sidecar over `parse_filename` and falls back to filename parsing for runs that predate it, so old results keep tabulating unchanged. Sidecars are skipped when the input directory is globbed for results.

A run whose realized ratio drifts from its target by more than 0.1% prints a `[BUDGET][WARNING]`; allocation policies are only comparable at equal realized compression, so use `allocation.realized_overall_ratio` rather than the requested `--compression_ratio` when comparing.

---

## Helper Scripts

### `allocation_report.py`

Explores compression-ratio allocations offline, from the spectra and Block Influence cached beside the whitening matrices. It replays the real `allocate_ratios` for any allocator × score × knob combination without a GPU and without loading model weights — matrix shapes come from the model config, every score is re-derived from its cached spectrum — so a whole sweep takes seconds against roughly an hour for one compression + evaluation run.

```bash
python allocation_report.py --model "Qwen/Qwen2.5-7B" --run_v2 \
    --compression_ratio 0.2 --group_criterion decoder \
    --sweep "score_metric=truncation,eff_rank_sq,entropy" \
    --sweep "offset=1.5,3.0"
```

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--model` | `str` | — | Model the spectra were cached for. Only its config is read. |
| `--save_path` | `str` | `./output` | Root holding `whitening_matrices/`. |
| `--whitening_mat_path` | `str` | `None` | Whitening directory to read, overriding the one derived from `--save_path` and `--model`. |
| `--run_v2` | `flag` | `False` | Read the V2 artifacts instead of V1. |
| `--compress_*` | `flag` | all | Restrict the report to some matrix families. Unlike `main.py`, giving none covers them all. |
| `--sweep KEY=V1,V2` | `str` | `[]` | Sweep one knob over several values, repeatable, taken as a cartesian product. |
| `--out_dir` | `str` | under `--save_path` | Where the CSV report is written. |
| `--plots` | `flag` | `False` | Also render a PNG of every figure, when matplotlib happens to be installed. The figure CSVs are written either way. |

Every allocation flag of `main.py` is accepted as the base configuration, and `--compression_ratio`, `--group_criterion`, `--inner_allocation`, `--outer_allocation`, `--score_metric`, `--offset`, `--outer_offset`, `--softmax_temp`, `--fusion_alpha`, `--max_ratio`, `--bypass_early_layers`, `--bypass_late_layers` and `--bypass_ratio` can be swept.

Output is CSV rather than figures, so `pgfplots` consumes it directly in the thesis: `summary.csv` (one row per variant), `matrices.csv` (per matrix: score, ratio, rank, truncation loss), `layers.csv` (per decoder layer, with its Block Influence), `budget/<variant>.log` (the captured `[BUDGET]` instrumentation) and `figures/*.csv` (one tidy table per figure).

#### Objectives

Variants are **not** ranked by a single number. An allocation is priced under six objectives, all oriented so that lower is better, and ordered by its mean rank across them:

| Objective | Measures | Optimized by construction by |
|---|---|---|
| `frobenius_tail` | squared energy discarded, summed | `truncation`, `truncation_sq` |
| `nuclear_tail` | discarded magnitude rather than energy | `norm\|1` |
| `spectral_tail` | largest single discarded direction, a minimax | `norm\|inf` |
| `relative_energy_lost` | mean per-matrix fraction of energy dropped, scale free | — |
| `eff_rank_lost` | mean per-matrix fraction of effective rank lost | `eff_rank`, `eff_rank_sq` |
| `influence_tail` | relative energy dropped, weighted by each block's Block Influence | `composite\|...\|block_influence` |

The right-hand column is the reason for the panel. `frobenius_tail` **is** the `truncation_sq` score summed over matrices, so ranking on it alone hands the truncation scores a win by definition. Reading a row across all six separates a variant that is broadly good from one that only wins the objective its own score metric optimizes, and that distinction is a reportable result rather than a hidden confound.

`influence_tail` is the only objective that is not a function of the spectra alone: Block Influence is measured on the dense model's residual stream, so no local score can reproduce it. It is the one end-to-end reading available without a GPU.

Each objective that is a plain sum over singular directions also carries `<objective>_oracle_ratio`, its value divided by the lowest reachable at the same budget, cap and active matrices (greedy in ascending loss per parameter removed, which is exact). The bound ignores the grouping, so it may spend the whole budget on a few matrices and the ratio runs to six figures — it says nothing in absolute terms. Its use is comparing **across budgets**: a `--sweep compression_ratio=0.2,0.5` moves the raw objective by orders of magnitude, and dividing by the bound removes the budget's own contribution. Within one budget the bound is constant and never reorders anything.

#### Figures

Each writes `figures/<name>.csv` always, and `figures/<name>.png` when `--plots` is given and matplotlib is installed:

`scores_by_depth`, `influence_by_depth`, `influence_vs_effrank` (+ its `_rho` companion), `spectra`, `layer_ratios`, `ratio_heatmap`, `ratio_by_type`, `cap_binding`, `objectives`, `oracle_gap`, `dispersion`.

Each variant is also checked against the invariants any policy must satisfy: realized removal matches the budget, ratios stay within `[0, --max_ratio]`, and the *value* of a constant score never changes the allocation. Two further checks apply only to ratio-space policies, which read nothing but the score: no group may give more removal to a higher-scoring matrix, and a constant score must collapse the allocation onto the flat ratio (the latter also requires a neutral outer level, since Block Influence is not flattened by it). A rank-space policy is exempt from both — it also prices a rank at `out + in`, and on a group of mixed shapes that can outweigh the score ordering, which is the family's bias rather than a defect.

For every policy the mean Spearman correlation between score and assigned ratio is reported as a `score_ratio_rho` column instead: `−1` means the allocation is perfectly monotone in the score, and values closer to zero measure how much the shape weighting or the per-matrix cap pulls it away.

Violations are reported per variant and the script exits non-zero; a budget drift usually means the configuration is infeasible (`--max_ratio` too low to reach the target once bypassed layers are charged), which is exactly what it is there to surface.

The script also reports the Spearman correlation between Block Influence and normalized effective rank per matrix family. Swift-SVD reports these as negatively correlated, which is what justifies fusing them into one composite score; confirming the sign on the model at hand is a precondition for using that fusion.

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
