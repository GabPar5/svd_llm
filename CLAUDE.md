# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for an MSc thesis: a reimplementation of **SVD-LLM** (V1 and V2) for Qwen2-style decoder LLMs (tested on Qwen 2.5 and LLaMA), extended with **heterogeneous compression-ratio allocation** (per-matrix ratios driven by spectral importance scores) — that allocation is the thesis contribution, not upstream SVD-LLM.

There is no package, no test suite, no linter/formatter config, and no dependency manifest. Everything runs as scripts from the repo root. Dependencies are expected to already exist in the active environment: `torch`, `transformers`, `datasets`, `lm_eval` (lm-evaluation-harness), `jax` (GPU, used only for large V2 eigendecompositions), `scipy`, `psutil`, `tqdm`, and optionally `peft` (only for the LoRA sequential update). Type checking is done with pyright (`# pyright: ignore` comments are used inline; there is no config file).

The thesis text itself lives in the sibling working directory `msc-thesis---s345139---parisini` — a separate calkit/DVC + LaTeX repo built with `calkit run` / `calkit latex build -e tex --no-check thesis/main.tex`. Do not mix the two repos.

## Commands

Everything goes through `main.py`; the mode is selected by argument combination, not by subcommand.

```bash
# Compress + evaluate in one run
python main.py --model "Qwen/Qwen2.5-7B" --save_path "./output" \
    --compress_mlp --compress_att_q --compress_att_k --compress_att_v --compress_att_out \
    --compression_ratio 0.2 --batch_size 16 --max_whitening_samples 2048 \
    --evaluate --eval_batch_size auto --eval_tasks "wikitext|0"

# Heterogeneous allocation (thesis path)
python main.py ... --het --group_criterion decoder --score_metric truncation --bypass_early_layers 2

# Compute and cache whitening matrices only (exits with SystemExit(0) before compression)
python main.py ... --whitening_only --whitening_start_layer 0 --whitening_end_layer 16

# Evaluate an existing checkpoint (skips compression entirely)
python main.py --model "Qwen/Qwen2.5-7B" --use_compressed \
    --compressed_model_path "./output/models/<name>.pt" --evaluate --eval_tasks "wikitext|0"

# Run the sequential low-rank update on a TAW-only checkpoint (requires --use_compressed + path)
python main.py ... --update_taw_only --sequential_update_method lora --sequential_lora_backend trainer

# Experiment grid: merges args/base_args.json into each entry of a stage file (args/ is tracked), runs main.py per config.
# EXPERIMENTS.md documents the stages; --dry_run previews, and an unresolved "__PLACEHOLDER__" aborts before the first run
python run_experiments.py args/experiments_stage2_score_grouping.json

# Explore allocations offline from the cached spectra, no GPU, seconds per sweep
python allocation_report.py --model "Qwen/Qwen2.5-7B" --run_v2 \
    --compression_ratio 0.2 --sweep "score_metric=truncation,eff_rank_sq"

# Reports from the eval JSONs
python generate_tables.py ./output/eval/<model_dir> -f latex -o tables.tex
python generate_tables.py ./output/eval/<model_dir> -f markdown -o report.md

# One table per EXPERIMENTS.md stage gate, resolving the placeholders the next stage waits on
python generate_tables.py ./output/eval/<model_dir> --report gates \
    --allocation_dir ./output/allocation_reports -o gates.md

# Text generation for qualitative comparison (one checkpoint, or a folder, sequentially)
python generate_text.py --base_model "Qwen/Qwen2.5-7B" --compressed_folder ./output/models/... --prompt "..."
```

`README.md` carries the full argument reference and the output layout, and is kept in sync with the argparse block — update both together when adding or renaming a flag. `main.py`'s argparse block stays the source of truth if they ever disagree.

## Architecture

### Layer roles

- **`main.py`** — argparse + orchestration only. It picks one of three entry paths (load original from HF / load compressed `.pt` from disk / compress from scratch), derives the run's `model_name` string, installs the `Logger` stdout tee (compression path only), then optionally runs the sequential update and evaluation. No compression math lives here.
- **`src/svd_llm.py`** — the pipeline: `get_whitening_matrices` -> optional score pass -> `allocate_ratios` -> per-matrix SVD + truncation -> optional sequential update -> checkpoint save. `compress_svd_llm` is the single function `main.py` calls.
- **`src/utils.py`** — the shared toolbox: enums (`DtypeMap`, `GroupBy`, `ScoreMetric`), dataset tokenization, budget math, checkpoint I/O, calibration replay, the paper-faithful `ppl_eval`, and the `check_*` numerical diagnostics.
- **`src/modules.py`** — `LowRank`: `W_u(W_v(x))` with input cast to the factor dtype and output cast back, which is what makes mixed `--model_dtype` / `--compressed_dtype` runs work.

Both `src` modules use star imports (`from .utils import *`), so names are effectively one flat namespace — which also means a helper meant to be shared must not start with `_`.

### Shared helpers to reach for

Reuse these instead of re-deriving the logic; each one is the single source of truth for something that used to be duplicated across entry points:

| Concern | Helpers (`src/utils.py` unless noted) |
|---|---|
| Run naming / paths | `build_run_name`, `sanitize_model_name`, `parse_dataset_spec` |
| Checkpoint I/O | `save_compressed_checkpoint`, `load_compressed_model`, `apply_lowrank`, `collect_non_persistent_buffers`, `restore_non_persistent_buffers` |
| Calibration replay | `capture_layer0_inputs`, `decoder_kwargs_to_device`, `decoder_layer_output`, `tree_map_tensors`, `group_keys_by_decoder_layer` |
| Budget & scoring | `compute_active_budget` (returns an `ActiveBudget` breakdown), `redundancy_from_scores`, `compute_spectrum_score`, `normalized_spectrum`, `spectrum_entropy`, `parse_composite_metric`, `compose_scores`, `normalize_block_influence` |
| Ratio allocation | `allocate_ratios` (the shell), `build_allocation_groups`, `build_group_scores`, `resolve_allocation_policies`, `allocation_knobs`, `select_policy_arguments`, `INNER_POLICIES` / `OUTER_POLICIES` and the policies they register |
| Allocation primitives | `waterfill_ratios` (ratio space), `bounded_proportional_split` (rank space), `clamp_group_budget`, `build_shape_map`, `rank_cost`, `matrix_shapes_from_config` |
| Sequential update | `validate_lora_batching`, `resolve_grad_accum_steps`, `build_lora_update_metadata`; `prepare_lora_update_data`, `run_local_u_update`, `lora_factor_targets`, `LORA_UPDATE_PHASES` (in `src/svd_llm.py`) |
| Device plumbing | `pin_memory_enabled`, `synchronize_device`, `cuda_cleanup`, `set_attn_implementation`, `vram_usage`, `ram_usage` |

### Compression pipeline

1. **Target selection.** `generate_paths` builds fully-qualified module paths (`model.layers.N.mlp.gate_proj`, …) from the `--compress_*` flags. These path strings are the primary key for everything downstream: whitening files, `score_map`, `ratio_map`, `rank_map`, state-dict keys.
2. **Whitening.** `get_whitening_matrices` captures decoder-layer-0 inputs once (via a `Catcher` module that raises `CatcherExit`), then replays activations layer by layer, accumulating `XXᵀ` in **fp64** through forward hooks. Per matrix it saves either the Cholesky factor (V1) or the `(U_s, L_s)` eigendecomposition (V2) as an individual `.pt` under `whitening_matrices/<model>/v1|v2/`. Runs can be chunked with `--whitening_start_layer/--whitening_end_layer`; between chunks, layer-input activations are persisted to `activation_checkpoints/` and validated on load against model name, version, `n_tokens`, and layer index.
3. **Ratio allocation.** Homogeneous: one ratio for all active matrices. Heterogeneous (`--het`): an extra `svdvals` pass computes a per-matrix score (`--score_metric`), then `allocate_ratios` (in `src/utils.py`, so the offline tool can reach it without importing the GPU-only pipeline) forms the groups (`--group_criterion` = `type` | `decoder` | `global` | `hierarchical`), hands each group a share of the budget through an **outer** policy (`OUTER_POLICIES`, `--outer_allocation`, default `param_share`) and splits that share across the group's matrices through an **inner** policy (`INNER_POLICIES`, `--inner_allocation`, default `waterfill` = redundancy weights `1 / log(score + offset)`), capped at `--max_ratio` per matrix. A policy declares everything it needs — data and knobs — as named parameters and `select_policy_arguments` hands it only those; `select_policy_knobs` derives from the same signature what the sidecar records. `hierarchical` groups by decoder block like `decoder` and additionally exposes each block's cached Block Influence to the outer policy, which is what makes the two-level allocation possible; with `param_share` it reproduces `decoder` exactly. The allocation preserves *removed parameters* (`Σ paramsᵢ · ratioᵢ = budget`), not the mean ratio, which is why `param_count_map` is threaded everywhere. `--bypass_early_layers` pins the first N decoder layers to `--bypass_ratio` and removes them from redistribution; the remaining budget is pushed onto the active matrices so the global target still holds. `--ratio_scope all` keeps the denominator at all q/k/v/o+MLP params even when only a subset is compressed.
4. **Truncation.** Per matrix: `rank = out*in*(1 - ratio) / (out + in)`, SVD of the whitened weight, split into `W_u`/`W_v`, then the `nn.Linear` is swapped for a `LowRank`. A ratio of exactly `0.0` skips SVD and leaves the layer dense (no `rank_map` entry). Math runs in fp64, factors are cast to `--compressed_dtype`. Each replacement is immediately checked by `check_weights_relative_difference`, `check_lowrank_equivalence`, and `check_layer_activation_error`.
5. **Sequential update** (optional, `--sequential_update`, or `--update_taw_only` on an existing checkpoint). Two implementations: `lora` (paper/upstream-faithful, tunes `W_u` then `W_v` with PEFT and merges after each phase; `trainer` or `custom` backend; fine-tunes on Alpaca via `--finetune_dataset`) and `local_u` (low-VRAM closed-form ridge solve for `W_u` only, keeping `W_v` fixed, layer-by-layer against a dense reference model).

### Memory discipline — preserve it

These models do not fit in VRAM alongside their decompositions, so the code is written around streaming, and edits must keep that property:

- Weights live on CPU; individual decoder layers (and the modules before them) are moved to GPU, used, and moved back.
- Whitening artifacts are written to disk one file per matrix and read back one at a time; `load_whitening_data` **pops** the path by default so an artifact is consumed once (`keep=True` is used for the score pass).
- `SOLVER_GPU_MAX_DIM = 32000` guards a real cuSOLVER 32-bit indexing failure: larger matrices route to scipy on CPU (V1 Cholesky) or to in-process JAX GPU `eigh` (V2, `eigh_jax_gpu_from_cpu`).
- Intermediates are explicitly set to `None` + `del`ed as soon as they are consumed, followed by `cuda_cleanup()`; `vram_usage()` / `ram_usage()` probes bracket every phase. The `[TAG]`-prefixed print instrumentation (`[WHITENING]`, `[BUDGET]`, `[SEQ-UPDATE]`, `[VRAM]`, …) is the primary debugging tool for multi-hour runs — extend it rather than removing it.
- fp32 default with TF32 explicitly disabled (`allow_tf32 = False`, `matmul_precision("highest")`) and fp64 accumulation are deliberate: fp32 `XXᵀ` accumulation produced negative eigenvalues.

### Checkpoint format

`.pt` files are plain `torch.save` dicts: `state_dict`, `rank_map`, `non_persistent_buffers`, `config`, `generation_config`, `svd_llm_metadata`. They are **not** HF-loadable (a code TODO wants this changed). Writing goes through `save_compressed_checkpoint`; reading goes through `load_compressed_model`, which performs the fixed four-step dance — instantiate from config on the `meta` device, `apply_lowrank(model, rank_map, state_dict)` to install `LowRank` modules with the right ranks, `to_empty()` + `load_state_dict(strict=True, assign=True)`, then `restore_non_persistent_buffers` for things absent from the state dict (RoPE `inv_freq`). Pass `audit=True` to also dump the non-state buffers and assert the base/LowRank dtypes (`generate_text.py` does, `main.py` does not).

### Filename convention is an interface

The run name encodes the full configuration: `<model>_q_k_v_out_mlp_<ratio_scope>_<ratio>_<het|hom>[_<grouping>][_<score>][_<bypassed>][_<inner_allocation>][_out<outer_allocation>][_cap<max_ratio>][_<knobs>][_upd_<method>][_v2]`. It determines the checkpoint filename, the eval JSON filename, and the log filename, and `generate_tables.py:parse_filename` parses it back out to build the table rows. `build_run_name` is the only place that constructs it — change it there, and update `parse_filename` to match, or tables silently mis-attribute rows.

Every token past `<bypassed>` is emitted only when its flag leaves its default, which is what keeps names that predate an option byte-identical. The `<knobs>` group (`_seed`, `_bypr`, `_fa`, `_off`, `_temp`, `_ooff`, listed in `KNOB_FILENAME_TOKENS`) additionally requires that the run *reads* the knob — relevance is decided from the policy signatures via `resolve_allocation_policies`, the same test the sidecar uses. A knob that a sweep varies but the name cannot express silently collapses that sweep onto one checkpoint, so any new swept flag needs a token here.

`parse_filename` reads the name all the way back out: positionally up to `<bypassed>` (both the bare integer and `byp<early>-<late>`), then by pattern over the suffix, into the same raw flag values the sidecar records — `GROUPING_FLAG_VALUES`, `INNER_ALLOCATION_TOKENS` / `OUTER_ALLOCATION_TOKENS` and `NAME_TOKEN_DEFAULTS` are the mirrors of `build_run_name`'s tokens, and a token missing from a name means its default. That is what places a run whose sidecar describes a later evaluation rather than its own compression, so a new token needs its mirror here or the runs carrying it drop out of every gate.

### Evaluation

`--eval_tasks "t1,t2|shots"` or `"t1,t2|s1,s2"`. `wikitext` and `c4` bypass lm-eval and go through `ppl_eval`, which reproduces the SVD-LLM paper's methodology exactly (all test docs joined with `\n\n`, non-overlapping fixed-length chunks, `exp(mean NLL)` over all token positions, non-finite batches skipped) — this is why perplexity numbers are comparable to the paper and must not be replaced with lm-eval's `word_perplexity`. Remaining tasks go through `lm_eval.simple_evaluate` with `HFLM` wrapping the in-memory model; both result sets are merged into one JSON under `output/eval/<model>/`. Note `c4` currently re-evaluates wikitext (open TODO in `main.py`).

### Output layout

Everything under `--save_path` (default `./output`, gitignored): `models/`, `eval/`, `logs/`, `whitening_matrices/<model>/<v1|v2>/` (plus its `spectra/` cache and `layer_importance.pt`), `activation_checkpoints/`, `calibration_datasets/`, `sequential_lora_trainer/`, `allocation_reports/<model>/`.

The three regenerable, bulky ones — `whitening_matrices/`, `activation_checkpoints/`, `sequential_lora_trainer/` — follow `--scratch_path` instead when it is set, resolved through the single helper `scratch_root(save_path, scratch_path)`; `allocation_report.py` takes the same flag so it still finds the spectra. `--no_save_checkpoint` skips the `.pt` and the tokenizer beside it while still writing the `<run_name>.config.json` sidecar, so `generate_tables.py` keeps working on a run that stored no checkpoint.

## Gotchas

- A new score metric has to be registered in three places: the `ScoreMetric` enum, the `match` in `compute_spectrum_score`, and `SCORING_TOKENS` in `generate_tables.py` (otherwise tables label it `unknown`). Accepted today: `truncation`, `truncation_sq`, `entropy`, `entropy_sq`, `eff_rank`, `eff_rank_sq`, the four `full_norm_*_tail_*` variants, `norm|<p>` (handled by `ScoreMetric._missing_`, with `p` parsed by `parse_norm_order`), and `composite|<local>|block_influence` (also `_missing_`, validated against `END_TO_END_SCORES`, split from the right so a `norm|p` local half survives). A composite score is fused by `compose_scores` *after* the score pass — `compute_spectrum_score` stays purely spectral — and a new local metric is therefore composite-ready for free, but needs its `composite|...` spellings added to `SCORING_TOKENS` too.
- A per-block score fused into a per-matrix one is constant inside a `decoder`/`hierarchical` group, so `--fusion_alpha 1` there is exactly homogeneous compression. `warn_on_degenerate_scores` prints a `[BUDGET][WARNING]` for that and any other case where scores vary by less than `DEGENERATE_SCORE_SPREAD` inside every group.
- `--whitening_only` terminates with `SystemExit(0)` from inside `compress_svd_llm`.
- Adding a compression-target flag means updating `generate_paths`, `matrix_shapes_from_config` (which has to name the same matrices, for offline analysis), `build_run_name`, `--group_patterns`, and `MATRIX_TOKEN_MAP` in `generate_tables.py`.
- A new allocation policy has to be written *and* registered in `INNER_POLICIES` / `OUTER_POLICIES`: `--inner_allocation`'s choices come from the registry, so an enum member without an entry is rejected with `NotImplementedError` rather than silently offered. It also has to be classified in `RATIO_SPACE_POLICIES` or `RANK_SPACE_POLICIES`, which is what `allocation_report.py` uses to decide whether a constant score must flatten the allocation, and listed in `generate_tables.py`'s `INNER_ALLOCATION_TOKENS` / `OUTER_ALLOCATION_TOKENS` so its filename token parses back.
- A new policy knob is a named parameter on the policy plus an entry in `allocation_knobs` plus a flag on `main.py` and `allocation_report.py`; the shell routes it by signature. A knob a stage *sweeps* additionally needs its filename token in `KNOB_FILENAME_TOKENS` and the mirror of that token in `generate_tables.py`'s `NAME_TOKEN_DEFAULTS`.
- Rank-space policies must go through `bounded_proportional_split`, not a clamp-and-redistribute loop. Pinning an entry to its lower bound early can strand budget that only that entry could have absorbed, which silently over-compresses.
- The code targets a transformers release that accepts `dtype=` on `from_pretrained`/`from_config` (older releases spell it `torch_dtype`), so it will not run on transformers 4.4x.
