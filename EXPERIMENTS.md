# Experiment grid

The staged design behind the results of the thesis. `args/` is gitignored, so this file is the
record of how the grid was configured and is the source for thesis section 4.1.

Every stage is one axis swept around a fixed configuration, with a **gate** at the end that resolves
the placeholders of the next stage. The grid is deliberately not a cross product: the full cross for
a single model at a single ratio is already 4 groupings x 6 scores x 4 inner policies = 96 runs.

Stages 1 to 8 evaluate **perplexity only** (`wikitext,c4|0`), which is what makes a ~100 run grid
affordable. Only stage 9 pays for the full benchmark suite.

Whitening is assumed to be already cached under `output/whitening_matrices/<model>/v2/`, together
with its `spectra/` cache and `layer_importance.pt`. Every stage below reads that cache and never
recomputes it.

## Running a stage

```bash
python run_experiments.py args/experiments_stage1_anchors.json            # run it
python run_experiments.py args/experiments_stage1_anchors.json --dry_run  # preview the commands
python run_experiments.py args/experiments_stage3_policies.json --base args/other_base.json
```

The runner merges `args/base_args.json` into each entry of the stage file, refuses to start if any
entry still holds an unresolved gate value, and continues to the next run when one fails rather than
aborting the queue.

The commands it prints are informational, not copy-paste ready: `--eval_tasks wikitext,c4|0` holds a
pipe that a shell would interpret. `subprocess` passes it as one argument, so runs are unaffected.

## Frozen across every run

Changing any of these invalidates comparability with everything already collected.

| Setting | Value | Why frozen |
|---|---|---|
| Model | `huggyllama/llama-7b` | single model until the Qwen whitening exists |
| Version | `--run_v2` | documented as a limitation in thesis 5.2.1 |
| Precision | fp32 weights, fp32 factors | fp32 accumulation is what keeps `XX^T` positive definite |
| Calibration | wikitext-2 train, `--max_length 2048` | |
| `--max_whitening_samples` | `2048` | truncation and `norm\|p` scores scale with the token count, so this cannot move between runs (thesis 5.2) |
| `--seed` | `6363` | fixes the calibration sample |
| Targets | all seven matrices, `--ratio_scope selected` | with all seven active, `selected` and `all` denominators coincide |
| Screening ratios | `0.2`, `0.5` | |
| Evaluation | `wikitext,c4\|0`, `--eval_max_length 4096` | |

## Placeholders

Stage files carry literal placeholders until their gate resolves them:

| Placeholder | Resolved by | Meaning |
|---|---|---|
| `__BEST_GROUPING__` | stage 2 | grouping criterion with the best mean rank |
| `__BEST_FLAT_GROUPING__` | stage 2 | better of `type` / `global`, never `decoder` |
| `__TOP1_SCORE__`, `__TOP2_SCORE__` | stage 2 (2b, 2c may promote) | the two best score metrics |
| `__BEST_INNER__` | stage 3 | best inner allocation policy |
| `__CHECKPOINT_PATH__` | stages 1 to 8 | a path under `output/models/huggyllama_llama_7b/` |

`run_experiments.py` refuses to start while any remain, so an unfilled gate cannot silently run the
wrong configuration.

## Reading results

```bash
python generate_tables.py ./output/eval/huggyllama_llama_7b -f markdown -o report.md
python generate_tables.py ./output/eval/huggyllama_llama_7b -f latex   -o tables.tex
```

Each run also writes `<run_name>.config.json` beside its checkpoint and its evaluation JSON, which
is authoritative for the dimensions the filename cannot carry, and records the **realized** removal
ratio alongside the target.

---

## Stage 1: anchors

**Purpose.** The floor every later comparison is measured against, and the first half of RQ1.

**Runs.** 3: dense LLaMA-7B, then homogeneous at 0.2 and 0.5.

```bash
python run_experiments.py args/experiments_stage1_anchors.json
```

**What to check.** On the dense baseline, wikitext and c4 perplexity must **differ**. Identical
values are the signature of the old `c4` bug, in which the c4 task re-evaluated wikitext, and would
mean the fix did not take effect in this environment.

**Gate.** None, this stage always runs first. Record the two homogeneous perplexities: they are the
`hom` row of every table in chapter 4.

---

## Stage 2: score x grouping (RQ2)

**Purpose.** Which grouping criterion and which spectral score make heterogeneous allocation work.
`inner_allocation` stays at `waterfill` throughout so the only axes are grouping and score.

**Runs.** 18: `{global, type, decoder} x {truncation, entropy, eff_rank} x {0.2, 0.5}`.

`hierarchical` is absent on purpose: with the default `param_share` outer policy it reproduces
`decoder` to the digit, so it only becomes a distinct configuration in stage 3.

```bash
python run_experiments.py args/experiments_stage2_score_grouping.json
```

**What to look at.** Rank the nine cells by wikitext perplexity at each ratio separately, then by
mean rank across the two. Also compare every cell against the stage 1 homogeneous row: this is the
first real read on RQ1.

> If **every** heterogeneous cell loses to homogeneous, do not conclude that RQ1 is negative yet.
> Run stage 4 first. Swift-SVD's own ablation has uncapped heterogeneous allocation losing to
> uniform, rescued only by a rank floor, and the default `--max_ratio 0.9` sits in exactly that
> regime.

**Gate.**
- `__BEST_GROUPING__` = grouping with the best mean rank across both ratios.
- `__TOP1_SCORE__`, `__TOP2_SCORE__` = the two best scores within that grouping, averaged over ratios.
- `__BEST_FLAT_GROUPING__` = the better of `type` and `global`. Never `decoder`, see stage 6.

---

## Stage 2b: squared spectra

**Purpose.** Whether weighting by energy (`_sq`) rather than amplitude changes the ranking. Feeds
thesis 3.1.1.

**Runs.** 6: `{truncation_sq, entropy_sq, eff_rank_sq} x {0.2, 0.5}` at `__BEST_GROUPING__`.

**Gate.** If a squared variant beats both scores chosen in stage 2, it replaces `__TOP1_SCORE__`
before stage 3 runs.

---

## Stage 2c: Schatten p-norms

**Purpose.** Fills thesis section 3.1.1.2, which currently carries `% TODO - no results`.

**Runs.** 8: `{norm|1, norm|3, norm|inf, norm|-inf} x {0.2, 0.5}` at `__BEST_GROUPING__`.

The four values span genuinely different signals: `norm|1` is the nuclear norm of the truncated
tail, `norm|inf` its largest singular value, `norm|-inf` its smallest. Expect `norm|1` to behave
close to `truncation` and the two extremes to diverge from it.

**Gate.** Reporting only, unless one of them beats `__TOP1_SCORE__`, in which case it is promoted.

---

## Stage 3: allocation policies (RQ3)

**Purpose.** Whether the policy that spends a group budget matters independently of the score that
ranks the matrices, and whether the outer level of the hierarchical allocator earns its place.

**Before running.** Use the offline tool to match the knobs, otherwise this stage confounds policy
*shape* with policy *aggressiveness*:

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --compression_ratio 0.2 --sweep "inner_allocation=waterfill,drank_lagrangian,swift_pool,softmax_temp"
```

Pick `--offset`, `--softmax_temp` and `--outer_offset` so that the four policies produce comparable
ratio dispersion, then set them in `base_args.json`. This costs seconds on CPU. Note that a knob left
at its default emits no filename token, so changing one here changes the run names of this stage
only.

**Runs.** 20:
- 12: `{drank_lagrangian, swift_pool, softmax_temp} x {top1, top2} x {0.2, 0.5}` at `__BEST_GROUPING__`
  (`waterfill` at those settings is already in stage 2)
- 8: `hierarchical` with `--outer_allocation waterfill` x all four inner policies x `{0.2, 0.5}`

**What to look at.** The second block against `decoder` + `param_share` from stage 2 is the
controlled ablation of the **outer level**, since the two criteria bucket matrices identically and
differ only in whether Block Influence gets to move budget between blocks. That comparison is the
thesis contribution's own test, so report it separately from the inner-policy comparison.

The offline tool also produces the mean-ratio-per-matrix-type figure for thesis 3.3.5, which is where
the rank-space bias of `drank_lagrangian` becomes visible against the ratio-space `waterfill`.

**Gate.** `__BEST_INNER__` = best inner policy at `__BEST_GROUPING__`, averaged over ratios.

---

## Stage 4: the per-matrix cap

**Purpose.** `--max_ratio` bounds how far any single matrix may be compressed. Swift-SVD reports that
without such a floor a heterogeneous allocation is *worse* than uniform, so this is a confound
sitting directly on RQ1, not a hyperparameter detail.

**Before running.** Check offline which caps actually bind at each target ratio, and drop any that
change nothing:

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --compression_ratio 0.2 --sweep "max_ratio=0.4,0.6,0.8,0.9"
```

**Runs.** 5: caps `{0.4, 0.6, 0.8}` at target 0.2 and `{0.6, 0.75}` at target 0.5.

**Gate.** If a lower cap wins, the RQ1 verdict is reported at that cap, and thesis 3.3 must state
that the cap is a first-order hyperparameter rather than a guard rail. Carry the winning cap into
stages 5, 6 and 8 by setting it in `base_args.json`.

---

## Stage 5: bypassing outer blocks (RQ4)

**Purpose.** Whether exempting the first or last N decoder blocks beats compressing everything, and
whether that gain **cannibalizes** the heterogeneous gain.

**Runs.** 24: six bypass settings x `{heterogeneous, homogeneous}` x `{0.2, 0.5}`.

Settings: `early 2`, `early 4`, `early 8`, `late 1`, `late 2`, `early 2 + late 1`. `--bypass_ratio`
stays at `0.0`, so a bypassed block is skipped entirely and its budget is pushed onto the rest.

**What to look at.** The homogeneous arm is not padding, it is the whole point. Compute the
heterogeneous gain over homogeneous *at each bypass setting* and compare it to the gain at bypass 0
from stage 2. If the gain shrinks as bypass grows, the two mechanisms are competing for the same
redundancy, which is the second half of RQ4.

---

## Stage 6: composite scores

**Purpose.** Whether fusing a per-matrix spectral score with the per-block Block Influence beats
either alone. This is the scalar counterpart of the hierarchical allocator tested in stage 3.

**Gate to open this stage.** `allocation_report.py` prints the Spearman correlation between Block
Influence and normalized effective rank per matrix family. Swift-SVD reports it as **negative**,
which is what makes the two signals complementary and justifies fusing them.

> If it comes out **positive** on LLaMA-7B, the two signals agree rather than complement, the
> `beta^alpha` convention is pushing both the same way, and these 12 runs are measuring something
> other than what they are meant to. Record the sign and revisit the fusion before spending the GPU
> time.

**Runs.** 12: `composite|{truncation, eff_rank}|block_influence` x `--fusion_alpha {0.25, 0.5, 0.75}`
x `{0.2, 0.5}` at `__BEST_FLAT_GROUPING__`.

**Why never `decoder`.** Block Influence is constant inside a decoder block, so under `decoder` or
`hierarchical` grouping a fused score cannot rank matrices within a group and the allocation
collapses to exactly homogeneous. `allocate_ratios` prints a `[BUDGET][WARNING]` when it detects
this, but the stage file avoids the situation rather than relying on the warning.

**Expect a flatter allocation.** `beta^0.5` spans `[1, 1.41]` while `log(e + local)` is typically 5
to 8, so the fused score has a much narrower dynamic range and lands closer to homogeneous under the
same `--offset`. That is a property of the fusion, not evidence the second signal is useless. Pick
`--offset` for these runs offline first.

---

## Stage 7: cross-model confirmation

**Blocked.** This is the one stage a single model cannot answer. The file names Qwen2.5-7B and
Qwen2.5-32B and stays unrunnable until their whitening artifacts exist.

**Runs.** 12: three finalist configurations x two models x two ratios.

**Gate.** Fill `__FINALIST{1,2,3}_*` from the best configurations found in stages 2 to 6.

---

## Stage 8: the ratio curve (RQ1)

**Purpose.** RQ1 asks how the heterogeneous gain *depends* on the target ratio. Two points give a
slope but no shape, so this adds three more rather than tripling every earlier stage.

**Runs.** 6: `{homogeneous, best heterogeneous}` at `{0.35, 0.65, 0.8}`. Combined with 0.2 and 0.5
already collected, that is a five point curve for both arms.

**What to look at.** Whether the gap widens, narrows or inverts with the ratio. At 0.8 a
heterogeneous allocation is heavily constrained by `--max_ratio`, so read this stage together with
stage 4.

---

## Stage 9: finalists on the full suite

**Purpose.** The headline table of chapter 4.

**Runs.** 8 evaluation-only runs. No recompression: each entry loads an existing checkpoint through
`--use_compressed --compressed_model_path`.

**Tasks.** `arc_easy, hellaswag, openbookqa, piqa, winogrande, gsm8k, truthfulqa_gen`. The last two
are generative and dominate the wall clock.

**Gate.** Fill `__CHECKPOINT_PATH___{1..8}` with real paths from `output/models/huggyllama_llama_7b/`.
Choose: dense, homogeneous at both ratios, and the best heterogeneous configuration from each of
stages 2, 3, 5 and 6.

`mathqa` is worth one attempt but is expected to fail, since `datasets` no longer permits custom
loading scripts. Report the failure rather than working around it.

---

## Stage 10: LoRA sequential update (RQ5)

**Purpose.** Whether heterogeneous allocation is a better *starting point* for fine-tuning, and
whether the gap over homogeneous widens or closes after the update.

**Runs.** 4: `{homogeneous, best heterogeneous} x {0.2, 0.5}`, each via `--update_taw_only` on an
existing checkpoint, so no recompression.

**What to look at.** The het-minus-hom gap before the update (from stage 2 or 3) against the same gap
after it. A gap that closes means the update recovers what a bad allocation lost, and RQ5 is answered
negatively.

---

## Totals

| Stage | Runs | Evaluation |
|---|---|---|
| 1 anchors | 3 | perplexity |
| 2 score x grouping | 18 | perplexity |
| 2b squared | 6 | perplexity |
| 2c Schatten | 8 | perplexity |
| 3 policies | 20 | perplexity |
| 4 cap | 5 | perplexity |
| 5 bypass | 24 | perplexity |
| 6 composite | 12 | perplexity |
| 7 cross-model | 12 | blocked |
| 8 ratio curve | 6 | perplexity |
| 9 benchmarks | 8 | full suite, no recompression |
| 10 LoRA | 4 | perplexity, no recompression |

**102 compression runs** on LLaMA-7B, plus 12 evaluation-only or update-only runs, plus 12 blocked.
