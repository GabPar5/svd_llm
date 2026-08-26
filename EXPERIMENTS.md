# Experiment grid

The staged design behind the results of the thesis, and the source for thesis section 4.1.

Every stage is one axis swept around a fixed configuration, with a **gate** at the end that resolves
the placeholders of the next stage. The grid is deliberately not a cross product: the full cross for
a single model at a single ratio is already 4 groupings x 6 scores x 4 inner policies = 96 runs.

Stages 1 to 8 evaluate **wikitext perplexity only**. At roughly 15 minutes per run including that
evaluation, the 204 compression runs below cost about 51 GPU hours, so the binding constraint on this
grid is not time but whether each run answers a distinct question.

Whitening is assumed to be already cached under `output/whitening_matrices/<model>/v2/`, together
with its `spectra/` cache and `layer_importance.pt`. Every stage below reads that cache and never
recomputes it — the one exception is stage 1b, which exists precisely to build a second one.

**Execution order**, which is also the section order:
`0, 1, 1b, 2, 2b, 2c, 3, 3c, 4, 4c, 4d, 5, 5b, 6, 6b, 7, 8, 9, 10`.

## The pilot, and why every number below is being re-measured

A 72-run pilot has been collected and then set aside. It is what designed this grid: it found the
screen, the two failure modes, the interaction that stage 4 is built around, and three defects in the
measurement itself. Its numbers appear throughout the findings below because they are the evidence
for the design.

**They are not results.** Two things disqualify them:

- **Two environments.** The sidecars split the corpus exactly by stage: 31 runs on the local machine
  (stages 1 to 2c) and 41 on Colab (stage 3 onward), with whitening caches that are not the same
  input. Every comparison that crosses the boundary is confounded, and the boundary falls between the
  outer level and its own baseline.
- **A quantised metric.** `ppl_eval` accumulated the per-token losses in fp16, so every pilot
  perplexity is an exact fp16 value: the reporting grid is 0.0039 at perplexity 7.8 against a field
  of configurations spanning 0.24. That is fixed now (the mean is taken in fp64), which makes new
  numbers finer-grained than old ones rather than comparable to them.

So the grid is run from scratch, on one machine, with the fixed metric. Read the findings below as
**the hypotheses the re-run is designed to confirm**, and expect the third decimal to move.

## What the pilot established

### The screen is the per-block ratio, not the per-matrix peak

This is the single most useful thing the pilot produced, and it replaces the peak threshold an earlier
draft of this document built its screens on.

Two runs at ratio 0.5 make the case on their own. Both reach a per-matrix peak of exactly 0.900,
which the old screen called fatal:

| at ratio 0.5, homogeneous 24.56 | matrices >= 0.85 | layers holding them | families | wikitext |
|---|---|---|---|---|
| `decoder` + `softmax_temp` + `eff_rank` | 31 | 26 of 32 | **q and k only** | **23.16** |
| `type` + `softmax_temp` + `eff_rank` | 11 | **0, 1, 2, 31** | **all seven** | **78.19** |

The better run has three times as many aggressive matrices. What separates them is that the first
spreads them one or two per block across the depth of the model and touches only the two most
redundant families, while the second hollows out the first three blocks across every family at once.

Ranking all the pilot runs against the parameter-weighted fraction each **block** loses:

| predictor | Spearman vs wikitext at 0.2 | at 0.5 |
|---|---|---|
| peak per-matrix ratio | +0.154 | +0.414 |
| **peak per-block ratio** | **+0.659** | **+0.730** |
| block 0's ratio | +0.401 | +0.528 |

And it is one-sided without exception at ratio 0.5: **every** run that pushed a block past 0.63
landed at 28.48 or worse, and the block at the peak was **layer 0 in every one of them**. So the
screen is a ceiling on what a block may lose, `BLOCK_RATIO_DANGER = 0.60` in `allocation_report.py`,
and it is exact offline because the preview replays the same allocator.

Weighting by parameters rather than averaging the seven ratios matters: an MLP matrix carries between
two and three times an attention matrix, so a mean of ratios is not the fraction the block loses.

### There are two failure modes, one per level of the allocator

`param_share` pins every `decoder` block at exactly the target, so the block screen cannot
discriminate inside that grouping — and yet the `decoder` rows span 23.16 to 44.47. Two independent
things can go wrong:

| | what breaks | owned by | pilot examples at 0.5 |
|---|---|---|---|
| **Depth failure** | a block loses too much in aggregate, always layer 0 | grouping, outer policy | `global`/`truncation` 39.09 (block 0 at 0.760), `type`/`softmax_temp` 78.19 (0.840) |
| **Within-block failure** | the block budget is right, its internal split is not | score, inner policy | `decoder`/`drank_lagrangian` 35.94 and 44.47, `decoder`/`eff_rank_sq` 34.44, `decoder`/`norm\|inf` 33.44 |

The two levels of the hierarchical allocator address one failure mode each, which is a stronger
framing for the contribution than "heterogeneity helps" and is the frame the thesis should use.

Within the 14 `decoder` rows, where the block budget is fixed by construction, the mean **MLP** ratio
correlates +0.556 with perplexity: the winners take budget off the MLP and give it to q and k
(`softmax_temp`: MLP 0.399, `v_proj` 0.529), the catastrophes do the reverse (`drank_lagrangian` with
`entropy`: MLP 0.617, `v_proj` 0.251). That is a tendency on 14 points rather than a law, and stage 6b
is the controlled test of it.

### The outer level is the largest controlled effect, and the score does not change its size

| at ratio 0.5, homogeneous 24.56 | wikitext | worst block | block 0 |
|---|---|---|---|
| `hierarchical` + `waterfill` outer, `eff_rank` | **22.19** | 0.551 (L27) | 0.346 |
| `hierarchical` + `waterfill` outer, `truncation` | **23.30** | 0.551 (L27) | 0.346 |
| `decoder` + `param_share`, `eff_rank` | 23.91 | 0.500 | 0.500 |
| `decoder` + `param_share`, `truncation` | 25.05 | 0.500 | 0.500 |

Against its matched control — same buckets of matrices, only Block Influence's freedom to move budget
between them differs — the outer level is worth **-1.72 with `eff_rank` and -1.75 with `truncation`**.

That refines rather than confirms the prediction this stage was written for. The prediction was that
`truncation` would gain more because its depth profile is worse. It did not: the outer level
**dominates** the depth profile and leaves the score to rank only within blocks, so its contribution
is the same whichever spectral score drives the inner split. The mean assigned ratio per depth shows
why the two hierarchical rows are interchangeable and where the flat groupings go wrong:

| | L0 | L15 | L27 | L31 |
|---|---|---|---|---|
| Block Influence | **0.4609** | 0.0929 | 0.0265 | **0.2929** |
| `decoder` + `param_share` | 0.505 | 0.503 | 0.506 | 0.510 |
| `hierarchical` + `waterfill` | **0.372** | 0.504 | **0.558** | **0.407** |
| `global` + `truncation` | **0.799** | 0.486 | 0.459 | 0.469 |

`global` + `truncation` runs a monotone *decreasing* profile — most removal at layer 0, least in the
deep blocks — which is anti-correlated with Block Influence. The outer level runs the profile Block
Influence asks for: protect both ends, spend in layers 24 to 29. The 15.8 perplexity between those
two rows is that inversion, and it is a figure rather than an argument.

Both hierarchical runs *lost* by 0.08 at ratio 0.2, which is exactly the two-machine offset measured
at that ratio and is not a result either way.

### Graded depth reweighting works; exempting a block does not

The pilot ran the control that separates "keep the tail off the early layers" from "the outer level
allocates better", by exempting the first two blocks under the profile that over-compresses them:

| at ratio 0.5, cap 0.7 | as is | with `--bypass_early_layers 2` |
|---|---|---|
| `global` / `truncation` | 33.06 | **34.78** (+1.72) |
| `type` / `truncation` | 33.56 | **34.84** (+1.28) |

Worse both times. The exempt blocks' budget lands on layers 2 and up, which the same inverted profile
already over-compresses — layer 2 goes from 0.594 to 0.637. So the failure is not layer 0 in
isolation, it is the whole depth profile, and a binary exemption at the top concentrates the error
instead of relieving it. An earlier draft of this document read the catastrophes as a layer-0 effect
and predicted this control would recover most of the damage; it does the opposite, which is what
isolates the result to the **allocator** rather than to any mechanism that happens to spare layer 0.

This is also why stage 5 is pinned to a grouping that does not reweight depth on its own.

### `drank_lagrangian` reallocates between matrix families, and `type` proves it

The prediction was that a rank-space policy pricing a rank at `out + in` can only bias an allocation
where a group mixes shapes. Mean assigned ratio per family at ratio 0.5 with `eff_rank`:

| family | `decoder` waterfill | `decoder` drank | `type` waterfill | `type` drank |
|---|---|---|---|---|
| `q_proj` | 0.533 | 0.384 | 0.500 | 0.500 |
| `k_proj` | 0.537 | 0.403 | 0.500 | 0.500 |
| `v_proj` | 0.501 | **0.248** | 0.500 | 0.500 |
| `o_proj` | 0.507 | **0.284** | 0.500 | 0.500 |
| `gate_proj` | 0.494 | **0.599** | 0.500 | 0.500 |
| `up_proj` | 0.490 | **0.584** | 0.500 | 0.500 |
| `down_proj` | 0.487 | **0.570** | 0.500 | 0.500 |
| **wikitext** | **23.91** | **35.94** | **25.23** | **28.48** |

Under `decoder`, where a group holds 4096x4096 attention beside 11008x4096 MLP, the policy moves
budget systematically off attention and onto the MLP and costs 12.0 perplexity with `eff_rank` and
20.2 with `entropy`. Under `type`, where every matrix in a group shares a shape, every family mean is
**exactly 0.500**: the bias is inert and the damage collapses to 3.25 and 0.44.

So the inner policy cannot be chosen independently of the grouping, which is why stage 4 runs three
arms instead of one and why `type` is in the grid as the control that makes the mechanism provable.

### A score's merit is a property of the score *and* the budget

At ratio 0.5 the order is `eff_rank` 23.91 < `entropy` 24.23 < `norm|1` 24.56 (exactly homogeneous)
< `truncation` 25.05, then a gap to `entropy_sq` 26.41, `norm|inf` 33.44 and `eff_rank_sq` 34.44. At
ratio 0.2 that order roughly **inverts**: `eff_rank_sq` is the best score in the pilot at 7.625,
`entropy_sq` also beats homogeneous at 7.742, and the plain scores cluster on it.

The aggressive scores allocate the same way at both budgets; what changes is whether that
aggressiveness lands a block inside the safe band. At 0.2 nothing reached a block ratio of 0.53; at
0.5 the same scores push blocks past 0.8. **The safe band does not scale with the budget**, which is
what makes an absolute ceiling plausible and is the RQ1 answer stage 8 is built to trace.

Neither 2b nor 2c promoted a score in the pilot, and the Schatten family said what it had to say:
`norm|1` reproduced homogeneous exactly and `norm|inf` was a catastrophe, so the conditional
`norm|-inf` companion has been dropped.

### The cap is a guard rail, not a tuning knob

Capping the winning configuration at 0.5 degrades it monotonically and slightly — 24.19, 24.09,
24.05, 24.05 at caps 0.6, 0.7, 0.8, 0.85 — and the last two are identical because the cap stops
binding at 0.8. Capping a catastrophe recovers part of the damage and leaves the rest: `global` /
`truncation` goes 39.09, 35.75, 33.06 at caps 0.9, 0.85, 0.7, while `decoder` / `truncation` sits at
25.05 at the same ceiling.

So `--max_ratio` earns its place by stopping a configuration that would run away, and every ratio
point it takes off one that would not is a small loss. It stays at **0.9** and stage 3 sweeps it where
it actually binds, which is on the flat groupings.

### The offline objectives rank backwards

Stage 2 measured the offline ordering against the measured one on all nine cells and **every row
disagreed**, close to inverted: `type`/`truncation` is offline-best and measured-worst,
`decoder`/`eff_rank` offline 8th of 9 and measured 1st.

There is a mechanism worth a paragraph of the thesis. The six objectives are all tail-energy measures,
minimised by concentrating removal where the spectrum decays fastest — which is the failure mode. They
reward it by construction, and with `score_ratio_rho` at `-1.0000` on every variant nothing else is
absorbing the difference.

`mean_rank` is therefore a reported negative result, not a screen. The offline pass keeps the
mechanical facts it is reliable on: feasibility, degeneracy, dispersion, cap binding, and above all
the block screen.

### The pipeline is deterministic given its whitening cache

Two pilot runs at ratio 0.2 differing only in a cap that never binds produced **bit-identical ratio
maps and bit-identical perplexity**. So run-to-run variance within one machine and one cache is zero,
and the error bars a thesis table needs are about the **calibration draw**, not about numerical noise.
That is what stage 1b measures, and it needs a fresh whitening directory to do it: the cache is keyed
by model and version alone, so a second seed pointed at the shared cache reads the first draw back and
measures nothing.

## The three tools, and the order to use them in

Every stage is the same loop, and each tool owns one step of it:

```
allocation_report.py  ->  run_experiments.py  ->  generate_tables.py --report gates
   prune and configure       spend the GPU            read the gate, fill the placeholder
   seconds, CPU              ~15 min per run          seconds, no GPU
```

| | `allocation_report.py` | `run_experiments.py` | `generate_tables.py --report gates` |
|---|---|---|---|
| Cost | seconds, CPU, no model weights | about 15 minutes per run, GPU | seconds, reads JSON |
| Answers | what ratios a configuration produces | what perplexity it produces | which configuration won |
| Reads | the cached spectra and Block Influence | the model | `output/eval/` and the offline CSVs |
| Settles | the run list, the knobs, feasibility | nothing by itself, it only measures | every `__PLACEHOLDER__` |

### What the offline report can and cannot settle

**It resolves no placeholder on its own.** Every `__*__` value in this grid is defined as a ranking by
perplexity, and perplexity needs the model. What the offline report does instead is decide **which
runs are worth making** and **what to hold them at**.

Read it as answering five questions, in this order:

1. **Is the configuration feasible?** A cap too low to reach the target once bypassed layers are
   charged shows up as a budget-drift violation, a `checks` entry in `summary.csv`, and a non-zero
   exit. Fix the configuration before running anything.
2. **Is the allocation about to blow up?** `max_block_ratio` in `summary.csv`, and the `worst block`
   column of the console table. This is the screen, and it is described in full below.
3. **Does the variant allocate anything?** A score constant inside every group produces the flat ratio
   whatever policy runs, so the run is homogeneous while looking heterogeneous. `allocate_ratios`
   prints a `[BUDGET][WARNING]` for it and `figures/dispersion.csv` shows a `ratio_std` near zero.
   Drop the cell: it will reproduce the stage 1 homogeneous number and take fifteen minutes to say so.
4. **Are two variants distinguishable?** Two configurations whose ratios agree to three decimals are
   one experiment, not two. `figures/map_distance.csv` reports every pair, and this is the one case
   where the offline pass can settle a gate outright: a candidate that cannot allocate differently
   cannot win a promotion.
5. **Are the variants comparable?** Policies compared at their default knobs differ in shape *and* in
   aggressiveness at once. `--offset` moves no policy's allocation and only `--softmax_temp` is live,
   so where two cannot be brought together, report `ratio_std` beside the result rather than implying
   it was controlled.

One gate is answered offline and only offline: **the Spearman sign in stage 6**. It is a go/no-go on
whether fusing Block Influence with a spectral score measures what it is meant to, and no perplexity
number substitutes for it.

### The block screen: what to read before every run

`figures/ratio_tail.csv` leads with `max_block_ratio`, the layer it sits on, block 0's own ratio, and
how many blocks cross the threshold. The console prints a warning naming every variant past it, and
the summary table marks it with `!`.

1. **A block above 0.60 is the failure mode.** On the pilot every run past 0.63 at ratio 0.5 cost at
   least 3 perplexity against homogeneous, and the block was layer 0 every time. Drop the variant.
2. **Then look at which layer.** A worst block in the middle or the late third of the model is a
   different animal from a worst block at 0 — `hierarchical` peaks at L27 and wins.
3. **The per-matrix `peak` is context, not a criterion.** It is kept in the same CSV because it bounds
   what any single truncation is doing, but a peak of 0.900 spread thinly across depth was the second
   best run in the pilot while a peak of 0.900 concentrated on three blocks was the worst.

**Do not use `ratio_std` for this.** `eff_rank_sq` at ratio 0.2 has the highest dispersion in its
sweep and is the best run at that ratio. Dispersion mislabels exactly the aggressive-but-safe
allocations that produce the gain. It remains the right tool for comparing two policies'
aggressiveness.

The threshold is one number fitted to 72 runs of one model at two budgets, and nothing in the pipeline
enforces it. Re-derive it per model. The one encouraging sign is that the band did **not** scale with
the budget between 0.2 and 0.5.

**The denominator is what the run compresses.** With all seven families targeted, `max_block_ratio` is
the fraction of the block. With a partial selection it is the fraction of the selection, so the
threshold does not apply and `selection_is_complete` makes the tool say so instead of warning. This
matters for stage 6b alone.

### The map-distance screen: two variants that are one experiment

`figures/map_distance.csv` gives, for every pair of variants in a sweep, the mean and the largest
per-matrix ratio difference plus each one's peak. A pair whose largest difference falls under 0.02 is
reported on the console as one experiment run twice.

The test is on the **largest** difference and not the mean. Raising the cap from 0.75 to 0.9 on
`type`/`truncation` at ratio 0.5 moves the allocation by 0.004 on average and by **0.15 on three
matrices** — and those three are the difference between a working model and one at 43 perplexity. A
screen that says "do not run this" has to be wrong in the safe direction.

It pays for itself immediately: it reports `hierarchical` + `param_share` as identical to `decoder`,
which is true by construction, and the cap sweep this document originally prescribed at ratio 0.2
collapses to one run because the cap never binds at the winner.

### Preview command

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --compression_ratio 0.2 \
    --sweep "group_criterion=global,type,decoder" \
    --sweep "score_metric=truncation,entropy,eff_rank" \
    --out_dir ./output/allocation_reports/stage2 --plots
```

`--sweep` is repeatable and taken as a cartesian product; anything not swept is passed as a plain flag
and held fixed, exactly as `run_experiments.py` would pass it. `--plots` adds PNGs when matplotlib is
installed; the CSVs are written either way.

**No `--compress_*` flag means every family**, unlike `main.py` where each one opts in, and with all
seven targeted the `all` and `selected` denominators coincide. That is why none of the per-stage
commands below carries either flag; stage 6b, whose experiment *is* the target selection, is the one
place both appear.

**`waterfill` gets its own invocation wherever it appears.** It is the only outer policy and it is
defined only on `hierarchical`, so crossing it with a `group_criterion` sweep fills the table with
`rejected` rows and, because a rejection counts as a failed invariant, takes the exit code with
them. Stages 3c, 4 and 4d therefore each carry two commands. A `map_distance` report that
`hierarchical` + `param_share` allocates identically to `decoder` is the expected result and not a
duplicated run: it is the equivalence stage 3c is built on.

**Name `--out_dir` after the stage it previews**: `stage2`, `stage3c`, `stage4`, and so on under
`output/allocation_reports/`. `generate_tables.py --allocation_dir` discovers stage directories by
that name and attaches each preview to the gate it belongs to. A suffix is allowed and keeps the same
stage, so `stage4_knobs` is read as stage 4; when both exist the unsuffixed one is used. Giving one
`--out_dir` per stage is also what stops each preview from overwriting the last.

### Reading its output

The console table is the summary: one row per variant, ordered by mean rank, with the swept axes named
once in the header and `worst block` carrying the screen. Each objective cell holds the value and, in
parentheses, its rank across the variants. The `checks` column is `ok` or the invariant that failed.

| File | Use |
|---|---|
| `summary.csv` | one row per variant: realized ratio, `mean_rank`, every objective with its rank and oracle ratio, ratio dispersion, `max_block_ratio` with its layer and block 0's, invariant violations |
| `matrices.csv` | one row per variant **per matrix**: score, assigned ratio, rank, truncation loss. The ratio map, and the only place two variants compare allocation by allocation |
| `layers.csv` | one row per variant per decoder block: params, removed params, block ratio, Block Influence |
| `figures/ratio_tail.csv` | **the screen**: `max_block_ratio` and its layer, block 0, blocks past the threshold, then the per-matrix tail as context |
| `figures/map_distance.csv` | how far apart two variants allocate, with each one's peak: the screen against paying twice for one experiment |
| `figures/dispersion.csv` | how widely each configuration spreads its ratios, for knob comparison and for spotting a degenerate cell |
| `figures/ratio_by_type.csv` | mean ratio per matrix family, where rank-space bias shows: thesis 3.3.5 |
| `figures/influence_vs_effrank_rho.csv` | Spearman rho per matrix family, the gate on stage 6 |
| `figures/layer_ratios.csv` | the assigned ratio against depth, which is the outer level's whole story |
| `figures/objectives.csv` | which variants win only the objective their own score optimizes |
| `figures/cap_binding.csv` | how many matrices `--max_ratio` actually pins |
| `figures/oracle_gap.csv` | each objective against its greedy lower bound, for comparing across budgets |
| `budget/<variant>.log` | the captured `[BUDGET]` instrumentation of that variant |

Variants are ranked by **mean rank across six objectives**, never by a single number: `frobenius_tail`
*is* the `truncation_sq` score summed over matrices, so ranking on it hands the truncation scores a
win by construction. And per the finding above, do not read that ordering as a prediction of the
perplexity ordering — it prices the allocation, not the model.

## Running a stage

```bash
python run_experiments.py args/experiments_stage1_anchors.json            # run it
python run_experiments.py args/experiments_stage1_anchors.json --dry_run  # preview the commands
python run_experiments.py args/experiments_stage4_policies.json --base args/other_base.json
```

The runner merges `args/base_args.json` into each entry of the stage file, refuses to start if any
entry still holds an unresolved gate value, and continues to the next run when one fails rather than
aborting the queue.

The commands it prints are informational, not copy-paste ready: a score like `norm|1` holds a pipe a
shell would interpret. `subprocess` passes it as one argument, so runs are unaffected.

`args/` is tracked, so the grid is reproducible from the repository. **Never put `--hf_token` in one of
these files**: pass it on the command line, or export it in the environment.

## Frozen across every run

Changing any of these invalidates comparability with everything already collected.
`args/base_args.json` is authoritative.

| Setting | Value | Why frozen |
|---|---|---|
| Model | `huggyllama/llama-7b` | single model until stage 7 |
| Version | `--run_v2` | documented as a limitation in thesis 5.2.1 |
| Precision | `float16` weights and factors | what the whole grid is being run at, so it is the comparable choice |
| Calibration | wikitext-2 train, `--max_length 2048` | |
| `--max_whitening_samples` | `256` | truncation and `norm\|p` scores scale with the token count, so this cannot move between runs (thesis 5.2) |
| `--seed` | `6363` | fixes the calibration sample; stage 1b is the only stage that varies it |
| Targets | all seven matrices, `--ratio_scope all` | stage 6b is the only stage that varies the selection, and `all` is what holds its budget comparable |
| `--max_ratio` | `0.9` | a guard rail, per the finding above; stage 3 sweeps it where it binds |
| Screening ratios | `0.2`, `0.5` | stage 8 extends the curve |
| Screening evaluation | `wikitext\|0`, `--eval_max_length 2048` | c4 and the suite arrive at stage 9. LLaMA-7B's context is 2048 |

The whitening cache is keyed by model and version alone, so raising `--max_whitening_samples` has no
effect unless `output/whitening_matrices/` is deleted first. That is also the mechanism stage 1b
exploits.

## Checkpoints, and what has to survive a cleanup

A compressed fp16 7B checkpoint is about 10 GB, so checkpoints are deleted once their run has been
evaluated. Two rules keep that from destroying a gate:

- **Keep every checkpoint a `__CKPT_*__` role names**: the two homogeneous anchors, the best score,
  outer, policy, bypass and composite runs, and the overall heterogeneous winner at each ratio.
  Stages 9 and 10 load these directly and cannot recompress.
- **Keep each stage's runner-up too.** Roles move: stage 2b or 2c promoting a score moves
  `__CKPT_BEST_SCORE_*__` onto a different run, and stage 3c re-resolving `__BEST_GROUPING__` moves
  several more.

The gate report's stage 9 roster has an `on disk` column for exactly this, so a cleanup can be checked
against it before stage 9 is queued. `--no_save_checkpoint` writes the sidecar without the `.pt`, so a
run that no role will ever name can skip the 10 GB and still appear in every table.

## Placeholders

Stage files carry literal placeholders until their gate resolves them.

| Placeholder | Resolved by | Meaning | What the offline preview contributes |
|---|---|---|---|
| `__BEST_GROUPING__` | stage 2, **re-resolved by 3c** | grouping criterion with the best mean rank | drops groupings whose scores are degenerate inside every group |
| `__BEST_OUTER__` | stage 3c | the outer policy that travels with it | `figures/layer_ratios.csv`, which shows whether the outer level moved anything |
| `__BEST_FLAT_GROUPING__` | stage 2 | better of `type` / `global`, never `decoder` | confirms both flat groupings actually spread their ratios |
| `__TOP1_SCORE__`, `__TOP2_SCORE__` | stage 2, promotable by 2b or 2c | the two best score metrics | drops a candidate whose ratio map matches an incumbent's, since it cannot win a promotion |
| `__BEST_INNER__` | stage 4, **under `__BEST_GROUPING__`** | best inner allocation policy | the block screen per policy, since they differ in aggressiveness by construction |
| `__BEST_BYPASS_EARLY__`, `__BEST_BYPASS_LATE__` | stage 5 | the bypass setting with the best gain over homogeneous | catches settings whose budget is infeasible once the exempt blocks are charged |
| `__CKPT_<ROLE>__` | stages 1 to 8 | a path under `output/models/huggyllama_llama_7b/` | nothing, these are outputs of runs |
| `__FINALIST{1,2,3}_*` | stages 2 to 8 | the three configurations worth another model, with the outer policy each needs | rerun stage 0 per model: the Spearman sign and the score-versus-depth shape are model properties |

`run_experiments.py` refuses to start while any remain, so an unfilled gate cannot silently run the
wrong configuration.

**`__BEST_GROUPING__` belongs to stage 3c, not stage 2.** Stage 2 can only nominate among the flat
criteria — the outer level is not on its ballot — so leaving the placeholder with it would hold every
later stage at `decoder` however far ahead `hierarchical` finished. The gate report resolves it twice
and the second answer wins, and `__BEST_OUTER__` travels with it because `hierarchical` +
`param_share` reproduces `decoder` exactly.

**No stage file past 2c names a score.** Stages 2b and 2c are promotion tests and either can move
`__TOP1_SCORE__` or `__TOP2_SCORE__` onto a squared score or a Schatten norm. Writing today's winner
into stage 4 would freeze the decision those two stages exist to revisit. The same holds for the
composite halves of stage 6, spelled `composite|__TOP1_SCORE__|block_influence`.

**Stage 4's arms are literal on purpose.** `hierarchical`, `decoder` and `type` are the ablation
itself, not a configuration inherited from a gate, so they are written out rather than placeheld.

**A `provisional` row is not an answer.** The gate report says how many candidates a placeholder was
chosen from, and a value decided by a table holding one entrant reads `provisional (1 candidate)`:
the report has recorded the only run that happened, not picked a winner.

## Resolving a gate

```bash
python generate_tables.py ./output/eval/huggyllama_llama_7b \
    --report gates --allocation_dir ./output/allocation_reports -o gates.md
```

One table per gate, in stage order, and a leading **Placeholders** table holding every value a stage
file waits on next to what the collected runs resolve it to. Add `--report both` for the result tables
in the same file, or `-f latex` for LaTeX.

What it does with the runs, so a table can be trusted before it is pasted into a stage file:

- Ranks **within each ratio** and averages the ranks, never comparing raw perplexity across budgets,
  and reports how many ratios priced each row.
- Carries the **gain over the homogeneous arm** at the same setting, which is the RQ1 read every stage
  repeats, and for stage 5 pairs the two arms bypass setting by bypass setting.
- **Holds fixed** whatever a stage is not sweeping, and reports which runs that excluded.
- **Intersects unmatched axes** before an aggregate: stage 3c drops any score only one arm was run at,
  because a mean rank over cells one arm has and another does not lets the wider arm win on its extra
  cells rather than on the factor under test.
- Splits stage 4 into **one panel per grouping arm** plus a table of the ranking across arms, since
  pooling them would rank the policies through the grouping instead of within it.
- Warns when a dimension moves inside a table without being one of its axes, when a run's realized
  removal drifted off its budget, and when a table mixes run environments.
- Attaches the offline preview of each stage from `--allocation_dir`, including the Spearman sign that
  gates stage 6 and, for stage 2, the offline ordering against the measured one.

The dimensions all come from the sidecar, the only place most of them exist. A run without one is
counted and left out rather than guessed at.

## Passing a gate, step by step

**1. Preview the stage offline**, sweeping the axis and the knobs together. Substitute placeholders by
hand: this tool takes flags, not stage files. A non-zero exit means an invariant failed.

**2. Prune and configure from the CSVs**, in this order: the exit code and `checks`, then
`max_block_ratio`, then the degeneracy warning, then `map_distance`, then `dispersion`. Write any
chosen knob into `args/base_args.json` and delete the dropped cells from the stage file.

**3. Run the stage**, `--dry_run` first.

**4. Read the gate.**

**5. Check the gate's own warnings before trusting it.** A `confounded` note means a dimension moved
that the stage was not comparing, and the fix is a run rather than a reading. A `priced at 1/2` row
was ranked on one ratio. A drift note means a run missed its budget and is not comparable at all. A
restriction note means an axis was intersected, and the dropped values are named.

**6. Copy the resolved value into the next stage file**, from the **Placeholders** table rather than
the body tables:

```bash
sed -i 's/__BEST_INNER__/softmax_temp/g' args/experiments_stage5_bypass.json
```

`run_experiments.py` refusing to start is the backstop: it means a placeholder was missed.

### When the offline pass is enough on its own

Four cases end at step 2, with no GPU time at all:

- **The configuration is infeasible.** Non-zero exit, `checks` populated.
- **A block crosses 0.60.** The screen has been right on every pilot run, and the cell is going to
  cost at least 3 perplexity.
- **The cell is degenerate.** A `[BUDGET][WARNING]` or a `ratio_std` near zero means the run
  reproduces the homogeneous number. Delete it and say so in the thesis: a heterogeneous allocation
  that cannot allocate is a finding about the score, not a missing data point.
- **Two candidates are the same experiment.** Ratio maps agreeing to three decimals cannot produce
  different perplexities.

### Which preview to run for which stage

| Stage | Preview `--out_dir` | Read | Decide |
|---|---|---|---|
| 0 | `stage0` | `figures/influence_vs_effrank_rho.csv` | whether stage 6 may run at all, and record the sign either way |
| 1, 1b | none | | a homogeneous run allocates nothing |
| 2 | `stage2` | `max_block_ratio` per cell | which cells will blow up, before spending 18 runs |
| 2b | `stage2b` | `matrices.csv`, `map_distance` | whether a `_sq` score allocates differently from the score it derives from |
| 2c | `stage2c` | `matrices.csv`, `figures/dispersion.csv` | whether a Schatten norm is signal or rounding noise |
| 3 | `stage3` | `figures/cap_binding.csv`, `map_distance` | that every cap binds, and that no two runs allocate the same way |
| 3c | `stage3c` + `stage3c_equivalence` | `figures/layer_ratios.csv`, `ratio_tail`; `map_distance` in the second | whether the outer level moves the worst block off layer 0, and that it is neutral under `param_share` |
| 4 | `stage4` + `stage4_hierarchical` | `max_block_ratio` per policy, `figures/ratio_by_type.csv` | which policies the cap still lets past the screen, and where the shape bias lands |
| 4c | `stage4c` | `max_block_ratio` against `--outer_offset` | the ladder's usable range, which is not monotone |
| 4d | `stage4d` + `stage4d_hierarchical` | `max_block_ratio` per temperature | which temperatures are runnable at each ratio and under each depth regime |
| 5, 5b | `stage5` | the exit code and `checks` | whether the bypassed budgets are feasible under the cap |
| 6 | `stage6` | `figures/dispersion.csv`, plus the stage 0 rho | the offset for the fused score, and that the three alphas allocate distinctly |
| 6b | `stage6b`, `_mlp`, `_attention` | the exit code | which selections can absorb the budget at all; the screen stands down here |
| 7 | `stage0` again, per model | the rho sign, `figures/scores_by_depth.csv` | whether the finalists transfer |
| 8 | `stage8` | the `<objective>_oracle_ratio` columns | the shape claim, made offline and never costing a run |
| 9, 10 | none | | both load existing checkpoints and allocate nothing |

Re-run stage 0 whenever the whitening cache changes.

---

## Stage 0: the offline pass

**Purpose.** Everything knowable without a GPU, plus the gate on stage 6. Costs seconds.

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --sweep "compression_ratio=0.2,0.5" \
    --out_dir ./output/allocation_reports/stage0 --plots
```

**Gate.** `figures/influence_vs_effrank_rho.csv`. Swift-SVD reports Block Influence and normalized
effective rank as **negatively** correlated, which is what makes the two signals complementary and
justifies fusing them in stage 6. On the pilot cache all seven families came out negative — `v_proj`
-0.9223, `k_proj` -0.7126, `q_proj` -0.7056, `down_proj` -0.7049, `gate_proj` -0.5913, `o_proj`
-0.3138, `up_proj` -0.1419 — so **stage 6 is open**. Record the sign per model either way: a positive
one means the two signals agree rather than complement and the fusion convention needs revisiting
before stage 6 is worth running.

**Also record** `figures/influence_by_depth.csv`. Layer 0 at 0.4609 and layer 31 at 0.2929 against
0.15 or less for everything else is the premise of the whole outer level, and it is a model property.

---

## Stage 1: anchors

**Runs.** 3, `args/experiments_stage1_anchors.json`: the dense model, and homogeneous at 0.2 and 0.5.

Every gain figure in the grid is measured against these two, and every checkpoint role that names a
homogeneous arm points at them, so they are kept on disk permanently.

**Preview.** None. A homogeneous run assigns the target ratio to every matrix, so there is no
allocation to inspect and `allocation_report.py` has nothing to add.

**Read the gate.**

```bash
python generate_tables.py output/eval/huggyllama_llama_7b --report gates \
    --allocation_dir output/allocation_reports -o output/gates/huggyllama_llama_7b/gates_stage1.md
```

**What to check in it.** That the dense perplexity is in the expected range for LLaMA-7B on wikitext,
and that both `__CKPT_HOM_*__` rows read `ready`. The c4-against-wikitext check that used to live here
has moved to stage 9, where c4 is actually evaluated.

**Gate.** `__CKPT_HOM_0.2__`, `__CKPT_HOM_0.5__`.

---

## Stage 1b: the replicate floor

**Purpose.** An error bar for every table in the thesis. The pilot showed the pipeline is
deterministic given its whitening cache, so the variance that matters is the **calibration draw**.

**Runs.** 8, `args/experiments_stage1b_replicates.json`: two extra seeds (7777, 8888) x
{homogeneous, `decoder` + `eff_rank`} x {0.2, 0.5}. The seed-6363 cells come free from stages 1 and 2.

Each entry also sets `--whitening_mat_path ./output/whitening_matrices_seed<N>`, which is not
optional: the cache is keyed by model and version, so a fresh seed pointed at the shared directory
reads the first draw back and reports zero variance. That means **two extra whitening passes**, which
are the expensive part of this stage.

**What to check in it.** The spread of the three seeds at each ratio and each configuration. That
number is the resolution of the whole grid: any later difference smaller than it is not a result, and
the gate report's low-spread note should be read against it rather than against a guess.

**Preview.** None. Both arms are already covered by the stage 1 and stage 2 previews, and a change
of seed changes the calibration draw rather than the allocator.

**Read the gate.**

```bash
python generate_tables.py output/eval/huggyllama_llama_7b --report gates \
    --allocation_dir output/allocation_reports -o output/gates/huggyllama_llama_7b/gates_stage1b.md
```

**Gate.** Reporting only. Nothing downstream is held on it, but every table that ranks two rows should
cite it.

---

## Stage 2: score x grouping (RQ2)

**Purpose.** The main effect, and the interaction the rest of the grid is organised around.

**Runs.** 18, `args/experiments_stage2_score_grouping.json`: {`global`, `type`, `decoder`} x
{`truncation`, `entropy`, `eff_rank`} x {0.2, 0.5}, inner policy `waterfill`, outer `param_share`.

`hierarchical` is deliberately absent: with `param_share` it reproduces `decoder` exactly, and with
`waterfill` it is stage 3c's experiment rather than this one's.

**Preview.**

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --sweep "compression_ratio=0.2,0.5" \
    --sweep "group_criterion=global,type,decoder" \
    --sweep "score_metric=truncation,entropy,eff_rank" \
    --out_dir ./output/allocation_reports/stage2 --plots
```

**Read the gate.**

```bash
python generate_tables.py output/eval/huggyllama_llama_7b --report gates \
    --allocation_dir output/allocation_reports -o output/gates/huggyllama_llama_7b/gates_stage2.md
```

**What to check in it.**

- The **grouping aggregate**, which is what `__BEST_GROUPING__` and `__BEST_FLAT_GROUPING__` come from
  until stage 3c revisits the first.
- Whether the grouping spread depends on the score. On the pilot it did, by an order of magnitude:
  1.3 perplexity under `eff_rank`, 0.6 under `entropy`, 8.5 under `truncation`. The useful statement
  is not "`decoder` is best" but **the grouping matters exactly as much as the score makes it
  matter** — a score whose values move sharply across depth hands the grouping a large lever.
- The **offline against measured** table. Expect it to invert; that disagreement is a thesis result and
  it bounds how far the free preview can substitute for GPU time.
- Any low-spread note at 0.2, read against stage 1b's floor.

**Gate.** `__BEST_GROUPING__`, `__BEST_FLAT_GROUPING__`, `__TOP1_SCORE__`, `__TOP2_SCORE__`,
`__CKPT_BEST_SCORE_*__`.

---

## Stage 2b: squared spectra

**Purpose.** Whether squaring a score's spectrum earns its aggressiveness.

**Runs.** 6, `args/experiments_stage2b_squared.json`: the three squared scores under
`__BEST_GROUPING__` at both ratios.

**Preview.**

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --group_criterion __BEST_GROUPING__ \
    --sweep "compression_ratio=0.2,0.5" \
    --sweep "score_metric=truncation,truncation_sq,entropy,entropy_sq,eff_rank,eff_rank_sq" \
    --out_dir ./output/allocation_reports/stage2b
```

Each unsquared score is swept beside its squared counterpart on purpose, so that
`map_distance` pairs them: a squared score whose ratio map matches the score it derives from
cannot win a promotion, and the run can be dropped without a GPU.

**What to check in it.** The promotion table, and specifically whether a score wins at one ratio and
loses at the other. On the pilot `eff_rank_sq` was the single best run at 0.2 (7.625 against 7.789)
and third-worst at 0.5 (34.44), which is the ratio-dependence finding above. **A score that wins only
at one budget must not take a `__TOP*_SCORE__` slot** — say so in the thesis and leave the incumbent
in place, because every later stage runs at both.

**Read the gate.**

```bash
python generate_tables.py output/eval/huggyllama_llama_7b --report gates \
    --allocation_dir output/allocation_reports -o output/gates/huggyllama_llama_7b/gates_stage2b.md
```

**Gate.** May promote either `__TOP1_SCORE__` or `__TOP2_SCORE__`.

---

## Stage 2c: Schatten p-norms

**Purpose.** Whether the score family generalizes beyond the three of stage 2.

**Runs.** 4, `args/experiments_stage2c_schatten.json`: `norm|1` and `norm|inf` at both ratios.

The pilot's answer was no — `norm|1` reproduced homogeneous to the digit and `norm|inf` was a
catastrophe (33.44 at 0.5, block 0 above the screen) — and the conditional `norm|-inf` companion has
been dropped rather than kept as a placeholder. Re-running the two is still worth it: they bracket the
family, and "the extremes of the Schatten family are homogeneous and catastrophic respectively" is a
cleaner sentence with fresh numbers behind it.

**Preview.**

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --group_criterion __BEST_GROUPING__ \
    --sweep "compression_ratio=0.2,0.5" \
    --sweep "score_metric=norm|1,norm|inf,__TOP1_SCORE__" \
    --out_dir ./output/allocation_reports/stage2c
```

The pipe in `norm|1` is inside a quoted `--sweep` value, so the shell leaves it alone. The
incumbent rides along to give `map_distance` something to pair the two norms against.

**Read the gate.**

```bash
python generate_tables.py output/eval/huggyllama_llama_7b --report gates \
    --allocation_dir output/allocation_reports -o output/gates/huggyllama_llama_7b/gates_stage2c.md
```

**Gate.** May promote either score placeholder. Same one-ratio rule as 2b.

---

## Stage 3: the per-matrix cap

**Purpose.** Whether `--max_ratio` is a first-order effect, as Swift-SVD reports, or a guard rail.

**Runs.** 10, `args/experiments_stage3_max_ratio.json`: caps {0.6, 0.7, 0.8, 0.85} x {`global`,
`type`} with `truncation` at 0.5, plus two caps at the winner.

**The sweep lives on the flat groupings because that is where the cap binds.** At the winning
configuration the peak sits around 0.72, so a cap of 0.8 or above changes nothing and `map_distance`
reports the runs as identical — the pilot's five caps at ratio 0.2 collapsed to one experiment. The two
rows at the winner are kept to show exactly that, not to sweep it.

**What to check in it.**

- `figures/cap_binding.csv` before running anything: a cap that pins no matrix is not an experiment.
- Whether capping a catastrophe **recovers all of the damage or part of it**. On the pilot it was
  part: `global`/`truncation` went 39.09, 35.75, 33.06 while `decoder`/`truncation` sat at 25.05 at
  the same ceiling. That gap is the grouping, not the cap, and it is the cleanest argument that the
  cap cannot substitute for allocating well.
- `max_block_ratio` at each cap. The cap bounds a matrix, not a block, which is why it only partly
  rescues a run whose block 0 is at 0.76.

**Preview.**

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --compression_ratio 0.5 --score_metric truncation \
    --sweep "group_criterion=global,type" \
    --sweep "max_ratio=0.6,0.7,0.8,0.85,0.9" \
    --out_dir ./output/allocation_reports/stage3
```

**Read the gate.**

```bash
python generate_tables.py output/eval/huggyllama_llama_7b --report gates \
    --allocation_dir output/allocation_reports -o output/gates/huggyllama_llama_7b/gates_stage3.md
```

**Gate.** `--max_ratio` into `args/base_args.json`. Expect it to stay at 0.9.

---

## Stage 3c: the outer level

**This is the thesis contribution's own test, and the highest-value stage in the grid.**

**Purpose.** Whether Block Influence moving budget *between* blocks beats not moving it at all.
`decoder` + `param_share` against `hierarchical` + `waterfill`: the two criteria bucket matrices
identically and differ only in that.

**Runs.** 6, `args/experiments_stage3c_outer_level.json`:

- 4: `hierarchical` + `--outer_allocation waterfill` x {`truncation`, `__TOP1_SCORE__`} x {0.2, 0.5},
  inner held at `waterfill`. `truncation` is in the set because it is where the depth lever is
  largest, so it is where protection has the most to recover.
- 2: the **early-layer control**, `global` and `type` with `truncation` at 0.5, capped at 0.7 and
  `--bypass_early_layers 2`. This separates "the tail must be kept off the early blocks" from "the
  outer level allocates better". On the pilot it came out **negative both times** (+1.72 and +1.28
  against the uncapped-at-0.7 siblings), which is the result that isolates the finding to the
  allocator. Re-running it matters more than most rows here.

Its baselines already exist in stage 2 and stage 3, and `hierarchical` + `param_share` reproduces
`decoder` exactly, so the outer policy is the only factor moving.

**Preview.** Two invocations rather than one crossed sweep, because `waterfill` is only defined on
`hierarchical` and crossing the two would fill a third of the table with `rejected` rows and take
the exit code with them:

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --sweep "compression_ratio=0.2,0.5" \
    --sweep "group_criterion=decoder,hierarchical" \
    --sweep "score_metric=truncation,__TOP1_SCORE__" \
    --out_dir ./output/allocation_reports/stage3c_equivalence
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --group_criterion hierarchical --outer_allocation waterfill \
    --sweep "compression_ratio=0.2,0.5" \
    --sweep "score_metric=truncation,__TOP1_SCORE__" \
    --out_dir ./output/allocation_reports/stage3c --plots
```

The first exists to be boring: `map_distance` should report every `hierarchical` cell as identical
to its `decoder` twin, which is the equivalence this stage rests on, and a difference there means
the outer level is not neutral under `param_share` and the ablation is not controlled.
The second is the arm under test. Read `figures/layer_ratios.csv` and `ratio_tail` from it: if the
outer level is doing what it should, `max_block_ratio` leaves layer 0 — on the pilot cache it moves
to L27 and block 0 falls from 0.500 to 0.346. A win without that movement is a win for something
else.

**What to check in it.**

- The gate's own restriction note. The aggregate only means something over the scores **both** arms
  were run at, and the report intersects them and says which it dropped.
- Whether the gain is the same at both scores. On the pilot it was (-1.72 and -1.75), which says the
  outer level dominates the depth profile and the score only ranks within blocks — a stronger claim
  than the one this stage was written to test, and worth stating as the finding.
- The two control rows against their siblings.
- Ratio 0.2 separately. The pilot's -0.08 there is the size of the environment offset; with one
  machine and the fp64 metric this is finally answerable.

**Read the gate.**

```bash
python generate_tables.py output/eval/huggyllama_llama_7b --report gates \
    --allocation_dir output/allocation_reports -o output/gates/huggyllama_llama_7b/gates_stage3c.md
```

**Gate.** `__BEST_GROUPING__` (re-resolved), `__BEST_OUTER__`, `__CKPT_BEST_OUTER_*__`. Whichever of
the two pairs wins is the configuration stages 4 onward are held at.

---

## Stage 4: allocation policies (RQ3)

**Purpose.** Whether the policy that spends a group's budget matters apart from the score that ranks
the matrices — and whether it can be chosen independently of the grouping. The pilot says it cannot,
which is why this stage has three arms.

**Runs.** 38, `args/experiments_stage4_policies.json`:

- 36: {`drank_lagrangian`, `swift_pool`, `softmax_temp`} x {`__TOP1_SCORE__`, `__TOP2_SCORE__`} x
  {`hierarchical`, `decoder`, `type`} x {0.2, 0.5}
- 2: `waterfill` under `hierarchical` at `__TOP2_SCORE__`, both ratios, so that arm is complete —
  stage 3c only ran the first score there.

`waterfill` under `decoder` and `type` comes from stage 2, and under `hierarchical` at the first score
from stage 3c.

**Why three arms.** `drank_lagrangian` allocates in rank space and prices a rank at `out + in`, so its
shape bias only bites when a group mixes shapes. Under `type` every bucket holds one family across 32
blocks and every shape is identical, so the bias is provably inert; under `decoder` and `hierarchical`
a bucket mixes 4096x4096 attention with 11008x4096 MLP and it is live. The pilot measured exactly
that — 12.0 and 20.2 perplexity of damage under `decoder`, 3.25 and 0.44 under `type`, with every
`type` family mean at exactly 0.500 — so `type` is the control that makes the mechanism provable
rather than a fourth data point.

**Also expected from the pilot:** `softmax_temp` at its default temperature was the best inner policy
under `decoder` (23.16 against `waterfill`'s 23.91) and the worst under `type` (78.19), because the
same peak lands one-or-two-per-block on q and k in the first case and hollows out blocks 0 to 2 across
all seven families in the second. If that reproduces, the interaction is the RQ3 answer and a single
`__BEST_INNER__` is meaningless without `__BEST_GROUPING__` beside it.

**Preview.**

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --score_metric __TOP1_SCORE__ \
    --sweep "compression_ratio=0.2,0.5" \
    --sweep "group_criterion=decoder,type" \
    --sweep "inner_allocation=waterfill,drank_lagrangian,swift_pool,softmax_temp" \
    --out_dir ./output/allocation_reports/stage4 --plots
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --score_metric __TOP1_SCORE__ \
    --group_criterion hierarchical --outer_allocation waterfill \
    --sweep "compression_ratio=0.2,0.5" \
    --sweep "inner_allocation=waterfill,drank_lagrangian,swift_pool,softmax_temp" \
    --out_dir ./output/allocation_reports/stage4_hierarchical
```

One score is enough for both: the preview answers the block screen per policy and the shape bias
in `figures/ratio_by_type.csv`, and neither depends on which score ranks the matrices. The gate
attaches the unsuffixed directory, so the hierarchical arm is read by hand.

**Read the gate.**

```bash
python generate_tables.py output/eval/huggyllama_llama_7b --report gates \
    --allocation_dir output/allocation_reports -o output/gates/huggyllama_llama_7b/gates_stage4.md
```

**What to check in it.**

- The **per-arm panels** first, never a pooled ranking: pooling ranks the policies through the
  grouping instead of within it.
- The **policy ranking across grouping arms** table, which states outright whether the arms agree. A
  disagreement is the result, not an obstacle.
- The aggregate, which reads `__BEST_INNER__` from the arm `__BEST_GROUPING__` names, and only that
  arm.
- `max_block_ratio` per policy from the offline preview. The four differ in aggressiveness by
  construction and `--offset` cannot equalise them, so if the frozen cap has not brought all four
  inside the screen the table is ranking aggressiveness rather than shape. Report `ratio_std` beside
  the result.
- `figures/ratio_by_type.csv`, which is where the rank-space bias becomes a figure: thesis 3.3.5.

**Gate.** `__BEST_INNER__`, `__CKPT_BEST_POLICY_*__`. If the ranking flips between arms, report the
interaction instead of a single winner.

---

## Stage 4c: how hard the outer level may reweight depth

**Purpose.** `--outer_offset` dials the strength of the depth reweighting that stage 3c showed is the
largest effect in the grid. Leaving it at its default would leave the contribution's size resting on
an untuned knob.

**Runs.** 10, `args/experiments_stage4c_outer_offset.json`: {1.05, 1.2, 2.0, 3.0, 6.0} x {0.2, 0.5}
at the stage 4 winner. The collected default 1.5 is the sixth point.

**The ladder is not monotone in danger**, which is the reason it needs measuring rather than reasoning
about. Offline on the pilot cache at ratio 0.5:

| `--outer_offset` | 1.05 | 1.2 | **1.5** | 2.0 | 3.0 | 6.0 |
|---|---|---|---|---|---|---|
| worst block | 0.841 | 0.641 | **0.581** | 0.600 | 0.660 | 0.702 |
| min ratio | 0.126 | 0.216 | 0.300 | 0.357 | 0.395 | 0.420 |
| `ratio_std` | 0.180 | 0.087 | 0.047 | 0.033 | 0.028 | 0.028 |

The default sits at a minimum of the worst block. Below it the reweighting is so strong that the
blocks it does not protect cross the screen; above it the allocation converges back on `param_share`,
so **6.0 should approach `decoder`** — a built-in continuity check that catches a mis-wired outer
level. 1.05 is included as the danger control precisely because the screen rejects it.

**What to check in it.** Whether the best offset is 1.5 or somewhere between 1.2 and 2.0, and whether
the optimum is the same at both ratios. If it is not, `--outer_offset` is budget-dependent and the
thesis has to say so rather than quoting one value.

**Preview.**

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --group_criterion hierarchical --outer_allocation waterfill \
    --score_metric __TOP1_SCORE__ --inner_allocation __BEST_INNER__ \
    --sweep "compression_ratio=0.2,0.5" \
    --sweep "outer_offset=1.05,1.2,1.5,2.0,3.0,6.0" \
    --out_dir ./output/allocation_reports/stage4c --plots
```

This is the preview the table above came from, so it doubles as a check that the cache still
reproduces it. `figures/layer_ratios.csv` is the figure: the ladder is a family of depth profiles.

**Read the gate.**

```bash
python generate_tables.py output/eval/huggyllama_llama_7b --report gates \
    --allocation_dir output/allocation_reports -o output/gates/huggyllama_llama_7b/gates_stage4c.md
```

**Gate.** Reporting, plus the offset every later stage runs at if it is not the default.

---

## Stage 4d: the temperature ladder

**Purpose.** `--softmax_temp` is the one live inner-policy knob, and the pilot's default-temperature
run was the best inner policy under `decoder`. The ladder asks whether that was the top of the range
or a point on the way up.

**Runs.** 16, `args/experiments_stage4d_temperature.json`: {0.15, 0.25, 0.35, 0.5} x {`decoder`,
`hierarchical`} x {0.2, 0.5}. The default 1.0 comes from stage 4.

This replaces the pilot's stage 3b, which ran the ladder at ratio 0.2 only and against a score that
turned out to be a one-ratio winner.

**Preview.**

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --inner_allocation softmax_temp --score_metric __TOP1_SCORE__ \
    --group_criterion decoder \
    --sweep "compression_ratio=0.2,0.5" \
    --sweep "softmax_temp=0.15,0.25,0.35,0.5,1.0" \
    --out_dir ./output/allocation_reports/stage4d
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --inner_allocation softmax_temp --score_metric __TOP1_SCORE__ \
    --group_criterion hierarchical --outer_allocation waterfill \
    --sweep "compression_ratio=0.2,0.5" \
    --sweep "softmax_temp=0.15,0.25,0.35,0.5,1.0" \
    --out_dir ./output/allocation_reports/stage4d_hierarchical
```

**Expect rejections, and honour them.** Lower temperatures pin matrices at the cap: on the pilot
cache, temperature 0.2 and 0.05 pinned a seventh of all matrices at ratio **0.2**, which the screen
rejects outright. Run only the temperatures that survive, and record which were rejected and why —
a rejected temperature is a finding about the policy.

**What to check in it.** Whether the temperature optimum differs between the two depth regimes. Under
`decoder` the aggressive allocation has nowhere to concentrate except within a block; under
`hierarchical` the outer level has already spent the depth budget, so the same temperature may be too
much. That interaction is the reason both arms are here.

**Read the gate.**

```bash
python generate_tables.py output/eval/huggyllama_llama_7b --report gates \
    --allocation_dir output/allocation_reports -o output/gates/huggyllama_llama_7b/gates_stage4d.md
```

**Gate.** Reporting, plus the temperature `__BEST_INNER__` is quoted at if it is `softmax_temp`.

---

## Stage 5: bypassing outer blocks (RQ4)

**Purpose.** Whether exempting the first or last N decoder blocks beats compressing everything, and
whether that gain **cannibalizes** the heterogeneous gain.

**Runs.** 36, `args/experiments_stage5_bypass.json`: nine settings x {heterogeneous, homogeneous} x
{0.2, 0.5}. Settings: `early {1, 4, 8}`, `late {1, 4, 8}`, then `2 + 2`, `4 + 4` and `2 + 1`, with
`--bypass_ratio` at `0.0` so an exempt block is skipped entirely and its budget pushed onto the rest.

The two ends carry the same depths on purpose, and that is what separates **placement** from
**amount**. Three settings exempt four blocks and three exempt eight, so inside each triple the budget
pushed onto the survivors is identical and only the placement differs. `2 + 1` is the asymmetric small
case and `early 1` / `late 1` bound how little it takes to matter.

**The heterogeneous arm is pinned to `decoder`, not `__BEST_GROUPING__`.** Under `hierarchical` the
outer level is already reweighting depth continuously, so a bypass there prices two depth mechanisms
at once and cannot answer RQ4. `decoder` flattens depth by construction, which makes it the only arm
where the bypass is the sole depth lever.

**Expect it to lose, and use the pilot's controls as the prior.** Exempting two blocks under an
inverted depth profile cost 1.72 and 1.28 perplexity. Under `decoder` + `eff_rank` the profile is
sane, so this stage is a fair test rather than a rerun of that — but the burden of proof is on the
bypass.

**Preview.**

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --group_criterion decoder --score_metric __TOP1_SCORE__ \
    --inner_allocation __BEST_INNER__ \
    --sweep "compression_ratio=0.2,0.5" \
    --sweep "bypass_early_layers=-1,1,2,4,8" \
    --sweep "bypass_late_layers=-1,1,2,4,8" \
    --out_dir ./output/allocation_reports/stage5
```

The sweep is the full cross of the two ends, which covers the nine settings the stage runs and the
combinations it does not, for the price of nothing. **It exits non-zero, and that is the point.**
Bypassing pushes an exempt block's budget onto the rest, and one cell cannot absorb it: exempting
eight blocks at each end at ratio 0.5 leaves sixteen of thirty-two blocks to carry the whole budget
and lands at a realized 0.45 against a target of 0.50, a ten percent drift the cap will not let it
close. That cell is not among the nine the stage runs, so the failure is information rather than a
blocker: read `checks` in `summary.csv` and confirm the drift sits only there before dismissing it.

**What to check in it.**

- The **paired table** and its `mean gain` column. The homogeneous arm is not padding: compute the
  heterogeneous gain over homogeneous *at each bypass setting* and compare it to the gain at bypass 0.
  A shrinking gain means the two mechanisms compete for the same redundancy, which is the second half
  of RQ4.
- The three matched-total triples, where only placement differs.
- `max_block_ratio` per setting: pushing an exempt block's budget onto the rest raises every other
  block, so a setting safe at bypass 0 can cross the screen once eight are exempt.
- Any missing heterogeneous cell, which means that setting was infeasible and the offline pass should
  have caught it.

**Read the gate.**

```bash
python generate_tables.py output/eval/huggyllama_llama_7b --report gates \
    --allocation_dir output/allocation_reports -o output/gates/huggyllama_llama_7b/gates_stage5.md
```

**Gate.** `__BEST_BYPASS_EARLY__`, `__BEST_BYPASS_LATE__`, `__CKPT_BEST_BYPASS_0.2__`.

**Stage 5b, the grouping probe.** 2 runs, `args/experiments_stage5b_bypass_grouping.json`: the winning
setting under `__BEST_FLAT_GROUPING__`. Bypassing means different things to the two groupings — under
`decoder` it deletes whole groups and `param_share` redistributes their budget between the survivors,
under a flat grouping it thins one pool — so a conclusion drawn at one may not transfer. The
homogeneous arm allocates nothing and is grouping-independent, which is why the probe is 2 runs.

---

## Stage 6: composite scores

**Purpose.** Whether fusing a per-matrix spectral score with per-block Block Influence beats either
alone. This is the **scalar** channel for the signal the outer level uses structurally, so with stage
3c behind it the comparison is a proper ablation of *how* to inject Block Influence rather than
whether to.

**Gate to open the stage.** The Spearman sign from stage 0. Negative in all seven families on the
pilot cache, so it is open.

**Runs.** 12, `args/experiments_stage6_composite.json`: `composite|__TOP1_SCORE__|block_influence`
under `__BEST_FLAT_GROUPING__` at `--fusion_alpha` {0.25, 0.5, 0.75} x {0.2, 0.5}, plus the same
under the second score.

**A flat grouping is required, and this is the part that is easy to get wrong.** Block Influence is
constant inside a decoder block, so it contributes nothing to the ranking *within* a
`decoder`/`hierarchical` group at any alpha, and `param_share` never reads a score *between* groups.
Under those criteria `--fusion_alpha` is a dispersion knob collinear with `--offset` and nothing more.
Block Influence has exactly two channels: a flat grouping, or the outer level. An earlier draft said a
fused score "cannot rank matrices within a group", which is only exactly true at alpha 1; the accurate
statement is this one.

**What to check in it.**

- The three alphas against each other, and `warn_on_degenerate_scores` for any that collapsed.
- The best composite against **both** `hierarchical` + `waterfill` from stage 3c and the plain score
  under the same flat grouping. Beating the plain score says fusion helps; beating the outer level
  says the scalar channel is the better way to inject the signal, which would be a surprise worth
  reporting loudly.
- `max_block_ratio`, since the fused score is what is supposed to keep the tail off the early blocks
  under a flat grouping. If it does not, the composite is not doing its job whatever the perplexity.

**Preview.**

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --group_criterion __BEST_FLAT_GROUPING__ \
    --score_metric "composite|__TOP1_SCORE__|block_influence" \
    --sweep "compression_ratio=0.2,0.5" \
    --sweep "fusion_alpha=0.25,0.5,0.75" \
    --out_dir ./output/allocation_reports/stage6 --plots
```

`figures/dispersion.csv` sets the offset for the fused score and shows whether the three alphas
allocate distinctly; `figures/influence_vs_effrank_rho.csv` from stage 0 is what opens the stage.

**Read the gate.**

```bash
python generate_tables.py output/eval/huggyllama_llama_7b --report gates \
    --allocation_dir output/allocation_reports -o output/gates/huggyllama_llama_7b/gates_stage6.md
```

**Gate.** `__CKPT_BEST_COMPOSITE_0.2__`.

---

## Stage 6b: which families the budget should land on

**Purpose.** The pilot's within-block finding — that the winners take budget off the MLP and give it
to q and k, and the catastrophes do the reverse — is a correlation over 14 runs. This is the
controlled version, and it is a new axis the grid never had.

**Runs.** 5, `args/experiments_stage6b_family_budget.json`, homogeneous on purpose:

| ratio | selection | what it asks |
|---|---|---|
| 0.2 | attention only | can the budget live entirely in attention |
| 0.2 | MLP only | can it live entirely in the MLP |
| 0.2 | everything except q and k | what it costs to protect the families the winners compress hardest |
| 0.5 | MLP only | the same at a budget where the MLP has to take 0.75 |
| 0.5 | everything except q and k | |

`--ratio_scope all` holds the **global** removed parameters at the target while the selection decides
which families absorb them, so the family axis moves alone. Homogeneous rather than heterogeneous for
the same reason: no score, no policy, nothing but the selection. Attention-only at 0.5 is absent
because attention does not hold enough parameters to absorb that budget — the offline pass reports it
as infeasible rather than drifting.

**The block screen does not apply here.** With a partial selection `max_block_ratio` is a fraction of
the selection, not of the block, and the threshold was fitted to full selections. The tool detects
this and says so instead of warning.

**What to check in it.** Whether excluding q and k costs more than excluding an equal mass elsewhere.
If it does, the spectral machinery is confirming something simpler — that early attention is low rank
— and the thesis should say so plainly and price the machinery against a hand-set family schedule.

**Preview.** One invocation per selection, since the target set is a flag rather than a sweepable
axis, and `--ratio_scope all` is what keeps the global budget comparable across them:

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 --ratio_scope all \
    --compress_att_v --compress_att_out --compress_mlp \
    --sweep "compression_ratio=0.2,0.5" \
    --out_dir ./output/allocation_reports/stage6b
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 --ratio_scope all \
    --compress_mlp \
    --sweep "compression_ratio=0.2,0.5" \
    --out_dir ./output/allocation_reports/stage6b_mlp
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 --ratio_scope all \
    --compress_att_q --compress_att_k --compress_att_v --compress_att_out \
    --compression_ratio 0.2 \
    --out_dir ./output/allocation_reports/stage6b_attention
```

Only the exit code and `checks` matter here: attention alone cannot absorb the budget at 0.5,
which is why that arm is one ratio rather than two, and the tool says so rather than drifting off
the target. The gate attaches the unsuffixed directory, so the other two are read by hand.

**Read the gate.**

```bash
python generate_tables.py output/eval/huggyllama_llama_7b --report gates \
    --allocation_dir output/allocation_reports -o output/gates/huggyllama_llama_7b/gates_stage6b.md
```

**Gate.** Reporting. This stage constrains the interpretation of RQ2 and RQ3 rather than any later
run.

---

## Stage 7: cross-model confirmation

**Purpose.** Whether the finalists transfer, which is the difference between a result and a property
of one checkpoint.

**Runs.** 18, `args/experiments_stage7_crossmodel.json`: per model, a dense reference, two homogeneous
anchors, and the three finalists at both ratios, for Qwen2.5-7B and Qwen2.5-32B.

**The per-model anchors are not optional.** Without them the gain column has nothing to subtract and
the transfer claim reduces to comparing a Qwen perplexity against a LLaMA one. The pilot's stage file
omitted them.

**Blocked on whitening.** Each model needs its own `get_whitening_matrices` pass and its own stage 0,
and 32B needs the JAX GPU eigendecomposition path.

**What to check in it.** Re-run stage 0 per model first and compare three things against LLaMA-7B: the
Spearman sign, the Block Influence profile across depth, and the score-versus-depth shape. The
finalists are only expected to transfer where those agree, and where they disagree the interesting
result is *which* of the three degrades.

**Preview.** Stage 0 again, once per model, which is the gate on whether the finalists are even
expected to transfer:

```bash
for model in "Qwen/Qwen2.5-7B" "Qwen/Qwen2.5-32B"; do
    python allocation_report.py --model "$model" --run_v2 \
        --sweep "compression_ratio=0.2,0.5" \
        --out_dir "./output/allocation_reports/stage0_$(basename "$model")" --plots
done
```

**Read the gate.**

```bash
python generate_tables.py output/eval/huggyllama_llama_7b --report gates \
    --allocation_dir output/allocation_reports -o output/gates/huggyllama_llama_7b/gates_stage7.md
```

**Gate.** Reporting. Nothing downstream depends on it.

---

## Stage 8: the ratio curve (RQ1)

**Purpose.** How the heterogeneous gain depends on the target ratio. The pilot's two budgets already
showed the gain is not monotone in any simple way — the aggressive scores win at 0.2 and collapse at
0.5 — so the curve is the RQ1 answer rather than a single number.

**Runs.** 12, `args/experiments_stage8_ratio_curve.json`: {0.1, 0.3, 0.4, 0.6, 0.7, 0.8} x
{homogeneous, the winner}. With 0.2 and 0.5 from stages 1 and 3c this is an eight-point curve.

This replaces the pilot's two-point onset probe. At fifteen minutes a run the full curve costs less
than four hours and answers a question the thesis asks in its first chapter.

**What to check in it.**

- Where the gain peaks. The pilot's working hypothesis is that it tracks the headroom between the
  allocation's worst block and the screen: at 0.1 there is nothing to gain because nothing is near
  collapse, at 0.8 there is nothing to gain because everything is.
- `max_block_ratio` at every point, plotted with the gain. If the two curves are mirror images the
  screen explains RQ1, which is a much stronger claim than a gain table.
- The `<objective>_oracle_ratio` columns, which make the shape claim offline and cost nothing.

**Preview.**

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --group_criterion __BEST_GROUPING__ --outer_allocation __BEST_OUTER__ \
    --score_metric __TOP1_SCORE__ --inner_allocation __BEST_INNER__ \
    --sweep "compression_ratio=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8" \
    --out_dir ./output/allocation_reports/stage8 --plots
```

The `<objective>_oracle_ratio` columns are the point: they divide out the budget's own
contribution, which is the only way an objective compares across eight of them.

**Read the gate.**

```bash
python generate_tables.py output/eval/huggyllama_llama_7b --report gates \
    --allocation_dir output/allocation_reports -o output/gates/huggyllama_llama_7b/gates_stage8.md
```

**Gate.** Reporting, and the headline figure of the results chapter.

---

## Stage 9: finalists on the full suite

**Purpose.** Everything the screening ratios deliberately skipped: c4, and the downstream benchmarks.

**Runs.** 9 evaluation-only, `args/experiments_stage9_benchmarks.json`. No compression — each entry
loads a `__CKPT_*__` checkpoint and evaluates
`wikitext,c4,arc_easy,hellaswag,openbookqa,piqa,winogrande,gsm8k,truthfulqa_gen|0`.

**Check the roster's `on disk` column first.** These runs cannot recompress, so a missing checkpoint
costs a compression run before this stage can start.

**What to check in it.**

- **wikitext against c4.** Every ranking in the grid was decided on wikitext alone. If the ordering
  holds on c4 the screening was sound; if it does not, that is a limitation the thesis must state and
  it bounds every earlier conclusion.
- Whether the perplexity ordering survives on the downstream tasks, and specifically whether the
  heterogeneous gain is visible there at all or is a perplexity-only effect.
- `merge_eval_results` means adding a task later cannot delete an earlier one, so a partial suite can
  be topped up rather than re-run.

**Preview.** None. These runs load an existing checkpoint and allocate nothing.

**Read the gate.**

```bash
python generate_tables.py output/eval/huggyllama_llama_7b --report gates \
    --allocation_dir output/allocation_reports -o output/gates/huggyllama_llama_7b/gates_stage9.md
```

**Gate.** Reporting. These are the thesis's headline tables.

---

## Stage 10: LoRA sequential update (RQ5)

**Purpose.** Whether the truncation-aware update closes the gap, and whether it closes more of it for
a heterogeneous allocation than a homogeneous one.

**Runs.** 4 update-only, `args/experiments_stage10_lora.json`: `--update_taw_only` on the two
homogeneous anchors and the two heterogeneous winners, `lora` method, `trainer` backend, Alpaca.

**What to check in it.** The gap closed on each arm, not the final perplexity. If the update closes
more of the homogeneous gap than the heterogeneous one, the two techniques compete; if it closes both
equally, they compose, and the thesis can claim the allocation and the update are independent
contributions.

**Preview.** None. The update reuses the allocation its checkpoint was built with.

**Read the gate.**

```bash
python generate_tables.py output/eval/huggyllama_llama_7b --report gates \
    --allocation_dir output/allocation_reports -o output/gates/huggyllama_llama_7b/gates_stage10.md
```

**Gate.** Reporting.

---

## Totals

In execution order.

| Stage | Runs | Evaluation | Offline preview |
|---|---|---|---|
| 0 offline | 0 | none | the stage itself |
| 1 anchors | 3 | wikitext | not needed |
| 1b replicates | 8 (+2 whitening passes) | wikitext | not needed |
| 2 score x grouping | 18 | wikitext | block screen, prune degenerate cells |
| 2b squared | 6 | wikitext | `map_distance` against the incumbent |
| 2c Schatten | 4 | wikitext | dispersion, and the block screen on `norm\|inf` |
| 3 cap | 10 | wikitext | cap binding, so no run repeats another |
| 3c outer level | 6 | wikitext | `layer_ratios`, to see the worst block leave layer 0 |
| 4 policies | 38 | wikitext | block screen per policy, `ratio_by_type` |
| 4c outer offset | 10 | wikitext | the ladder's usable range, which is not monotone |
| 4d temperature | 16 | wikitext | which temperatures survive the screen |
| 5 bypass | 36 | wikitext | catches infeasible budgets |
| 5b bypass x grouping | 2 | wikitext | same preview as 5 |
| 6 composite | 12 | wikitext | **gated** on the Spearman sign, sets the offset |
| 6b family budget | 5 | wikitext | feasibility only; the screen stands down |
| 7 cross-model | 18 | wikitext, blocked on whitening | rerun stage 0 per model |
| 8 ratio curve | 12 | wikitext | the shape claim, made offline |
| 9 benchmarks | 9 eval-only | full suite **and c4** | none |
| 10 LoRA | 4 update-only | wikitext | none |

**204 compression runs**, plus 13 evaluation-only or update-only, plus two extra whitening passes for
stage 1b and one per model for stage 7. At roughly 15 minutes a run that is about **51 GPU hours** for
the compressions.

GPU time is not the binding constraint, so these counts describe the grid rather than budget it: spend
runs wherever they resolve something. What the offline pass buys is not saved hours but the guarantee
that each run buys a distinct experiment — the block screen rejects a cell that is going to fail, and
the map-distance screen rejects a cell that repeats one already collected, which is how the five caps
this document once prescribed at ratio 0.2 turned out to be one run.
