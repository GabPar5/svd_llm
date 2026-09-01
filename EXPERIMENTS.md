# Experiment grid

The staged design behind the results of the thesis, and the source for thesis section 4.1.

Every stage is one axis swept around a fixed configuration, with a **gate** at the end that resolves
the placeholders of the next stage. The grid is deliberately not a cross product: the full cross for
a single model at a single ratio is already 4 groupings x 6 scores x 4 inner policies = 96 runs.

Stages 1 to 8 evaluate **wikitext perplexity only**. At roughly 15 minutes per run including that
evaluation, the 223 compression runs below cost about 56 GPU hours on the 7B models, so the binding
constraint on this grid is not time but whether each run answers a distinct question. Stage 7d is the
exception: its seven runs are on a 32B model and are rationed accordingly.

Whitening is assumed to be already cached under `output/whitening_matrices/<model>/v2/`, together
with its `spectra/` cache and `layer_importance.pt`. Every stage below reads that cache and never
recomputes it — the one exception is stage 1b, which exists precisely to build a second one.

**Execution order**, which is also the section order:
`0, 1, 1b, 2, 2b, 2c, 3, 3c, 4, 4c, 4d, 5, 5b, 6, 6b, 7, 7b, 7c, 7d, 7e, 7f, 8, 9, 10`.

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

> This conclusion held, and it is LLaMA-specific. Stage 7f re-measured it under the configuration the
> grid eventually elected and found the same thing -- `cap 0.6` costs 17% at ratio 0.5 -- while stages
> 7d and 7e found the opposite on both Qwen models, where the same bound is the single largest effect
> in the grid. Read this section as a result about multi-head attention rather than about the
> allocator, and do not carry "it stays at 0.9" onto a grouped-query model.

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

**It goes blind on a grouped-query model, and stage 7b is where that was found.** The MLP carries 87%
of a Qwen2.5-7B block against 67% of a LLaMA-7B one, so once the outer level has fixed each block's
budget there is almost nothing left for the family split to move at block level. Four Qwen runs
spanning **12.0 to 48.0** wikitext share a `max_block_ratio` of **0.2831 and a block 0 of 0.0413, to
the last digit**. The screen has exactly zero discriminative power there, and the companion below is
what replaces it.

**Stage 7b showed it is worse than blind: on a grouped-query model it is inverted.** Three of its runs
at ratio 0.2 share a block profile to the last digit — peak 0.283 at layer 16, block 0 at 0.041 — and
span **8.93 to 11.99**. Two at ratio 0.5 share a peak of 0.708 and span **37.00 to 151.15**. Worse,
`BLOCK_RATIO_DANGER = 0.60` flags the best run in the entire Qwen grid, the 37.00 sitting at 0.708,
and clears the homogeneous 67.16 it beats by 45%. On a grouped-query model the block screen must be
**replaced** by the KV rank screen, not read alongside it; on a multi-head model it stands as
described above.

### The KV rank screen: the companion for a grouped-query model

`figures/family_tail.csv` gives, per variant and per family, the parameter-weighted mean ratio, the
peak, the ratio as a multiple of the target, and the **retained rank fraction** — how much of its full
rank the family keeps. That last column is the one to read, because `--max_ratio` is not shape
invariant: a cap of 0.9 leaves a square matrix 5.0% of its rank, a 512x3584 projection 8.75% and an
18944x3584 one 8.4%, so comparing families on ratio alone compares three different amounts of
truncation.

The screen fires when a `k_proj` or `v_proj` keeps less than `KV_RANK_FRACTION_DANGER = 0.20` of its
rank, and **only on a grouped-query model**. Across the eight Qwen2.5-7B runs collected so far it
orders the field exactly (Spearman -1.000 on the five at ratio 0.2):

| target | wikitext | min rank fraction | on | max k/v ratio |
|---|---|---|---|---|
| 0.2 | **10.72** homogeneous | 0.400 | q_proj | 0.200 |
| 0.2 | **12.00** | 0.289 | q_proj | 0.650 |
| 0.2 | **34.13** | 0.141 | k_proj | 0.839 |
| 0.2 | **42.17** | 0.0875 | k_proj | 0.900 |
| 0.2 | **47.98** | 0.0875 | k_proj | 0.900 |
| 0.5 | **67.15** homogeneous | 0.250 | q_proj | 0.500 |
| 0.5 | **151.15** | 0.0500 | q_proj | 0.900 |
| 0.5 | **157.64** | 0.104 | q_proj | 0.881 |

Everything at or below 0.141 measured at least three times the homogeneous perplexity; everything at
or above 0.289 stayed within 12% of it. Five points at one budget is thin evidence and stage 7c adds
sixteen more, so read the threshold as provisional.

**Stage 7c re-fits it, and the fit holds.** Within one score at ratio 0.5 the ordering is exactly
monotone in retained KV rank: 0.0875 measured 53.60, 0.125 measured 45.46, 0.200 measured 44.63. Every
run below 0.15 at that budget again measured at least twice its anchor. Across scores it still does
not rank -- `truncation_rel` sits at 0.0875 and beats `eff_rank_rel` at the same value -- so the screen
stays a one-sided bound on damage and never a ranking.

**Stage 7b's seven runs sharpen it and bound what it can be used for.** As a one-sided *damage* bound
it survives: the two new runs below the band, 0.088 at ratio 0.5, measured 151.15 against a
homogeneous 67.16, and every run at or above 0.35 stayed within 12% of its anchor or beat it. As a
*ranking* it does not. At ratio 0.5 the ordering inverts outright — `--max_ratio 0.6` keeps 0.350 and
measures 51.65 while homogeneous keeps 0.4375 and measures 67.16 — because once k and v are restrained
the heterogeneous signal is worth more than the rank it costs them. Use the screen to reject a
configuration, never to rank two that both pass it.

**Why it is restricted to grouped-query attention, and this is the finding rather than a caveat.**
On LLaMA-7B the same screen at the same threshold fires on **61 of 83 runs at ratio 0.5, including the
best one** — 18.89 perplexity with `k_proj` down to 5% of its rank — while the runs it passes start at
22.09. Under MHA each query head owns its key and value, so a rank cut on `k_proj` damages one head
and stops there. Under GQA each KV head is read by `heads / kv_heads` query heads, seven of them on
Qwen2.5-7B, and the same cut reaches all of them. `kv_sharing_from_config` is what the tool branches
on, and on an MHA model it prints that the screen does not apply instead of warning.

**The screen and the knob are the same number.** `--min_rank_fraction f` caps every matrix at
`1 - f * (out + in) / max(out, in)`, which is exactly "keep at least `f` of full rank", so a variant
the screen names at 0.0875 is silenced by `--min_rank_fraction 0.2` and the offline report confirms it
before any GPU time is spent.

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
| `figures/family_tail.csv` | **the KV screen**: per family, the weighted mean and peak ratio, the ratio over target, and the retained rank fraction that the screen reads |
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
python run_experiments.py args/experiments_stage8_ratio_curve.json --skip_completed  # resume it
```

The runner merges `args/base_args.json` into each entry of the stage file, refuses to start if any
entry still holds an unresolved gate value, and continues to the next run when one fails rather than
aborting the queue.

**Resuming a stage.** Every invocation reports how many of its entries are already complete, and
`--skip_completed` acts on that. Complete means the run's evaluation JSON holds **every task that
entry asks for**, not merely that the file exists — a run collected on wikitext alone still executes
for an entry that has since been widened to the full suite, which is what makes it safe to add tasks
to a stage that has already run. Anything undecidable is executed rather than skipped: a run with no
evaluation to compare against (`--whitening_only`, or `--evaluate` off) and a run whose name does not
follow from its arguments both count as incomplete.

The flag is off by default on purpose. A run name encodes the configuration and nothing about the
code that produced it, so after changing the pipeline every name is unchanged and a whole stage would
look complete. Re-running a stage to pick up a code change is exactly the case where skipping is
wrong, and it is the common one in this repository.
The commands it prints are informational, not copy-paste ready: a score like `norm|1` holds a pipe a
shell would interpret. `subprocess` passes it as one argument, so runs are unaffected.

`args/` is tracked, so the grid is reproducible from the repository. **Never put `--hf_token` in one of
these files**: pass it on the command line, or export it in the environment.

## Frozen across every run

Changing any of these invalidates comparability with everything already collected.
`args/base_args.json` is authoritative.

| Setting | Value | Why frozen |
|---|---|---|
| Model | `huggyllama/llama-7b` | single model until stage 7; Qwen2.5-7B from 7 to 7c, Qwen2.5-32B at 7d |
| Version | `--run_v2` | documented as a limitation in thesis 5.2.1 |
| Precision | **per model**: `float16` for LLaMA-7B, `bfloat16` for both Qwen2.5 models | the precision each model was trained at. This is the one setting frozen per model rather than globally, so stages 7 to 7d are read as within-model gains and never as perplexities compared across the boundary. `base_args.json` holds `float16`, and every Qwen stage file overrides both `--model_dtype` and `--compressed_dtype` |
| Calibration | wikitext-2 train, `--max_length 2048` | |
| `--max_whitening_samples` | `256` | truncation and `norm\|p` scores scale with the token count, so this cannot move between runs (thesis 5.2) |
| `--seed` | `6363` | fixes the calibration sample; stage 1b is the only stage that varies it |
| Targets | all seven matrices, `--ratio_scope all` | stage 6b is the only stage that varies the selection, and `all` is what holds its budget comparable |
| `--max_ratio` | `0.9` | a guard rail, per the finding above; stage 3 sweeps it where it binds, and stage 7b sweeps it again on a grouped-query model where it is the dominant lever |
| `--outer_offset` | `1.05` | the incumbent arm; stage 4c sweeps it and `__BEST_OUTER_OFFSET__` carries the answer into `base_args.json` |
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
| `__BEST_OUTER_OFFSET__` | stage 4c | how hard Block Influence may reweight depth | `max_block_ratio` against the ladder, which is not monotone |
| `__BEST_BYPASS_EARLY__`, `__BEST_BYPASS_LATE__` | stage 5 | the bypass setting with the best gain over homogeneous | catches settings whose budget is infeasible once the exempt blocks are charged |
| `__CKPT_<ROLE>__` | stages 1 to 8 | a path under `output/models/huggyllama_llama_7b/` | nothing, these are outputs of runs |
| `__FINALIST{1,2,3}_*` | stages 2 to 8 | the three configurations worth another model, with the outer policy each needs | rerun stage 0 per model: the Spearman sign and the score-versus-depth shape are model properties |
| `__FINALIST1_SCORE_REL__` | stage 7, **derived** | the shape-invariant sibling of `__FINALIST1_SCORE__` | nothing, it is a spelling rather than a ranking |
| `__GQA_WINNER1_*` | stage 7c | the repaired configuration worth a second grouped-query model | the KV rank screen re-derived for that model |

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
itself, not a configuration inherited from a gate, so they are written out rather than placeheld. The
same rule keeps `type` and `decoder` literal in stage 7b, and `entropy_rel` and `truncation_rel`
literal in stage 7c: each is the axis its stage sweeps.

**`__FINALIST1_SCORE_REL__` is derived, not ranked.** Stage 7c pairs the score fix with the other two,
and pairing it with any score but the finalist's own would move two things at once. The stage 7 gate
therefore emits it by appending `_rel` to whatever won, and leaves it unresolved when the winner has
no shape-invariant spelling — a `norm|p` or a composite. If that happens, stage 7c's fix-1 rows have
no meaning as written and the stage needs rethinking rather than a substitution.

**`__BEST_OUTER_OFFSET__` follows `--max_ratio`.** Both are knobs a stage sweeps and every later stage
inherits, so the gate report resolves them and step 2 below writes them into `args/base_args.json`.
Stages 4c, 7b, 7c and 7d name the placeholder because they run before that file is updated; every
other stage reads the value from `base_args.json` and never mentions it.

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

**3. Run the stage**, `--dry_run` first. Add `--skip_completed` when resuming one that was
interrupted, or when entries were added to a stage already partly collected.

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
| 7b | `stage7b`, `stage7b_ratio0.5` | `figures/family_tail.csv`, `max_block_ratio` | which cap arms are feasible, and which cause the runs separate |
| 7c | `stage7c` | `figures/family_tail.csv` per score | whether a shape-invariant score lifts the KV rank off the danger band |
| 7d | `stage0` and `stage7c` for the second model | the KV screen re-derived at that sharing factor | whether the repair is expected to hold at a different `G` |
| 8 | `stage8` | the `<objective>_oracle_ratio` columns | the shape claim, made offline and never costing a run |
| 9, 10 | none | | both load existing checkpoints and allocate nothing |

Re-run stage 0 whenever the whitening cache changes.

---

## Stage 0: the offline pass

**Purpose.** Everything knowable without a GPU, plus the gate on stage 6. Costs seconds.

**Runs.** None. This stage *is* the preview.

**Preview.**

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

**Read the gate.** Stage 0 runs nothing, so it has no results table of its own. What it does have is
the offline preview every later gate attaches, so the command below is worth running once here to
confirm the stage directory is discovered before any GPU time is spent:

```bash
python generate_tables.py output/eval/huggyllama_llama_7b --report gates \
    --allocation_dir output/allocation_reports -o output/gates/huggyllama_llama_7b/gates_stage0.md
```

The **Offline preview of stage 0** section it prints is the same rho table as above, read back through
`--allocation_dir`. If it is missing, `--out_dir` did not match the name the gate looks for.

---

## Stage 1: anchors

**Runs.** 3, `args/experiments_stage1_anchors.json`: the dense model, and homogeneous at 0.2 and 0.5.

Every gain figure in the grid is measured against these two, and every checkpoint role that names a
homogeneous arm points at them, so they are kept on disk permanently.

**Preview.** A homogeneous run assigns the target ratio to every matrix, so this settles nothing
about the allocation. It is still the cheapest confirmation that the whitening cache the whole grid
depends on is readable and complete, which is worth having before the first GPU run:

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --sweep "compression_ratio=0.2,0.5" \
    --out_dir ./output/allocation_reports/stage1 --plots
```

Expect `ratio_std` at exactly 0 and `min_assigned_ratio` equal to `max_assigned_ratio`. Anything else
means a cached spectrum is missing and the run would silently allocate around the gap.

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

**Preview.** A change of seed changes the calibration draw rather than the allocator, so the
allocation is already covered by the stage 1 and stage 2 previews. What is worth previewing is the
*second cache*, since this stage is the only one that builds one and a half-written cache would show
up here as a changed allocation rather than as an error:

```bash
for seed in 7777 8888; do
    python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
        --whitening_mat_path "./output/whitening_matrices_seed${seed}" \
        --group_criterion decoder --score_metric eff_rank \
        --sweep "compression_ratio=0.2,0.5" \
        --out_dir "./output/allocation_reports/stage1b_seed${seed}"
done
```

Run it after each extra whitening pass and before the compression runs that consume it.

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

---

## Stage 5b: does the bypass conclusion depend on the grouping

**Purpose.** Bypassing means different things to the two groupings — under `decoder` it deletes whole
groups and `param_share` redistributes their budget between the survivors, under a flat grouping it
thins one pool — so a conclusion drawn at one may not transfer.

**Runs.** 2, `args/experiments_stage5b_bypass_grouping.json`: the winning bypass setting from stage 5,
re-run under `__BEST_FLAT_GROUPING__` at both ratios. The homogeneous arm allocates nothing and is
grouping-independent, which is why the probe is 2 runs rather than 4.

**What to check in it.** Whether the sign of the bypass gain survives the change of grouping. If it
does, stage 5's answer is a property of bypassing; if it flips, it is a property of `decoder` and the
thesis has to say so.

**Preview.** The same one stage 5 runs — the exit code and `checks` are what matter, because a
bypassed budget pushed onto a thinner pool is where infeasibility appears:

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --group_criterion __BEST_FLAT_GROUPING__ --score_metric __TOP1_SCORE__ \
    --inner_allocation __BEST_INNER__ \
    --bypass_early_layers __BEST_BYPASS_EARLY__ --bypass_late_layers __BEST_BYPASS_LATE__ \
    --sweep "compression_ratio=0.2,0.5" \
    --out_dir ./output/allocation_reports/stage5b --plots
```

**Read the gate.**

```bash
python generate_tables.py output/eval/huggyllama_llama_7b --report gates \
    --allocation_dir output/allocation_reports -o output/gates/huggyllama_llama_7b/gates_stage5b.md
```

**Gate.** Reporting, and it bounds how stage 5's answer may be phrased.

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

**Runs.** 9, `args/experiments_stage7_crossmodel.json`: on Qwen2.5-7B, a dense reference, two
homogeneous anchors, and the three finalists at both ratios.

**Qwen2.5-32B has moved to stage 7d.** This stage used to carry both models, 18 runs. Its answer is
negative, so the nine 32B runs would have spent the most expensive GPU time in the grid confirming
that configurations already known to lose also lose at a second scale. Stage 7d carries onto 32B only
what stage 7c repairs, which is both cheaper and a better question.

**The per-model anchors are not optional.** Without them the gain column has nothing to subtract and
the transfer claim reduces to comparing a Qwen perplexity against a LLaMA one. The pilot's stage file
omitted them.

**Precision is pinned per model, in the stage file.** Every entry sets `--model_dtype bfloat16` and
`--compressed_dtype bfloat16`, because that is the precision Qwen2.5 was trained at, while
`args/base_args.json` holds `float16` for LLaMA. Without the override the stage would inherit
`float16` and silently confound the transfer comparison with a precision change. This is the one
setting frozen per model rather than globally, and it is why the thesis reads this section as two
within-model gains rather than as one comparison of perplexities.

**Blocked on whitening.** The model needs its own `get_whitening_matrices` pass and its own stage 0.

**What to check in it.** Re-run stage 0 per model first and compare three things against LLaMA-7B: the
Spearman sign, the Block Influence profile across depth, and the score-versus-depth shape. The
finalists are only expected to transfer where those agree, and where they disagree the interesting
result is *which* of the three degrades.

**Preview.** Stage 0 again for the second model, which is the gate on whether the finalists are even
expected to transfer:

```bash
python allocation_report.py --model "Qwen/Qwen2.5-7B" --run_v2 \
    --sweep "compression_ratio=0.2,0.5" \
    --out_dir ./output/allocation_reports/stage0_Qwen2.5-7B --plots
```

**Read the gate.** Against the second model's own directory, since the anchors it subtracts are that
model's:

```bash
python generate_tables.py output/eval/Qwen_Qwen2.5_7B --report gates \
    --allocation_dir output/allocation_reports -o output/gates/Qwen_Qwen2.5_7B/gates_stage7.md
```

**Gate.** `__FINALIST{1,2,3}_*` are resolved on the *LLaMA* directory, by stages 2 to 6; this stage
spends them rather than setting them. What it does resolve is `__FINALIST1_SCORE_REL__`, the spelling
stage 7c needs, and whether stage 7b has to exist at all: it does only if the transfer fails.

---

## Stage 7b: why the finalists do not transfer to grouped-query attention

**Purpose.** Stage 7 answered its question and the answer was no. Every finalist *loses* to
homogeneous on Qwen2.5-7B, at both budgets and by a wide margin:

| wikitext | dense | 0.2 hom | 0.2 best het | 0.5 hom | 0.5 best het |
|---|---|---|---|---|---|
| LLaMA-7B | 5.68 | 7.79 | **7.47** (-4%) | 24.58 | **17.32** (-30%) |
| Qwen2.5-7B | 6.85 | 10.72 | **12.00** (+12%) | 67.15 | **151.15** (+125%) |

This stage establishes *why*, which is what turns a failed transfer into a result. Four causes were
identified offline, and the runs below separate them.

**Cause 1, the score is not scale invariant.** `eff_rank` is bounded by `min(out, in)`, `entropy` by
its log, `truncation` by the whitened Frobenius norm. All seven LLaMA families share `min(out, in) =
4096`, so raw scores are comparable there *by accident*. Under GQA `k_proj` and `v_proj` carry a
spectrum seven times shorter and score low for a purely dimensional reason:

| family | LLaMA eff_rank | ÷ len | Qwen eff_rank | ÷ len |
|---|---|---|---|---|
| q_proj | 1796 | 0.439 | 1475 | 0.412 |
| k_proj | 1692 (#1 most compressed) | 0.413 | **288** (#1) | 0.562 |
| v_proj | 2526 (#4) | 0.617 | **420** (#2) | **0.820** |
| o_proj | 2308 | 0.564 | 1819 | 0.508 |
| down_proj | 3279 (#7) | 0.800 | 2894 | 0.807 |

Length-normalized, Qwen's `v_proj` has the **flattest spectrum in the model** — by the metric's own
logic the matrix to protect hardest — and the allocator compresses it second hardest. The ordering
inverts entirely, and it inverts only where a group mixes shapes.

**Cause 2, the budget stops restraining k/v.** `k_proj` is 8.3% of a LLaMA block and **0.79% of a
Qwen block**. Since the allocation preserves `Σ pᵢ·rᵢ`, a matrix is restrained in proportion to its
parameter weight, so on Qwen sending k and v to the cap costs 1.5% of the block budget. This is an
amplifier rather than an independent cause: it is what lets the wrong score act unopposed. At target
0.2 LLaMA's allocator tops out at k = 0.55-0.63 while Qwen's saturates the 0.900 cap.

**Cause 3, GQA amplifies the damage.** 28 query heads read 4 KV heads. At r = 0.9 `k_proj` keeps rank
44 across 4 heads, 11 dimensions per head of 128, and every one of the 28 query heads reads that
space. Under MHA the damage is head-local, which is why LLaMA's *best* run at 0.5 takes `k_proj` to 5%
of its rank and still measures 18.89.

**Cause 4, the block screen is blind here.** Documented above; it is why this stage reports
`family_tail.csv` rather than `ratio_tail.csv`.

**Runs.** 7, `args/experiments_stage7b_gqa_diagnostics.json`, all on Qwen2.5-7B against the recorded
stage 7 anchors, every entry pinned to `bfloat16` for the reason stage 7 gives:

| # | run | isolates | prediction |
|---|---|---|---|
| 1 | `type` + `eff_rank` + `softmax_temp` @0.2 | cause 1 | near homogeneous |
| 2 | `decoder` + same @0.2 | matched control for 1 | degraded, k 0.433 / v 0.413 |
| 3 | k/v excluded, hierarchical finalist @0.2 | causes 2+3 | recovers most of the loss |
| 4, 5 | finalist at `--max_ratio 0.35`, `0.5` @0.2 | cause 2 | monotone recovery |
| 6, 7 | run 3 and `--max_ratio 0.6` @0.5 | budget dependence | — |

**Runs 1 and 2 are the decisive pair.** Same score, same policy, same budget, differing only in
whether a group ever compares two matrices of different shape. Under `type` every family mean is
exactly 0.200 by construction, so no cross-shape comparison happens and cause 1 is fully neutralized;
under `decoder` the same score drives k to 0.433 and v to 0.413.

**What the preview already settled.** `--max_ratio 0.35` at a 0.5 target is infeasible (30% budget
drift) and was raised to 0.6; `--max_ratio 0.5` at a 0.5 target collapses to exactly homogeneous and
was dropped. Neither cost a GPU minute.

### Collected

All seven landed. `KV rank` below is the smaller of the two retained rank fractions on `k_proj` and
`v_proj`, `peak block` the parameter-weighted worst block.

At ratio 0.2, against homogeneous **10.72** and dense 6.85:

| # | run | wikitext | vs hom | k mean | v mean | KV rank | peak block |
|---|---|---|---|---|---|---|---|
| 3 | **k and v excluded** | **8.93** | **-16.7%** | -- | -- | 1.000 | 0.283 |
| 4 | `--max_ratio 0.35` | 10.93 | +2.0% | 0.326 | 0.324 | 0.569 | 0.283 |
| 1 | `type` | 11.60 | +8.2% | 0.200 | 0.200 | 0.521 | 0.355 |
| 5 | `--max_ratio 0.5` | 11.66 | +8.8% | 0.418 | 0.406 | 0.438 | 0.283 |
| -- | stage 7 finalist | 11.99 | +11.9% | 0.440 | 0.419 | 0.306 | 0.283 |
| 2 | `decoder` | 12.88 | +20.1% | 0.433 | 0.413 | 0.446 | 0.200 |

At ratio 0.5, against homogeneous **67.16**:

| # | run | wikitext | vs hom | k mean | v mean | KV rank | peak block |
|---|---|---|---|---|---|---|---|
| 6 | **k and v excluded** | **37.00** | **-44.9%** | -- | -- | 1.000 | 0.708 |
| 7 | `--max_ratio 0.6` | 51.65 | -23.1% | 0.578 | 0.578 | 0.350 | 0.600 |
| -- | stage 7 finalist | 151.15 | +125.1% | 0.836 | 0.828 | 0.088 | 0.708 |

**Causes 2 and 3 dominate, and by more than the stage predicted.** Run 3 was written to "recover most
of the loss". It does not recover a loss, it produces the **first heterogeneous win on Qwen at either
budget**, and a large one. Both k/v-excluded runs are budget-exact against their anchors — sidecar
`actual_removed_params` equals `target_removed_params` to the last digit and `realized_overall_ratio`
reads 0.2000 and 0.5000 — so this is the same parameter removal pushed onto q, o and the MLP, not a
cheaper model. At 0.5 it does that while taking q to a mean of 0.805 and o to 0.746, ratios Qwen
absorbs without complaint. It is specifically k and v that are fragile, and `--ratio_scope all` is
what makes the comparison legitimate.

**Cause 1 is confirmed, and is not the whole story.** Runs 1 and 2 separate in the predicted direction
and by a wide margin: `decoder`, where a group compares matrices of different shape, loses 20.1%;
`type`, where by construction it never does, loses 8.2%. But run 1 was predicted to land *near*
homogeneous and it does not. Its block profile says why — L0 at 0.333 and L1 at 0.355 against a 0.200
target. Neutralizing the shape defect moved the failure onto the depth axis rather than removing it.
On Qwen at 0.2 heterogeneity costs along both axes, 2.4x more along the family one.

**The cap responds monotonically, which is cause 2's signature.** 11.99 -> 11.66 -> 10.93 as the cap
tightens from 0.9 through 0.5 to 0.35, and 151.15 -> 51.65 at the aggressive budget from that one
knob.

**Restraint alone recovers the gain at 0.5.** Run 7 beats homogeneous by 23%, which is the first sign
that the LLaMA result is reachable here: the heterogeneous signal was never the problem, the freedom
to act on it in the wrong place was.

**Gate.** Which cause dominates, which sets what stage 7c has to fix. Cause 1 is confirmed if run 1
lands near homogeneous while run 2 does not.

**What it answered.** Cause 1 confirmed in direction but not in size, causes 2 and 3 dominant. Stage
7c's fix 1 attacks cause 1 and is therefore necessary but, on these numbers, not sufficient on its
own; the offline preview in that stage says the same thing at 0.5.

**Read the gate.**

```bash
python generate_tables.py output/eval/Qwen_Qwen2.5_7B --report gates \
    --allocation_dir output/allocation_reports -o output/gates/Qwen_Qwen2.5_7B/gates_stage7b.md
```

Read the **Stage 7 gate: finalists** table in it, not a 7b-specific one: these runs are ranked beside
the finalists they diagnose, which is the comparison that matters. The `--max_ratio` rows are held out
of that table by design and reported in the note beneath it.

**Preview.**

One invocation per budget, because a cap is only meaningful above its target and
`--sweep` is a cartesian product: 0.35 against a 0.5 target is the infeasible cell this preview
exists to reject, and a rejection takes the exit code with it.

```bash
python allocation_report.py --model "Qwen/Qwen2.5-7B" --run_v2 \
    --group_criterion hierarchical --inner_allocation softmax_temp \
    --outer_allocation waterfill --outer_offset 1.05 --score_metric eff_rank \
    --compression_ratio 0.2 --sweep "max_ratio=0.9,0.5,0.35" \
    --out_dir ./output/allocation_reports/stage7b --plots

python allocation_report.py --model "Qwen/Qwen2.5-7B" --run_v2 \
    --group_criterion hierarchical --inner_allocation softmax_temp \
    --outer_allocation waterfill --outer_offset 1.05 --score_metric eff_rank \
    --compression_ratio 0.5 --sweep "max_ratio=0.9,0.6" \
    --out_dir ./output/allocation_reports/stage7b_ratio0.5 --plots
```

---

## Stage 7c: the three fixes, and whether each one earns its place

**Purpose.** Measure each fix alone and in combination, against the stage 7 anchors. Every one is off
by default, so a fix that does not pay is simply not used.

**Fix 1, `*_rel` scores (cause 1).** Six new metrics dividing out the ceiling the spectrum length
imposes. On LLaMA under `softmax_temp`, which min-max normalizes within the group, they are
**bit-identical to the raw form**: verified at `max_abs_diff = 0.000000` over all 224 matrices, since
a division by a constant survives min-max normalization and every LLaMA family shares its bound. On
Qwen the allocation moves to the shape LLaMA's winner has:

| family | `eff_rank` | `eff_rank_rel` |
|---|---|---|
| q_proj | 0.281 | 0.373 |
| k_proj | **0.440** | **0.278** |
| v_proj | **0.419** | **0.153** |
| o_proj | 0.249 | 0.307 |

**Fix 2, `--min_rank_fraction` (a fourth defect, found while building fix 1).** A scalar cap is not
shape invariant, so `--max_ratio 0.9` permits a square matrix down to 5.0% of its rank and a 512x3584
projection only to 8.75% — a 1.75x spread in what the same guard rail allows. `--min_rank_fraction f`
replaces it with `ratio <= 1 - f·(out + in)/max(out, in)`, which is one statement everywhere; `f =
0.05` reproduces `--max_ratio 0.9` on a square matrix exactly, and `0.0` is the old behaviour.

Once fix 1 is applied it binds `q_proj`, not k/v, because q is then the most rank-starved matrix in
the run: at ratio 0.5 it moves q from 0.797 to 0.691 at `f = 0.125` and to 0.568 at `f = 0.2` while k
and v barely move. It is swept at 0.5 only, because at a 0.2 target the peak allocation is 0.58 and no
ceiling loose enough to be a guard rail binds at all.

An earlier draft of this section concluded from that binding pattern that the floor was "the wrong
tool for restraining a KV projection". **The inference does not follow and the runs contradict it.**
Restraining `q_proj` is worth 16.7% on its own at ratio 0.5, which is the second largest effect in
this stage. What binds and what pays are different questions.

**Fix 3, `--head_block_svd` (cause 3).** `k_proj` and `v_proj` factored one KV head at a time against
the same whitening factor, so the reconstruction is block diagonal in head space and a head's rank
cannot be spent on another's. It is the cheaper parametrization — `heads·r·(in + head_dim)` against
`r_joint·(in + heads·head_dim)`, so 392 total rank against 358 at a 0.2 target — but block diagonality
is also a constraint, and on synthetic weights it loses 3-4% when heads share their input structure
and wins up to +35% when they do not and the truncation bites. **Whether it pays is empirical**, which
is why it is probed alone, with fix 1, and homogeneous.

An offline gate was written for it and **it was wrong**, which is worth keeping rather than deleting.
The argument: a projection whose heads all spanned one row space could not exceed an effective rank of
`head_dim`, and Qwen's `k_proj` reaches 2.25x `head_dim` and `v_proj` 3.28x, so the heads carry
substantially independent row spaces and the block form should win. The runs refute it in every
pairing. Effective rank above `head_dim` says the heads are not collinear; it does not say the joint
factorization was failing to exploit what they share. The test is not a predictor and no stage should
be gated on it.

**Runs.** 18, `args/experiments_stage7c_gqa_fixes.json`. Six at ratio 0.2 (each `*_rel` score, fix 3
alone, fix 1+3, and homogeneous + fix 3) and twelve at 0.5 (the same, the `--min_rank_fraction` sweep,
and the two cap arms the first sixteen runs argued for). Every one was checked feasible offline before
the stage was written. `entropy_rel` and
`truncation_rel` are written literally because they are the score ablation itself; every other
allocation dimension is placeheld on the stage 7 finalist, and fix 1's own score is spelled
`__FINALIST1_SCORE_REL__` so it can never name a different family of score from the run it repairs.

### What the preview predicts, before the stage runs

At **ratio 0.2** the fix works, and works by reproducing the configuration stage 7b found by hand.
`eff_rank_rel` moves the allocation to within 3% relative of run 3 on every family run 3 compresses:

| family | `eff_rank` (measured 11.99) | `eff_rank_rel` (offline) | 7b run 3, k/v excluded (measured **8.93**) |
|---|---|---|---|
| q_proj | 0.281 | **0.373** | 0.385 |
| o_proj | 0.249 | **0.307** | 0.316 |
| gate_proj | 0.215 | **0.226** | 0.230 |
| up_proj | 0.178 | **0.169** | 0.170 |
| down_proj | 0.170 | **0.152** | 0.154 |
| k_proj | 0.440 | 0.278 | 0 (dense) |
| v_proj | 0.419 | **0.153** | 0 (dense) |

The KV rank fraction rises from 0.306 to 0.472, clear of the screen. A score derived from an argument
about units, with nothing fitted, lands on the allocation a hand-built probe had to be told to find.

At **ratio 0.5 the fix is expected to be insufficient on its own, and the stage as written cannot show
why.** `eff_rank_rel` protects `v_proj` (0.828 -> 0.393) and leaves `k_proj` saturating the 0.900 cap,
so the KV rank fraction stays at **0.0875**, inside the danger band the screen was fitted on. Neither
of the other two fixes rescues it: `--min_rank_fraction` binds `q_proj` first, dropping it 0.797 ->
0.691 -> 0.568 while lifting KV rank only to 0.125 and then 0.200. What does work offline is the plain
cap — `eff_rank_rel` with `--max_ratio 0.6` gives KV rank 0.350 with q at 0.573, and stage 7b measured
the *unrepaired* score at that cap at 51.65 against a homogeneous 67.16.

**So read a negative at 0.5 as a statement about the arm list, not about fix 1.** The stage has no
`__FINALIST1_SCORE_REL__` x `--max_ratio` cell; adding one is the obvious follow-up if the 0.5 rows
disappoint, and it costs two runs.

**Homogeneous + `--head_block_svd` is the cleanest probe of fix 3**: one variable against a recorded
anchor, with no allocation involved at all.

**Preview.** The allocation half only — fix 3 changes how a ratio is realized, never the ratio, so it
is invisible to the offline report and its rows coincide with the plain ones:

```bash
python allocation_report.py --model "Qwen/Qwen2.5-7B" --run_v2 \
    --group_criterion hierarchical --inner_allocation softmax_temp \
    --outer_allocation waterfill --outer_offset 1.05 \
    --outer_offset __BEST_OUTER_OFFSET__ \
    --sweep "compression_ratio=0.2,0.5" \
    --sweep "score_metric=__FINALIST1_SCORE__,__FINALIST1_SCORE_REL__,entropy_rel,truncation_rel" \
    --sweep "min_rank_fraction=0.0,0.125,0.2" \
    --out_dir ./output/allocation_reports/stage7c --plots
```

Read `figures/family_tail.csv` per score: it is the KV rank screen, and on a grouped-query model it
replaces the block screen rather than supplementing it.

### Collected

Sixteen of the eighteen have landed. The verdict differs for each fix, which is what the stage was
built to produce.

**Fix 1 wins at both budgets, and flips the sign.** Every shape-invariant score beats the homogeneous
anchor; every raw one loses to it. This is the first configuration on Qwen where the *allocator* wins
rather than a hand-built target selection.

| | ratio 0.2 (hom 10.72) | ratio 0.5 (hom 67.16) |
|---|---|---|
| `eff_rank` -> `eff_rank_rel` | 12.00 (+11.9%) -> **9.76 (-9.0%)** | 151.15 (+125%) -> **53.60 (-20.2%)** |
| `entropy` -> `entropy_rel` | 12.19 (+13.7%) -> **9.80 (-8.6%)** | 149.25 (+122%) -> **57.88 (-13.8%)** |
| `truncation_rel` | **9.93 (-7.4%)** | **49.39 (-26.4%)** |

The measured allocation reproduces the offline preview above to three decimals, as it must — the
allocator is deterministic given its cache — so the offline report is now a trusted predictor on this
model rather than a plausibility check.

**Fix 2 pays at ratio 0.5, on top of fix 1.** On `eff_rank_rel`: 53.60 with no floor, **45.46** at
`f = 0.125`, **44.63** at `f = 0.2`. It also helps the raw score (151.15 -> 110.13) without getting it
anywhere near the anchor, so it is a second-order repair and not a substitute for fix 1.

**Fix 3 loses in every pairing it was given.** Seven matched pairs, and the ratio maps are identical
within each one, so this compares the *realization* with the allocation held fixed:

| budget | on top of | without `hb` | with `hb` | change |
|---|---|---|---|---|
| 0.2 | homogeneous | 10.72 | 11.17 | +4.1% |
| 0.2 | `eff_rank` | 12.00 | 13.82 | +15.2% |
| 0.2 | `eff_rank_rel` | 9.76 | 10.12 | +3.7% |
| 0.5 | homogeneous | 67.16 | **124.51** | **+85.4%** |
| 0.5 | `eff_rank` | 151.15 | 226.87 | +50.1% |
| 0.5 | `eff_rank_rel` | 53.60 | 59.01 | +10.1% |
| 0.5 | `eff_rank_rel` + `mrf 0.125` | 45.46 | 53.68 | +18.1% |

The damage grows with the budget, which is what a constraint does: block diagonality forbids spending
rank on directions several heads share, and that forbidding costs most when there is least rank to
spend. **`--head_block_svd` is a negative result. It stays off, no stage past this one runs it, and
the thesis keeps it as future work with the offline gate's failure recorded beside it.**

**What the repair does not reach.** Excluding k and v from compression entirely — stage 7b run 3,
same budget, redistributed onto q, o and the MLP — is still the best configuration on this model:

| budget | homogeneous | best repaired allocation | k and v excluded | repair captures |
|---|---|---|---|---|
| 0.2 | 10.72 | 9.76 | **8.93** | 54% |
| 0.5 | 67.16 | 44.63 | **37.00** | 75% |

So fix 1 and fix 2 together close half to three quarters of the distance to simply not touching the
KV projections. That is cause 3 measured rather than argued: the metric defect is repaired and the
architectural amplification is not, which is exactly the bound the thesis limitation on GQA states.

**The two cap arms exist because of these rows.** Within `eff_rank_rel` at 0.5 the ordering is monotone
in retained KV rank (0.0875 -> 53.60, 0.125 -> 45.46, 0.200 -> 44.63), and `--max_ratio 0.6` reaches
**KV 0.350 while leaving `q_proj` at 0.573**, which the floor cannot do at any setting: `f = 0.3` buys
KV 0.300 only by dropping q to 0.390. The cap restrains k and v directly where the floor restrains
whatever is most rank-starved, so the two are complementary rather than alternatives.

**Read the gate.**

```bash
python generate_tables.py output/eval/Qwen_Qwen2.5_7B --report gates \
    --allocation_dir output/allocation_reports -o output/gates/Qwen_Qwen2.5_7B/gates_stage7c.md
```

**Gate.** Whether heterogeneous allocation beats homogeneous on a grouped-query model once the score
is scale free, and which of the three fixes the thesis keeps. A fix that does not beat its own control
is reported as a negative result and left at its default. The gate ranks every run carrying a fix and
resolves `__GQA_WINNER1_*` from the best row that allocates, which is what stage 7d spends. A
homogeneous run carrying a fix is ranked beside them under the score `none (homogeneous)`: it records
no score of its own, and dropping it for that would discard the cleanest control the stage has.

The gate selects rows by the fix they carry — a `_rel` score, a non-zero `--min_rank_fraction`, or
`--head_block_svd` — rather than by the model, so it stays empty on the LLaMA directory instead of
answering stage 7d out of runs that never had the defect. The placeholders name the *allocation*
only: if a winning row also sets `--min_rank_fraction` or `--head_block_svd`, carry that flag into
stage 7d by hand when substituting, and the gate prints a note saying so.

---

## Stage 7d: does the repair hold at a second sharing factor

**Purpose.** Stage 7c repairs one grouped-query model. Whether that is a result about grouped-query
attention or about Qwen2.5-7B is decided here, and it is the question stage 7 was originally carrying
32B for.

**Why a second model settles something the first cannot.** The argument behind the shape-invariant
scores is written in terms of `G`, the number of query heads served by one key and value head, and it
predicts an effect that grows with `G`. Qwen2.5-7B has 28 query heads over 4 KV heads, `G = 7`.
Qwen2.5-32B has 40 over 8, **`G = 5`**. Two intermediate values turn the dependence on `G` from
derived into measured, which is exactly the gap the thesis limitation on GQA currently records.

**Runs.** 18, `args/experiments_stage7d_gqa_scale.json`, all Qwen2.5-32B and all pinned to
`bfloat16`. The first nine are the original transfer probe — a dense reference, homogeneous anchors at
0.2 and 0.5, and three arms at both ratios. The nine that follow were added once those came back, and
they answer a second question the first nine raised rather than repeating them: whether the *bounds*
transfer, and where along the ratio axis the heterogeneous arm stops paying.

**The third arm carries the other gate's score.** Stage 7 and stage 7c both elect a configuration
from Qwen2.5-7B and they agree on grouping, inner and outer policy but not on the score: 7c ranks the
repairs and elects `__GQA_WINNER1_SCORE__`, stage 7 ranks the finalists and elects
`__FINALIST1_SCORE__`. Running both at `G = 5` is what says whether that disagreement survives a
change of model, and it costs two runs rather than a third configuration. Note that the 7c gate fills
`__GQA_WINNER1_*` only: the pivot row below its winner is usually the same score under a different
bound, so a genuine second arm has to come from elsewhere, and stage 7 is where it comes from.

**The bounds are a swept axis here, so they are literal rather than placeheld.** `__GQA_WINNER1_*`
and `__BEST_OUTER_OFFSET__` still carry the allocation, exactly as before. `--max_ratio` and
`--min_rank_fraction` are written out as values, on the same principle stages 7b and 7c already
follow: a stage spells out the axis it sweeps and inherits everything else. This is also the trap
that cost the first nine runs their headline — the 7c gate resolves an *allocation*, and its own
note says so, but the elected row on Qwen2.5-7B also carried `--min_rank_fraction 0.2`, and the
placeholders could not express it. Every one of the first nine therefore ran at `--max_ratio 0.9`
with no floor, which is the configuration stage 7c had already retired.

**Two of the three arms differ in what they compress, not in how they allocate.** Stage 7c elects one
allocation, so `__GQA_WINNER1_*` fills both; a second set of allocation placeholders would have
carried the same configuration twice and spent 32B time on a knob. The second arm is instead the
target selection that stage 7c could not beat — `--compress_att_k false --compress_att_v false` — which
is the open question this stage can actually settle: whether an allocator that scores k and v
correctly still loses to one that refuses to compress them, at a second sharing factor. Every matrix
left after that exclusion has `min(out, in) = d_model`, so the shape defect cannot act inside it and
the winner's score is the right one to carry rather than a third configuration.

**The anchors are the point, not overhead.** 32B's homogeneous perplexity is not predictable from
7B's, so without them there is no gain to report and the stage says nothing.

**The nine added runs, and what each is for.** They are ordered in the file by what they decide, so a
stage stopped early still holds the cells the thesis depends on:

- **The bound ladder at 0.5** (`--max_ratio 0.6`, `--max_ratio 0.7`, `--min_rank_fraction 0.2`). On
  Qwen2.5-7B, on the repaired score at this budget, the ladder runs 53.60 at `cap 0.9`, 45.46 at
  `f = 0.125`, 44.63 at `f = 0.2`, 42.89 at `cap 0.7` and **39.21 at `cap 0.6`**. The cap is the
  stronger of the two bounds there, and neither was ever applied to 32B. Three runs decide whether
  that ordering is a property of the method or of the 7B checkpoint.
- **`--max_ratio 0.6` with k and v excluded**, the best-known combination of the two things that
  worked, which no model has yet been run at.
- **`--max_ratio 0.6` at ratio 0.2**, so the elected bound is priced at both ends of the curve rather
  than only where the unbounded arm failed.
- **Ratios 0.3 and 0.4, both arms.** With the two runs above, these close a four-point curve at one
  configuration (0.2, 0.3, 0.4, 0.5) against a homogeneous anchor at each. The collected nine put
  the sign change somewhere between 0.2 and 0.5 and cannot say where; this locates it. It is the only
  addition that is not a bound, and the reason it is worth 32B time is in the next paragraph.

**Why the crossover is the interesting quantity.** Normalizing each model by its own dense reference
and working in nats, the share of the compression damage that reallocation recovers orders the three
models the same way at both collected budgets, and the ordering is by how far the homogeneous anchor
has already degraded rather than by size or by `G`: at 0.2, LLaMA-7B recovers 12.0% at 1.37x dense,
32B 15.9% at 1.44x, and 7B 20.9% at 1.56x; at 0.5, 32B *loses* 21.2% at 3.66x, LLaMA recovers 12.2%
at 4.33x, and 7B 23.6% at 9.80x. If that reading is right the crossover is a property of the anchor,
not of the model, and it should sit at the budget where 32B's homogeneous arm reaches roughly the
damage LLaMA carries at 0.4 to 0.5. Ratios 0.3 and 0.4 are where that prediction is falsifiable.

### Collected

All eighteen, on wikitext. Gain is the homogeneous anchor of the same model minus the row:

| arm | 0.2 | gain | 0.5 | gain |
| --- | --- | --- | --- | --- |
| dense | 5.02 | -- | 5.02 | -- |
| homogeneous | 7.24 | -- | 18.35 | -- |
| `eff_rank`, raw | 7.24 | -0.00 | 40.80 | -22.46 |
| `entropy_rel` | 6.83 | +0.40 | 26.15 | -7.80 |
| `eff_rank_rel` | 6.83 | +0.41 | 24.14 | -5.79 |
| `eff_rank_rel`, `--min_rank_fraction 0.2` | -- | -- | 16.77 | +1.58 |
| `eff_rank_rel`, `--max_ratio 0.7` | -- | -- | 16.09 | +2.25 |
| `eff_rank_rel`, `--max_ratio 0.6` | **6.83** | **+0.41** | **15.28** | **+3.07** |
| `eff_rank_rel`, k and v excluded | 6.69 | +0.55 | 20.77 | -2.42 |
| `eff_rank_rel`, `--max_ratio 0.6`, k and v excluded | -- | -- | **13.78** | **+4.57** |

**The reversal was the bound, not the sharing factor.** It is gone. `--max_ratio 0.6` at ratio 0.5
turns the -5.79 the unbounded arm measured into **+3.07**, a 16.7% perplexity reduction against the
anchor, and the exclusion arm under the same bound reaches +4.57, or 24.9%. There is no longer a model
or a budget in this grid where the allocation loses. The sentence this stage was written to test --
that the gain reverses at a second sharing factor -- is false, and what was measured instead is that
the cap is not a tuning knob but a precondition: without it the repair is worse than not repairing at
all, and with it the repair transfers.

**The cap alone is worth more than everything else in the pipeline.** At ratio 0.5 the score, the
grouping, the inner policy and the outer offset together move 18.35 to 24.14, which is backwards by
5.79. Adding `--max_ratio 0.6` and changing nothing else moves 24.14 to 15.28, a gain of 8.86. The
same asymmetry holds on 7B, where the uncapped configuration is worth 13.56 against its anchor and the
cap on top of it is worth a further 14.39. Whatever §4 says about scores and policies, the bound is the
larger effect on both grouped-query models.

**The cap beats the floor, and the two are not interchangeable.** `--max_ratio 0.6` (15.28) against
`--min_rank_fraction 0.2` (16.77) at the same budget: the cap wins by 8.9%. This is the same ordering
7B reports (39.21 against 44.63, 12.1%), so the answer §3.3.1 promised is consistent across both
models, and it is the cap. The mechanism is the one the offline pass predicted -- the floor leaves the
depth profile bit-identical and works only inside blocks, while the cap reshapes depth as well -- and
`--max_ratio 0.7` landing between them (16.09) is the ladder behaving monotonically in the bound.

**The cap is inert at the low budget, measured rather than argued.** At ratio 0.2 the `cap 0.6` and
`cap 0.9` runs return **the same perplexity to four decimals** (6.8277), because no block reaches 0.6
to be clipped. The bound is a high-budget instrument, and a thesis table that reports it at 0.2 is
reporting the uncapped allocation under another name.

**The curve crosses nowhere: the allocation wins at all four budgets.** The four-point sweep at
`cap 0.6`, which is the shape RQ1 asks for on a second model:

| ratio | homogeneous | `cap 0.6` | gain | reduction |
| --- | --- | --- | --- | --- |
| 0.2 | 7.24 | 6.83 | +0.41 | 5.7% |
| 0.3 | 8.58 | 7.91 | +0.67 | 7.8% |
| 0.4 | 10.77 | 10.05 | +0.72 | 6.7% |
| 0.5 | 18.35 | 15.28 | +3.07 | 16.7% |

The gain is roughly flat from 0.2 to 0.4 and then more than doubles at 0.5, which is where the anchor
itself starts to fall apart. Read that as the allocation buying most where homogeneous compression has
least headroom left, not as a slope.

**The full suite agrees, and c4 agrees harder.** On the two budgets that carry the seven reported
tasks:

| ratio | arm | wikitext | c4 | mean of the five tasks |
| --- | --- | --- | --- | --- |
| 0.2 | homogeneous | 7.24 | 20.90 | 0.7038 |
| 0.2 | `cap 0.6` | **6.83** | **17.07** | **0.7129** |
| 0.5 | homogeneous | 18.35 | 165.89 | 0.4766 |
| 0.5 | `cap 0.6` | **15.28** | **90.41** | **0.4886** |

c4 improves by 18.3% at 0.2 and **45.5%** at 0.5, in both cases by more than wikitext does. Calibration
is wikitext-2 train, so the held-out corpus is where the allocation is *least* able to be overfitting
and it is where the gain is largest. That is the strongest single argument in the grid against the
objection that these numbers are a calibration artefact, and it wants stating in §4 explicitly.

**Perplexity was a sound screen.** Spearman between wikitext perplexity and the mean of the five
multiple-choice tasks is **-1.00** across the five 32B rows carrying both, and -0.97 and -0.96 on
LLaMA-7B (n = 11) and Qwen2.5-7B (n = 16). Every ranking in stages 1 to 8 was decided on wikitext
alone; this is the evidence that the decision would not have changed.

**One thing the screens do not do is predict the sign.** At ratio 0.5 and `cap 0.9` the KV screen
fires just as hard on Qwen2.5-7B (`k_proj` 0.0875) as on 32B (0.0833), and 7B still beat its anchor
while 32B did not. The screens price damage; whether damage loses is decided by how much headroom the
homogeneous arm had left. Report them that way rather than as a predictor of the outcome.

**Blocked on whitening.** 32B needs its own `get_whitening_matrices` pass, and it is the one model in
the grid that routes through the in-process JAX GPU `eigh` — `SOLVER_GPU_MAX_DIM = 32000` sends its
larger V2 eigendecompositions there. Build it in chunks with `--whitening_start_layer` /
`--whitening_end_layer` and `--whitening_only`; that pass, not the nine runs, is this stage's cost.

**What it answered.** The stage is complete. Every question it was written to ask is settled by the
table above except the last, which is methodological and still open:

- **The sign, and whether the reversal survives the bound.** Both yes. The elected configuration beats
  32B's own anchor at all four budgets under `cap 0.6`, and the reversal at `cap 0.9` was the bound.
  The cross-model claim is therefore one about bounds, not about scale.
- **Cap against floor.** The cap, by 8.9% here and 12.1% on 7B. Section 4.7 can report a single answer
  for both models rather than hedging, and the reason is the one §3.3.1 gives: the floor leaves the
  depth profile bit-identical (the block peak is 0.7077 on 7B at both `f = 0` and `f = 0.2`) and works
  only inside blocks, while the cap reshapes depth as well.
- **Where the curve crosses.** Nowhere. There is no crossover to report, which is a cleaner result than
  a sign flip: the allocation is ahead at 0.2, 0.3, 0.4 and 0.5, and the gain widens where the anchor
  degrades fastest.
- **Which arm wins.** The exclusion arm, at both budgets, exactly as on 7B: 6.69 against 6.83 at 0.2,
  and 13.78 against 15.28 at 0.5 under the cap. The ordering does **not** reverse at `G = 5`, which is
  evidence that what k and v exclusion buys is about k and v being shared at all rather than about one
  sharing factor.
- **The size against `G`.** The prediction holds, and is confounded. The gain is 16.7% at `G = 5`
  against 41.6% at `G = 7` at ratio 0.5, so the repair is worth less where the sharing factor is
  smaller, which is what the argument predicts. But `G` moves together with depth (64 blocks against
  28), with width, and above all with how much headroom the anchor has left -- 32B's homogeneous arm
  sits at 3.66x dense perplexity at ratio 0.5 where 7B's sits at 9.80x. The thesis should report the
  ordering and name the confound rather than attributing the size to `G`.
- **The KV rank screen, re-derived.** *Still open.* The 0.141 danger threshold was fitted on eight
  Qwen2.5-7B runs and is a model property, exactly like `BLOCK_RATIO_DANGER`. Re-fit it on the 32B rows
  rather than inheriting it, and say in the thesis which of the two numbers any claim rests on.

**Preview.** Stage 0 for the new model, then the elected allocation on it:

```bash
python allocation_report.py --model "Qwen/Qwen2.5-32B" --run_v2 \
    --sweep "compression_ratio=0.2,0.5" \
    --out_dir ./output/allocation_reports/stage0_Qwen2.5-32B --plots

python allocation_report.py --model "Qwen/Qwen2.5-32B" --run_v2 \
    --group_criterion __GQA_WINNER1_GROUPING__ --score_metric __GQA_WINNER1_SCORE__ \
    --inner_allocation __GQA_WINNER1_INNER__ --outer_allocation __GQA_WINNER1_OUTER__ \
    --outer_offset __BEST_OUTER_OFFSET__ \
    --sweep "compression_ratio=0.2,0.5" \
    --out_dir ./output/allocation_reports/stage7d --plots
```

The second one is the cheap gate: if `figures/family_tail.csv` shows the elected configuration landing
`k_proj` or `v_proj` back inside the danger band at `G = 5`, the repair is 7B-specific and that is
worth knowing before the whitening pass, not after. It did: at `--max_ratio 0.9` and no floor the
elected allocation leaves `k_proj` 0.0833 of its rank.

The two sweeps behind the added runs, which is where the `--max_ratio 0.6` choice comes from:

```bash
python allocation_report.py --model "Qwen/Qwen2.5-32B" --run_v2 \
    --compress_mlp --compress_att_q --compress_att_k --compress_att_v --compress_att_out \
    --compression_ratio 0.5 \
    --group_criterion __GQA_WINNER1_GROUPING__ --score_metric __GQA_WINNER1_SCORE__ \
    --inner_allocation __GQA_WINNER1_INNER__ --outer_allocation __GQA_WINNER1_OUTER__ \
    --outer_offset __BEST_OUTER_OFFSET__ \
    --sweep "max_ratio=0.6,0.7,0.9" \
    --out_dir ./output/allocation_reports/stage7d_cap --plots

python allocation_report.py --model "Qwen/Qwen2.5-32B" --run_v2 \
    --compress_mlp --compress_att_q --compress_att_k --compress_att_v --compress_att_out \
    --compression_ratio 0.5 \
    --group_criterion __GQA_WINNER1_GROUPING__ --score_metric __GQA_WINNER1_SCORE__ \
    --inner_allocation __GQA_WINNER1_INNER__ --outer_allocation __GQA_WINNER1_OUTER__ \
    --outer_offset __BEST_OUTER_OFFSET__ \
    --sweep "min_rank_fraction=0.0,0.125,0.2" \
    --out_dir ./output/allocation_reports/stage7d_floor --plots
```

Only `max_ratio=0.6` comes out of the first with neither screen firing. The second is what makes the
floor comparable: it moves `k_proj` from 0.0833 to 0.1250 to 0.2000 without touching the depth
profile, which is the distinction the thesis draws between the two bounds.

**The 32B whitening cache may be missing its version directory.** `allocation_report.py` resolves
spectra at `whitening_matrices/<model>/<v1|v2>/spectra`, and a cache moved off a scratch drive can
land at `whitening_matrices/<model>/spectra` with the `v2` level dropped, which reads as "no cached
spectra". Symlink rather than move, so the compression runs and the offline tool agree:

```bash
ln -sfnT "$PWD/output/whitening_matrices/Qwen_Qwen2.5_32B" \
    "$PWD/output/whitening_matrices/Qwen_Qwen2.5_32B/v2"
```

**Running only the added work.** `--skip_completed` skips a run whose evaluation JSON already holds
every task it asks for, so the stage can simply be re-run:

```bash
python run_experiments.py args/experiments_stage7d_gqa_scale.json --dry_run --skip_completed
python run_experiments.py args/experiments_stage7d_gqa_scale.json --skip_completed
```

All 18 are collected, so the flag now skips every one of them and the recipe is kept as the record of
how the stage was resumed: it left 12 invocations of the 18, skipping entries 4 to 9 and re-running 1
to 3, because the anchors and the dense reference had asked for the full suite while only wikitext was
collected for them -- coverage, rather than the file existing, is what the flag tests. Run it without
the flag first if you want the same report without acting on it.

**Read the gate.**

```bash
python generate_tables.py output/eval/Qwen_Qwen2.5_32B --report gates \
    --allocation_dir output/allocation_reports -o output/gates/Qwen_Qwen2.5_32B/gates_stage7d.md
```

**Gate.** Reporting, and it decides whether the thesis claims a result about grouped-query attention
or about one checkpoint.

---

## Stage 7e: closing the Qwen2.5-7B grid

**Purpose.** Three gaps that an audit of the collected Qwen2.5-7B runs turned up. None of them is a
new question; each is an axis the grid already argues about but priced at one budget, or on a
configuration the thesis has since retired.

**The reported rows had no downstream data.** This is the one that would have reached the thesis. Of
the eleven Qwen2.5-7B runs carrying the full suite, every heterogeneous one is a *pre-repair*
configuration -- raw `eff_rank`, raw `entropy`, `swift_pool`, `softmax_temp 0.5` -- and all of them
lose to the homogeneous anchor at both budgets. Every configuration the thesis elects had wikitext
alone. A downstream table built from that data would have shown heterogeneous allocation losing on
every task, which is the result stages 7b and 7c exist to correct. Five runs put the seven reported
tasks on the rows the thesis actually reports.

**The cap ladder existed only at ratio 0.5 on the repaired score.** At 0.2 the caps were swept on raw
`eff_rank` (0.35, 0.5, 0.9), before the shape-invariant scores existed. `--max_ratio 0.6` and `0.35`
on the elected score at 0.2 make the bound claim hold at both ends of the curve rather than one, and
0.6 is also what stage 7d prices on the 32B, so the two models are read on the same bound.

**Grouping and temperature were swept only at ratio 0.2, and on the retired score.** Both are re-run
on the elected score at both budgets. Four runs for `decoder` and `type`, two for `softmax_temp 0.5`.
`decoder` and `type` take the default `param_share` outer level and read no offset, because the
`waterfill` outer level needs a score per block and only `hierarchical` supplies one -- so a grouping
comparison necessarily moves the outer level too. That is not a confound to fix but a property of the
design: since `hierarchical` with `param_share` reproduces `decoder` exactly, the comparison is
precisely "does the outer water-fill earn its place".

**Runs.** 12, `args/experiments_stage7e_qwen7b_closeout.json`, all Qwen2.5-7B and all `bfloat16`. Two
of them re-run a configuration already collected in order to add the six missing tasks to its JSON;
`--skip_completed` reports those as incomplete and the rest as new, so the stage can be run as it
stands.

**It consumes `__GQA_WINNER1_*`, not `__FINALIST1_SCORE_REL__`.** Stage 7c's own placeholders have
drifted: `__FINALIST1_SCORE_REL__` derives from the stage 7 finalist, and now that the
shape-invariant scores have won stage 7 that finalist is itself `entropy_rel`, so the placeholder no
longer resolves to the `eff_rank_rel` the collected 7c runs used. Stage 7e therefore names the
configuration stage 7c *elected*, which is stable, and the same placeholder stage 7d carries.

**The two exclusion arms change score label without changing the allocation.** The collected k/v
exclusion runs on 7B used raw `eff_rank` while 32B used `eff_rank_rel`, and stage 7e re-runs them
under the `_rel` spelling so both models carry the same axis value. This costs nothing in
information: once k and v are excluded every remaining matrix has `min(d_out, d_in) = d_model`, so the
`_rel` divisor is constant inside each group and `softmax_temp`'s min-max normalization removes it
exactly. Checked offline across all 140 matrices at both budgets, the two allocations are identical to
the last bit, so the re-run should reproduce 8.93 and 37.00 and is worth having as a measured
confirmation of that argument.

### Collected

All twelve, on wikitext, against the anchors 10.72 at ratio 0.2 and 67.15 at 0.5:

| arm | 0.2 | gain | 0.5 | gain |
| --- | --- | --- | --- | --- |
| homogeneous | 10.72 | -- | 67.15 | -- |
| elected `eff_rank_rel` | 9.76 | +0.97 | 53.60 | +13.56 |
| `--max_ratio 0.6` | 9.76 | +0.97 | **39.21** | **+27.95** |
| `--max_ratio 0.35` | 9.78 | +0.94 | -- | -- |
| `--min_rank_fraction 0.2` | -- | -- | 44.63 | +22.52 |
| `--softmax_temp 0.5` | 9.95 | +0.77 | 105.70 | -38.54 |
| `decoder` | 10.13 | +0.60 | 80.71 | -13.56 |
| `type` | 11.60 | -0.88 | 1540.43 | -1473.27 |
| k and v excluded | **8.93** | **+1.80** | **37.00** | **+30.15** |

**The three arguments it was built to confirm all held, one of them to the last bit.**

- **The cap is inert at the low budget.** `cap 0.6` and `cap 0.9` return the same 9.7555 at ratio 0.2,
  and `cap 0.35` is very slightly *worse* (9.78) -- tight enough to bind, and binding costs. The bound
  belongs to the high-budget regime on this model too.
- **The `_rel` re-spelling of the exclusion arm changed nothing.** It reproduced 8.9277 and 36.9999
  exactly, matching the raw-score runs to four decimals at both budgets. Once k and v are excluded every
  remaining matrix has `min(d_out, d_in) = d_model`, so the `_rel` divisor is constant inside each group
  and `softmax_temp`'s min-max normalization removes it. The offline check said the allocations were
  identical across all 140 matrices; the measurement agrees. The same equivalence shows up a third time
  under `type` grouping at 0.2, where `eff_rank` and `eff_rank_rel` both return 11.5994.
- **The outer water-fill earns its place.** `hierarchical` with `waterfill` (53.60 at ratio 0.5) against
  `decoder` with `param_share` (80.71) and `type` (1540.43). Since `hierarchical` with `param_share`
  reproduces `decoder` exactly, that gap *is* the outer level, and at the high budget it is the
  difference between beating the anchor by 20% and losing to it by 20%.

**The suite rows now describe the method.** Adding the seven tasks to the elected rows was the point of
the stage, and it changes the sign of what a downstream table says:

| ratio | arm | wikitext | c4 | mean of the five tasks |
| --- | --- | --- | --- | --- |
| 0.2 | homogeneous | 10.72 | 40.85 | 0.5817 |
| 0.2 | `cap 0.6` | 9.76 | 32.56 | 0.5984 |
| 0.2 | k and v excluded | **8.93** | **22.99** | **0.6458** |
| 0.5 | homogeneous | 67.15 | 407.31 | 0.3721 |
| 0.5 | `cap 0.6` | 39.21 | 230.72 | 0.3872 |
| 0.5 | k and v excluded | **37.00** | **187.48** | **0.4163** |

The exclusion arm at ratio 0.2 recovers **48.7%** of the accuracy the homogeneous anchor gives up
against dense (0.5817 to 0.6458, dense 0.7134), against 12.7% for the full-matrix arm. As on 32B the c4
gain exceeds the wikitext gain in every row, by 20.3% against 9.0% at 0.2 and 43.4% against 41.6% at
0.5.

**Preview.** The cap ladder at the low budget, which is the only genuinely new allocation here:

```bash
python allocation_report.py --model "Qwen/Qwen2.5-7B" --run_v2 \
    --compress_mlp --compress_att_q --compress_att_k --compress_att_v --compress_att_out \
    --compression_ratio 0.2 \
    --group_criterion __GQA_WINNER1_GROUPING__ --score_metric __GQA_WINNER1_SCORE__ \
    --inner_allocation __GQA_WINNER1_INNER__ --outer_allocation __GQA_WINNER1_OUTER__ \
    --outer_offset __BEST_OUTER_OFFSET__ \
    --sweep "max_ratio=0.35,0.6,0.9" \
    --out_dir ./output/allocation_reports/stage7e_cap --plots
```

**Read the gate.**

```bash
python generate_tables.py output/eval/Qwen_Qwen2.5_7B --report gates \
    --allocation_dir output/allocation_reports -o output/gates/Qwen_Qwen2.5_7B/gates_stage7e.md
```

**Gate.** Reporting. Nothing downstream waits on it, which is why it runs after stage 7d rather than
before: it closes the 7B grid rather than deciding anything the 32B needs.

---

## Stage 7f: closing the LLaMA-7B grid

**Purpose.** One asymmetry and one hole, both artefacts of the order the grid was run in: the bound
ladder was discovered on the two Qwen models, after LLaMA-7B had already been swept.

**The cap has never been tested on LLaMA under the configuration the thesis elects.** Both Qwen models
resolve `--max_ratio` to **0.6** from their own stage 3 gate, and it is the largest single effect in the
grid -- worth 14.39 perplexity on 7B and 8.86 on 32B at ratio 0.5, in both cases more than the score,
the grouping, the inner policy and the outer offset achieve between them. LLaMA's gate reports
`--max_ratio 0.9`, *provisional, one candidate*: its cap sweep ran under `decoder` and `global` with the
default `waterfill` inner level, and in that configuration the cap cannot act. Offline, under `decoder`
with `param_share` outside it, every LLaMA block sits at exactly 0.500 at ratio 0.5, so the block peak
never approaches 0.6 -- which makes the collected `cap0.6` run (24.03) matching its uncapped twin
(23.91) the expected null rather than evidence against the bound. Under the elected `hierarchical` with
`waterfill` outside it the same budget spreads the blocks to **0.814 (L27)**, past
`BLOCK_RATIO_DANGER`. That is the regime the cap exists for, and it has never been measured on an MHA
model.

**The elected finalist at ratio 0.2 has no downstream data.** The defect stage 7e found on 7B, one row
wide. Of the eleven LLaMA runs carrying the full suite, the heterogeneous ones at ratio 0.2 are
`swift_pool`, `byp0-1`, `global` and a composite; the elected
`hierarchical / eff_rank / softmax_temp / waterfill / 1.05` has wikitext alone. At 0.5 it does have the
suite, so the c4 and accuracy columns of the thesis's LLaMA table currently exist at one budget out of
the two it reports. The checkpoint is on disk, so this costs an evaluation and no compression.

**LLaMA needs no shape-invariant score, and that is a theorem rather than an omission.** Every matrix
in LLaMA-7B has `min(d_out, d_in) = 4096`: `q`, `k`, `v` and `o` are square at `d_model`, and
`gate`/`up`/`down` are `11008 x 4096` or its transpose. A spectral score's units carry `min(d_out,
d_in)`, so dividing it out is a global constant rescale, which `softmax_temp` removes anyway. The
`_rel` family that stages 7b and 7c exist to introduce is provably a no-op here -- the reason LLaMA has
no `_rel` runs and needs none. Say it that way in §4.7 rather than leaving the model looking
under-swept.

**Runs.** 3, `args/experiments_stage7f_llama_closeout.json`: one evaluation-only and two compressions,
all LLaMA-7B and all `float16` inherited from `args/base_args.json`.

**It names the allocation through `__FINALIST1_*` but writes the offset as a literal.** LLaMA's gate
resolves the four finalist placeholders to `hierarchical / eff_rank / softmax_temp / waterfill`, so the
file stays self-describing. `--outer_offset` is the literal `1.05` instead of `__BEST_OUTER_OFFSET__`,
because stage 4c leaves that placeholder unresolved on LLaMA -- the gate prints the whole ladder but
elects nothing, having priced the offsets across eight budgets without aggregating them -- and
`run_experiments.py` refuses to start while any placeholder is unresolved. 1.05 is what the elected run
itself carries.

**Why ratio 0.4 as well as 0.5.** 0.4 is the first budget at which the block screen fires on LLaMA
under the elected configuration, and it is the cheapest place to watch the cap begin to act. The block
peak across the curve at `cap 0.9`, from the preview below: 0.326 at ratio 0.2, 0.488 at 0.3, **0.651**
at 0.4, **0.814** at 0.5, then 0.900 at 0.6, 0.7 and 0.8. The 0.4 entry asks for wikitext only; only
the 0.5 entry carries the suite, because 0.5 is the budget the thesis reports.

**The 0.4 entry did not isolate the cap when it was written, and stage 8 closed that.** It carries
`--outer_offset 1.05` to match the elected configuration, while the uncapped run collected at ratio 0.4
belonged to the stage 8 ratio curve and sat at the default 1.5, so the pair moved two variables at
once. Re-running the curve's six budgets at 1.05 supplied the missing arm, and the 0.4 comparison is
now single-variable like the 0.5 one.

**What it deliberately does not cover.** Ratios 0.6 to 0.8. There the uncapped allocation is pinned
against `--max_ratio` itself -- the block peak is exactly 0.900 at all three, at L24, L20 and L16 -- and
the collected stage 8 gains are correspondingly non-monotone: 29.5% at ratio 0.5, then 14.7% at 0.6 and
12.7% at 0.7 before 45.6% at 0.8. Three more capped runs would very likely make that curve read
cleanly, and they are worth having if time appears. They are not in this file because RQ1's shape claim
is already made and the budgets the thesis reports are 0.2 and 0.5.

### Collected

All three. Ratio 0.5 is the clean test, both arms held at `--outer_offset 1.05`:

| ratio | arm | wikitext | c4 | mean of the five tasks |
| --- | --- | --- | --- | --- |
| 0.2 | homogeneous | 7.79 | 15.99 | 0.5837 |
| 0.2 | elected, `cap 0.9` | **7.56** | **14.62** | **0.5991** |
| 0.5 | homogeneous | 24.58 | 134.76 | 0.4119 |
| 0.5 | elected, `cap 0.9` | **17.32** | **65.97** | **0.4523** |
| 0.5 | `cap 0.6` | 20.32 | 94.38 | 0.4232 |

**The cap does not transfer to multi-head attention. It costs 17% there.** This is the branch the
questions below were written against, and it fired harder than the wording allowed for: `cap 0.6` does
not merely gain less on LLaMA, it *loses*, on all three measures at once -- 17.3% on wikitext, 43.1% on
c4, 2.9 points of accuracy -- against the same allocation left uncapped. Both Qwen models elect the
bound from their own stage 3 gate and gain 14.39 and 8.86 perplexity from it; LLaMA gives up 3.00. The
bound is a grouped-query instrument, and §4.7 has to say so in those terms rather than reporting
`--max_ratio` as a hyperparameter that happened to be tuned per model. That is the stronger claim of
the two, and it puts the cap and the k/v exclusion arm in one story instead of two: both stop the
allocator over-spending on attention that is shared, one bluntly and one exactly, which is also why
exclusion beats the cap on both Qwen models and why neither belongs on LLaMA.

**The allocation-level diagnostics do not explain the sign, and should not be made to.** The cap does
the same thing to both architectures. It lifts the minimum retained rank fraction of the attention
projections out of the 0.05 to 0.09 band and into 0.20 to 0.35, and pushes the freed damage onto the
MLP: `mlp.down_proj` moves from a mean ratio
of 0.362 to 0.444 on LLaMA and 0.389 to 0.448 on Qwen2.5-7B, while `k_proj` is relieved from 0.748 to
0.580 and 0.689 to 0.551. Same intervention, opposite outcome. Report the boundary as an empirical
fact -- grouped-query attention cannot survive being run to the cap and multi-head attention can --
and note that this is the second time in the grid the screens priced damage correctly without
predicting whether it loses, the first being the 7B-against-32B comparison recorded under stage 7d.

**The 0.4 point now measures the same thing, and the cost scales with the clip.** Once stage 8 supplied
the uncapped arm at offset 1.05, ratio 0.4 reads 11.04 uncapped against 11.35 at `cap 0.6`: the cap
costs 2.8% there against 17.3% at ratio 0.5. That ordering is what the block peak predicts, since the
peak the cap has to clip is 0.651 at ratio 0.4 and 0.814 at 0.5 -- a 0.05 clip against a 0.21 one. So
on LLaMA the cap never helps, and how much it hurts is set by how far it has to pull the deepest block
back. Two budgets, both negative, is a firmer statement than the single point this stage was designed
around.

**The rest of the stage came out as designed.** The ratio 0.2 row now carries c4 and the five tasks,
and the c4 gain exceeds the wikitext gain here too, 8.6% against 3.0%, so that pattern holds on all
three models and on both architectures.

**The elected row is not the wikitext argmax at ratio 0.2, and it is the accuracy argmax.** Nine of the
74 collected LLaMA runs at that budget beat its 7.5625 on wikitext, mostly the low-temperature and
bypass arms. Only two of those nine carry the full suite -- `swift_pool` at 7.4967 and `byp0-1` at
7.5605 -- and the elected row beats both on the five tasks, 0.5991 against 0.5975 and 0.5859. Two pairs
is thin, so do not build a claim about wikitext disagreeing with accuracy on it. What it does support
is narrower and useful: a per-budget wikitext argmax is not automatically the accuracy argmax, which is
an argument for the gate's mean-rank-across-budgets choice over picking a winner at each ratio, and a
reason the seven remaining wikitext-only rows above it would need the suite before any of them could
displace the reported configuration.

**Preview.** Where the screen fires across the curve, then the cap ladder at the reported budget:

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --compress_mlp --compress_att_q --compress_att_k --compress_att_v --compress_att_out \
    --group_criterion __FINALIST1_GROUPING__ --score_metric __FINALIST1_SCORE__ \
    --inner_allocation __FINALIST1_INNER__ --outer_allocation __FINALIST1_OUTER__ \
    --outer_offset 1.05 --max_ratio 0.9 \
    --sweep "compression_ratio=0.2,0.3,0.4,0.5,0.6,0.7,0.8" \
    --out_dir ./output/allocation_reports/stage7f_screen --plots

python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --compress_mlp --compress_att_q --compress_att_k --compress_att_v --compress_att_out \
    --compression_ratio 0.5 \
    --group_criterion __FINALIST1_GROUPING__ --score_metric __FINALIST1_SCORE__ \
    --inner_allocation __FINALIST1_INNER__ --outer_allocation __FINALIST1_OUTER__ \
    --outer_offset 1.05 \
    --sweep "max_ratio=0.6,0.7,0.9" \
    --out_dir ./output/allocation_reports/stage7f_cap --plots
```

Both were run before the stage was written. `cap 0.6` is the only rung of 0.6, 0.7 and 0.9 that clears
the block screen -- it lands the peak at 0.600 (L18) against 0.700 (L23) and 0.814 (L27) -- and it wins
every offline objective except `influence_tail`. The KV rank screen does not apply: under MHA every
query head owns its key and value, so the damage a truncated `k_proj` does stays inside one head. That
is the whole reason this stage is one bound and not the three fixes of stage 7c.

**What it answered.**

- **Does `cap 0.6` clear LLaMA's anchor at ratio 0.5?** Yes (20.32 against 24.58), but that is the
  wrong comparison. Against the uncapped arm it *loses*, 20.32 against 17.32, and it loses at ratio 0.4
  too once stage 8 supplies a matched arm there, 11.35 against 11.04. The bound is not about depth: the
  block screen is defined on per-block ratios and never mentions heads, yet the bound it motivates pays
  only where k and v are shared. Whatever the screen measures, it is not what makes the cap worth
  applying.
- **The ratio 0.2 suite row.** Collected. The elected row is the accuracy argmax at that budget
  without being the wikitext argmax, which is recorded above with the caveat it needs.
- **Does the c4 gain exceed the wikitext gain?** Yes, 8.6% against 3.0%. The pattern now holds on all
  three models, so the calibration-artefact objection is answered across both architectures rather
  than on grouped-query models alone.

**Read the gate.**

```bash
python generate_tables.py output/eval/huggyllama_llama_7b --report gates \
    --allocation_dir output/allocation_reports -o output/gates/huggyllama_llama_7b/gates_stage7f.md
```

**Gate.** Reporting, and it decided both things it was asked. §4.7 states the cap as a property of
grouped-query attention, not of the allocator. And stage 10's heterogeneous arms stay where
`__CKPT_HET_*__` already points: the uncapped run is LLaMA's best at both budgets, so the repointing
that stage warns about is not needed on this model.

---

## Stage 8: the ratio curve (RQ1)

**Purpose.** How the heterogeneous gain depends on the target ratio. The pilot's two budgets already
showed the gain is not monotone in any simple way — the aggressive scores win at 0.2 and collapse at
0.5 — so the curve is the RQ1 answer rather than a single number.

**Runs.** 18, `args/experiments_stage8_ratio_curve.json`: {0.1, 0.3, 0.4, 0.6, 0.7, 0.8} x
{homogeneous, the winner}, then the same six heterogeneous budgets again at `--outer_offset 1.05`. With
0.2 and 0.5 from stages 1 and 3c this is an eight-point curve traced twice, once per offset.

This replaces the pilot's two-point onset probe. At fifteen minutes a run the full curve costs less
than four hours and answers a question the thesis asks in its first chapter.

**The curve exists at two offsets, and the thesis reports the elected one.** The first twelve runs name
no `--outer_offset` and no `--softmax_temp`, so those six ratios ran at 1.5 and 1.0 while stages 3c and
4c had swept both around the same allocation at 0.2 and 0.5 only -- leaving two budgets with a dozen
runs sharing a grouping, a score and an inner policy where the other six had one. The six added runs
close that: every budget now has the arm the rest of the grid is measured on, and the mixed-arm curve
the gate used to hold at defaults is superseded by a curve that needs no holding at all.

**The tie-break that made this visible is worth remembering.** Before the arms were separated,
`build_pivot`'s tie-break (first by run name) let `ooff1.05` stand in for ratio 0.5 alone, and the
printed curve read `+1.72, +7.26, +8.45` across 0.4, 0.5, 0.6 — a kink at exactly the budget the rest
of the grid is measured on, produced by nothing but alphabetical order. Held at one arm it read `+1.72,
+4.02, +8.45`. The gate still holds `max_ratio`, `outer_offset` and `softmax_temp` at their dominant
values and carries `outer_allocation`, `outer_offset` and `softmax_temp` in the configuration column,
so the arm is named in the table rather than inferred. **Read the `held at` notes**: a curve whose
configuration changes between rows is a different claim from a curve that does not, and the notes are
what tell the two apart. The one switch that survives is real — `swift_pool` wins at 0.2 where
`softmax_temp` wins everywhere else — and that is what the note beneath the table is for.

### Collected

Eight budgets, both offsets, on wikitext. `peak` is the offline block maximum at `--outer_offset 1.05`,
which is what the gain turns out to track:

| ratio | peak | homogeneous | het, offset 1.05 | gain | het, offset 1.5 | gain |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 0.163 | 7.01 | 6.94 | 1.0% | 6.94 | 1.1% |
| 0.2 | 0.326 | 7.79 | 7.56 | 3.0% | 7.59 | 2.7% |
| 0.3 | 0.488 | 9.36 | 8.62 | 7.9% | 8.79 | 6.1% |
| 0.4 | 0.651 | 13.47 | **11.04** | **18.1%** | 11.75 | 12.7% |
| 0.5 | 0.814 | 24.58 | **17.32** | **29.5%** | 20.55 | 16.4% |
| 0.6 | 0.900 | 57.35 | **40.03** | **30.2%** | 48.90 | 14.7% |
| 0.7 | 0.900 | 178.82 | 133.19 | 25.5% | 156.07 | 12.7% |
| 0.8 | 0.900 | 841.88 | 475.00 | 43.6% | 457.66 | 45.6% |

**The gain rises with the budget, peaks near 0.5 to 0.6, and the shape is the block peak's.** The
outer water-fill hands the deepest block a constant **1.628x** the target ratio -- 0.163, 0.326, 0.488,
0.651, 0.814 are exactly `1.628 r` -- until `--max_ratio 0.9` intervenes at ratio 0.55 and pins it.
Over the free stretch the gain tracks that peak monotonically, 1.0 to 3.0 to 7.9 to 18.1 to 29.5, and
once the peak is pinned the gain flattens and then falls: 30.2 at 0.6, 25.5 at 0.7. So the screen does
explain RQ1, but only where it is free to move, and the answer to "are the two curves mirror images" is
yes up to the point the cap starts clipping and no after it. That is a sharper claim than the gain
table alone and it is the one §4 should make.

**Ratio 0.8 is not part of the curve.** 841.88 against 475.00 is a comparison between two destroyed
models -- 148x and 84x dense perplexity -- and the 43.6% that arithmetic produces means nothing about
allocation. It is also the one budget where offset 1.5 wins. Report the curve over 0.1 to 0.7 and show
0.8 only to say where the method stops applying.

**The offset is worth more the higher the budget, which is why the old curve understated the method.**
Moving from 1.5 to 1.05 is worth -0.1% at ratio 0.1, then 0.3%, 1.9%, 6.1%, 15.8%, 18.1%, 14.7%, and
-3.8% at 0.8. The default-knob curve therefore reported a gain that peaked at 16.4% where the elected
arm reaches 30.2%. Its dip at 0.6 was the offset and not the allocation -- 14.7% becomes 30.2%, so the
gain rises there rather than falling -- while the decline at 0.7 survives the change and is real. Any
figure built from the pre-existing twelve runs alone would have made the method look weakest at exactly
the budget where it is strongest.

**What it answered.**

- **Where the gain peaks.** Between ratio 0.5 and 0.6, and the pilot's hypothesis was half right: there
  is nothing to gain at 0.1 because nothing is near collapse, but at 0.8 the gain is large and
  meaningless rather than small. The correct statement is that the gain grows while the allocation has
  a block it can still protect and collapses as a *measurement* once both arms are broken.
- **`max_block_ratio` against the gain.** Mirror images over 0.1 to 0.5, decoupled from 0.6 up, with
  `--max_ratio 0.9` as the exact point of decoupling. The screen explains RQ1 on the free stretch.
- **The `<objective>_oracle_ratio` columns**, which make the shape claim offline and cost nothing.

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

**The two generation tasks are collected but not reported, and are not run again.** Stage 7d and any
later full-suite pass use the seven-task suite without `gsm8k` and `truthfulqa_gen`. They are
`generate_until` tasks, so they decode autoregressively and dominate the cost of a suite on a 32B
model, and what they measure here does not survive inspection:

- `gsm8k` **strict-match is 0.0000 in every compressed run of both models**, and 0.0000 for dense
  LLaMA-7B too. Dense Qwen2.5-7B reaches 0.0083 with a standard error of 0.0025, which is noise. These
  are base models evaluated zero-shot, so that is the expected result rather than a surprise.
- `gsm8k` **flexible-extract is non-zero and points the wrong way**: on Qwen2.5-7B at ratio 0.2 the
  homogeneous run scores 0.0485 against 0.0227 and 0.0053 for the heterogeneous runs that beat it on
  every other measure. It regex-matches any number in the output, so on a degraded model it rewards
  producing digits at all. A metric that inverts the ordering is worse than a missing one.
- `truthfulqa_gen`'s **`bleu_acc` inverts on Qwen**: dense scores 0.0747 and *every* compressed run
  scores higher, up to 0.3647. It compares BLEU against a true answer with BLEU against a false one,
  and when both collapse the comparison is close to a coin flip. `bleu_max` does fall monotonically
  (3.62 to 1.12 to 0.27), but it is a text-overlap score on a base model and the thesis argues nothing
  about it.

The five multiple-choice tasks behave: `arc_easy`, `hellaswag`, `openbookqa`, `piqa` and `winogrande`
all fall monotonically with perplexity on both models. Those plus `wikitext` and `c4` are the seven
the thesis reports, identically for all three models, which is what keeps a downstream table
comparable across them. The stage 9 file keeps the nine-task string because its LLaMA-7B and
Qwen2.5-7B runs were collected with it, and a stage file that no longer describes what it produced is
worse than a wide one.

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

**Preview.** These runs load an existing checkpoint and allocate nothing, so there is no allocation
to preview. The roster is the thing to check instead, and it comes from the gate rather than from
`allocation_report.py`:

```bash
python generate_tables.py output/eval/huggyllama_llama_7b --report gates \
    --allocation_dir output/allocation_reports -o output/gates/huggyllama_llama_7b/gates_stage9.md \
    && grep -n "on disk" output/gates/huggyllama_llama_7b/gates_stage9.md
```

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

**It runs on LLaMA-7B, which is the only model that can.** The roster's `on disk` column decides this
rather than preference: LLaMA has all four arms, Qwen2.5-7B has the two heterogeneous ones and neither
anchor, and Qwen2.5-32B has none -- and 32B checkpoints are not kept, so putting this stage on 32B
would mean four recompressions of a 32B model before the first update starts.

**Check the two heterogeneous paths before launching.** `__CKPT_HET_*__` is resolved by the stage 7
gate, which is deliberately held at `--max_ratio 0.9` so its rows stay comparable, so the placeholder
names an *unbounded* checkpoint: on Qwen2.5-7B `entropy_rel`, which the bound ladder has since beaten by
32%, and on 32B `--min_rank_fraction 0.2`, beaten by 9%. **On LLaMA no repointing is needed**, and
stage 7f is what settles that rather than luck: the cap loses on this model, so the uncapped `eff_rank`
run the placeholder names is the best checkpoint at both budgets and the file runs as it stands. Should
this stage ever move to a Qwen model, repoint both entries first. RQ5 asks whether the update erases
the allocation's advantage; an update applied to an allocation the thesis does not report answers that
about the wrong object, and the answer would look artificially favourable to the update, because there
is more damage left in a worse checkpoint for it to recover.

**Normalize the gap before concluding anything.** At ratio 0.5 on LLaMA the homogeneous arm carries
31% more excess NLL than the heterogeneous one -- `ln(24.58 / 5.68)` against `ln(17.32 / 5.68)` -- so it
has more damage available to recover and will close more *absolute* perplexity for arithmetic reasons
alone. Read the closed gap against `ln(dense)`, the normalization §4 already uses for the cross-model
recovery share, rather than comparing the two arms' raw improvements.

**What to check in it.** The gap closed on each arm, not the final perplexity. If the update closes
more of the homogeneous gap than the heterogeneous one, the two techniques compete; if it closes both
equally, they compose, and the thesis can claim the allocation and the update are independent
contributions.

**Preview.** The update reuses the allocation its checkpoint was built with, so there is nothing new
to allocate. To read the allocation each arm inherits, point the report at the configuration the
checkpoint was compressed under rather than re-deriving it:

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --group_criterion __BEST_GROUPING__ --score_metric __TOP1_SCORE__ \
    --inner_allocation __BEST_INNER__ --outer_allocation __BEST_OUTER__ \
    --outer_offset __BEST_OUTER_OFFSET__ \
    --sweep "compression_ratio=0.2,0.5" \
    --out_dir ./output/allocation_reports/stage10
```

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
| 7 cross-model | 9 | wikitext, blocked on whitening | rerun stage 0 for the model |
| 7b GQA diagnostics | 7 | wikitext | `family_tail`, and which caps are feasible |
| 7c GQA fixes | 18 | wikitext | `family_tail` per score |
| 7d second sharing factor | 18 | wikitext; full suite on the dense, anchor and elected-bound rows | stage 0, the elected allocation, and the cap and floor sweeps, on 32B |
| 7e Qwen2.5-7B closeout | 12 | wikitext; full suite on the five reported rows | the cap ladder at 0.2 and the grouping arms, on 7B |
| 7f LLaMA-7B closeout | 2 (+1 eval-only) | wikitext; full suite on the elected 0.2 row and the capped 0.5 row | the block screen across the curve, and the cap ladder under the elected outer level, on LLaMA |
| 8 ratio curve | 18 | wikitext | the shape claim, made offline |
| 9 benchmarks | 9 eval-only | full suite **and c4** | none |
| 10 LoRA | 4 update-only | wikitext | none |

**254 compression runs**, plus 14 evaluation-only or update-only, plus two extra whitening passes for
stage 1b, one for Qwen2.5-7B and one for Qwen2.5-32B. At roughly 15 minutes a run that is about **55
GPU hours** for the compressions on the two smaller models; stage 7d's eighteen runs are on a 32B model
and cost several times that each, which is why they are the only ones the grid rations. Nine of those
eighteen were added after the first nine came back, and stage 7d is the one place in the grid where
the full evaluation suite rides along with the compression rather than following as stage 9: 32B
checkpoints are not kept, so an eval-only pass over them would mean compressing a second time.

GPU time is not the binding constraint, so these counts describe the grid rather than budget it: spend
runs wherever they resolve something. What the offline pass buys is not saved hours but the guarantee
that each run buys a distinct experiment — the block screen rejects a cell that is going to fail, and
the map-distance screen rejects a cell that repeats one already collected, which is how the five caps
this document once prescribed at ratio 0.2 turned out to be one run.
