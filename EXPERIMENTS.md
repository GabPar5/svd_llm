# Experiment grid

The staged design behind the results of the thesis, and the source for thesis section 4.1.

Every stage is one axis swept around a fixed configuration, with a **gate** at the end that resolves
the placeholders of the next stage. The grid is deliberately not a cross product: the full cross for
a single model at a single ratio is already 4 groupings x 6 scores x 4 inner policies = 96 runs.

Stages 1 to 8 evaluate **wikitext perplexity only**, which is what makes a ~100 run grid affordable.
Only stage 9 pays for c4 and the full benchmark suite, which is also where the wikitext-against-c4
check lives.

Whitening is assumed to be already cached under `output/whitening_matrices/<model>/v2/`, together
with its `spectra/` cache and `layer_importance.pt`. Every stage below reads that cache and never
recomputes it.

**Execution order**, matching the section order again now that the cap comes first:
`0, 1, 2, 2b, 2c, 3, 3b, 3c, 4, onset, 5, 5b, 6, 7, 9, 10`.
Stage 3 runs **before** stage 4 and freezes `--max_ratio` into `args/base_args.json`; the reason is
in the findings below. The onset probe lives in the stage 8 section and file for continuity.

## What stages 1, 2, 2b and 2c established

All four have run: 28 heterogeneous cells plus two homogeneous anchors and the dense reference.
Their results are the premise of everything below, so they are recorded here rather than left in a
report.

### The peak assigned ratio is a threshold, not the whole story

Sorting the 28 heterogeneous runs collected before the cap sweep by the **largest removal ratio
assigned to any single matrix** separates them without exception:

| peak assigned ratio | runs | wikitext against the homogeneous anchor |
|---|---|---|
| 0.23 to 0.83 | 23 | -0.65 to +0.67 |
| 0.87 to 0.90 | 5 | **+1.85 to +18.82** |

The boundary sits between 0.8325 (`global`/`eff_rank`, which *beat* homogeneous by 0.28) and 0.8743
(`decoder`/`norm|inf`, which lost 8.88). `global`/`truncation` and `type`/`truncation` at ratio 0.5
each pinned **3 matrices out of 224** at the 0.9 cap and lost 14.53 and 18.82.

**The cap sweep then tested whether the peak was the cause, and it is only part of it.** Capping the
two catastrophes recovers between a third and a half of the damage and leaves the rest:

| at ratio 0.5, homogeneous 24.56 | peak 0.90 | peak 0.85 | peak 0.70 |
|---|---|---|---|
| `global` / `truncation` | 39.09 | 35.75 | **33.06** |
| `type` / `truncation` | 43.38 | 35.31 | **33.56** |
| `decoder` / `truncation` | — | — | **25.05** |

At a peak of 0.70 all three allocate to the same ceiling, and `decoder` is still **eight perplexity
better**. So the peak is a threshold that amplifies a bad allocation, not the thing that makes it bad.
An earlier draft of this document read the catastrophes as pure peak damage and demoted the grouping
to a second-order effect; the cap sweep refutes that. For `truncation` the grouping is worth about 8
perplexity at matched peak, which is more than ten times the best heterogeneous gain anywhere in the
grid.

**And within the safe band a higher peak is better, not worse.** Capping the winning configuration
degrades it monotonically:

| `decoder`/`eff_rank` at 0.5 | cap 0.6 | 0.7 | 0.8 | 0.85 |
|---|---|---|---|---|
| peak | 0.600 | 0.700 | 0.724 | 0.724 |
| wikitext | 24.19 | 24.09 | 24.05 | 24.05 |

Caps at 0.8 and above stop binding, which is why the last two rows are identical. So `--max_ratio` is
a **guard rail and not a tuning knob**: it earns its place by stopping a configuration that would run
past 0.85, and every ratio point it takes off a configuration that would not is a small loss.

### It is the tail, and the tail lands on the earliest blocks

The peak is the same number for all three groupings once a cap equalizes it, and the outcomes are
still eight perplexity apart. What differs is how much of the allocation sits behind that peak, and
where:

| at ratio 0.5, cap 0.7, `truncation` | matrices at the cap | above 0.6 | layers holding the top 8 | wikitext |
|---|---|---|---|---|
| `decoder` | **1** | 8 | 0, 1, 2, 3, 4, 5, 6 | **25.05** |
| `type` | 8 | 18 | 0, 1 | 33.56 |
| `global` | 8 | 22 | 0, 1, 2, 3 | 33.06 |

`decoder` reaches the ceiling with one matrix and spreads its eight aggressive allocations one per
block. `global` and `type` put eight matrices *at* the ceiling and concentrate their whole tail in
the first two to four blocks. Uncapped, the same three matrices of layer 0 are the ones pinned at
0.9.

So the mechanism is not the peak and not the grouping as such: **the flat groupings over-compress the
earliest blocks, and `param_share` prevents that structurally** by giving every block the same average
ratio, which caps how much of the tail any one block can hold.

**Block Influence already knows this.** From the stage 0 cache, layer 0 scores 0.4602 against 0.29 for
layer 31 and 0.15 or less for everything else — it is three times the next block and by far the most
important in the model:

| layer | 0 | 1 | 2 | 4 | 8 | 16 | 24 | 31 |
|---|---|---|---|---|---|---|---|---|
| Block Influence | **0.4602** | 0.1414 | 0.1511 | 0.1361 | 0.1255 | 0.0874 | 0.0334 | 0.2909 |

The outer level of the hierarchical allocator gives high-influence blocks *less* removal. It therefore
does deliberately, and with a signal, what `decoder` does only as a side effect of flattening: protect
layer 0 from the tail. That is the thesis contribution's motivation, and it is now an empirical
prediction rather than an analogy — which is why stage 3c runs it next.

### The peak is predictable offline, exactly

`max_ratio_assigned` in `summary.csv` matched the realized peak of every one of the 28 runs to four
decimals — it is the same allocator over the same cached spectra. So every one of the five damaged
runs was identifiable before it was made, at zero cost. `figures/cap_binding.csv` says the same
thing as a count: the four runs that pinned anything are four of the five that failed.

### Where the heterogeneity actually goes

Under `decoder` + `eff_rank`, only the query and key projections move away from the flat ratio. At
target 0.2 every other family sits within two ratio points of it:

| family | mean ratio | range |
|---|---|---|
| `q_proj` | 0.2132 | 0.2029 to 0.2897 |
| `k_proj` | 0.2150 | 0.2050 to 0.2870 |
| `v_proj` | 0.2004 | 0.1856 to 0.2037 |
| `o_proj` | 0.2030 | 0.1903 to 0.2069 |
| `gate_proj` | 0.1976 | 0.1842 to 0.1995 |
| `up_proj` | 0.1958 | 0.1814 to 0.1982 |
| `down_proj` | 0.1948 | 0.1734 to 0.2254 |

And the deviation is concentrated at the input end. Layer 0's `q_proj` has an effective rank of **109
out of 4096** and takes 0.290; layer 1 takes 0.240; from layer 4 onward every `q_proj` sits between
0.207 and 0.216.

So the winning allocation is, to a good approximation, **compress q and k slightly harder than
everything else, and compress the first two blocks' q and k much harder**. That is worth saying
plainly rather than leaving implicit in a ratio map, and it suggests a control worth running: a
hand-set family schedule that reproduces those means with no score at all. If it matches, the spectral
machinery is an expensive way to discover that early attention is low rank, and the thesis should say
so.

**On comparability**, which is the obvious objection: `q`, `k`, `v` and `o` are all 4096x4096 and the
MLP matrices are 11008x4096, so every matrix in the model has 4096 singular values and the effective
ranks are on one scale. The low q/k values are a spectral property, not a units artifact. What
`eff_rank` does not do is normalize by the matrix's own maximum; `normalized_effective_rank` already
exists in `src/utils.py` for the Block Influence correlation, and registering it as a score metric
would make a scale-free variant sweepable if that turns out to matter.

**Ratios near zero are a different phenomenon.** They appear under the `softmax_temp` inner policy at
low temperature, where the allocation goes bimodal: at temperature 0.05 every family has a matrix at
0.0000 and one at the cap. That is the policy's aggressiveness, not a property of any one family.

### How big the grouping effect is, and it depends on the score

At matched peak, `decoder` wins at every score, but by amounts that differ by an order of magnitude:

| at ratio 0.5 | `decoder` | `global` | `type` | spread |
|---|---|---|---|---|
| `eff_rank`, peaks 0.72 to 0.83 | **23.91** | 24.28 | 25.23 | 1.3 |
| `entropy`, peaks 0.58 to 0.61 | **24.23** | 24.38 | 24.80 | 0.6 |
| `truncation`, all capped to peak 0.70 | **25.05** | 33.06 | 33.56 | **8.5** |

The offline map distance predicted exactly this ordering before any of it was measured. Capped alike,
`decoder` and `global` differ by a mean of 0.0085 per matrix under `eff_rank` and 0.0364 under
`truncation`, four times more, and the measured spreads are 0.37 and 8.01. **Where the allocations
converge the outcomes converge, and where they do not the grouping is worth more than everything else
in the grid put together.**

So the useful statement is not "`decoder` is the best grouping" but: **the grouping matters exactly as
much as the score makes it matter.** A score whose values move sharply across depth, like
`truncation`, hands the grouping a large lever, because the grouping is what decides whether depth is
allowed to absorb budget. A score that is flatter across depth, like `eff_rank`, leaves almost nothing
for the grouping to do.

`decoder` also keeps a mechanical advantage worth stating separately: `param_share` gives every block
the same average ratio, which bounds how far the allocation can push any single matrix. It is the
grouping least able to produce a dangerous peak, which is a different virtue from ranking well.

### A note on the two machines

The 11 cap runs were made on Colab and the other 31 elsewhere, and the whitening caches are not
identical: at ratio 0.2 the three capped `decoder`/`eff_rank` runs all give peak 0.28969 and 7.85
while the uncapped run from the other machine gives 0.28947 and 7.77. The offset is around 0.08 at
ratio 0.2 and 0.14 at 0.5. It is temporary and the grid is going back to one machine, so it is
recorded here only so that a 0.1 discrepancy between an old and a new number is not mistaken for a
result. The gate report prints a note on any table that mixes the two.

### Ratio 0.2 is not flat, and aggressive allocation is what wins there

Stage 2's nine cells spanned 0.03 perplexity at 0.2, which looked like noise. Stage 2b settles it:
`eff_rank_sq` reaches **7.62** against a homogeneous 7.79, a 0.17 gain — five times the entire
spread of stage 2 — and `entropy_sq` reaches 7.74. Both are the most aggressive allocations at that
ratio, with peaks of 0.533 and 0.473 against 0.29 for `eff_rank`.

The same two scores are near-worst at 0.5, where their aggressiveness pushes the peak to the 0.9
cap. So a score's merit is not a property of the score: **it is whether that score's aggressiveness
lands the peak inside the safe band at that budget**, and the band does not scale with the budget.
At 0.2 no run went past 0.533, so the headroom between there and ~0.85 has never been tested, and
that is where the remaining gain at 0.2 most likely is.

This is a better RQ1 answer than a monotone gain claim, and it is the working hypothesis the rest of
the grid tests.

## The three tools, and the order to use them in

Every stage is the same loop, and each tool owns one step of it:

```
allocation_report.py  ->  run_experiments.py  ->  generate_tables.py --report gates
   prune and configure       spend the GPU            read the gate, fill the placeholder
   seconds, CPU               ~1h per run              seconds, no GPU
```

| | `allocation_report.py` | `run_experiments.py` | `generate_tables.py --report gates` |
|---|---|---|---|
| Cost | seconds, CPU, no model weights | about an hour per run, GPU | seconds, reads JSON |
| Answers | what ratios a configuration produces | what perplexity it produces | which configuration won |
| Reads | the cached spectra and Block Influence | the model | `output/eval/` and the offline CSVs |
| Settles | the run list, the knobs, feasibility | nothing by itself, it only measures | every `__PLACEHOLDER__` |

### What the offline report can and cannot settle

**It resolves no placeholder on its own.** Every `__*__` value in this grid is defined as a ranking
by perplexity, and perplexity needs the model. What the offline report does instead is decide
**which runs are worth making** and **what to hold them at**, which is where the hours are actually
saved.

Read it as answering four questions, in this order:

1. **Is the configuration even feasible?** A cap too low to reach the target once bypassed layers
   are charged shows up as a budget-drift violation, a `checks` entry in `summary.csv`, and a
   non-zero exit. Fix the configuration before running anything.
2. **Does the variant allocate anything?** A score that is constant inside every group produces the
   flat ratio whatever policy runs, so the run is homogeneous while looking heterogeneous.
   `allocate_ratios` prints a `[BUDGET][WARNING]` for it, and `figures/dispersion.csv` shows a
   `ratio_std` near zero. Drop the cell: it will reproduce the stage 1 homogeneous number and take
   an hour to say so.
3. **Are two variants distinguishable?** Two configurations whose ratios agree to three decimals are
   one experiment, not two. Compare them in `matrices.csv`, which carries the assigned ratio per
   matrix. This is what makes the stage 2b pairs (`truncation` against `truncation_sq`) worth
   checking before spending six runs on them, and it is the one case where the offline pass can
   settle a gate outright: a candidate that cannot allocate differently cannot win a promotion, so
   `__TOP1_SCORE__` stays where stage 2 put it.
4. **Are the variants comparable to each other?** Policies compared at their default knobs differ in
   shape *and* in aggressiveness at once. Read `ratio_std` from `figures/dispersion.csv` to see by how
   much. Equalising it is mostly not possible: `--offset` moves no policy's allocation and only
   `--softmax_temp` is live, so where two policies cannot be brought together, report the dispersion
   beside the result rather than implying it was controlled.
5. **Is the allocation about to blow up?** See the peak screen below, which is the one screen that
   held on every collected run.

One gate is answered offline and only offline: **the Spearman sign in stage 6**. It is a go/no-go on
whether fusing Block Influence with a spectral score measures what it is meant to, and no perplexity
number can substitute for it.

### Do not rank with it

Stage 2 measured the offline ordering against the measured one on all nine cells, and **every single
row disagreed** — not randomly, but close to inverted. `type/truncation` is offline-best and
measured-worst (43.38 at ratio 0.5); `decoder/eff_rank` is offline 8th of 9 and measured 1st.

There is a mechanism, and it is worth a paragraph of the thesis. The six objectives are all
tail-energy measures, minimised by concentrating removal where the spectrum decays fastest — exactly
the aggressive cross-depth allocation that destroys the model. They systematically reward the
failure mode. With `score_ratio_rho` at `-1.0000` on every variant, nothing else is absorbing the
difference.

So `mean_rank` is a reported negative result, not a screen. Keep the offline pass for the mechanical
facts it is reliable on: feasibility, degeneracy, dispersion, cap binding. Expect the ranking to
invert.

### The tail screen: what to read before every run

`figures/ratio_tail.csv` gives, per variant, the peak, how many matrices sit above 0.6, 0.7, 0.8 and
0.85, and which layers hold the top eight. Read all three columns, in this order:

1. **Peak above 0.85 is fatal.** On the 28 runs collected before the cap sweep this classified all 28
   correctly, with the boundary observed between 0.8325 (safe, and better than homogeneous) and
   0.8743 (lost 8.88). Drop the variant.
2. **Mass behind the peak.** A variant with one matrix at 0.70 and one with eight are not the same
   experiment: at ratio 0.5 they measure 25.05 and 33.06. Compare `above_0.6` and `above_0.7`.
3. **Which layers.** A tail concentrated in layers 0 to 3 is the failure mode; a tail spread one
   matrix per block is not. `layers_of_top_8` says which it is at a glance.

The peak is exact rather than indicative: it equalled the realized peak of all 28 runs to four
decimals, because the preview replays the same allocator.

**Do not use `ratio_std` for this.** An earlier draft proposed it and the 2b runs refute it:
`eff_rank_sq` at ratio 0.2 has the *highest* dispersion in its sweep (0.104) and is the **best** run
at that ratio (7.62). Dispersion mislabels exactly the aggressive-but-safe allocations that produce
the gain. It remains the right tool for comparing two policies' aggressiveness.

The thresholds rest on 42 runs from one model at two budgets, and nothing in the tooling enforces
them. Re-derive them per model, and note that the safe band did **not** scale with the budget between
0.2 and 0.5, which is what makes an absolute line plausible.

### The map-distance screen: two variants that are one experiment

`figures/map_distance.csv` gives, for every pair of variants in a sweep, the mean and the largest
per-matrix ratio difference between their allocations, plus each one's peak. A pair whose largest
difference falls under 0.02 is reported on the console as one experiment run twice.

The test is on the **largest** difference and not the mean, and the reason is worth keeping in mind
when reading the file. Raising the cap from 0.75 to 0.9 on `type`/`truncation` at ratio 0.5 moves the
allocation by 0.004 on average and by **0.15 on three matrices** — and those three matrices are the
difference between a working model and one at 43.38 perplexity. A screen that tells you not to run
something has to be wrong in the safe direction.

It pays for itself immediately. The cap sweep this document originally prescribed for stage 3 turns
out to produce a largest difference of exactly 0.0000 across all five caps at ratio 0.2, because the
cap never binds at the winning configuration: four of those runs were the same run.

### Preview command

```bash
python allocation_report.py \
    --model "huggyllama/llama-7b" --run_v2 \
    --compression_ratio 0.2 \
    --sweep "group_criterion=global,type,decoder" \
    --sweep "score_metric=truncation,entropy,eff_rank" \
    --out_dir ./output/allocation_reports/stage2 \
    --plots
```

`--sweep` is repeatable and taken as a cartesian product; anything not swept is passed as a plain
flag and held fixed, exactly as `run_experiments.py` would pass it. `--plots` adds PNGs when
matplotlib is installed; the CSVs are written either way.

**Name `--out_dir` after the stage it previews**: `stage2`, `stage2b`, `stage4`, and so on under
`output/allocation_reports/`. `generate_tables.py --allocation_dir` discovers stage directories by
that name and attaches each preview to the gate it belongs to. A suffix is allowed and keeps the
same stage, so `stage4_knobs` is read as stage 4; when both exist the unsuffixed one is used. A
directory named anything else is simply not picked up. Giving one `--out_dir` per stage is also what
stops each preview from overwriting the last.

### Reading its output

The console table is the summary: one row per variant, ordered by mean rank, with the swept axes
named once in the header. Each objective cell holds the objective value and, in parentheses, its
rank across the variants. The `checks` column is `ok` or the invariant that failed.

| File | Use |
|---|---|
| `summary.csv` | one row per variant: realized ratio, `mean_rank`, every objective with its rank and oracle ratio, ratio dispersion, invariant violations |
| `matrices.csv` | one row per variant **per matrix**: score, assigned ratio, rank, truncation loss. This is the ratio map, and the only place two variants can be compared allocation by allocation |
| `layers.csv` | one row per variant per decoder block: params, removed params, block ratio, Block Influence |
| `figures/objectives.csv` | which variants win only the objective their own score optimizes |
| `figures/dispersion.csv` | how widely each configuration spreads its ratios, for knob matching and for spotting a degenerate cell |
| `figures/cap_binding.csv` | how many matrices `--max_ratio` actually pins |
| `figures/influence_vs_effrank_rho.csv` | Spearman rho per matrix family, the gate on stage 6 |
| `figures/ratio_by_type.csv` | mean ratio per matrix family, where rank-space bias shows |
| `figures/oracle_gap.csv` | each objective against its greedy lower bound, for comparing across budgets |
| `figures/map_distance.csv` | how far apart two variants allocate, with each one's peak: the screen against paying twice for one experiment |
| `figures/ratio_tail.csv` | the peak, how many matrices sit above 0.6 to 0.85, and which layers hold the top eight: the screen against a run that is going to fail |
| `budget/<variant>.log` | the captured `[BUDGET]` instrumentation of that variant |

Variants are ranked by **mean rank across six objectives**, never by a single number. The obvious
single number, `frobenius_tail`, *is* the `truncation_sq` score summed over matrices, so ranking on
it hands the truncation scores a win by construction. A variant that wins one column and trails the
others is winning on its own terms, which is a result to report rather than to resolve.

**Do not read the offline ordering as a prediction of the perplexity ordering.** It prices the
allocation, not the model. Where the two disagree is a thesis result in its own right, and the gate
report prints the comparison for stage 2 once both halves exist.

## Running a stage

```bash
python run_experiments.py args/experiments_stage1_anchors.json            # run it
python run_experiments.py args/experiments_stage1_anchors.json --dry_run  # preview the commands
python run_experiments.py args/experiments_stage4_policies.json --base args/other_base.json
```

The runner merges `args/base_args.json` into each entry of the stage file, refuses to start if any
entry still holds an unresolved gate value, and continues to the next run when one fails rather than
aborting the queue.

The commands it prints are informational, not copy-paste ready: a score like `norm|1` holds a pipe
that a shell would interpret. `subprocess` passes it as one argument, so runs are unaffected.

`args/` is tracked, so the grid is reproducible from the repository. **Never put `--hf_token` in one
of these files**: pass it on the command line, or export it in the environment.

## Frozen across every run

Changing any of these invalidates comparability with everything already collected.

`args/base_args.json` is authoritative for these.

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
| Screening evaluation | `wikitext\|0`, `--eval_max_length 4096` | c4's validation shard costs as much again per run, and stage 9 is where a second corpus earns it. LLaMA-7B's context is 2048, so this is clamped to it |

### What the collected runs used

The sidecars of the 31 runs from stages 1 to 2c record four values that differ from the table above:
`float16` for both dtypes, `--max_whitening_samples 256`, `--ratio_scope all` and
`--eval_max_length 2048`. Two of those are immaterial — with all seven matrices active the `all` and
`selected` denominators coincide, and 4096 is clamped to LLaMA-7B's 2048 context either way — so the
differences that carry are the precision and the calibration token count.

They are consistent across all 31 runs, so everything collected is comparable with itself. Keep them
in view when a later stage is compared against those numbers, and note that the whitening cache is
keyed by model and version alone: raising `--max_whitening_samples` has no effect unless
`output/whitening_matrices/` is deleted first.

## Checkpoints, and what has to survive a cleanup

A compressed fp32 7B checkpoint is about 20 GB, so checkpoints are deleted once their run has been
evaluated. Two rules keep that from destroying a gate:

- **Keep every checkpoint a `__CKPT_*__` role names**: the two homogeneous anchors, the best score,
  policy, bypass and composite runs, and the overall heterogeneous winner at each ratio. Stages 9
  and 10 load these directly and cannot recompress.
- **Keep each stage's runner-up too.** Roles move: stage 2b or 2c promoting a score moves
  `__CKPT_BEST_SCORE_*__` onto a different run, and if that run is gone it costs an hour to rebuild.

The gate report's stage 9 roster has an `on disk` column for exactly this, so a cleanup can be
checked against it before stage 9 is queued.

**Every checkpoint from stages 1 to 2c has already been deleted.**
`output/models/huggyllama_llama_7b/` holds the sidecars and the tokenizer but no `.pt`, so the
roster reads `on disk: no` for every role. Stages 9 and 10 load checkpoints and cannot recompress,
so whichever runs end up filling the `__CKPT_*__` roles have to be **rebuilt at about an hour
each**. Two consequences:

- Budget roughly 10 extra compression runs for stage 9 and 10 inputs, or re-run those specific
  configurations with `--evaluate` off once the gates have settled.
- Apply the retention rules from now on. The cheapest moment to keep a checkpoint is the run that
  produced it.

## Placeholders

Stage files carry literal placeholders until their gate resolves them. Every one is filled from the
gate report, and the offline preview only narrows the candidates that go into it:

| Placeholder | Resolved by | Meaning | What the offline preview contributes |
|---|---|---|---|
| `__BEST_GROUPING__` | stage 2 | grouping criterion with the best mean rank | drops groupings whose scores are degenerate inside every group |
| `__BEST_FLAT_GROUPING__` | stage 2 | better of `type` / `global`, never `decoder` | confirms both flat groupings actually spread their ratios |
| `__TOP1_SCORE__`, `__TOP2_SCORE__` | stage 2 (2b, 2c may promote) | the two best score metrics | drops a candidate whose ratio map matches an incumbent's, since it cannot win a promotion |
| `__BEST_INNER__` | stage 4 | best inner allocation policy | **required**: the four policies differ in aggressiveness, and the preview says which of them the frozen cap still lets past 0.85 |
| `--max_ratio` | stage 3, into `args/base_args.json` | the cap every later stage runs at | **the first-order effect**: `max_ratio_assigned` gives the peak each cap produces, exactly |
| `__BEST_BYPASS_EARLY__`, `__BEST_BYPASS_LATE__` | stage 5 | the bypass setting with the best gain over homogeneous | catches settings whose budget is infeasible once the exempt blocks are charged |
| `__CKPT_<ROLE>__` | stages 1 to 8 | a path under `output/models/huggyllama_llama_7b/` | nothing, these are outputs of runs |
| `__FINALIST{1,2,3}_*` | stages 2 to 6 | the three configurations worth testing on another model | rerun stage 0 per model: the Spearman sign and score-versus-depth shape are model properties, and the finalists may not transfer |

`run_experiments.py` refuses to start while any remain, so an unfilled gate cannot silently run the
wrong configuration. The gate report prints the same list with the resolved value beside each one,
and `waiting on runs` where a stage has not produced its answer yet.

**No stage file past 2c names a score.** Stages 2b and 2c are promotion tests, and either can move
`__TOP1_SCORE__` or `__TOP2_SCORE__` onto a squared score or a Schatten norm. Writing today's winner
into stage 4 would freeze the decision those two stages exist to revisit, so every entry from stage
3 onward carries the placeholder and is substituted at the moment it runs. The same holds for the
composite halves of stage 6, which are spelled `composite|__TOP1_SCORE__|block_influence`.

**A `provisional` row is not an answer.** The gate report reports how many candidates a placeholder
was chosen from, and a value decided by a table holding one entrant reads `provisional (1
candidate)`. That is the report saying it has recorded the only run that has happened, not picked a
winner.

## Reading evaluation results

```bash
python generate_tables.py ./output/eval/huggyllama_llama_7b -f markdown -o report.md
python generate_tables.py ./output/eval/huggyllama_llama_7b -f latex   -o tables.tex
```

Each run also writes `<run_name>.config.json` beside its checkpoint and its evaluation JSON, which
is authoritative for the dimensions the filename cannot carry, and records the **realized** removal
ratio alongside the target.

## Resolving a gate

```bash
python generate_tables.py ./output/eval/huggyllama_llama_7b \
    --report gates --allocation_dir ./output/allocation_reports -o gates.md
```

One table per gate below, in stage order, and a leading **Placeholders** table holding every value a
stage file waits on next to what the collected runs resolve it to. A gate still waiting on runs says
so rather than defaulting, so an unresolved row means that stage cannot run yet. Add `--report both`
to get the result tables in the same file, or `-f latex` to get the gate tables as LaTeX.

What it does with the runs, so a table can be trusted before it is pasted into a stage file:

- Ranks **within each ratio** and averages the ranks, never comparing raw perplexity across budgets,
  and reports how many ratios priced each row: stage 3 runs different caps at different ratios, and
  a mean rank from one ratio is not comparable to one from both.
- Carries the **gain over the homogeneous arm** at the same setting, which is the RQ1 read every
  stage repeats, and for stage 5 pairs the two arms bypass setting by bypass setting.
- **Holds fixed** whatever a stage is not sweeping, and reports which runs that excluded. Stage 3's
  cap sweep matches stage 4's table in every dimension except the cap, so left in it would decide
  stage 4's gate through the cap. Stage 5 reads the setting to hold from its own bypassed runs,
  since its bypass-0 reference otherwise sits among every other stage's runs.
- Warns when a dimension moves inside a table without being one of its axes, and when a run's
  realized removal drifted off the budget it is being compared at.
- Attaches the offline preview of each stage from `--allocation_dir`, including the Spearman sign
  that gates stage 6, the dispersion used to match the stage 4 knobs, and the cap binding behind
  stage
  4. For stage 2 it also prints the offline ordering against the measured one, which is the
  disagreement worth reporting: it bounds how far the free preview can substitute for an hour of GPU
  per cell.

The dimensions all come from the sidecar, the only place most of them exist. A run without one is
counted and left out rather than guessed at.

## Passing a gate, step by step

The same six steps for every stage. Stage 4 is used here because its offline pass both prunes the
runs and settles a configuration.

**1. Preview the stage offline.** Sweep the axis, and sweep the knobs with it:

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --group_criterion __BEST_GROUPING__ --sweep "compression_ratio=0.2,0.5" \
    --sweep "inner_allocation=waterfill,drank_lagrangian,swift_pool,softmax_temp" \
    --sweep "softmax_temp=0.05,0.2,1.0" \
    --out_dir ./output/allocation_reports/stage4_knobs
```

Substitute the placeholder by hand here: this tool takes flags, not stage files, and does not read
`args/`. A non-zero exit means an invariant failed and the configuration is not runnable yet.

**2. Prune and configure from the CSVs.** In order:

- **`max_ratio_assigned` in `summary.csv` first.** Drop every variant whose peak exceeds 0.85. This
  is the screen that would have saved the five damaged runs already collected, and it is exact.
- **The degenerate-score warning and a `ratio_std` near zero.** Either means the run reproduces the
  homogeneous number.
- **`figures/dispersion.csv` to compare aggressiveness**, not to equalise it. `--offset` moves
  nothing and only `--softmax_temp` is live, so where two policies cannot be brought together,
  report the dispersion beside the result instead of pretending it was controlled.

Then write any chosen knob into `args/base_args.json` and delete the dropped cells from the stage
file.

**3. Run the stage.**

```bash
python run_experiments.py args/experiments_stage4_policies.json --dry_run  # check the commands first
python run_experiments.py args/experiments_stage4_policies.json
```

**4. Read the gate.**

```bash
python generate_tables.py ./output/eval/huggyllama_llama_7b \
    --report gates --allocation_dir ./output/allocation_reports -o gates.md
```

**5. Check the gate's own warnings before trusting it.** A `confounded` note means a dimension moved
that the stage was not comparing, and the fix is a run rather than a reading. A `priced at 1/2` row
was ranked on one ratio and is not comparable to a row ranked on both. A drift note means a run
missed its budget and is not comparable at all.

**6. Copy the resolved value into the next stage file.** Take it from the **Placeholders** table,
not from the body tables, and replace the literal string:

```bash
sed -i 's/__BEST_INNER__/drank_lagrangian/g' args/experiments_stage5_bypass.json
```

Then repeat from step 1 for the next stage. `run_experiments.py` refusing to start is the backstop:
it means a placeholder was missed.

### When the offline pass is enough on its own

Three cases end at step 2, with no GPU time at all:

- **The configuration is infeasible.** Non-zero exit, `checks` populated. Fix the cap, the bypass or
  the target and preview again.
- **The cell is degenerate.** A `[BUDGET][WARNING]` or a `ratio_std` near zero means the run would
  reproduce the homogeneous number. Delete the cell and say so in the thesis: a heterogeneous
  allocation that cannot allocate is a finding about the score, not a missing data point.
- **Two candidates are the same experiment.** Ratio maps agreeing to three decimals cannot produce
  different perplexities. Keep one, and for a stage 2b or 2c candidate that means the incumbent
  `__TOP1_SCORE__` holds without a run.

### Which preview to run for which stage

| Stage | Preview `--out_dir` | Read | Decide |
|---|---|---|---|
| 0 | `stage0` | `figures/influence_vs_effrank_rho.csv` | whether stage 6 may run at all, and record the sign either way |
| 1 | none | | a homogeneous run allocates nothing |
| 2 | `stage2` | `max_ratio_assigned` per cell | which cells will blow up, before spending 18 runs |
| 2b | `stage2b` | `matrices.csv`, the ratio column per matrix | whether a `_sq` score allocates differently from the score it derives from |
| 2c | `stage2c` | `matrices.csv`, `figures/dispersion.csv` | whether `norm\|-inf` is signal or rounding noise, which decides the 2 conditional runs |
| 4 | `stage4_knobs` | `max_ratio_assigned`, `figures/ratio_by_type.csv` | that the frozen cap brings all four policies inside the safe band; `--offset` is inert |
| 3, 3b | `stage3`, `stage3b` | `max_ratio_assigned`, `figures/map_distance.csv` | that every cap binds, and that no two runs allocate the same way |
| 5, 5b | `stage5` | the exit code and `checks` | whether the bypassed budgets are feasible under the cap |
| 6 | `stage6` | `figures/dispersion.csv`, plus the stage 0 rho | the offset for the fused score, and that the three alphas allocate distinctly |
| 7 | `stage0` again, per model | the rho sign, `figures/scores_by_depth.csv` | whether the finalists transfer to that model |
| 8 onset | `stage8` | the `<objective>_oracle_ratio` columns of `summary.csv` | the shape claim itself, which is made offline and never costs a run |
| 9, 10 | none | | both stages load existing checkpoints and allocate nothing |

The preview never needs a GPU and never touches the model weights, so re-running one after changing
a knob costs seconds. Re-run stage 0 whenever the whitening cache changes.

---

## Stage 0: the offline pass

**Purpose.** Everything that can be known without a GPU, plus the two gates that govern later
stages. Costs seconds. Run it before anything else and re-run it whenever the whitening cache
changes.

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --compression_ratio 0.2 \
    --sweep "group_criterion=global,type,decoder,hierarchical" \
    --sweep "score_metric=truncation,entropy,eff_rank" \
    --out_dir ./output/allocation_reports/stage0 --plots
```

**What to check.**
1. The header reports how many selected matrices have a cached spectrum. Anything below 100% means
   the cache was built by a run that bypassed layers, and sweeps below that bypass are not fully
   explorable.
2. `figures/influence_vs_effrank_rho.csv`: **the gate on stage 6**. Swift-SVD reports this
   correlation as negative, which is what makes the two signals complementary. Record the sign now.
3. Any variant with a `[BUDGET][WARNING]` about degenerate scores is homogeneous in all but name.
4. The script exits non-zero if any variant violates an allocation invariant.

**Thesis figures produced here.** `scores_by_depth` (3.1.1), `influence_by_depth` (fills the
`bidepth:fig` placeholder in 3.1.2), `influence_vs_effrank` (4.2.1), `spectra` (3.1.1),
`ratio_by_type` (3.3.5).

---

## Stage 1: anchors

**Purpose.** The floor every later comparison is measured against, and the first half of RQ1.

**Runs.** 3: dense LLaMA-7B, then homogeneous at 0.2 and 0.5.

```bash
python run_experiments.py args/experiments_stage1_anchors.json
```

**Offline preview.** None needed. A homogeneous run gives every matrix the same ratio, so there is
no allocation to inspect.

**Runs so far.** Done: dense 5.68, homogeneous 7.79 at 0.2 and 24.56 at 0.5, on wikitext.

**What to check, at stage 9.** On the dense baseline, wikitext and c4 perplexity must **differ**.
Identical values are the signature of the old `c4` bug, in which the c4 task re-evaluated wikitext.
Screening evaluates wikitext alone, so the check runs when c4 first does, at stage 9, against the
dense row there. The gate report says which of the two situations it is looking at rather than
reading an absent c4 as a failure.

**Read the gate.**

```bash
python generate_tables.py output/eval/huggyllama_llama_7b --report gates \
    --allocation_dir output/allocation_reports \
    -o output/gates/huggyllama_llama_7b/gates_stage1.md
```

**What to check in it.**

- The **Stage 1 gate** table: dense against both homogeneous anchors, and the realized ratio equal
  to the target on both.
- The c4 note. While screening it should say c4 was not evaluated, which is expected. If it ever
  says the two perplexities are identical, stop: the c4 task is re-evaluating wikitext.
- `__CKPT_HOM_0.2__` and `__CKPT_HOM_0.5__` resolve, and their `on disk` column in the **stage 9
  roster** says `yes`. It currently says `no`, because those checkpoints were deleted.

**Gate.** Nothing to decide, this stage always runs first, but it does resolve `__CKPT_HOM_0.2__`
and `__CKPT_HOM_0.5__` for stages 9 and 10. Record the two homogeneous perplexities: they are the
`hom` row of every table in chapter 4, and the gate report subtracts them as the `gain` column of
every heterogeneous table that follows.

---

## Stage 2: score x grouping (RQ2)

**Purpose.** Which grouping criterion and which spectral score make heterogeneous allocation work.
`inner_allocation` stays at `waterfill` throughout so the only axes are grouping and score.

**Runs.** 18: `{global, type, decoder} x {truncation, entropy, eff_rank} x {0.2, 0.5}`.

`hierarchical` is absent on purpose: with the default `param_share` outer policy it reproduces
`decoder` to the digit, so it only becomes a distinct configuration in stage 4.

**Offline preview.**

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --sweep "compression_ratio=0.2,0.5" \
    --sweep "group_criterion=global,type,decoder" \
    --sweep "score_metric=truncation,entropy,eff_rank" \
    --out_dir ./output/allocation_reports/stage2 --plots
```

Drop any cell that trips the degenerate-score warning or whose dispersion is near zero: it will
reproduce the homogeneous result and cost an hour to say so. Note the offline `mean_rank` ordering
before running, then compare it against the perplexity ordering afterwards. Where the two disagree
is itself a thesis result, since it bounds how far any offline proxy can substitute for evaluation.

```bash
python run_experiments.py args/experiments_stage2_score_grouping.json
```

**What to look at.** Rank the nine cells by wikitext perplexity at each ratio separately, then by
mean rank across the two. Also compare every cell against the stage 1 homogeneous row: this is the
first real read on RQ1.

> If **every** heterogeneous cell loses to homogeneous, do not conclude that RQ1 is negative yet.
> Run stage 3 first. Swift-SVD's own ablation has uncapped heterogeneous allocation losing to
> uniform, rescued only by a rank floor, and the default `--max_ratio 0.9` sits in exactly that
> regime.

**Read the gate.**

```bash
python generate_tables.py output/eval/huggyllama_llama_7b --report gates \
    --allocation_dir output/allocation_reports \
    -o output/gates/huggyllama_llama_7b/gates_stage2.md
```

**What to check in it.**

- The **Stage 2 gate** table, then `mean rank` per grouping and per score in the two aggregate
  tables below it. Those aggregates are what `__BEST_GROUPING__` and the two scores are read off.
- Any `not resolvable` note. It fired on ratio 0.2 for the nine stage 2 cells alone, and stopped
  firing once 2b and 2c widened the spread, which is the honest reading: the nine cells did not
  separate but the score family does.
- Any `realized removal drifted off target` note, which invalidates the row it names.
- The **offline against measured** table. Expect disagreement; it is a result, not a fault.
- Cross-check the peak of every row against `max_ratio_assigned` in
  `output/allocation_reports/stage2/summary.csv` before believing any ranking: a row whose peak
  passed 0.85 is reporting cap damage, not its score.

**Gate.** All four values come from the gate report, which shows the aggregate each one is read off
in its own table rather than only the answer:

- `__BEST_GROUPING__` = grouping with the best mean rank across both ratios. A grouping holds three
  scores here, so it is judged by the mean of their mean ranks.
- `__TOP1_SCORE__`, `__TOP2_SCORE__` = the two best scores within that grouping, averaged over
  ratios.
- `__BEST_FLAT_GROUPING__` = the better of `type` and `global`. Never `decoder`, see stage 6.
- `__CKPT_BEST_SCORE_0.2__`, `__CKPT_BEST_SCORE_0.5__` = the rank-1 checkpoint at each ratio, for
  stage 9. A promotion in 2b or 2c moves these too.

Only the nine cells of this stage decide it: the families added by 2b and 2c are promotion tests
against this gate's own winner, so counting them here would make the promotion circular.

---

## Stage 2b: squared spectra

**Purpose.** Whether weighting by energy (`_sq`) rather than amplitude changes the ranking. Feeds
thesis 3.1.1.

**Runs.** 6: `{truncation_sq, entropy_sq, eff_rank_sq} x {0.2, 0.5}` at `__BEST_GROUPING__`.

**Offline preview.** Sweep the six scores together and compare the *ratio maps*, not the scores. If
`truncation` and `truncation_sq` produce ratios that agree to three decimals, the pair is one
experiment and not two.

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --group_criterion __BEST_GROUPING__ \
    --sweep "compression_ratio=0.2,0.5" \
    --sweep "score_metric=truncation,truncation_sq,entropy,entropy_sq,eff_rank,eff_rank_sq" \
    --out_dir ./output/allocation_reports/stage2b
```

**Read the gate.**

```bash
python generate_tables.py output/eval/huggyllama_llama_7b --report gates \
    --allocation_dir output/allocation_reports \
    -o output/gates/huggyllama_llama_7b/gates_stage2b.md
```

**What to check in it.**

- The **Stage 2b gate** table, which holds both incumbents next to the three squared scores. The
  promotion note says whether either slot moved.
- `gain@0.2` against `gain@0.5` on the same row. `eff_rank_sq` gains 0.16 at one budget and loses
  9.88 at the other, which is the ratio-dependent inversion this stage exists to surface. A mean
  rank that averages the two hides it, so read the per-ratio columns.
- The peak of every row, again from `summary.csv`. The two squared scores that lost at 0.5 both
  pinned at the cap.

**Gate.** Both incumbents are in the table, and the top two of the combined ranking become
`__TOP1_SCORE__` and `__TOP2_SCORE__`. Either can move: a squared score that beats the second-placed
amplitude score takes that slot even if it does not take the first. `__CKPT_BEST_SCORE_*__` follows
whenever the top score changes.

---

## Stage 2c: Schatten p-norms

**Purpose.** Fills thesis section 3.1.1.2, which currently carries `% TODO - no results`.

**Runs.** 4: `{norm|1, norm|inf} x {0.2, 0.5}` at `__BEST_GROUPING__`, plus 2 conditional.

`norm|1` is the nuclear norm of the truncated tail and `norm|inf` its largest singular value: two
genuinely different signals. `norm|3` is dropped because it interpolates between them, which is not a
research question. `norm|-inf`, its smallest singular value, sits in
`args/experiments_stage2c_schatten_neg_inf.json` and runs only if the preview clears it.

**Offline preview.** `norm|-inf` is numerically the most fragile quantity in the whole score family.
Check that its scores are not all within rounding distance of each other before spending 2 runs on
it, and check the dispersion of all three against the screen above.

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --group_criterion __BEST_GROUPING__ --compression_ratio 0.5 \
    --sweep "score_metric=norm|1,norm|inf,norm|-inf" \
    --out_dir ./output/allocation_reports/stage2c
```

**Read the gate.**

```bash
python generate_tables.py output/eval/huggyllama_llama_7b --report gates \
    --allocation_dir output/allocation_reports \
    -o output/gates/huggyllama_llama_7b/gates_stage2c.md
```

**What to check in it.**

- The **Stage 2c gate** table and its promotion note.
- Whether `norm|-inf` was run at all. It pins **116 of 224** matrices in the offline preview, far past
  the peak screen, so it stays in `experiments_stage2c_schatten_neg_inf.json` unless a lower cap
  makes it feasible. Not running it is the result.
- `norm|inf` is the cautionary row: peak 0.8743 at ratio 0.5, just past the boundary, and it lost 8.88.

**Gate.** Same promotion test as 2b, over the same two slots.

---

## Stage 3: the peak curve

**Runs immediately after 2c, before stage 4.** The collected runs make the peak the first-order
effect rather than a hyperparameter detail: it separated all 28 heterogeneous cells into safe and
damaged with no exceptions, and three matrices out of 224 at the 0.9 cap cost 14 to 19 perplexity.
Nothing downstream can be compared until the peak is understood and frozen somewhere safe.

**Purpose.** Trace perplexity against peak, and find where the safe band ends. This is the largest
effect in the data and it is currently measured only at whatever peaks the scores happened to produce.

**The cap can only lower a peak.** That governs the whole stage design. `--max_ratio` clips an
allocation, so a curve needs a driver whose *uncapped* peak already sits at the top of the range:

- At ratio 0.5, `type`/`truncation` peaks at 0.90 uncapped, so caps of 0.55 through 0.85 all bind and
  the peak equals the cap exactly. It also happens to be the worst run collected, at 43.38, so the
  curve runs from catastrophic back to safe on one configuration.
- At the winning `decoder`/`eff_rank` the uncapped peak is 0.7242 at 0.5 and **0.2897 at 0.2**, so the
  cap is inert there. Every cap between 0.6 and 0.9 produces a byte-identical allocation at ratio 0.2,
  which `figures/map_distance.csv` reports as a largest difference of 0.0000. The four such runs an
  earlier draft prescribed were one run.

**Offline preview.** Already run, in `output/allocation_reports/stage3/`. Confirm the cap binds before
adding a cap to the grid: `max_ratio_assigned` equal to the cap means it binds, equal to something
lower means the run duplicates one you already have.

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --group_criterion type --score_metric truncation --inner_allocation waterfill \
    --compression_ratio 0.5 --sweep "max_ratio=0.55,0.65,0.75,0.85" \
    --out_dir ./output/allocation_reports/stage3
```

**Mostly collected already.** Eleven cap runs exist: `decoder`/`eff_rank` at caps `{0.7, 0.8, 0.85}`
at ratio 0.2 and `{0.6, 0.7, 0.8, 0.85}` at 0.5, and `global`/`truncation` and `type`/`truncation` at
caps `{0.7, 0.85}` at 0.5. They are what the findings above are built on. All eleven ran on the second
machine, so they are internally comparable and must not be read against the 31 earlier runs.

**Runs remaining.** 4, all at ratio 0.5, filling the low end of the curve:

- 2: `type`/`truncation` at caps `{0.55, 0.65}`
- 2: `global`/`truncation` at the same two

Together with the collected 0.70, 0.85 and uncapped points that gives five points per driver, from
below the safe band to past it. Ratio 0.2 has no cap runs, because nothing at that budget reaches a
cap worth setting; its arm is 3b.

The three `decoder`/`eff_rank` cap runs at ratio 0.2 are worth keeping as a record even though they
changed nothing: all three produce a peak of 0.28969 and measure 7.85, which is the clearest evidence
in the grid that the cap is inert at the winning configuration.

**What this replaces.** An earlier draft called this the rescue test and asked whether a tighter cap
saves the two cells that blew up. That question is still here, and is answered by the first eight
runs; what changed is that it is now read as a curve rather than as a yes or no, because the peak is
a continuous variable and the interesting part is where its damage begins.

## Stage 3b: peak headroom at 0.2

**Purpose.** No collected run at ratio 0.2 pushed its peak past 0.533, and the two that came closest
are the two best runs at that budget: `eff_rank_sq` at peak 0.533 gaining 0.17, `entropy_sq` at 0.473
gaining 0.05. If the safe band is absolute at around 0.85 rather than proportional to the budget,
there is untested headroom at 0.2 and the gain should keep growing with the peak until it collapses.
This is the cleanest test of the mechanism in the whole grid.

**The lever is aggressiveness, not the cap**, since the cap cannot raise a peak. The `softmax_temp`
inner policy has a temperature that sets it continuously, and the preview gives this ladder at ratio
0.2 with `eff_rank_sq` under `decoder`, cap 0.9:

| `--softmax_temp` | 1.0 | 0.7 | 0.5 | 0.35 | 0.25 | 0.15 |
|---|---|---|---|---|---|---|
| peak | 0.3055 | 0.3524 | 0.4148 | 0.5071 | 0.6205 | 0.8362 |
| smallest ratio | 0.079 | 0.053 | 0.030 | 0.013 | 0.004 | 0.0003 |

**Runs.** 4: temperatures `{0.5, 0.35, 0.25, 0.15}` at ratio 0.2, giving peaks 0.41, 0.51, 0.62 and
0.84 against the homogeneous 7.79 and the best collected 7.62.

**Read it with the second row in view.** Under a fixed budget a peak cannot rise without something
else falling, so the aggressive end also leaves matrices almost uncompressed — 0.0003 at temperature
0.15. The curve therefore measures the whole shape of the allocation, not the peak alone, and a
collapse at the top could be either the peak or the near-zero tail. If it collapses, the follow-up
that separates them is a cap: clip the same temperature at 0.6 and see whether the damage goes.

**Read the gate.**

```bash
python generate_tables.py output/eval/huggyllama_llama_7b --report gates \
    --allocation_dir output/allocation_reports \
    -o output/gates/huggyllama_llama_7b/gates_stage3.md
```

**What to check in it.**

- The **Stage 3 gate** table, read as a curve: perplexity against the cap, which at this driver is
  the peak exactly. Where it turns is the end of the safe band, and the collected runs put that
  between 0.83 and 0.87.
- Whether the two drivers turn at the **same** peak. `type` and `global` differ only in where the
  budget comes from, so a common turning point says the boundary is a property of the model rather
  than of the allocation, which is the stronger claim.
- How far the rescue goes. If capping pulls `type`/`truncation` from 43.38 back to within a
  perplexity of homogeneous, the catastrophes were peak damage alone and the stage 2 grouping ranking
  has to be re-read at the safe cap. If a gap remains, that residual is the score genuinely
  misallocating, and it is the honest size of the score effect.
- The two spine rows at caps 0.6 and 0.7 against the uncapped 23.91. Clipping the winner's peak from
  0.7242 down should cost a little if the peak is doing useful work, and cost nothing if it is not.
- `figures/map_distance.csv` before believing any row is new: a largest difference under 0.02 against
  a run you already have means the cap changed nothing.

**Gate.** `--max_ratio` = the cap at the top of the safe band, carried into `args/base_args.json` for
every later stage. Report the RQ1 verdict at that cap, and if it is below 0.9 say in thesis 3.3 that
the cap is a first-order hyperparameter rather than a guard rail.

---

## Stage 3c: the outer level, and the early-layer control

**This is the thesis contribution's own test, and after the cap sweep it is the highest-value stage in
the grid.** It was a block inside stage 4; it is promoted here because the findings turned it from an
analogy into a prediction.

**The prediction.** The flat groupings lose eight perplexity by concentrating their tail on layers 0
to 3, and layer 0 carries a Block Influence of 0.4602 against 0.15 or less for almost everything else.
The outer level gives high-influence blocks less removal, so it should protect exactly the blocks that
`global` and `type` over-compress, deliberately and from a signal, where `decoder` only manages it
as a side effect of flattening every block alike. If the outer level is worth anything, this
is where it shows.

**Runs.** 6:

- 4: `hierarchical` + `--outer_allocation waterfill` x `{truncation, __TOP1_SCORE__}` x
  `{0.2, 0.5}`, inner held at `waterfill`. `truncation` is in the set because it is where the
  grouping lever is largest, so it is where protection has the most to recover.
- 2: the **early-layer control**, `global` and `type` with `truncation` at 0.5, capped at 0.7 and
  `--bypass_early_layers 2`. This asks whether simply exempting the first two blocks recovers the
  eight perplexity, which separates "the tail must be kept off the early layers" from "the outer level
  allocates better".

**Its baselines already exist**: `decoder` + `param_share` at both scores and both ratios, and the
capped `global`/`type` runs from stage 3. `hierarchical` + `param_share` reproduces `decoder` exactly,
so the outer policy is the only factor moving.

**Read the gate.**

```bash
python generate_tables.py output/eval/huggyllama_llama_7b --report gates \
    --allocation_dir output/allocation_reports \
    -o output/gates/huggyllama_llama_7b/gates_stage3c.md
```

**What to check in it.**

- The **Stage 4 gate: the outer level** table, which is where these rows land. Compare each against
  `decoder` + `param_share` at the same score and ratio: that pair differs in one factor only.
- `figures/ratio_tail.csv` for the hierarchical runs before reading any perplexity. If the outer level
  is doing what it should, `layers_of_top_8` no longer starts at 0 and `above_0.7` falls.
- The two control rows against their uncapped and capped siblings. If bypassing two blocks recovers
  most of the eight perplexity, the story is about the early layers and any mechanism that protects
  them will do; if it does not, the outer level's allocation is doing something a blunt exemption
  cannot.
- Whether `truncation` benefits more than `__TOP1_SCORE__`. The prediction is yes, because the lever
  is proportional to how sharply the score varies across depth.

**Gate.** Reporting, plus whichever of `decoder` + `param_share` and `hierarchical` + `waterfill` wins
becomes the configuration stages 4 onward are held at.

---

## Stage 4: allocation policies (RQ3)

**Runs after stage 3**, under the cap stage 3 froze. At `--max_ratio 0.9` this table would rank the
policies by which of them reaches the cap, and the preview below already answers that for free.

**Purpose.** Whether the policy that spends a group budget matters independently of the score that
ranks the matrices, and whether the outer level of the hierarchical allocator earns its place.

**Offline preview: already run**, in `output/allocation_reports/stage4_knobs/`. It swept the four
policies against `--offset {1.2, 1.5, 2.0}` and `--softmax_temp {0.05, 0.2, 1.0}` at ratio 0.2, and
it settles two things.

**`--offset` is inert.** It does not move the allocation of any of the four policies: `waterfill`
shifts its dispersion from 0.0184 to 0.0182 across the whole range, and `drank_lagrangian` and
`swift_pool` are identical to four decimals because neither declares the knob, so
`select_policy_arguments` never hands it to them. The prescription in an earlier draft — match the
policies' dispersion by tuning `--offset` — is not achievable, and the only live knob is
`--softmax_temp`, which belongs to one policy.

**The policies differ in aggressiveness by construction, and that is the finding.** At ratio 0.2
with the cap at 0.9:

| policy | `ratio_std` | peak assigned | pinned at cap |
|---|---|---|---|
| `waterfill` | 0.018 | 0.304 | 0 |
| `softmax_temp`, temp 1.0 (default) | 0.082 | 0.411 | 0 |
| `drank_lagrangian` | 0.159 | 0.536 | 0 |
| `swift_pool` | 0.247 | 0.794 | 0 |
| `softmax_temp`, temp 0.2 | 0.304 | **0.900** | 31 |
| `softmax_temp`, temp 0.05 | 0.342 | **0.900** | 33 |

So `softmax_temp` stays at its default 1.0: the other two temperatures pin a seventh of all matrices
at the cap *at ratio 0.2*, which the peak screen rejects outright. And since dispersion cannot be
equalised across the other three, report it beside the result and treat aggressiveness as a property
of each policy rather than a nuisance to be tuned away. The same sweep at 0.5 is worth running
before the stage, because every peak above will be higher there and `swift_pool` starts from 0.794.

**Runs.** 32:
- 24: `{drank_lagrangian, swift_pool, softmax_temp}` x `{__TOP1_SCORE__, __TOP2_SCORE__}` x
  `{__BEST_GROUPING__, __BEST_FLAT_GROUPING__}` x `{0.2, 0.5}` (`waterfill` at those settings is
  already in stage 2)
- 8: `hierarchical` with `--outer_allocation waterfill` x all four inner policies x `{0.2, 0.5}`

**Why two groupings.** The policy ranking is *guaranteed* to depend on the grouping, because
`drank_lagrangian` allocates in rank space and prices a rank at `out + in`, so its shape bias only
bites when a group mixes shapes. Under `type` every bucket holds one matrix family across 32 blocks
and every shape is identical, so the bias is inert; under `decoder` a bucket mixes 4096x4096
attention with 11008x4096 MLP and it is live. Freezing the grouping before this stage would put that
confound directly on RQ3. Running both costs 12 extra runs and settles it.

**What to look at.** The second block against `decoder` + `param_share` from stage 2 is the
controlled ablation of the **outer level**, since the two criteria bucket matrices identically and
differ only in whether Block Influence gets to move budget between blocks. That comparison is the
thesis contribution's own test, so report it separately from the inner-policy comparison.

Stage 2 predicts it loses: depth movement hurt when spectra drove it, and the outer level exists to
re-introduce depth movement driven by Block Influence instead. A loss is therefore a result about
the signal, not a bug in the allocator. Should it come out flat rather than negative, the follow-up
is `--outer_offset`, which dials how much depth movement the outer level is allowed to create and
can be previewed offline for nothing.

`figures/ratio_by_type.csv` is where `drank_lagrangian`'s rank-space bias becomes visible against
ratio-space `waterfill`: it prices a rank at `out + in`, so on a group of mixed shapes it can
compress a family harder than its score alone would justify. That figure is thesis 3.3.5.

**Read the gate.**

```bash
python generate_tables.py output/eval/huggyllama_llama_7b --report gates \
    --allocation_dir output/allocation_reports \
    -o output/gates/huggyllama_llama_7b/gates_stage4.md
```

**What to check in it.**

- The **Stage 4 gate: inner allocation policies** table first, then its aggregate table, which is
  what `__BEST_INNER__` is read off.
- Whether the policy ranking is the same under both groupings. If it flips, report the interaction
  rather than a single winner: it means policy and grouping cannot be chosen independently.
- Any `confounded` note. `softmax_temp` is the one policy with a live knob, so a run at a
  non-default temperature next to runs at the default will trip it.
- The peak of every policy, from `output/allocation_reports/stage4_knobs/summary.csv`. At ratio 0.2
  and cap 0.9 the peaks are 0.30 (`waterfill`), 0.54 (`drank_lagrangian`), 0.79 (`swift_pool`) and
  0.41 (`softmax_temp` at its default). If the frozen cap has not brought all four inside the safe
  band, the table is ranking aggressiveness rather than shape.
- The **Stage 4 gate: the outer level** table, separately. It is the thesis contribution's own test
  and answers a different question from the inner comparison.

**Gate.** `__BEST_INNER__` = best inner policy, averaged over the scores, groupings and ratios it
was run at. If the ranking flips between the two groupings, report that instead of a single winner:
it means the policy and the grouping cannot be chosen independently, which is itself an RQ3 answer.

---

## Stage 5: bypassing outer blocks (RQ4)

**Purpose.** Whether exempting the first or last N decoder blocks beats compressing everything, and
whether that gain **cannibalizes** the heterogeneous gain.

**Runs.** 36: nine bypass settings x `{heterogeneous, homogeneous}` x `{0.2, 0.5}`.

Settings: `early {1, 4, 8}`, `late {1, 4, 8}`, then `2 + 2`, `4 + 4` and `2 + 1`. `--bypass_ratio`
stays at `0.0`, so a bypassed block is skipped entirely and its budget is pushed onto the rest.

The two ends carry the same depths on purpose, and that is what separates **placement** from
**amount**. Three settings exempt four blocks (`early 4`, `late 4`, `2 + 2`) and three exempt eight
(`early 8`, `late 8`, `4 + 4`), so inside each triple the budget pushed onto the remaining blocks is
identical and the only difference is where the exempt blocks sit. Without matched totals, a
first-against-last comparison prices placement and pushed-back budget at the same time. `2 + 1` is
the asymmetric small case, and `early 1` / `late 1` bound how little it takes to matter.

**Offline preview: this one catches infeasibility.** Bypassing blocks pushes their budget onto the
rest, and at ratio 0.5 with 8 blocks skipped the remainder may not be able to absorb it under the
cap. That surfaces as a budget-drift violation and a non-zero exit, for free.

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --group_criterion __BEST_GROUPING__ --score_metric __TOP1_SCORE__ \
    --inner_allocation __BEST_INNER__ \
    --sweep "compression_ratio=0.2,0.5" \
    --sweep "bypass_early_layers=-1,1,4,8" --sweep "bypass_late_layers=-1,1,4,8" \
    --out_dir ./output/allocation_reports/stage5
```

**What to look at.** The homogeneous arm is not padding, it is the whole point. Compute the
heterogeneous gain over homogeneous *at each bypass setting* and compare it to the gain at bypass 0
from stage 2. If the gain shrinks as bypass grows, the two mechanisms are competing for the same
redundancy, which is the second half of RQ4. The gate report pairs the two arms per setting and
states this comparison outright, taking the configuration to hold from the bypassed runs themselves.

**Read the gate.**

```bash
python generate_tables.py output/eval/huggyllama_llama_7b --report gates \
    --allocation_dir output/allocation_reports \
    -o output/gates/huggyllama_llama_7b/gates_stage5.md
```

**What to check in it.**

- The **Stage 5 gate** table, which pairs both arms at each bypass setting, and the `mean gain`
  column.
- The RQ4 note comparing the gain at bypass 0 against the best bypassed setting. A shrinking gain
  means bypassing and heterogeneity compete for the same redundancy.
- The three matched-total triples. Inside each, the budget pushed onto the remaining blocks is
  identical and only the placement differs, so a difference there is about placement alone.
- Any missing `het@` cell, which means that setting was infeasible and the offline preview should
  have caught it.
- The peak per setting: pushing a bypassed block's budget onto the rest raises every other matrix,
  so a setting that was safe at bypass 0 can cross 0.85 once 8 blocks are exempt.

**Gate.** `__BEST_BYPASS_EARLY__` and `__BEST_BYPASS_LATE__` = the setting with the best mean gain
over homogeneous, and `__CKPT_BEST_BYPASS_0.2__` = its heterogeneous checkpoint at 0.2, for stage 9.

**Stage 5b, the grouping probe.** 2 runs, `args/experiments_stage5b_bypass_grouping.json`: the
winning setting again under `__BEST_FLAT_GROUPING__`, heterogeneous only. Bypassing means different
things to the two groupings — under `decoder` it deletes whole groups and `param_share`
redistributes their budget between the survivors, under a flat grouping it thins one pool — so a
conclusion drawn at one may not transfer. The homogeneous arm allocates nothing and is
grouping-independent, which is why the probe is 2 runs rather than 4.

---

## Stage 6: composite scores

**Purpose.** Whether fusing a per-matrix spectral score with the per-block Block Influence beats
either alone. This is the scalar counterpart of the hierarchical allocator tested in stage 4.

**Gate to open this stage.** `figures/influence_vs_effrank_rho.csv` from stage 0 holds the Spearman
correlation between Block Influence and normalized effective rank, per matrix family. Swift-SVD
reports it as **negative**, which is what makes the two signals complementary and justifies fusing
them.

> If it comes out **positive** on LLaMA-7B, the two signals agree rather than complement, the
> `beta^alpha` convention is pushing both the same way, and these 12 runs are measuring something
> other than what they are meant to. Record the sign and revisit the fusion before spending the GPU
> time.

**Offline preview: this stage needs its own `--offset`.** `beta^0.5` spans `[1, 1.41]` while
`log(e + local)` is typically 5 to 8, so the fused score has a much narrower dynamic range and lands
closer to homogeneous under the same offset. That is a property of the fusion, not evidence the
second signal is useless, but it does mean the offset that suited stage 2 will under-spread here.

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --group_criterion __BEST_FLAT_GROUPING__ --compression_ratio 0.2 \
    --sweep "score_metric=composite|truncation|block_influence,composite|eff_rank|block_influence" \
    --sweep "fusion_alpha=0.0,0.25,0.5,0.75,1.0" --sweep "offset=1.05,1.2,1.5" \
    --out_dir ./output/allocation_reports/stage6 --plots
```

Pick the offset that gives dispersion comparable to the stage 2 winner, and confirm the three chosen
alphas actually produce distinct allocations rather than assuming 0.25 / 0.5 / 0.75 do. The
`fusion_alpha=1.0` point should collapse to homogeneous under a per-block grouping, which is a
useful check that the sweep is wired correctly.

**Runs.** 12: `composite|{__TOP1_SCORE__, __TOP2_SCORE__}|block_influence` x
`--fusion_alpha {0.25, 0.5, 0.75}` x `{0.2, 0.5}` at `__BEST_FLAT_GROUPING__`. The local halves are
held as placeholders like every other score past 2c, so a promotion in 2b or 2c is fused here too.

**Why never `decoder`.** Not because the grouping is bad, it won stage 2, but because it closes both
channels through which Block Influence could act. The fusion is

```
s_i = beta_b^alpha * log(e + local_i)^(1 - alpha),   beta in [1, 2] across blocks
```

Under `decoder`, one group *is* one block, so `beta_b` is a single constant `c` inside it and
`s_i = c * L_i^(1-alpha)`. Raising to a positive power and multiplying by a positive constant are
both monotone, so **the within-group ordering is exactly the local-only ordering at every alpha**:
Block Influence cannot reorder anything. And between groups, `param_share` splits by parameter count
and never reads a score at all, so the block-level signal has nowhere to go there either.

What is left is a rescaling that passes through the redundancy weight `1 / log(s + offset)` and
moves the ratios slightly. That makes `--fusion_alpha` a dispersion knob collinear with `--offset`,
so a win under `decoder` would be an offset artifact wearing a hypothesis's clothes. At `alpha = 1`
the score is constant outright and the allocation is exactly homogeneous, which `allocate_ratios`
catches with a `[BUDGET][WARNING]`; the harmful case is the one it does not catch.

Block Influence has exactly two channels: a **flat grouping**, where a group spans all 32 blocks so
beta varies inside it, and the **outer level** of stage 4. This stage is the first; stage 4's second
block is the second.

**What to look at.** `influence_tail` in the offline objective panel is circular with this score
metric, exactly as `frobenius_tail` is circular with `truncation`. Do not read a composite win on
that column as evidence.

Stage 2 predicts this stage loses: it re-opens cross-depth movement, and it does so under a grouping
already measured as worse than `decoder`. That makes it a control rather than a candidate, and a
negative result here is the scalar half of the same finding the outer level tests.

**Read the gate.**

```bash
python generate_tables.py output/eval/huggyllama_llama_7b --report gates \
    --allocation_dir output/allocation_reports \
    -o output/gates/huggyllama_llama_7b/gates_stage6.md
```

**What to check in it.**

- The **Block Influence against effective rank** table first. It is the gate on the whole stage, and
  it comes out negative on all seven matrix families here, which is what justifies fusing the two
  signals.
- The **Stage 6 gate: composite scores** table, and whether any row beats the plain score it was
  fused from. Stage 2's finding predicts it does not.
- Any note about a per-block grouping having been used. Under `decoder` the fusion cannot reorder
  anything and `--fusion_alpha` degenerates into a dispersion knob.
- The peak per alpha. A higher alpha flattens the fused score, so the peak should fall as alpha
  rises; if it does not, the sweep is not wired the way it reads.

**Gate.** `__CKPT_BEST_COMPOSITE_0.2__` = the best composite checkpoint at 0.2, for stage 9. The
composite scores also enter the stage 7 finalist ranking on equal footing with the plain ones.

---

## Stage 7: cross-model confirmation

**Blocked.** This is the one stage a single model cannot answer. The file names Qwen2.5-7B and
Qwen2.5-32B and stays unrunnable until their whitening artifacts exist.

**Runs.** 12: three finalist configurations x two models x two ratios.

**Offline preview.** Once the whitening exists, run stage 0 against each new model *first*. The
Spearman sign and the score-versus-depth shape are model properties, and if they differ from
LLaMA-7B then the finalists chosen here may not transfer, which is itself worth reporting.

**Gate.** Fill the three finalists from the best configurations found in stages 2 to 6. Each needs a
grouping, a score and an inner policy:

- `__FINALIST1_GROUPING__`, `__FINALIST1_SCORE__`, `__FINALIST1_INNER__`
- `__FINALIST2_GROUPING__`, `__FINALIST2_SCORE__`, `__FINALIST2_INNER__`
- `__FINALIST3_GROUPING__`, `__FINALIST3_SCORE__`, `__FINALIST3_INNER__`

---

## Stage 8: the onset probe (RQ1)

**Runs it after stage 3**, so it inherits the frozen cap. It keeps the stage 8 section and file.

**Purpose.** Not a curve. Stage 2 measured no distinguishable effect at 0.2 and a 2.7% gain at 0.5,
so the question RQ1 can actually answer is **where between them the effect switches on**, and
whether it survives past 0.5. Two points locate that; a five-point curve would not, for a reason
worth stating in the thesis.

**Why not a curve.** There is no variance estimate anywhere in this design. `--seed 6363` is frozen
upstream of the whitening cache, so a repeat measurement means recomputing whitening, hours per
point. A five-point curve with no error bars supports a direction claim at best, and 0.2 against 0.5
already gives the direction. 0.35 is interpolation between two points already owned, and 0.8 puts
the model past 100 perplexity where the comparison stops meaning anything.

**Runs.** 4: `{homogeneous, best heterogeneous}` at `{0.35, 0.65}`.

**Offline preview: the shape claim lives here, for free.** Raw objective values move by orders of
magnitude with the budget, which drowns the difference between two policies at the same ratio. The
`<objective>_oracle_ratio` columns divide out the budget's own contribution and leave the points
comparable, so sweep as many budgets as you like and report the shape as an allocation-level result,
clearly labelled as not a perplexity claim.

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --group_criterion __BEST_GROUPING__ --score_metric __TOP1_SCORE__ \
    --inner_allocation __BEST_INNER__ \
    --sweep "compression_ratio=0.2,0.35,0.5,0.65,0.8" \
    --out_dir ./output/allocation_reports/stage8 --plots
```

**What to look at.** Whether 0.35 sits with 0.2 (no measurable effect) or with 0.5 (a real gain).
That boundary is the honest scope of the RQ1 claim, and if it falls above 0.35 the thesis has to say
that the window where allocation matters and the window where the model is usable may not overlap.
The gate report prints one row per budget and names the winning configuration at each: if the name
changes between rows, that change is itself the answer.

**Read the gate.**

```bash
python generate_tables.py output/eval/huggyllama_llama_7b --report gates \
    --allocation_dir output/allocation_reports \
    -o output/gates/huggyllama_llama_7b/gates_stage8.md
```

**What to check in it.**

- The **Stage 8 gate** table, one row per budget, with the winning configuration named per row.
- Whether 0.35 sits with 0.2 or with 0.5. That boundary is the honest scope of the RQ1 claim.
- Whether the configuration name changes between rows. If it does, no single allocation is best
  across budgets, which is the finding rather than an inconvenience.

**Gate.** None, this stage resolves no placeholder. It is read, not consumed.

---

## Stage 9: finalists on the full suite

**Purpose.** The headline table of chapter 4.

**Runs.** 9 evaluation-only runs: the dense reference plus 8 checkpoints. No recompression, each
entry loads an existing checkpoint through `--use_compressed --compressed_model_path`.

**Tasks.** `arc_easy, hellaswag, openbookqa, piqa, winogrande, gsm8k, truthfulqa_gen`. The last two
are generative and dominate the wall clock.

**Offline preview.** None: nothing is allocated here.

**Read the gate.**

```bash
python generate_tables.py output/eval/huggyllama_llama_7b --report gates \
    --allocation_dir output/allocation_reports \
    -o output/gates/huggyllama_llama_7b/gates_stage9.md
```

**What to check in it.**

- The **Stage 9 roster**, specifically the `on disk` and `full suite done` columns, before queueing
  anything. Every role currently reads `no` on disk.
- The benchmark tables from `--report benchmarks`, which is where c4 finally appears beside
  wikitext.
- The dense row's wikitext against c4: this is where the check deferred from stage 1 happens.

**Gate.** Fill each placeholder with a real path from `output/models/huggyllama_llama_7b/`. The role
names what each slot is for:

| Placeholder | Comes from |
|---|---|
| `__CKPT_HOM_0.2__`, `__CKPT_HOM_0.5__` | stage 1 |
| `__CKPT_BEST_SCORE_0.2__`, `__CKPT_BEST_SCORE_0.5__` | stage 2, or 2b / 2c if one promoted |
| `__CKPT_BEST_POLICY_0.2__`, `__CKPT_BEST_POLICY_0.5__` | stage 4 |
| `__CKPT_BEST_BYPASS_0.2__` | stage 5 |
| `__CKPT_BEST_COMPOSITE_0.2__` | stage 6 |

`mathqa` is worth one attempt but is expected to fail, since `datasets` no longer permits custom
loading scripts. Report the failure rather than working around it.

---

## Stage 10: LoRA sequential update (RQ5)

**Purpose.** Whether heterogeneous allocation is a better *starting point* for fine-tuning, and
whether the gap over homogeneous widens or closes after the update.

**Runs.** 4: `{homogeneous, best heterogeneous} x {0.2, 0.5}`, each via `--update_taw_only` on an
existing checkpoint, so no recompression.

**Read the gate.**

```bash
python generate_tables.py output/eval/huggyllama_llama_7b --report gates \
    --allocation_dir output/allocation_reports \
    -o output/gates/huggyllama_llama_7b/gates_stage10.md
```

**What to check in it.**

- The **Stage 10 gate** table: before, after and recovered per arm.
- The het-minus-hom gap before the update against the same gap after it. A gap that closes means the
  update recovers what a bad allocation lost, and RQ5 is answered negatively.

**Gate.** `__CKPT_HOM_0.2__` and `__CKPT_HOM_0.5__` are the stage 1 checkpoints; `__CKPT_HET_0.2__`
and `__CKPT_HET_0.5__` are whichever heterogeneous configuration won overall, so this stage waits on
stages 2 to 6.

**Offline preview.** None: the allocation is already fixed in the checkpoint being updated.

**What to look at.** The het-minus-hom gap before the update (from stage 2 or 3) against the same
gap after it. A gap that closes means the update recovers what a bad allocation lost, and RQ5 is
answered negatively.

---

## Totals

In execution order, with the two completed stages marked.

| Stage | Runs | Evaluation | Offline preview |
|---|---|---|---|
| 0 offline | 0 | none | the stage itself |
| 1 anchors | 3 **done** | wikitext | not needed |
| 2 score x grouping | 18 **done** | wikitext | peak screen, prune degenerate cells |
| 2b squared | 6 **done** | wikitext | compare ratio maps |
| 2c Schatten | 4 **done** (+2 rejected by the peak screen) | wikitext | check `norm\|-inf` is not noise |
| 3 peak curve | 4 (+11 **done**) | wikitext | confirms every cap binds, so no run repeats another |
| 3b peak headroom | 4 | wikitext | the temperature ladder, already measured |
| 3c outer level | 6 | wikitext | `ratio_tail`, to see whether the tail leaves layer 0 |
| 4 policies | 24 | wikitext | its knob preview is already run and settles them |
| 8 onset | 4 | wikitext | oracle-normalized, and where the shape claim lives |
| 5 bypass | 36 | wikitext | catches infeasible budgets |
| 5b bypass x grouping | 2 | wikitext | same preview as 5 |
| 6 composite | 12 | wikitext | **gated** on the Spearman sign, sets the offset |
| 7 cross-model | 12 | blocked | rerun stage 0 per model |
| 9 benchmarks | 9 | full suite **and c4** | needs its checkpoints rebuilt first |
| 10 LoRA | 4 | wikitext, no recompression | not applicable |

**97 compression runs remaining** on LLaMA-7B, on top of the 42 already collected, plus 13
evaluation-only or update-only runs, plus 12 blocked, plus **roughly 10 recompressions** to rebuild
the stage 9 and 10 checkpoint inputs that were deleted.

GPU time is not the binding constraint on this grid, so these counts describe it rather than budget
it: spend runs wherever they resolve something. What the offline pass buys is not saved hours but the
guarantee that each hour buys a distinct experiment. The peak screen rejects a cell that is going to
fail, and the map-distance screen rejects a cell that repeats one already collected — which is how
four of the runs this document used to prescribe for stage 3 turned out to be one run.
