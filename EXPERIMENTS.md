# Experiment grid

The staged design behind the results of the thesis, and the source for thesis section 4.1.

Every stage is one axis swept around a fixed configuration, with a **gate** at the end that resolves
the placeholders of the next stage. The grid is deliberately not a cross product: the full cross for
a single model at a single ratio is already 4 groupings x 6 scores x 4 inner policies = 96 runs.

Stages 1 to 8 evaluate **perplexity only** (`wikitext,c4|0`), which is what makes a ~100 run grid
affordable. Only stage 9 pays for the full benchmark suite.

Whitening is assumed to be already cached under `output/whitening_matrices/<model>/v2/`, together
with its `spectra/` cache and `layer_importance.pt`. Every stage below reads that cache and never
recomputes it.

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

**It resolves no placeholder on its own.** Every `__*__` value in this grid is defined as a ranking by
perplexity, and perplexity needs the model. What the offline report does instead is decide **which
runs are worth making** and **what to hold them at**, which is where the hours are actually saved.

Read it as answering four questions, in this order:

1. **Is the configuration even feasible?** A cap too low to reach the target once bypassed layers are
   charged shows up as a budget-drift violation, a `checks` entry in `summary.csv`, and a non-zero
   exit. Fix the configuration before running anything.
2. **Does the variant allocate anything?** A score that is constant inside every group produces the
   flat ratio whatever policy runs, so the run is homogeneous while looking heterogeneous.
   `allocate_ratios` prints a `[BUDGET][WARNING]` for it, and `figures/dispersion.csv` shows a
   `ratio_std` near zero. Drop the cell: it will reproduce the stage 1 homogeneous number and take an
   hour to say so.
3. **Are two variants distinguishable?** Two configurations whose ratios agree to three decimals are
   one experiment, not two. Compare them in `matrices.csv`, which carries the assigned ratio per
   matrix. This is what makes the stage 2b pairs (`truncation` against `truncation_sq`) worth checking
   before spending six runs on them, and it is the one case where the offline pass can settle a gate
   outright: a candidate that cannot allocate differently cannot win a promotion, so `__TOP1_SCORE__`
   stays where stage 2 put it.
4. **Are the variants comparable to each other?** Policies compared at their default knobs differ in
   shape *and* in aggressiveness at once. Match `ratio_std` across them from
   `figures/dispersion.csv`, then freeze `--offset`, `--softmax_temp` and `--outer_offset` in
   `args/base_args.json`. That is the offline pass configuring the GPU stage, not just previewing it.

One gate is answered offline and only offline: **the Spearman sign in stage 6**. It is a go/no-go on
whether fusing Block Influence with a spectral score measures what it is meant to, and no perplexity
number can substitute for it.

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

**Name `--out_dir` after the stage it previews**: `stage2`, `stage2b`, `stage3`, and so on under
`output/allocation_reports/`. `generate_tables.py --allocation_dir` discovers stage directories by
that name and attaches each preview to the gate it belongs to. A suffix is allowed and keeps the same
stage, so `stage3_knobs` is read as stage 3; when both exist the unsuffixed one is used. A directory
named anything else is simply not picked up. Giving one `--out_dir` per stage is also what stops each
preview from overwriting the last.

### Reading its output

The console table is the summary: one row per variant, ordered by mean rank, with the swept axes named
once in the header. Each objective cell holds the objective value and, in parentheses, its rank across
the variants. The `checks` column is `ok` or the invariant that failed.

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
python run_experiments.py args/experiments_stage3_policies.json --base args/other_base.json
```

The runner merges `args/base_args.json` into each entry of the stage file, refuses to start if any
entry still holds an unresolved gate value, and continues to the next run when one fails rather than
aborting the queue.

The commands it prints are informational, not copy-paste ready: `--eval_tasks wikitext,c4|0` holds a
pipe that a shell would interpret. `subprocess` passes it as one argument, so runs are unaffected.

`args/` is tracked, so the grid is reproducible from the repository. **Never put `--hf_token` in one
of these files**: pass it on the command line, or export it in the environment.

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

Stage files carry literal placeholders until their gate resolves them. Every one is filled from the
gate report, and the offline preview only narrows the candidates that go into it:

| Placeholder | Resolved by | Meaning | What the offline preview contributes |
|---|---|---|---|
| `__BEST_GROUPING__` | stage 2 | grouping criterion with the best mean rank | drops groupings whose scores are degenerate inside every group |
| `__BEST_FLAT_GROUPING__` | stage 2 | better of `type` / `global`, never `decoder` | confirms both flat groupings actually spread their ratios |
| `__TOP1_SCORE__`, `__TOP2_SCORE__` | stage 2 (2b, 2c may promote) | the two best score metrics | drops a candidate whose ratio map matches an incumbent's, since it cannot win a promotion |
| `__BEST_INNER__` | stage 3 | best inner allocation policy | **required**: the knobs must be matched on `ratio_std` first, or the comparison is confounded |
| `--max_ratio` | stage 4, into `args/base_args.json` | the cap every later stage runs at | drops caps that pin no matrix, from `cap_binding.csv` |
| `__CKPT_<ROLE>__` | stages 1 to 8 | a path under `output/models/huggyllama_llama_7b/` | nothing, these are outputs of runs |
| `__FINALIST{1,2,3}_*` | stages 2 to 6 | the three configurations worth testing on another model | rerun stage 0 per model: the Spearman sign and score-versus-depth shape are model properties, and the finalists may not transfer |

`run_experiments.py` refuses to start while any remain, so an unfilled gate cannot silently run the
wrong configuration. The gate report prints the same list with the resolved value beside each one, and
`waiting on runs` where a stage has not produced its answer yet.

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
  and reports how many ratios priced each row: stage 4 runs different caps at different ratios, and
  a mean rank from one ratio is not comparable to one from both.
- Carries the **gain over the homogeneous arm** at the same setting, which is the RQ1 read every
  stage repeats, and for stage 5 pairs the two arms bypass setting by bypass setting.
- **Holds fixed** whatever a stage is not sweeping, and reports which runs that excluded. Stage 4's
  cap sweep matches stage 3's table in every dimension except the cap, so left in it would decide
  stage 3's gate through the cap. Stage 5 reads the setting to hold from its own bypassed runs, since
  its bypass-0 reference otherwise sits among every other stage's runs.
- Warns when a dimension moves inside a table without being one of its axes, and when a run's
  realized removal drifted off the budget it is being compared at.
- Attaches the offline preview of each stage from `--allocation_dir`, including the Spearman sign that
  gates stage 6, the dispersion used to match the stage 3 knobs, and the cap binding behind stage 4.
  For stage 2 it also prints the offline ordering against the measured one, which is the disagreement
  worth reporting: it bounds how far the free preview can substitute for an hour of GPU per cell.

The dimensions all come from the sidecar, the only place most of them exist. A run without one is
counted and left out rather than guessed at.

## Passing a gate, step by step

The same six steps for every stage. Stage 3 is used here because it is the one stage whose offline
pass is mandatory and changes the runs rather than only pruning them.

**1. Preview the stage offline.** Sweep the axis, and sweep the knobs with it:

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --group_criterion __BEST_GROUPING__ --compression_ratio 0.2 \
    --sweep "inner_allocation=waterfill,drank_lagrangian,swift_pool,softmax_temp" \
    --sweep "offset=1.2,1.5,2.0" --sweep "softmax_temp=0.05,0.2,1.0" \
    --out_dir ./output/allocation_reports/stage3_knobs
```

Substitute the placeholder by hand here: this tool takes flags, not stage files, and does not read
`args/`. A non-zero exit means an invariant failed and the configuration is not runnable yet.

**2. Prune and configure from the CSVs.** Read `figures/dispersion.csv` and pick the knob values that
bring the four policies to a comparable `ratio_std`; drop any cell that tripped the degenerate-score
warning or whose `ratio_std` is near zero. Write the chosen knobs into `args/base_args.json`, and
delete the dropped cells from the stage file.

**3. Run the stage.**

```bash
python run_experiments.py args/experiments_stage3_policies.json --dry_run  # check the commands first
python run_experiments.py args/experiments_stage3_policies.json
```

**4. Read the gate.**

```bash
python generate_tables.py ./output/eval/huggyllama_llama_7b \
    --report gates --allocation_dir ./output/allocation_reports -o gates.md
```

**5. Check the gate's own warnings before trusting it.** A `confounded` note means a dimension moved
that the stage was not comparing, and the fix is a run rather than a reading. A `priced at 1/2` row
was ranked on one ratio and is not comparable to a row ranked on both. A drift note means a run missed
its budget and is not comparable at all.

**6. Copy the resolved value into the next stage file.** Take it from the **Placeholders** table, not
from the body tables, and replace the literal string:

```bash
sed -i 's/__BEST_INNER__/drank_lagrangian/g' args/experiments_stage4_max_ratio.json
```

Then repeat from step 1 for the next stage. `run_experiments.py` refusing to start is the backstop: it
means a placeholder was missed.

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
| 2 | `stage2` | the console mean rank, `figures/dispersion.csv` | which of the nine cells to drop before spending 18 runs |
| 2b | `stage2b` | `matrices.csv`, the ratio column per matrix | whether a `_sq` score allocates differently from the score it derives from |
| 2c | `stage2c` | `matrices.csv`, `figures/dispersion.csv` | whether `norm\|-inf` is signal or rounding noise |
| 3 | `stage3_knobs` | `figures/dispersion.csv`, `figures/ratio_by_type.csv` | the knobs to freeze in `args/base_args.json`, **mandatory** |
| 4 | `stage4` | `figures/cap_binding.csv` | which caps pin any matrix, so which caps are worth a run |
| 5 | `stage5` | the exit code and `checks` | whether the bypassed budgets are feasible under the cap |
| 6 | `stage6` | `figures/dispersion.csv`, plus the stage 0 rho | the offset for the fused score, and that the three alphas allocate distinctly |
| 7 | `stage0` again, per model | the rho sign, `figures/scores_by_depth.csv` | whether the finalists transfer to that model |
| 8 | `stage8` | the `<objective>_oracle_ratio` columns of `summary.csv` | how to compare five budgets without the budget dominating |
| 9, 10 | none | | both stages load existing checkpoints and allocate nothing |

The preview never needs a GPU and never touches the model weights, so re-running one after changing a
knob costs seconds. Re-run stage 0 whenever the whitening cache changes.

---

## Stage 0: the offline pass

**Purpose.** Everything that can be known without a GPU, plus the two gates that govern later stages.
Costs seconds. Run it before anything else and re-run it whenever the whitening cache changes.

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

**What to check.** On the dense baseline, wikitext and c4 perplexity must **differ**. Identical
values are the signature of the old `c4` bug, in which the c4 task re-evaluated wikitext, and would
mean the fix did not take effect in this environment.

**Gate.** Nothing to decide, this stage always runs first, but it does resolve `__CKPT_HOM_0.2__` and
`__CKPT_HOM_0.5__` for stages 9 and 10. Record the two homogeneous perplexities: they are the `hom`
row of every table in chapter 4, and the gate report subtracts them as the `gain` column of every
heterogeneous table that follows.

---

## Stage 2: score x grouping (RQ2)

**Purpose.** Which grouping criterion and which spectral score make heterogeneous allocation work.
`inner_allocation` stays at `waterfill` throughout so the only axes are grouping and score.

**Runs.** 18: `{global, type, decoder} x {truncation, entropy, eff_rank} x {0.2, 0.5}`.

`hierarchical` is absent on purpose: with the default `param_share` outer policy it reproduces
`decoder` to the digit, so it only becomes a distinct configuration in stage 3.

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
> Run stage 4 first. Swift-SVD's own ablation has uncapped heterogeneous allocation losing to
> uniform, rescued only by a rank floor, and the default `--max_ratio 0.9` sits in exactly that
> regime.

**Gate.** All four values come from the gate report, which shows the aggregate each one is read off in
its own table rather than only the answer:

- `__BEST_GROUPING__` = grouping with the best mean rank across both ratios. A grouping holds three
  scores here, so it is judged by the mean of their mean ranks.
- `__TOP1_SCORE__`, `__TOP2_SCORE__` = the two best scores within that grouping, averaged over ratios.
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

**Gate.** If a squared variant beats both scores chosen in stage 2, it replaces `__TOP1_SCORE__`
before stage 3 runs.

---

## Stage 2c: Schatten p-norms

**Purpose.** Fills thesis section 3.1.1.2, which currently carries `% TODO - no results`.

**Runs.** 8: `{norm|1, norm|3, norm|inf, norm|-inf} x {0.2, 0.5}` at `__BEST_GROUPING__`.

The four values span genuinely different signals: `norm|1` is the nuclear norm of the truncated
tail, `norm|inf` its largest singular value, `norm|-inf` its smallest.

**Offline preview.** Worth doing carefully here, because `norm|-inf` reads the smallest singular
value of the tail, which is numerically the most fragile quantity in the whole score family. Check
that its scores are not all within rounding distance of each other before spending 2 runs on it.

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --group_criterion __BEST_GROUPING__ --compression_ratio 0.2 \
    --sweep "score_metric=norm|1,norm|3,norm|inf,norm|-inf" \
    --out_dir ./output/allocation_reports/stage2c
```

**Gate.** Reporting only, unless one of them beats `__TOP1_SCORE__`, in which case it is promoted.

---

## Stage 3: allocation policies (RQ3)

**Purpose.** Whether the policy that spends a group budget matters independently of the score that
ranks the matrices, and whether the outer level of the hierarchical allocator earns its place.

**Offline preview: mandatory, and it changes the configuration.** Comparing four policies at their
default knobs compares their *shape* and their *aggressiveness* at once, which is a confound sitting
directly on RQ3.

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --group_criterion __BEST_GROUPING__ --compression_ratio 0.2 \
    --sweep "inner_allocation=waterfill,drank_lagrangian,swift_pool,softmax_temp" \
    --sweep "offset=1.2,1.5,2.0" --sweep "softmax_temp=0.05,0.2,1.0" \
    --out_dir ./output/allocation_reports/stage3_knobs
```

Read `figures/dispersion.csv` and pick `--offset`, `--softmax_temp` and `--outer_offset` so the four
policies produce comparable ratio standard deviations, then set them in `args/base_args.json`. A knob
left at its default emits no filename token, so changing one here changes the run names of this stage
only.

**Runs.** 20:
- 12: `{drank_lagrangian, swift_pool, softmax_temp} x {top1, top2} x {0.2, 0.5}` at `__BEST_GROUPING__`
  (`waterfill` at those settings is already in stage 2)
- 8: `hierarchical` with `--outer_allocation waterfill` x all four inner policies x `{0.2, 0.5}`

**What to look at.** The second block against `decoder` + `param_share` from stage 2 is the
controlled ablation of the **outer level**, since the two criteria bucket matrices identically and
differ only in whether Block Influence gets to move budget between blocks. That comparison is the
thesis contribution's own test, so report it separately from the inner-policy comparison.

`figures/ratio_by_type.csv` is where `drank_lagrangian`'s rank-space bias becomes visible against
ratio-space `waterfill`: it prices a rank at `out + in`, so on a group of mixed shapes it can
compress a family harder than its score alone would justify. That figure is thesis 3.3.5.

**Gate.** `__BEST_INNER__` = best inner policy at `__BEST_GROUPING__`, averaged over ratios.

---

## Stage 4: the per-matrix cap

**Purpose.** `--max_ratio` bounds how far any single matrix may be compressed. Swift-SVD reports that
without such a floor a heterogeneous allocation is *worse* than uniform, so this is a confound
sitting directly on RQ1, not a hyperparameter detail.

**Offline preview.** `figures/cap_binding.csv` reports how many matrices each cap actually pins. A
cap that pins nothing cannot change an allocation, so drop it before it costs an hour.

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --group_criterion __BEST_GROUPING__ --score_metric __TOP1_SCORE__ \
    --inner_allocation __BEST_INNER__ \
    --sweep "compression_ratio=0.2,0.5" --sweep "max_ratio=0.4,0.6,0.75,0.8,0.9" \
    --out_dir ./output/allocation_reports/stage4
```

**Runs.** 5: caps `{0.4, 0.6, 0.8}` at target 0.2 and `{0.6, 0.75}` at target 0.5.

**Gate.** If a lower cap wins, the RQ1 verdict is reported at that cap, and thesis 3.3 must state
that the cap is a first-order hyperparameter rather than a guard rail. Carry the winning cap into
stages 5, 6 and 8 by setting it in `args/base_args.json`.

---

## Stage 5: bypassing outer blocks (RQ4)

**Purpose.** Whether exempting the first or last N decoder blocks beats compressing everything, and
whether that gain **cannibalizes** the heterogeneous gain.

**Runs.** 24: six bypass settings x `{heterogeneous, homogeneous}` x `{0.2, 0.5}`.

Settings: `early 2`, `early 4`, `early 8`, `late 1`, `late 2`, `early 2 + late 1`. `--bypass_ratio`
stays at `0.0`, so a bypassed block is skipped entirely and its budget is pushed onto the rest.

**Offline preview: this one catches infeasibility.** Bypassing blocks pushes their budget onto the
rest, and at ratio 0.5 with 8 blocks skipped the remainder may not be able to absorb it under the
cap. That surfaces as a budget-drift violation and a non-zero exit, for free.

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --group_criterion __BEST_GROUPING__ --score_metric __TOP1_SCORE__ \
    --inner_allocation __BEST_INNER__ \
    --sweep "compression_ratio=0.2,0.5" \
    --sweep "bypass_early_layers=-1,2,4,8" --sweep "bypass_late_layers=-1,1,2" \
    --out_dir ./output/allocation_reports/stage5
```

**What to look at.** The homogeneous arm is not padding, it is the whole point. Compute the
heterogeneous gain over homogeneous *at each bypass setting* and compare it to the gain at bypass 0
from stage 2. If the gain shrinks as bypass grows, the two mechanisms are competing for the same
redundancy, which is the second half of RQ4. The gate report pairs the two arms per setting and states
this comparison outright, taking the configuration to hold from the bypassed runs themselves.

**Gate.** `__CKPT_BEST_BYPASS_0.2__` = the heterogeneous checkpoint of the best-gain bypass setting at
0.2, for stage 9.

---

## Stage 6: composite scores

**Purpose.** Whether fusing a per-matrix spectral score with the per-block Block Influence beats
either alone. This is the scalar counterpart of the hierarchical allocator tested in stage 3.

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
`fusion_alpha=1.0` point should collapse to homogeneous under a per-block grouping, which is a useful
check that the sweep is wired correctly.

**Runs.** 12: `composite|{truncation, eff_rank}|block_influence` x `--fusion_alpha {0.25, 0.5, 0.75}`
x `{0.2, 0.5}` at `__BEST_FLAT_GROUPING__`.

**Why never `decoder`.** Block Influence is constant inside a decoder block, so under `decoder` or
`hierarchical` grouping a fused score cannot rank matrices within a group and the allocation
collapses to exactly homogeneous. `allocate_ratios` prints a `[BUDGET][WARNING]` when it detects
this, but the stage file avoids the situation rather than relying on the warning.

**What to look at.** `influence_tail` in the offline objective panel is circular with this score
metric, exactly as `frobenius_tail` is circular with `truncation`. Do not read a composite win on
that column as evidence.

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

## Stage 8: the ratio curve (RQ1)

**Purpose.** RQ1 asks how the heterogeneous gain *depends* on the target ratio. Two points give a
slope but no shape, so this adds three more rather than tripling every earlier stage.

**Runs.** 6: `{homogeneous, best heterogeneous}` at `{0.35, 0.65, 0.8}`. Combined with 0.2 and 0.5
already collected, that is a five point curve for both arms.

**Offline preview: this is where the oracle earns its place.** Raw objective values move by orders of
magnitude with the budget, which drowns the difference between two policies at the same ratio. The
`<objective>_oracle_ratio` columns divide out the budget's own contribution and leave the five points
comparable.

```bash
python allocation_report.py --model "huggyllama/llama-7b" --run_v2 \
    --group_criterion __BEST_GROUPING__ --score_metric __TOP1_SCORE__ \
    --inner_allocation __BEST_INNER__ \
    --sweep "compression_ratio=0.2,0.35,0.5,0.65,0.8" \
    --out_dir ./output/allocation_reports/stage8 --plots
```

**What to look at.** Whether the gap widens, narrows or inverts with the ratio. At 0.8 a
heterogeneous allocation is heavily constrained by `--max_ratio`, so read this stage together with
stage 4 and with `figures/cap_binding.csv`. The gate report prints the curve as one row per budget and
names the winning configuration at each: if the name changes between rows, that change is the answer.

**Gate.** None, this stage resolves no placeholder. It is read, not consumed.

---

## Stage 9: finalists on the full suite

**Purpose.** The headline table of chapter 4.

**Runs.** 9 evaluation-only runs: the dense reference plus 8 checkpoints. No recompression, each
entry loads an existing checkpoint through `--use_compressed --compressed_model_path`.

**Tasks.** `arc_easy, hellaswag, openbookqa, piqa, winogrande, gsm8k, truthfulqa_gen`. The last two
are generative and dominate the wall clock.

**Offline preview.** None: nothing is allocated here.

**Gate.** Fill each placeholder with a real path from `output/models/huggyllama_llama_7b/`. The role
names what each slot is for:

| Placeholder | Comes from |
|---|---|
| `__CKPT_HOM_0.2__`, `__CKPT_HOM_0.5__` | stage 1 |
| `__CKPT_BEST_SCORE_0.2__`, `__CKPT_BEST_SCORE_0.5__` | stage 2, or 2b / 2c if one promoted |
| `__CKPT_BEST_POLICY_0.2__`, `__CKPT_BEST_POLICY_0.5__` | stage 3 |
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

**Gate.** `__CKPT_HOM_0.2__` and `__CKPT_HOM_0.5__` are the stage 1 checkpoints; `__CKPT_HET_0.2__`
and `__CKPT_HET_0.5__` are whichever heterogeneous configuration won overall, so this stage waits on
stages 2 to 6.

**Offline preview.** None: the allocation is already fixed in the checkpoint being updated.

**What to look at.** The het-minus-hom gap before the update (from stage 2 or 3) against the same gap
after it. A gap that closes means the update recovers what a bad allocation lost, and RQ5 is answered
negatively.

---

## Totals

| Stage | Runs | Evaluation | Offline preview |
|---|---|---|---|
| 0 offline | 0 | none | the stage itself |
| 1 anchors | 3 | perplexity | not needed |
| 2 score x grouping | 18 | perplexity | prune degenerate cells |
| 2b squared | 6 | perplexity | compare ratio maps |
| 2c Schatten | 8 | perplexity | check `norm\|-inf` is not noise |
| 3 policies | 20 | perplexity | **mandatory**, sets the knobs |
| 4 cap | 5 | perplexity | which caps bind |
| 5 bypass | 24 | perplexity | catches infeasible budgets |
| 6 composite | 12 | perplexity | **gated** on the Spearman sign, sets the offset |
| 7 cross-model | 12 | blocked | rerun stage 0 per model |
| 8 ratio curve | 6 | perplexity | oracle-normalized comparison |
| 9 benchmarks | 9 | full suite, no recompression | not applicable |
| 10 LoRA | 4 | perplexity, no recompression | not applicable |

**102 compression runs** on LLaMA-7B, plus 13 evaluation-only or update-only runs, plus 12 blocked.
Every stage is previewed offline first, at a cost of seconds.
