# Closing the gap between the thesis index and `svd_llm`

## Context

The thesis skeleton (`msc-thesis---s345139---parisini/thesis/main.tex`) promises a chapter comparing
several heterogeneous compression-ratio **allocation algorithms** and two families of **score
metrics**, and its research questions ask about the first *and last* N decoder blocks. Before this
plan the code implemented exactly one allocator (a parameter-weighted water-fill on
`1 / log(score + offset)`), only spectral scores, and only early-layer bypass, so RQ3 and half of
RQ4 were unanswerable.

The headline addition is **not** a reproduction: it is a two-level *hierarchical* allocator — budget
split across decoder blocks by end-to-end importance, then redistributed inside each block by local
spectral scores — which has no direct counterpart in the literature.

Every design decision below was chosen explicitly rather than assumed; §6 records them for
traceability.

---

## 1. Status

| # | Step | State |
|---|---|---|
| 1 | `--max_ratio` flag; `--bypass_late_layers` + `is_bypassed_key` | done |
| 2 | Run configuration sidecar + `generate_tables.py` fallback | done |
| 3 | c4 evaluation fix | done |
| 4 | Spectra cache (raw) + Block Influence in the whitening replay | done |
| 5 | Offline allocation tool (`allocation_report.py`) | done |
| 6 | Allocator refactor: shell/policy split, two registries | done |
| 7 | `drank_lagrangian`, `swift_pool`, `softmax_temp` | done |
| 8 | `--group_criterion hierarchical` + the outer level | done |
| 9 | Composite score grammar + `compose_scores` + `--fusion_alpha` | done |
| 10 | `README.md` / argparse sync; this file | done |
| 11 | LaTeX drafting (§2.4, §3, `biblio.bib`, §5.2, §5.2.2) | pending |
| 12 | Experiment grid (`args/*.json`) | pending |

`README.md` carries the full argument reference and has been checked against the argparse blocks of
`main.py`, `allocation_report.py`, `generate_tables.py` and `generate_text.py` — 96 flags, defaults
included.

---

## 2. What shipped

### Foundations

- **`--max_ratio`** promoted from a hardcoded `0.9` to the single per-matrix bound shared by every
  policy. This is why Swift-SVD's `δ` rank floor was dropped rather than reimplemented.
- **`--bypass_late_layers`**, usable together with `--bypass_early_layers`, both charged at the one
  `--bypass_ratio`. Half of RQ4 depended on this and nothing else.
- **Run configuration sidecar.** `<run_name>.config.json` next to both the checkpoint and the
  evaluation JSON, holding the resolved arguments, the realized allocation (target *and* actual
  removal, the full `ratio_map`, the policies and the knobs that applied) and the checkpoint
  metadata. `generate_tables.py` prefers it and falls back to `parse_filename`, so results collected
  before it keep tabulating. The HF token is stripped before writing.
- **c4 evaluation fix.** `main.py` mapped the `c4` task to the wikitext dataset, so every C4
  perplexity collected so far is a duplicate wikitext measurement. It now reads a real C4 validation
  shard (`en/c4-validation.00000-of-00008.json.gz`, which is what upstream SVD-LLM loads; the full
  split is ~364k documents and cannot be joined). **Perplexity evaluations must be re-run** —
  checkpoints are on disk, so no re-compression is needed.

### Caching, and the offline tool

- **Spectra cache.** The score pass caches the **raw** singular values of each whitened matrix under
  `whitening_matrices/<model>/<v1|v2>/spectra/`, keyed by matrix path and validated on load against
  the calibration size. Every score metric is derivable from the spectrum, so one cache serves all
  of them and a repeated score pass skips the decomposition entirely.
- **Block Influence for free.** `β = 1 − E[cos(x_in, x_out)]` per decoder block, accumulated inside
  the existing whitening replay, which already holds both the block input and its output. No extra
  forward pass, no model copy; two fp64 scalars per block, reduced per batch. Raw sums are stored so
  chunked runs merge exactly. This is a real cost argument for the thesis against Dobi-SVD (needs
  training) and ASVD (needs perplexity probes).
- **`allocation_report.py`.** Replays the real `allocate_ratios` over the cached spectra and
  importance for any allocator × score × knob combination, on CPU, in seconds, against roughly an
  hour for one compression and evaluation. Matrix shapes come from the model config, and the
  spectrum length independently confirms them. Emits CSV for `pgfplots` plus the captured `[BUDGET]`
  logs, ranks variants by predicted truncation loss, and enforces the invariants in §7. This is the
  agreed verification mechanism; there are no test files.

### Allocation as a pluggable family

`allocate_ratios` moved to `src/utils.py` (so the offline tool reaches it without importing the
GPU-only pipeline) and split into a shell and two policies. The shell owns grouping, the budget
split and every `[BUDGET]` line; an **outer** policy divides the budget across groups and an
**inner** policy divides each group's share across its matrices. A policy declares what it needs —
data and knobs alike — as named parameters, and the shell hands it only what it declares, which is
also how the sidecar records the knobs that actually applied.

| Inner policy | Origin | Rule | Family |
|---|---|---|---|
| `waterfill` (default) | SVD-LLM V2 Alg. 1, parameter-weighted | `ratio ∝ 1 / log(score + offset)` | ratio |
| `softmax_temp` | MoDeGPT Eq. 10-11 | `ratio ∝ softmax(−score / ε)` | ratio |
| `swift_pool` | Swift-SVD Alg. 2 | `ratio = max_ratio − pool · score / Σ(score · params)` | ratio |
| `drank_lagrangian` | D-Rank Eq. 3-7 | `rank ∝ sqrt(score / ω)`, `ω = out + in` | rank |

| Outer policy | Rule |
|---|---|
| `param_share` (default) | proportional to parameters, so every group gets the same average ratio |
| `waterfill` | the same water-fill one level up, on per-block Block Influence, with its own `--outer_offset` |

`--group_criterion hierarchical` groups by decoder block like `decoder` and additionally exposes
each block's Block Influence to the outer policy. With `param_share` it reproduces `decoder` to the
digit, which is the controlled ablation of the outer level itself.

### Composite scores

`--score_metric composite|<local>|block_influence` fuses the two signals into one number, following
Swift-SVD Eq. 12: `s = β^α · log(e + local)^(1−α)`, with `β` min-max normalized into `[1,2]` and
`--fusion_alpha` defaulting to 0.5. The fusion happens after the score pass, so
`compute_spectrum_score` stays purely spectral and any future local metric is composite-ready. The
grammar splits from the right, so a local half carrying its own separator
(`composite|norm|-inf|block_influence`) still parses.

Comparing this against the hierarchical allocator **is** §4.2: the hierarchical allocator keeps the
two signals separate at two granularities, the composite collapses them into one for a flat
allocator.

---

## 3. What implementation turned up

These are findings, not plans. Each one changes what the thesis should say.

### 3.1 D-Rank's rank-space bias is severe on mixed-shape groups

D-Rank allocates retained parameters `∝ sqrt(score · ω)`. On a group of identical shapes — which is
what `--group_criterion type` produces, and what D-Rank itself assumes, since its groups are one
matrix concatenated across layers under a shared basis — this coincides with the ratio-space
policies. On a group mixing shapes it does not: a small matrix costs fewer parameters per rank, so
it saturates first. Measured at a 0.2 target with `--group_criterion decoder`, mean ratio per type:

```
policy               q_proj   k_proj   v_proj   o_proj  gate_proj  up_proj  down_proj
waterfill            0.2173   0.2173   0.2173   0.2173     0.1913   0.2016     0.1913
drank_lagrangian     0.0311   0.0000   0.0000   0.0311     0.2444   0.2704     0.2444
swift_pool           0.3165   0.3165   0.3165   0.3165     0.1148   0.2644     0.1148
softmax_temp         0.2375   0.2375   0.2375   0.2375     0.1669   0.2322     0.1669
```

`drank_lagrangian` drives the attention projections to nearly dense and loads the whole budget onto
the MLP. §3.3.5 must state that `drank_lagrangian × type` is D-Rank as published while
`drank_lagrangian × decoder|global|hierarchical` is a different method, and §4 must not compare the
two as though they were one.

A consequence for the verification harness: "a uniform score leaves every matrix at the flat ratio"
is a **ratio-space** invariant, not a universal one. What holds for every policy is weaker — the
*value* of a constant score must not change the allocation.

### 3.2 A per-block score cannot differentiate inside a block

With `--group_criterion decoder` or `hierarchical`, `--fusion_alpha 1` gives every matrix in a block
the same score, and the allocation becomes **exactly homogeneous** (measured ratio standard
deviation 0.000). Under `type` or `global` it varies normally (0.022).

This is the sharpest argument for the hierarchical allocator and belongs in §3.1.3 or §4.2: an
end-to-end signal fused into a per-matrix score is wasted under per-block grouping, because it can
only do work at the outer level. `allocate_ratios` prints a `[BUDGET][WARNING]` whenever scores vary
by less than 0.1% inside every group, so a run cannot silently be homogeneous.

### 3.3 The composite score is much flatter than the raw one

`β^0.5` spans `[1, 1.41]` while `log(e + local)` is typically 5 to 8 for a truncation score. On a
small fixture the per-matrix ratio spread fell from a standard deviation of 0.111 (raw truncation)
to 0.039 (`α = 0`) to 0.021 (`α = 0.5`). Under the same `--offset` a composite allocation therefore
sits closer to homogeneous. §4.2 must not mistake reduced dispersion for reduced benefit; pick an
`--offset` for composite runs with `allocation_report.py` first.

### 3.4 Divergences from the published algorithms, to be declared

- **SVD-LLM V2 Alg. 1** preserves the mean ratio; `waterfill` preserves removed *parameters*
  (`Σ paramsᵢ · ratioᵢ = budget`) and caps at `--max_ratio`. The two agree when a group's matrices
  are the same size. The literal Alg. 1 is not implemented.
- **MoDeGPT Eq. 11** likewise preserves the mean ratio, and its `s` is Block Influence, already in
  `[0,1]`. Ours span orders of magnitude by metric, so scores are min-max normalized per group
  before the softmax; the largest allocation weight then exceeds the smallest by `exp(1/ε)` whatever
  the metric.
- **Swift-SVD Alg. 2** writes `k̄ = (mn/(m+n))·ρ` with `ρ` a *retention* ratio, the opposite
  convention from ours; we keep `rank = out·in·(1−ratio)/(out+in)`. Its `δ` floor is replaced by
  `--max_ratio`, a far lower floor (an effective cap of 0.6 at a 0.2 target, against 0.9), so
  `swift_pool` runs more aggressively than the paper. Its pool is in raw rank units, which only
  preserves a parameter budget on same-shape groups; the reduction is applied to the ratio instead,
  so the two agree exactly under `--group_criterion type`. The 11-candidate grid search over `α` is
  not implemented — at about an hour per run it is not affordable, and `α = 0.5` is reported as a
  deliberately unexplored hyperparameter.
- **D-Rank** contributes its Lagrangian only. Its grouped horizontal concatenation under a shared
  basis (Basis Sharing) and its Q/K-to-V rebalancing are not implemented, since both change the
  decomposition rather than the allocation. Its `R_eff` is exactly our `eff_rank_sq`.
- **ERC-SVD** is cited as motivation, but for the **early** end, not the late one. Under a fixed
  overall ratio it compresses only the *last* `k` blocks and leaves the first `N - k` untouched,
  which is `--bypass_early_layers N-k`, not `--bypass_late_layers`. Its Alg. 3 charges the
  compressed blocks `R_l = N·R_o/k`, the same push-the-budget-onto-the-rest arithmetic as
  `compute_active_budget`, and searches over `k`; we fix the count by hand instead. Neither that
  search nor its residual compensation is implemented. The late-N arm is motivated by the
  importance profile rather than by ERC-SVD: ShortGPT and MoDeGPT both report the first *and* last
  blocks as the most critical.
- **§3.3.2** is written as an *equivalence*: pinning `rᵢ` for the bypassed set and redistributing —
  what `compute_active_budget` already does — gives the same answer as constraining the §3.3.1
  objective, for all the separable objectives above. One implementation, one source of bugs.

### 3.5 Open, and blocking

- **The sign of the Block Influence / effective-rank correlation.** Swift-SVD Fig. 3 reports it
  *negative*, which is what justifies fusing the two signals rather than treating them as one. If it
  comes out positive on Qwen2.5-7B, the `β^α` convention is wrong for this setup and every composite
  run is misdirected. `allocation_report.py` prints Spearman ρ per matrix family; it needs one
  whitening pass on a GPU box. **Do this before any composite run.**
- **Score-scale confound.** `L = L · sqrt(n_calibration_tokens)` makes truncation and `norm|p`
  scores depend on `--max_whitening_samples`, while entropy and effective-rank scores are
  scale-invariant. The decision is to keep the rescale, **freeze `--max_whitening_samples` across
  every thesis run**, and document it in §5.2.

---

## 4. Thesis prose (step 11)

Drafted in `msc-thesis---s345139---parisini`, a separate calkit/DVC repo — never mix the builds.

- **§2.4**, the subsections marked `TODO - read paper`: Plain SVD, DRONE, FWSVD, ASVD, ERC-SVD,
  Basis Sharing / D-RANK, SWIFT-SVD.
- **§3**, only what actually ships: §3.1.2, §3.1.3, §3.3.1-§3.3.5. §3.3.1 is writable as an
  optimization problem now that `softmax_temp` exists as its exemplar, which also gives a principled
  reading of `1/log` as its unregularized cousin.
- **`thesis/biblio.bib`** entries for every newly cited paper, matching the existing
  authoryear/biblatex setup and key style (`svdllm:pap`, `transformer:pap`).
- **§5.2** the score-scale limitation of §3.5.
- **§5.2.2 (GQA)** a free contrast worth using: Qwen2.5-7B already has GQA (28 heads, 4 KV heads)
  while LLaMA-7B is full MHA, so the chosen model set supports a measured discussion rather than a
  caveat.
- Everything marked as a divergence in §3.4 has to be declared where the method is described, not
  buried in a limitations section.

---

## 5. Experiment grid (step 12)

Models: **Qwen2.5-7B, LLaMA-7B, Qwen2.5-32B**. Ratios: **0.2 and 0.5**.

Grid on **perplexity only** (wikitext + the fixed c4). Finalists re-evaluated on **arc_easy,
hellaswag, openbookqa, piqa, winogrande, gsm8k, truthfulqa_gen**; `mathqa` will be attempted but is
likely blocked by `datasets` no longer permitting custom loading scripts, which will be reported
rather than worked around.

Runs are driven by `run_experiments.py` from `args/base_args.json` and `args/experiments.json`, both
gitignored — so the grid itself has to be described in thesis §4.1.

Unblocked immediately and needing no new code: the Schatten p-norm sweep (`--score_metric "norm|1"`,
`norm|3`, `norm|inf`, `norm|-inf`) for §3.1.1.2, and dropping individual `--compress_*` flags with
`--ratio_scope all` for §3.2.1.1.

Sequence the grid so the cheap offline sweeps come first: `allocation_report.py` ranks every
allocator × score × knob combination in seconds, and only the finalists need GPU time.

---

## 6. Decisions locked

| Area | Decision |
|---|---|
| RQ5 / LoRA | Sequential update **is** the LoRA arm; no separate downstream fine-tune |
| Inner allocators | `waterfill` (default), `drank_lagrangian`, `swift_pool`, `softmax_temp`; **no** `v2_literal`, no `zscore_affine` |
| Outer allocators | `param_share` (default), `waterfill` — two separate enums, not one registry |
| Hierarchy | Exposed as `--group_criterion hierarchical`; both levels pluggable |
| Importance | Block Influence `1 − cos`, per decoder block, always computed, cached beside whitening |
| Composite | Both hierarchical and scalar; geometric fusion; grammar in `--score_metric`; `--fusion_alpha` 0.5 |
| `swift_pool` generalization | Ratio-reduction ∝ score, so it stays ratio-space and matches the paper under `type` |
| `softmax_temp` scale | Scores min-max normalized to `[0,1]` per group before the softmax |
| D-Rank | Lagrangian only — no basis sharing, no Q/K-to-V rebalancing |
| Swift-SVD | Fixed `α = 0.5`; `δ` dropped; no grid search |
| Per-matrix cap | One shared `--max_ratio` for all policies |
| Bypass | Both ends, one shared `--bypass_ratio` |
| ERC-SVD | Motivation only |
| Budget invariant | Warn and clamp, realized ratio recorded |
| Score scale | Keep the rescale, freeze `--max_whitening_samples`, document in §5.2 |
| Spectra cache | Raw values, rescaled at score time; plus the offline tool with predicted truncation loss |
| `svd_llm.py` merge-the-passes TODO | Correct the comment, do not merge the passes |
| Config | Standalone `<run_name>.config.json` sidecar |
| Knobs | One explicit flag per knob |
| c4 bug | Fixed, in scope |
| Generative-eval TODO in `main.py` | Out of scope |
| Verification | Offline tool only, no test files |
| Prose | Code plus draft LaTeX (§2.4, §3, `biblio.bib`) |

---

## 7. Invariants the offline tool enforces

Any policy added later is checked by construction rather than by inspection:

1. Realized removal matches the budget.
2. Ratios stay within `[0, --max_ratio]`.
3. The *value* of a constant score never changes the allocation.
4. Ratio-space policies only: no group gives more removal to a higher-scoring matrix, and a constant
   score collapses the allocation onto the flat ratio (the latter also needs a neutral outer level).
   Rank-space policies are exempt from both, for the reason in §3.1; the mean Spearman correlation
   between score and ratio is reported for every policy instead.

A budget drift almost always means the configuration is infeasible — `--max_ratio` too low to reach
the target once bypassed layers are charged — which is what the check exists to surface.

Two further properties are worth re-checking after any change to the allocator, because they are
what make the comparisons meaningful: `waterfill` must stay bit-identical to its pre-refactor
behaviour, and `hierarchical` with `param_share` must stay bit-identical to `decoder`.
