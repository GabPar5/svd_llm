"""
Offline exploration of heterogeneous compression-ratio allocations.

Replays `allocate_ratios` over the spectra and Block Influence cached beside the
whitening matrices, for any allocator x score x knob combination, without a GPU
and without loading model weights: matrix shapes come from the model config and
every score is re-derived from the cached spectrum. A whole sweep runs in
seconds, which is what makes the allocation study affordable at all, since one
real compression + evaluation run costs about an hour.

Emits CSV rather than figures, so `pgfplots` can consume it directly in the
thesis; PNG previews are written too when matplotlib happens to be installed.
"""

from src.utils import *

import argparse
import csv
import io
import itertools
import math
import os
import re
import torch
from collections import defaultdict
from contextlib import redirect_stdout
from scipy.stats import spearmanr
from transformers import AutoConfig # pyright: ignore[reportPrivateImportUsage]
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

# Deviation from the target removal above which a variant is not comparable to
# the others, matching the threshold `compress_svd_llm` warns at
BUDGET_TOLERANCE = 1e-3

# Ratios come out of a water-fill in fp64, so equality needs a little slack
RATIO_TOLERANCE = 1e-9

# Sweepable knobs and how to read their values off the command line
SWEEPABLE: Dict[str, Any] = {
    "compression_ratio": float,
    "group_criterion": str,
    "inner_allocation": str,
    "score_metric": str,
    "offset": float,
    "max_ratio": float,
    "min_rank_fraction": float,
    "bypass_early_layers": int,
    "bypass_late_layers": int,
    "bypass_ratio": float,
    "outer_allocation": str,
    "softmax_temp": float,
    "outer_offset": float,
    "fusion_alpha": float,
}

# Part of every variant's configuration, but held fixed across a sweep
FIXED_CONFIG = ( "ratio_scope", )

# Two variants are one experiment when *no* matrix moves more than this between
# them. The test is on the largest per-matrix difference and not the mean,
# because the mean hides exactly the case that matters: raising the cap from 0.75
# to 0.9 on LLaMA-7B moves the allocation by 0.004 on average and by 0.15 on the
# three matrices that decide the outcome, which is the difference between a
# working model and one at 43 perplexity. A screen that says "do not run this"
# has to be wrong in the safe direction
DUPLICATE_MAP_MAX_DELTA = 0.02

# A wide sweep can hold hundreds of near-identical pairs. The CSV keeps them all,
# the console prints enough to act on
MAX_REPORTED_DUPLICATES = 10

# The fraction of a decoder block's parameters that may be removed before the
# block stops carrying its function. Measured on LLaMA-7B over 39 allocations at
# ratio 0.5: all 12 that pushed a block past 0.63 landed at 28.5 perplexity or
# worse against a homogeneous 24.56, and the block at the peak was layer 0 in
# every one of them. It screens twice as well as the peak per-matrix ratio it
# replaces (Spearman +0.73 against +0.41), because a matrix can be truncated hard
# as long as its siblings in the same block are not.
#
# The test is one-sided, and reading it as a guarantee is the trap. Staying under
# the threshold rules out the depth failure and nothing else: `param_share` pins
# every block at the target by construction, and those allocations still span 23
# to 44 perplexity on how they split a block internally. A pass here means the
# grouping and the outer policy are safe, not the score and the inner policy
BLOCK_RATIO_DANGER = 0.60

# The companion screen for a grouped-query model, on the fraction of its full
# rank a key or value projection keeps. Fitted to eight Qwen2.5-7B runs at two
# budgets, where everything at or below 0.141 measured at least three times the
# homogeneous perplexity and everything at or above 0.289 stayed within 12%
KV_RANK_FRACTION_DANGER = 0.20

MATRIX_TYPES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


class Inputs(NamedTuple):
    """Everything a sweep reads once and then reuses for every variant"""
    layers_str: List[str]
    param_count_map: Dict[str, int]
    shapes: Dict[str, Tuple[int, int]]
    spectra: Dict[str, torch.Tensor]
    n_calibration_tokens: int
    importance: Optional[Dict[int, float]]
    num_layers: int
    selected_params: int
    target_total_params: int
    head_partition: Dict[str, int]
    kv_sharing: int


class Variant(NamedTuple):
    """
    One point of the sweep: a full allocation configuration and its name.

    `name` is the identifier, and becomes a budget-log filename and a CSV key.
    `label` carries the swept values alone, for a console table that names the
    axes once in its header instead of repeating every flag name on every row.
    """
    name: str
    config: Dict[str, Any]
    label: str
    axes: Tuple[str, ...]


class VariantResult(NamedTuple):
    ratio_map: Dict[str, float]
    score_map: Dict[str, float]
    budget_log: str
    realized_ratio: float
    objectives: Dict[str, float]
    target_removed: float
    active_keys: List[str]
    checks: List[str]
    score_ratio_rho: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare compression-ratio allocations offline, from cached spectra",
    )

    parser.add_argument('--model', type=str, required=True, help='LLM the spectra were cached for')
    parser.add_argument('--save_path', type=str, default='./output', help='Root holding whitening_matrices/')
    parser.add_argument(
        '--scratch_path',
        type=str,
        default='./output',
        help='Root holding whitening_matrices/ when the compression run was given a --scratch_path',
    )
    parser.add_argument(
        '--whitening_mat_path',
        type=str,
        default=None,
        help='Whitening directory to read, overriding the one derived from --save_path and --model',
    )
    parser.add_argument('--run_v2', action='store_true', help='Read the v2 whitening artifacts instead of v1')
    parser.add_argument('--hf_token', type=str, default=None, help='Huggingface token, to read a gated config')

    parser.add_argument('--compress_mlp', action='store_true', help='Include the MLP matrices')
    parser.add_argument('--compress_att_q', action='store_true', help='Include the query projection')
    parser.add_argument('--compress_att_k', action='store_true', help='Include the key projection')
    parser.add_argument('--compress_att_v', action='store_true', help='Include the value projection')
    parser.add_argument('--compress_att_out', action='store_true', help='Include the attention output projection')

    parser.add_argument('--compression_ratio', type=float, default=0.2, help='Overall target compression ratio')
    parser.add_argument(
        '--ratio_scope',
        type=str,
        default='selected',
        choices=[ 'selected', 'all' ],
        help='Whether the target ratio is measured over the selected matrices or over all compressible ones',
    )
    parser.add_argument(
        '--group_criterion',
        type=str,
        default='type',
        choices=[criterion.value for criterion in GroupBy],
        help='Criterion used to group weight matrices',
    )
    parser.add_argument(
        '--inner_allocation',
        type=str,
        default=InnerAllocation.WATERFILL.value,
        choices=[policy.value for policy in INNER_POLICIES],
        help='Policy that splits a group budget across the matrices inside it',
    )
    parser.add_argument(
        '--outer_allocation',
        type=str,
        default=OuterAllocation.PARAM_SHARE.value,
        choices=[policy.value for policy in OUTER_POLICIES],
        help='Policy that splits the budget across groups. "waterfill" needs --group_criterion hierarchical',
    )
    parser.add_argument(
        '--score_metric',
        type=str,
        default='truncation',
        help='Score metric, as accepted by main.py, including "composite|<local>|block_influence"',
    )
    parser.add_argument('--fusion_alpha', type=float, default=0.5, help='Weight of the end-to-end half of a composite score metric')
    parser.add_argument('--offset', type=float, default=1.5, help='Offset added to scores before the log, read by waterfill')
    parser.add_argument('--outer_offset', type=float, default=1.5, help='Same offset, applied to Block Influence by the waterfill outer policy')
    parser.add_argument('--softmax_temp', type=float, default=1.0, help='Temperature of the softmax_temp policy, over min-max normalized scores')
    parser.add_argument('--max_ratio', type=float, default=0.9, help='Per-matrix upper bound on the ratio')
    parser.add_argument(
        '--min_rank_fraction',
        type=float,
        default=DEFAULT_MIN_RANK_FRACTION,
        help='Fraction of full rank every matrix must retain, which tightens --max_ratio per shape',
    )
    parser.add_argument('--bypass_early_layers', type=int, default=-1, help='Starting layers excluded from redistribution')
    parser.add_argument('--bypass_late_layers', type=int, default=-1, help='Ending layers excluded from redistribution')
    parser.add_argument('--bypass_ratio', type=float, default=0.0, help='Ratio pinned on the bypassed layers')
    parser.add_argument(
        '--group_patterns',
        type=str,
        default="q_proj:self_attn.q_proj;k_proj:self_attn.k_proj;v_proj:self_attn.v_proj;o_proj:self_attn.o_proj;gate_proj:mlp.gate_proj;up_proj:mlp.up_proj;down_proj:mlp.down_proj",
        help='Group patterns used when grouping by type',
    )

    parser.add_argument(
        '--sweep',
        type=str,
        action='append',
        default=[],
        metavar='KEY=V1,V2',
        help=(
            "Sweep one knob over several values, repeatable, taken as a cartesian "
            f"product. Sweepable: {', '.join(sorted(SWEEPABLE))}"
        ),
    )
    parser.add_argument('--out_dir', type=str, default=None, help='Where to write the CSV report, defaults under --save_path')
    parser.add_argument(
        '--plots',
        action='store_true',
        help='Also render PNG previews of every figure, when matplotlib is installed. The figure CSVs are written either way',
    )

    return parser.parse_args()


def resolve_targets(args: argparse.Namespace) -> Dict[str, bool]:
    """
    Which matrix families the report covers.

    A report over nothing is useless, so an invocation that names no target
    covers them all, unlike the compression entry point where each flag opts in.
    """
    targets = {
        "mlp": args.compress_mlp,
        "q": args.compress_att_q,
        "k": args.compress_att_k,
        "v": args.compress_att_v,
        "attention_output": args.compress_att_out,
    }

    if not any(targets.values()):
        print("[REPORT] No --compress_* flag given, covering every compressible matrix")
        targets = {name: True for name in targets}

    return targets


def spearman(xs: List[float], ys: List[float]) -> float:
    """Rank correlation, NaN when either side is constant or the sample is tiny"""
    if len(xs) < 3:
        return float("nan")

    # Indexing rather than `.statistic`, which scipy only grew recently
    return float(spearmanr(xs, ys)[0]) # pyright: ignore[reportIndexIssue]


def matrix_type_of(key: str) -> str:
    """`model.layers.7.mlp.gate_proj` -> `mlp.gate_proj`"""
    return key.partition(f".layers.{get_layer_idx_from_key(key)}.")[2]


def block_ratios(inputs: Inputs, ratio_map: Dict[str, float]) -> Dict[int, float]:
    """
    Parameter-weighted removed fraction per decoder block, over the targeted matrices.

    This is the quantity that predicts whether an allocation survives, so it is
    derived in one place and read by the summary, the tail screen and layers.csv.
    Weighting by parameters rather than averaging the ratios matters: an MLP
    matrix carries between two and three times an attention matrix, so the mean
    of the seven ratios in a block is not the fraction the block actually loses.

    The denominator is what the run compresses, not what the block holds, so a
    partial selection reports a fraction of that selection — which is why
    `selection_is_complete` guards the threshold rather than this function.
    """
    removed: Dict[int, float] = defaultdict(float)
    params: Dict[int, float] = defaultdict(float)

    for key in inputs.layers_str:
        layer = get_layer_idx_from_key(key)
        count = inputs.param_count_map[key]
        removed[layer] += count * ratio_map.get(key, 0.0)
        params[layer] += count

    return {layer: removed[layer] / params[layer] for layer in sorted(params) if params[layer]}


def peak_block(block_map: Dict[int, float]) -> Tuple[int, float]:
    """The block losing the most, and how much. `(-1, nan)` on an empty map"""
    if not block_map:
        return -1, float("nan")

    layer = max(block_map, key=lambda item: block_map[item])
    return layer, block_map[layer]


def selection_is_complete(inputs: Inputs) -> bool:
    """
    Whether every matrix family in a block is a compression target.

    `BLOCK_RATIO_DANGER` was calibrated on runs compressing all seven, so on a
    partial selection the same number means something else: an attention-only run
    at an overall 0.2 removes 0.60 of the attention it targets while the block
    itself loses 0.2, and screening that as doomed would be reading one
    denominator against a threshold fitted to another
    """
    return {matrix_type_of(key) for key in inputs.layers_str} == set(MATRIX_TYPES)


def load_inputs(args: argparse.Namespace) -> Inputs:
    """Read the model config and the cached spectra, and reconcile the two"""
    version_str = "v2" if args.run_v2 else "v1"
    wm_dir = args.whitening_mat_path or whitening_dir(
        scratch_root(args.save_path, args.scratch_path),
        args.model,
        version_str,
    )

    print(f"[REPORT] Whitening directory: {wm_dir}")

    config = AutoConfig.from_pretrained(args.model, token=args.hf_token, trust_remote_code=True)
    num_layers = int(config.num_hidden_layers)
    shapes_by_type = matrix_shapes_from_config(config)

    targets = resolve_targets(args)
    layers_str = generate_paths(
        mlp=targets["mlp"],
        q=targets["q"],
        k=targets["k"],
        v=targets["v"],
        attention_output=targets["attention_output"],
        layers_number=num_layers,
    )

    shapes = {key: shapes_by_type[matrix_type_of(key)] for key in layers_str}
    param_count_map = {key: out * inp for key, (out, inp) in shapes.items()}

    selected_params = sum(param_count_map.values())
    target_total_params = selected_params

    if args.ratio_scope == "all":
        # The denominator covers every compressible matrix, whether selected or
        # not, exactly as `compress_svd_llm` builds it
        all_keys = generate_paths(True, True, True, True, True, num_layers)
        target_total_params = sum(
            shapes_by_type[matrix_type_of(key)][0] * shapes_by_type[matrix_type_of(key)][1]
            for key in all_keys
        )

    spectra, n_calibration_tokens = load_spectra_cache(wm_dir)

    if not spectra:
        raise SystemExit(
            f"No cached spectra under {spectra_dir(wm_dir)}. Run a heterogeneous "
            f"compression once, or a scoring pass, to populate it",
        )

    # The spectrum length is min(out, in), which independently confirms the
    # shapes taken from the config: a wrong head_dim assumption shows up here
    mismatched = [
        key for key, spectrum in spectra.items()
        if key in shapes and len(spectrum) != min(shapes[key])
    ]
    if mismatched:
        raise SystemExit(
            f"{len(mismatched)} cached spectra disagree with the shapes derived from "
            f"the model config, e.g. {mismatched[0]}: got {len(spectra[mismatched[0]])}, "
            f"expected {min(shapes[mismatched[0]])}. The cache belongs to another model",
        )

    covered = sum(1 for key in layers_str if key in spectra)
    print(
        f"[REPORT] {covered}/{len(layers_str)} selected matrices have a cached spectrum "
        f"(calibration tokens: {n_calibration_tokens:,})",
    )

    if covered < len(layers_str):
        print(
            "[REPORT][WARNING] Matrices without a spectrum cannot be scored and will "
            "receive the flat active ratio. Bypassed layers of the run that built this "
            "cache are the usual reason, so a sweep over --bypass_early_layers below "
            "that run's value is not fully explorable",
        )

    importance = load_layer_importance(
        wm_dir=wm_dir,
        model_name=args.model,
        version_str=version_str,
        n_tokens=int(n_calibration_tokens or 0),
        num_layers=num_layers,
    )
    print(f"[REPORT] Block Influence: {'available' if importance else 'not cached'}")

    return Inputs(
        layers_str=layers_str,
        param_count_map=param_count_map,
        shapes=shapes,
        spectra=spectra,
        n_calibration_tokens=int(n_calibration_tokens or 1),
        importance=importance,
        num_layers=num_layers,
        selected_params=selected_params,
        target_total_params=target_total_params,
        head_partition=head_partition_map(layers_str, config),
        kv_sharing=kv_sharing_from_config(config),
    )


def build_variants(args: argparse.Namespace) -> List[Variant]:
    """Expand the --sweep specifications into the cartesian product of variants"""
    base = {key: getattr(args, key) for key in ( *SWEEPABLE, *FIXED_CONFIG )}

    axes: Dict[str, List[Any]] = {}
    for spec in args.sweep:
        key, _, raw = spec.partition("=")
        key = key.strip()

        if key not in SWEEPABLE:
            raise SystemExit(f"Cannot sweep '{key}'. Sweepable: {', '.join(sorted(SWEEPABLE))}")

        values = [SWEEPABLE[key](value.strip()) for value in raw.split(",") if value.strip()]

        if not values:
            raise SystemExit(f"No values given for --sweep {key}")

        axes[key] = values

    variants: List[Variant] = []
    keys = sorted(axes)

    for combination in itertools.product(*(axes[key] for key in keys)):
        swept = dict(zip(keys, combination))
        config = { **base, **swept }
        name = "base"
        label = "base"

        if swept:
            name = "__".join(f"{key}-{value}" for key, value in swept.items())
            # The name becomes a filename, and composite metrics carry a
            # separator no filename should hold
            name = re.sub(r"[^A-Za-z0-9.-]+", "_", name)
            label = " / ".join(str(value) for value in swept.values())

        variants.append(Variant(name=name, config=config, label=label, axes=tuple(keys)))

    return variants


def truncation_rank(out_features: int, in_features: int, ratio: float, spectrum_length: int) -> int:
    """Rank a ratio buys, clamped exactly as the compression pipeline clamps it"""
    rank = int((out_features * in_features * (1.0 - ratio)) / (out_features + in_features))
    return max(1, min(rank, spectrum_length - 1))


def build_score_map(
        inputs: Inputs,
        score_metric: str,
        active_keys: List[str],
        probe_ratio: float,
        fusion_alpha: float = 0.5
) -> Dict[str, float]:
    """
    Re-derive every score from the cached spectra, at the run's probe ratio.

    This mirrors the score pass of `compress_svd_llm`, rescale and composite
    fusion included, so a variant explored here allocates exactly as the same
    flags would on a GPU.
    """
    metric = ScoreMetric(score_metric)
    composite = parse_composite_metric(metric)
    local_metric = composite.local if composite is not None else metric

    score_map: Dict[str, float] = {}

    for key in active_keys:
        spectrum = inputs.spectra.get(key)

        if spectrum is None:
            continue

        out_features, in_features = inputs.shapes[key]
        rank = truncation_rank(out_features, in_features, probe_ratio, len(spectrum))

        # Recover the unnormalized singular values, as the score pass does
        rescaled = spectrum.to(torch.float64) * math.sqrt(inputs.n_calibration_tokens)
        score_map[key] = compute_spectrum_score(local_metric, rescaled, rank)

    if composite is not None:
        score_map = compose_scores(score_map, inputs.importance, fusion_alpha)

    return score_map


def rescaled_spectrum(inputs: Inputs, key: str) -> torch.Tensor:
    """Undo the cache's normalization, exactly as the score pass does"""
    return inputs.spectra[key].to(torch.float64) * math.sqrt(inputs.n_calibration_tokens)


def retained_ranks(inputs: Inputs, ratio_map: Dict[str, float]) -> Dict[str, int]:
    """
    Rank each matrix keeps under an allocation.

    A ratio of exactly 0.0 leaves the layer dense, so it keeps its whole
    spectrum rather than dropping out of the accounting: an allocation that
    spends its budget elsewhere has to be credited for what it left alone.
    """
    ranks: Dict[str, int] = {}

    for key in inputs.layers_str:
        spectrum = inputs.spectra.get(key)

        if spectrum is None:
            continue

        ratio = ratio_map.get(key, 0.0)

        if ratio <= 0.0:
            ranks[key] = len(spectrum)
            continue

        out_features, in_features = inputs.shapes[key]
        ranks[key] = truncation_rank(out_features, in_features, ratio, len(spectrum))

    return ranks


def objective_frobenius_tail(inputs: Inputs, ratio_map: Dict[str, float]) -> float:
    """Squared energy discarded, summed over matrices. The whitened reconstruction error"""
    return sum(
        float(rescaled_spectrum(inputs, key)[rank:].pow(2).sum())
        for key, rank in retained_ranks(inputs, ratio_map).items()
    )


def objective_nuclear_tail(inputs: Inputs, ratio_map: Dict[str, float]) -> float:
    """Discarded magnitude rather than energy, so a long thin tail costs more"""
    return sum(
        float(rescaled_spectrum(inputs, key)[rank:].sum())
        for key, rank in retained_ranks(inputs, ratio_map).items()
    )


def objective_spectral_tail(inputs: Inputs, ratio_map: Dict[str, float]) -> float:
    """
    Largest single direction any matrix throws away.

    A minimax rather than a sum, so unlike every other objective here it is
    driven by the one worst matrix and ignores how the rest were treated.
    """
    worst = 0.0

    for key, rank in retained_ranks(inputs, ratio_map).items():
        spectrum = rescaled_spectrum(inputs, key)

        if rank < len(spectrum):
            worst = max(worst, float(spectrum[rank]))

    return worst


def objective_relative_energy_lost(inputs: Inputs, ratio_map: Dict[str, float]) -> float:
    """
    Mean per-matrix fraction of energy discarded.

    Scale free, which is what separates it from the Frobenius tail: there the
    few largest matrices dominate the total whatever happens to the rest.
    """
    fractions = []

    for key, rank in retained_ranks(inputs, ratio_map).items():
        energy = inputs.spectra[key].to(torch.float64).pow(2)
        total = float(energy.sum())

        if total > 0.0:
            fractions.append(float(energy[rank:].sum()) / total)

    return sum(fractions) / len(fractions) if fractions else 0.0


def objective_eff_rank_lost(inputs: Inputs, ratio_map: Dict[str, float]) -> float:
    """
    Mean per-matrix fraction of effective rank lost.

    Measures how much of the spectrum's *diversity* the truncation removes,
    which a matrix with one dominant direction can lose almost none of even
    under a heavy ratio.
    """
    losses = []

    for key, rank in retained_ranks(inputs, ratio_map).items():
        spectrum = inputs.spectra[key].to(torch.float64)
        full = float(torch.exp(spectrum_entropy(normalized_spectrum(spectrum, squared=True))))
        kept = float(torch.exp(spectrum_entropy(normalized_spectrum(spectrum[:rank], squared=True))))

        if full > 0.0:
            losses.append(1.0 - kept / full)

    return sum(losses) / len(losses) if losses else 0.0


def objective_influence_tail(inputs: Inputs, ratio_map: Dict[str, float]) -> float:
    """
    Relative energy discarded, weighted by the Block Influence of each block.

    The only objective here that is not a function of the spectra alone. Block
    Influence is measured on the dense model's residual stream, so no local
    score can reproduce it, which is what makes this the one end-to-end reading
    available without a GPU.
    """
    if not inputs.importance:
        return float("nan")

    total = 0.0

    for key, rank in retained_ranks(inputs, ratio_map).items():
        influence = inputs.importance.get(get_layer_idx_from_key(key))

        if influence is None:
            continue

        energy = inputs.spectra[key].to(torch.float64).pow(2)
        denominator = float(energy.sum())

        if denominator > 0.0:
            total += influence * float(energy[rank:].sum()) / denominator

    return total


def marginal_frobenius(inputs: Inputs, key: str) -> torch.Tensor:
    return rescaled_spectrum(inputs, key).pow(2)


def marginal_nuclear(inputs: Inputs, key: str) -> torch.Tensor:
    return rescaled_spectrum(inputs, key)


def marginal_relative_energy(inputs: Inputs, key: str) -> torch.Tensor:
    energy = inputs.spectra[key].to(torch.float64).pow(2)
    total = float(energy.sum())
    scale = len([k for k in inputs.layers_str if k in inputs.spectra])
    return energy / (total * scale) if total > 0.0 and scale else torch.zeros_like(energy)


def marginal_influence(inputs: Inputs, key: str) -> torch.Tensor:
    energy = inputs.spectra[key].to(torch.float64).pow(2)
    total = float(energy.sum())
    influence = (inputs.importance or {}).get(get_layer_idx_from_key(key), 0.0)
    return energy * influence / total if total > 0.0 else torch.zeros_like(energy)


class Objective(NamedTuple):
    """
    One way of pricing an allocation. All are oriented so that lower is better.

    `marginal` gives the loss of discarding each singular direction of a matrix,
    and only an objective that is a plain sum over directions has one. Those are
    exactly the objectives the greedy oracle below can solve.

    `circular_with` names the score metrics that optimize this objective by
    construction. A variant scored by one of them starts with a structural
    advantage here, which is the whole reason the report never ranks on a single
    objective.
    """
    name: str
    aggregate: Callable[[Inputs, Dict[str, float]], float]
    marginal: Optional[Callable[[Inputs, str], torch.Tensor]]
    circular_with: str


OBJECTIVES = (
    Objective("frobenius_tail", objective_frobenius_tail, marginal_frobenius, "truncation, truncation_sq"),
    Objective("nuclear_tail", objective_nuclear_tail, marginal_nuclear, "norm|1"),
    Objective("spectral_tail", objective_spectral_tail, None, "norm|inf"),
    Objective("relative_energy_lost", objective_relative_energy_lost, marginal_relative_energy, "-"),
    Objective("eff_rank_lost", objective_eff_rank_lost, None, "eff_rank, eff_rank_sq"),
    Objective("influence_tail", objective_influence_tail, marginal_influence, "composite|...|block_influence"),
)


def oracle_cost(
        inputs: Inputs,
        objective: Objective,
        target_removed: float,
        active_keys: List[str],
        max_ratio: float
) -> float:
    """
    Lowest this objective can reach over the same matrices, budget and cap.

    An additive objective is minimized by discarding directions in ascending
    order of loss per parameter removed. The loss of a direction decreases with
    its index, so that ordering always discards a contiguous tail from each
    matrix and the result is realizable as a rank truncation.

    What it ignores is the grouping: it may spend the whole budget on a handful
    of matrices and leave the rest dense. These objectives really are minimized
    by that degenerate allocation, so the ratio to the bound runs to six figures
    and says nothing about how good an allocation is in absolute terms.

    Its use is normalization across budgets. A sweep over `compression_ratio`
    moves the raw objective by orders of magnitude, which drowns the difference
    between two policies at the same ratio; dividing by the bound removes the
    budget's own contribution and leaves them comparable. Within one budget the
    bound is a constant, so it never reorders anything.
    """
    if objective.marginal is None or not math.isfinite(target_removed):
        return float("nan")

    weights: List[torch.Tensor] = []
    costs: List[torch.Tensor] = []

    for key in active_keys:
        if key not in inputs.spectra:
            continue

        out_features, in_features = inputs.shapes[key]
        # The cap bounds how far any single matrix may be truncated, and it
        # already clamps the rank to at least 1, so the leading direction is
        # never discardable
        floor_rank = truncation_rank(out_features, in_features, max_ratio, len(inputs.spectra[key]))
        weight = objective.marginal(inputs, key)[floor_rank:]

        if not len(weight):
            continue

        weights.append(weight)
        costs.append(torch.full_like(weight, float(out_features + in_features)))

    if not weights:
        return float("nan")

    weight = torch.cat(weights)
    cost = torch.cat(costs)
    order = torch.argsort(weight / cost)
    cumulative = torch.cumsum(cost[order], dim=0)
    taken = int(torch.searchsorted(cumulative, torch.tensor(target_removed, dtype=torch.float64)).item()) + 1

    return float(weight[order][:taken].sum())


def mean_objective_rank(ranks: Dict[str, Dict[str, float]], variant_name: str) -> float:
    """
    Average rank of one variant across the objectives that could price it.

    Averaging ranks rather than values is what makes the objectives comparable
    at all: they carry different units and span different orders of magnitude,
    so no weighted sum of them would mean anything.
    """
    values = [
        ranks[objective.name][variant_name] for objective in OBJECTIVES
        if not math.isnan(ranks[objective.name][variant_name])
    ]

    return sum(values) / len(values) if values else float("nan")


def compute_oracles(
        inputs: Inputs,
        variants: List[Variant],
        results: Dict[str, VariantResult]
) -> Dict[str, Dict[str, float]]:
    """
    Lower bound per variant per objective, at that variant's own budget.

    A sweep over `compression_ratio` gives its variants different budgets, so
    one oracle per objective would not be comparable across them. The bound only
    depends on the budget, which is what the cache keys on.
    """
    cache: Dict[Tuple[str, int, float, int], float] = {}
    oracles: Dict[str, Dict[str, float]] = {}

    for variant in variants:
        result = results[variant.name]
        target = result.target_removed
        max_ratio = variant.config["max_ratio"]
        per_objective: Dict[str, float] = {}

        for objective in OBJECTIVES:
            cache_key = (
                objective.name,
                int(round(target)) if math.isfinite(target) else -1,
                max_ratio,
                hash(tuple(sorted(result.active_keys))),
            )

            if cache_key not in cache:
                cache[cache_key] = oracle_cost(inputs, objective, target, result.active_keys, max_ratio)

            per_objective[objective.name] = cache[cache_key]

        oracles[variant.name] = per_objective

    return oracles


def rank_variants(values: Dict[str, float]) -> Dict[str, float]:
    """
    Competition ranks over one objective, 1 being best, ties sharing a rank.

    NaN stays NaN rather than sorting to an end, so a variant an objective could
    not price does not silently look either best or worst at it.
    """
    finite = {name: value for name, value in values.items() if not math.isnan(value)}
    ordered = sorted(finite.values())

    return {
        name: float(ordered.index(value) + 1) if name in finite else float("nan")
        for name, value in values.items()
    }


def allocate(inputs: Inputs, config: Dict[str, Any], group_patterns: Dict[str, List[str]], score_map: Dict[str, float]) -> Tuple[Dict[str, float], str]:
    """Run the real allocator, capturing its [BUDGET] instrumentation"""
    log = io.StringIO()

    with redirect_stdout(log):
        ratio_map = allocate_ratios(
            group_criterion=config["group_criterion"],
            score_map=score_map,
            layers_str=inputs.layers_str,
            target_ratio=config["compression_ratio"],
            param_count_map=inputs.param_count_map,
            offset=config["offset"],
            group_patterns=group_patterns,
            bypass_early_layers=config["bypass_early_layers"],
            bypass_ratio=config["bypass_ratio"],
            max_ratio=config["max_ratio"],
            min_rank_fraction=config["min_rank_fraction"],
            target_total_params=inputs.target_total_params,
            bypass_late_layers=config["bypass_late_layers"],
            num_layers=inputs.num_layers,
            inner_allocation=config["inner_allocation"],
            outer_allocation=config["outer_allocation"],
            shape_map=inputs.shapes,
            importance_map=inputs.importance,
            softmax_temp=config["softmax_temp"],
            outer_offset=config["outer_offset"],
        )

    return ratio_map, log.getvalue()


def check_variant(
        inputs: Inputs,
        config: Dict[str, Any],
        group_patterns: Dict[str, List[str]],
        ratio_map: Dict[str, float],
        score_map: Dict[str, float],
        budget: ActiveBudget
) -> Tuple[List[str], float]:
    """
    The invariants an allocation has to satisfy whatever policy produced it.

    Each returned string is one violation, so a policy added later is checked by
    construction instead of by inspection of its output. Also returns how
    strongly the allocation tracks the score, averaged over groups, which is a
    measurement rather than a check.
    """
    problems: List[str] = []

    removed = sum(inputs.param_count_map[key] * ratio_map.get(key, 0.0) for key in inputs.layers_str)
    target_removed = budget.target_removed
    drift = abs(removed - target_removed) / max(1.0, target_removed)

    if drift > BUDGET_TOLERANCE:
        problems.append(f"budget drift {drift:.4%} (target {target_removed:,.0f}, actual {removed:,.0f})")

    cap_map = build_cap_map(
        inputs.layers_str,
        config["max_ratio"],
        inputs.shapes,
        config["min_rank_fraction"],
    )
    out_of_bounds = [
        key for key, ratio in ratio_map.items()
        if ratio < -RATIO_TOLERANCE or ratio > cap_map.get(key, config["max_ratio"]) + RATIO_TOLERANCE
    ]
    if out_of_bounds:
        problems.append(
            f"{len(out_of_bounds)} ratios outside their per-matrix ceiling, e.g. {out_of_bounds[0]}",
        )

    grouping = build_allocation_groups(
        group_criterion=GroupBy(config["group_criterion"]),
        active_keys=budget.active_keys,
        score_map=score_map,
        group_patterns=group_patterns,
    )

    is_ratio_space = InnerAllocation(config["inner_allocation"]) in RATIO_SPACE_POLICIES
    # A rank floor gives each shape its own ceiling, and a matrix pinned at a
    # tighter one can end up below a less important matrix that had room left.
    # That inverts the score ordering for the same reason a rank-space policy
    # does: a second, shape-derived term now competes with the score
    uniform_ceilings = len(set(round(cap, 9) for cap in cap_map.values())) <= 1
    correlations = []

    for group_name, keys in grouping.groups.items():
        rho = spearman([score_map[key] for key in keys], [ratio_map[key] for key in keys])

        if math.isnan(rho):
            continue

        correlations.append(rho)

        # A ratio-space policy reads nothing but the score, so a more important
        # matrix must never be compressed harder. A rank-space one also prices a
        # rank at out + in, and that can outweigh the score ordering on a group
        # of mixed shapes, which is the family's bias rather than a defect
        if is_ratio_space and uniform_ceilings and rho > RATIO_TOLERANCE:
            problems.append(f"group {group_name} allocates more removal to higher scores (rho={rho:+.3f})")

    mean_rho = sum(correlations) / len(correlations) if correlations else float("nan")

    # A constant score carries no preference, so which constant it is must not
    # change the allocation. This holds for every policy, in either space
    low_map, _ = allocate(inputs, config, group_patterns, {key: 1.0 for key in score_map})
    high_map, _ = allocate(inputs, config, group_patterns, {key: 7.0 for key in score_map})
    drift = max(
        (abs(low_map[key] - high_map.get(key, 0.0)) for key in low_map),
        default=0.0,
    )

    if drift > 1e-6:
        problems.append(f"the value of a constant score changes the allocation (by {drift:.2e})")

    # A ratio-space policy has to collapse onto the flat ratio once its scores
    # carry nothing. Three cases are exempt: rank-space policies still favour
    # matrices that cost fewer parameters per rank, a non-neutral outer policy is
    # still allocating by Block Influence, which this does not flatten, and a
    # rank floor clips whichever shapes the flat ratio would overrun and spreads
    # what they cannot take onto the rest
    is_neutral_outer = OuterAllocation(config["outer_allocation"]) is OuterAllocation.PARAM_SHARE

    if is_ratio_space and is_neutral_outer and uniform_ceilings:
        flat_active = [low_map[key] for key in budget.active_keys if key in low_map]

        if flat_active:
            spread = max(flat_active) - min(flat_active)
            if spread > 1e-6:
                problems.append(f"uniform scores do not collapse to a flat ratio (spread {spread:.2e})")

    return problems, mean_rho


def evaluate_variant(
        inputs: Inputs,
        variant: Variant,
        group_patterns: Dict[str, List[str]]
) -> VariantResult:
    config = variant.config

    budget = compute_active_budget(
        layers_str=inputs.layers_str,
        param_count_map=inputs.param_count_map,
        target_ratio=config["compression_ratio"],
        bypass_early_layers=config["bypass_early_layers"],
        bypass_ratio=config["bypass_ratio"],
        max_ratio=config["max_ratio"],
        target_total_params=inputs.target_total_params,
        bypass_late_layers=config["bypass_late_layers"],
        num_layers=inputs.num_layers,
        cap_map=build_cap_map(
            inputs.layers_str,
            config["max_ratio"],
            inputs.shapes,
            config["min_rank_fraction"],
        ),
    )

    score_map = build_score_map(
        inputs,
        config["score_metric"],
        budget.active_keys,
        budget.active_ratio,
        config["fusion_alpha"],
    )
    ratio_map, budget_log = allocate(inputs, config, group_patterns, score_map)

    removed = sum(inputs.param_count_map[key] * ratio_map.get(key, 0.0) for key in inputs.layers_str)
    checks, score_ratio_rho = check_variant(inputs, config, group_patterns, ratio_map, score_map, budget)

    return VariantResult(
        ratio_map=ratio_map,
        score_map=score_map,
        budget_log=budget_log,
        realized_ratio=removed / inputs.target_total_params if inputs.target_total_params else 0.0,
        objectives={objective.name: objective.aggregate(inputs, ratio_map) for objective in OBJECTIVES},
        target_removed=budget.target_removed,
        active_keys=list(budget.active_keys),
        checks=checks,
        score_ratio_rho=score_ratio_rho,
    )


def write_reports(
        out_dir: str,
        inputs: Inputs,
        variants: List[Variant],
        results: Dict[str, VariantResult],
        ranks: Dict[str, Dict[str, float]],
        mean_rank: Dict[str, float],
        oracles: Dict[str, Dict[str, float]]
) -> None:
    """One row per variant, per matrix and per decoder layer, ready for pgfplots"""
    os.makedirs(out_dir, exist_ok=True)
    budget_dir = os.path.join(out_dir, "budget")
    os.makedirs(budget_dir, exist_ok=True)

    summary_path = os.path.join(out_dir, "summary.csv")
    with open(summary_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        objective_columns = [
            column
            for objective in OBJECTIVES
            for column in ( objective.name, f"{objective.name}_rank", f"{objective.name}_oracle_ratio" )
        ]
        writer.writerow([
            "variant", *sorted(SWEEPABLE), "outer_allocation", "ratio_scope",
            "realized_ratio", "mean_rank", *objective_columns,
            "min_ratio", "max_ratio_assigned", "mean_ratio", "ratio_std",
            "max_block_ratio", "max_block_layer", "block0_ratio",
            "score_ratio_rho", "checks",
        ])

        for variant in variants:
            result = results[variant.name]
            ratios = torch.tensor(
                [result.ratio_map.get(key, 0.0) for key in inputs.layers_str],
                dtype=torch.float64,
            )

            blocks = block_ratios(inputs, result.ratio_map)
            hot_layer, hot_ratio = peak_block(blocks)

            objective_cells: List[str] = []
            for objective in OBJECTIVES:
                value = result.objectives[objective.name]
                rank = ranks[objective.name][variant.name]
                oracle = oracles[variant.name][objective.name]
                gap = ""

                # Only meaningful where a lower bound exists and is not
                # zero, which rules out the non-additive objectives
                if math.isfinite(value) and math.isfinite(oracle) and oracle > 0.0:
                    gap = f"{value / oracle:.6f}"

                objective_cells += [
                    f"{value:.6e}" if math.isfinite(value) else "",
                    f"{rank:.0f}" if not math.isnan(rank) else "",
                    gap,
                ]

            writer.writerow([
                variant.name,
                *[variant.config[key] for key in sorted(SWEEPABLE)],
                variant.config["outer_allocation"],
                variant.config["ratio_scope"],
                f"{result.realized_ratio:.6f}",
                f"{mean_rank[variant.name]:.4f}" if not math.isnan(mean_rank[variant.name]) else "",
                *objective_cells,
                f"{ratios.min().item():.6f}",
                f"{ratios.max().item():.6f}",
                f"{ratios.mean().item():.6f}",
                f"{ratios.std().item():.6f}",
                f"{hot_ratio:.6f}" if math.isfinite(hot_ratio) else "",
                str(hot_layer) if hot_layer >= 0 else "",
                f"{blocks[0]:.6f}" if 0 in blocks else "",
                f"{result.score_ratio_rho:+.4f}" if not math.isnan(result.score_ratio_rho) else "",
                "; ".join(result.checks),
            ])

    matrices_path = os.path.join(out_dir, "matrices.csv")
    with open(matrices_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "variant", "key", "layer", "matrix_type", "out_features", "in_features",
            "params", "score", "ratio", "rank", "truncation_loss",
        ])

        for variant in variants:
            result = results[variant.name]
            score_map = result.score_map

            for key in inputs.layers_str:
                out_features, in_features = inputs.shapes[key]
                ratio = result.ratio_map.get(key, 0.0)
                spectrum = inputs.spectra.get(key)

                rank = ""
                loss = ""
                if spectrum is not None and ratio > 0.0:
                    rank_value = truncation_rank(out_features, in_features, ratio, len(spectrum))
                    rescaled = spectrum.to(torch.float64) * math.sqrt(inputs.n_calibration_tokens)
                    rank = str(rank_value)
                    loss = f"{float(rescaled[rank_value:].pow(2).sum().item()):.6e}"

                writer.writerow([
                    variant.name,
                    key,
                    get_layer_idx_from_key(key),
                    matrix_type_of(key),
                    out_features,
                    in_features,
                    inputs.param_count_map[key],
                    f"{score_map[key]:.6e}" if key in score_map else "",
                    f"{ratio:.6f}",
                    rank,
                    loss,
                ])

    layers_path = os.path.join(out_dir, "layers.csv")
    with open(layers_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([ "variant", "layer", "params", "removed_params", "layer_ratio", "block_influence" ])

        for variant in variants:
            result = results[variant.name]
            blocks = block_ratios(inputs, result.ratio_map)
            per_layer: Dict[int, List[str]] = defaultdict(list)

            for key in inputs.layers_str:
                per_layer[get_layer_idx_from_key(key)].append(key)

            for layer in sorted(per_layer):
                keys = per_layer[layer]
                params = sum(inputs.param_count_map[key] for key in keys)
                influence = inputs.importance.get(layer) if inputs.importance else None

                writer.writerow([
                    variant.name,
                    layer,
                    params,
                    f"{params * blocks[layer]:.0f}" if layer in blocks else "",
                    f"{blocks[layer]:.6f}" if layer in blocks else "",
                    f"{influence:.6f}" if influence is not None else "",
                ])

    for variant in variants:
        log_path = os.path.join(budget_dir, f"{variant.name}.log")
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write(results[variant.name].budget_log)

    print(f"\n[REPORT] Wrote {summary_path}")
    print(f"[REPORT] Wrote {matrices_path}")
    print(f"[REPORT] Wrote {layers_path}")
    print(f"[REPORT] Wrote per-variant [BUDGET] logs under {budget_dir}")


def report_importance(inputs: Inputs) -> None:
    """
    Relate Block Influence to spectral redundancy, per matrix family.

    Swift-SVD reports these two signals as negatively correlated, which is what
    justifies fusing them into one score. The sign is worth confirming on the
    model at hand before any composite allocation is run against it.
    """
    if not inputs.importance:
        print("\n[IMPORTANCE] No cached Block Influence, skipping the correlation report")
        return

    print("\n[IMPORTANCE] Block Influence vs normalized effective rank, per matrix type")
    print(f"  {'matrix type':<22} {'matrices':>9} {'spearman rho':>13}")

    for matrix_type in MATRIX_TYPES:
        influences: List[float] = []
        effective_ranks: List[float] = []

        for key, spectrum in inputs.spectra.items():
            if not key.endswith(matrix_type):
                continue

            layer = get_layer_idx_from_key(key)

            if layer not in inputs.importance:
                continue

            spectrum = spectrum.to(torch.float64)
            effective_rank = torch.exp(spectrum_entropy(normalized_spectrum(spectrum, squared=True))).item()

            influences.append(inputs.importance[layer])
            effective_ranks.append(effective_rank / len(spectrum))

        rho = spearman(influences, effective_ranks)

        if not math.isnan(rho):
            print(f"  {matrix_type:<22} {len(influences):>9} {rho:>+13.4f}")

    print(
        "  A negative rho reproduces Swift-SVD Fig. 3: blocks that transform the residual\n"
        "  stream most carry the least redundant spectra. A positive one means the two\n"
        "  signals agree rather than complement, and the fusion convention needs revisiting",
    )


def load_pyplot() -> Optional[Any]:
    """`pyplot` on the Agg backend, or None when matplotlib is not installed"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "\n[REPORT] matplotlib is not installed, so no PNG was rendered. Every figure "
            "still wrote its CSV, which is what the thesis consumes through pgfplots",
        )
        return None

    return plt


def write_figure_csv(fig_dir: str, name: str, header: List[str], rows: List[List[Any]]) -> None:
    with open(os.path.join(fig_dir, f"{name}.csv"), "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def save_figure(fig_dir: str, name: str, figure: Any, plt: Any) -> None:
    figure.tight_layout()
    figure.savefig(os.path.join(fig_dir, f"{name}.png"), dpi=150)
    plt.close(figure)


def present_matrix_types(inputs: Inputs) -> List[str]:
    """The matrix families this report actually covers, in a stable order"""
    present = {matrix_type_of(key) for key in inputs.layers_str}
    return [matrix_type for matrix_type in MATRIX_TYPES if matrix_type in present]


def keys_by_layer(inputs: Inputs) -> Dict[int, List[str]]:
    per_layer: Dict[int, List[str]] = defaultdict(list)

    for key in inputs.layers_str:
        per_layer[get_layer_idx_from_key(key)].append(key)

    return per_layer


def swept_keys(variants: List[Variant]) -> List[str]:
    """Knobs that actually vary across the sweep, which is what a curve plots against"""
    return [
        key for key in sorted(SWEEPABLE)
        if len({variant.config[key] for variant in variants}) > 1
    ]


def normalized_effective_rank(spectrum: torch.Tensor) -> float:
    """Effective rank as a fraction of the full rank, so shapes stay comparable"""
    spectrum = spectrum.to(torch.float64)
    return float(torch.exp(spectrum_entropy(normalized_spectrum(spectrum, squared=True)))) / len(spectrum)


def figure_scores_by_depth(fig_dir: str, inputs: Inputs, variants: List[Variant], results: Dict[str, VariantResult], plt: Any) -> None:
    """
    Score against decoder depth, one line per matrix family.

    Min-max normalized within each family, because the raw scores of a q
    projection and a down projection differ by orders of magnitude and would
    otherwise plot as two flat lines at opposite ends.
    """
    rows: List[List[Any]] = []

    for variant in variants:
        score_map = results[variant.name].score_map

        for matrix_type in present_matrix_types(inputs):
            scored = [
                ( get_layer_idx_from_key(key), score_map[key] )
                for key in inputs.layers_str
                if matrix_type_of(key) == matrix_type and key in score_map
            ]

            if not scored:
                continue

            values = [value for _, value in scored]
            lowest, span = min(values), max(values) - min(values)

            for layer, value in sorted(scored):
                normalized = (value - lowest) / span if span > 0 else 0.5
                rows.append([ variant.name, layer, matrix_type, f"{value:.6e}", f"{normalized:.6f}" ])

    write_figure_csv(fig_dir, "scores_by_depth", [ "variant", "layer", "matrix_type", "score", "normalized_score" ], rows)

    if plt is None or not rows:
        return

    # One panel per variant would not fit a page for a wide sweep, so the plot
    # shows the first variant and the CSV carries the rest
    variant_name = variants[0].name
    figure, axis = plt.subplots(figsize=(9, 5))

    for matrix_type in present_matrix_types(inputs):
        points = sorted(
            ( int(row[1]), float(row[4]) ) for row in rows
            if row[0] == variant_name and row[2] == matrix_type
        )

        if points:
            axis.plot([p[0] for p in points], [p[1] for p in points], marker="o", markersize=3, label=matrix_type)

    axis.set_xlabel("decoder layer")
    axis.set_ylabel("normalized score")
    axis.set_title(f"Score across depth per matrix family ({variant_name})")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=7)
    save_figure(fig_dir, "scores_by_depth", figure, plt)


def figure_influence_by_depth(fig_dir: str, inputs: Inputs, plt: Any) -> None:
    """Block Influence against depth, with spectral redundancy on the second axis"""
    if not inputs.importance:
        return

    per_layer = keys_by_layer(inputs)
    rows: List[List[Any]] = []

    for layer in sorted(per_layer):
        influence = inputs.importance.get(layer)

        if influence is None:
            continue

        ranks = [normalized_effective_rank(inputs.spectra[key]) for key in per_layer[layer] if key in inputs.spectra]
        mean_rank = sum(ranks) / len(ranks) if ranks else float("nan")
        rows.append([ layer, f"{influence:.6f}", f"{mean_rank:.6f}" if ranks else "" ])

    write_figure_csv(fig_dir, "influence_by_depth", [ "layer", "block_influence", "mean_normalized_eff_rank" ], rows)

    if plt is None or not rows:
        return

    layers = [int(row[0]) for row in rows]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(layers, [float(row[1]) for row in rows], marker="o", markersize=3, color="tab:blue", label="Block Influence")
    axis.set_xlabel("decoder layer")
    axis.set_ylabel("Block Influence", color="tab:blue")
    axis.grid(alpha=0.3)

    twin = axis.twinx()
    ranks = [float(row[2]) for row in rows if row[2]]

    if len(ranks) == len(rows):
        twin.plot(layers, ranks, marker="s", markersize=3, color="tab:red", label="normalized effective rank")
        twin.set_ylabel("normalized effective rank", color="tab:red")

    axis.set_title("Block Influence and spectral redundancy across depth")
    save_figure(fig_dir, "influence_by_depth", figure, plt)


def figure_influence_vs_effrank(fig_dir: str, inputs: Inputs, plt: Any) -> None:
    """
    The Swift-SVD Fig. 3 relationship, per matrix family.

    A negative rho is what justifies fusing the two signals into one composite
    score, so this figure is the gate on every composite run.
    """
    if not inputs.importance:
        return

    rows: List[List[Any]] = []
    correlations: Dict[str, float] = {}

    for matrix_type in present_matrix_types(inputs):
        influences: List[float] = []
        effective_ranks: List[float] = []

        for key in inputs.layers_str:
            if matrix_type_of(key) != matrix_type or key not in inputs.spectra:
                continue

            layer = get_layer_idx_from_key(key)
            influence = inputs.importance.get(layer)

            if influence is None:
                continue

            effective_rank = normalized_effective_rank(inputs.spectra[key])
            influences.append(influence)
            effective_ranks.append(effective_rank)
            rows.append([ matrix_type, layer, f"{influence:.6f}", f"{effective_rank:.6f}" ])

        correlations[matrix_type] = spearman(influences, effective_ranks)

    write_figure_csv(fig_dir, "influence_vs_effrank", [ "matrix_type", "layer", "block_influence", "normalized_eff_rank" ], rows)
    write_figure_csv(
        fig_dir,
        "influence_vs_effrank_rho",
        [ "matrix_type", "spearman_rho" ],
        [ [ matrix_type, f"{rho:+.6f}" ] for matrix_type, rho in correlations.items() if not math.isnan(rho) ],
    )

    if plt is None or not rows:
        return

    types = present_matrix_types(inputs)
    columns = min(4, len(types))
    figure_rows = math.ceil(len(types) / columns)
    figure, axes = plt.subplots(figure_rows, columns, figsize=(3.2 * columns, 3.0 * figure_rows), squeeze=False)

    for index, matrix_type in enumerate(types):
        axis = axes[index // columns][index % columns]
        points = [ ( float(row[2]), float(row[3]) ) for row in rows if row[0] == matrix_type ]
        axis.scatter([p[0] for p in points], [p[1] for p in points], s=12, alpha=0.7)
        rho = correlations.get(matrix_type, float("nan"))
        axis.set_title(f"{matrix_type}\nrho={rho:+.3f}" if not math.isnan(rho) else matrix_type, fontsize=8)
        axis.set_xlabel("Block Influence", fontsize=7)
        axis.set_ylabel("norm. eff. rank", fontsize=7)
        axis.tick_params(labelsize=6)
        axis.grid(alpha=0.3)

    for index in range(len(types), figure_rows * columns):
        axes[index // columns][index % columns].axis("off")

    save_figure(fig_dir, "influence_vs_effrank", figure, plt)


def figure_spectra(fig_dir: str, inputs: Inputs, plt: Any) -> None:
    """
    Singular value spectra on a log axis, for one representative block.

    Every spectrum would be hundreds of thousands of CSV rows, so this samples
    the first, middle and last block and thins each spectrum.
    """
    per_layer = keys_by_layer(inputs)
    layers = sorted(per_layer)

    if not layers:
        return

    sampled = sorted({ layers[0], layers[len(layers) // 2], layers[-1] })
    rows: List[List[Any]] = []

    for layer in sampled:
        for key in per_layer[layer]:
            spectrum = inputs.spectra.get(key)

            if spectrum is None:
                continue

            spectrum = rescaled_spectrum(inputs, key)
            step = max(1, len(spectrum) // 256)

            for index in range(0, len(spectrum), step):
                rows.append([ layer, matrix_type_of(key), index, f"{float(spectrum[index]):.6e}" ])

    write_figure_csv(fig_dir, "spectra", [ "layer", "matrix_type", "index", "singular_value" ], rows)

    if plt is None or not rows:
        return

    middle = sampled[len(sampled) // 2]
    figure, axis = plt.subplots(figsize=(9, 5))

    for matrix_type in present_matrix_types(inputs):
        points = sorted(
            ( int(row[2]), float(row[3]) ) for row in rows
            if row[0] == middle and row[1] == matrix_type
        )

        if points:
            axis.plot([p[0] for p in points], [p[1] for p in points], label=matrix_type)

    axis.set_yscale("log")
    axis.set_xlabel("singular value index")
    axis.set_ylabel("singular value")
    axis.set_title(f"Whitened spectra, decoder block {middle}")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=7)
    save_figure(fig_dir, "spectra", figure, plt)


def figure_layer_ratios(fig_dir: str, inputs: Inputs, variants: List[Variant], results: Dict[str, VariantResult], plt: Any) -> None:
    """Parameter-weighted removal ratio against depth, one line per variant"""
    per_layer = keys_by_layer(inputs)
    rows: List[List[Any]] = []

    for variant in variants:
        ratio_map = results[variant.name].ratio_map

        for layer in sorted(per_layer):
            keys = per_layer[layer]
            params = sum(inputs.param_count_map[key] for key in keys)
            removed = sum(inputs.param_count_map[key] * ratio_map.get(key, 0.0) for key in keys)
            rows.append([ variant.name, layer, f"{removed / params:.6f}" if params else "" ])

    write_figure_csv(fig_dir, "layer_ratios", [ "variant", "layer", "layer_ratio" ], rows)

    if plt is None or not rows:
        return

    figure, axis = plt.subplots(figsize=(9, 5))

    for variant in variants:
        points = sorted(( int(row[1]), float(row[2]) ) for row in rows if row[0] == variant.name and row[2])

        if points:
            axis.plot([p[0] for p in points], [p[1] for p in points], marker="o", markersize=3, label=variant.name)

    axis.set_xlabel("decoder layer")
    axis.set_ylabel("parameter-weighted removal ratio")
    axis.set_title("Allocation across decoder layers")
    axis.grid(alpha=0.3)

    if len(variants) <= 8:
        axis.legend(fontsize=7)

    save_figure(fig_dir, "layer_ratios", figure, plt)


def figure_ratio_heatmap(fig_dir: str, inputs: Inputs, variants: List[Variant], results: Dict[str, VariantResult], plt: Any) -> None:
    """Layer by matrix family, the whole allocation of a variant in one panel"""
    types = present_matrix_types(inputs)
    per_layer = keys_by_layer(inputs)
    layers = sorted(per_layer)
    rows: List[List[Any]] = []

    for variant in variants:
        ratio_map = results[variant.name].ratio_map

        for layer in layers:
            for matrix_type in types:
                matching = [key for key in per_layer[layer] if matrix_type_of(key) == matrix_type]

                if matching:
                    rows.append([ variant.name, layer, matrix_type, f"{ratio_map.get(matching[0], 0.0):.6f}" ])

    write_figure_csv(fig_dir, "ratio_heatmap", [ "variant", "layer", "matrix_type", "ratio" ], rows)

    if plt is None or not rows:
        return

    shown = variants[:6]
    figure, axes = plt.subplots(len(shown), 1, figsize=(10, 2.2 * len(shown)), squeeze=False)

    for index, variant in enumerate(shown):
        axis = axes[index][0]
        ratio_map = results[variant.name].ratio_map
        grid = [
            [
                next(
                    ( ratio_map.get(key, 0.0) for key in per_layer[layer] if matrix_type_of(key) == matrix_type ),
                    float("nan"),
                )
                for layer in layers
            ]
            for matrix_type in types
        ]
        image = axis.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
        axis.set_yticks(range(len(types)))
        axis.set_yticklabels(types, fontsize=6)
        axis.set_title(variant.name, fontsize=8)
        axis.set_xlabel("decoder layer", fontsize=7)
        figure.colorbar(image, ax=axis, label="ratio")

    save_figure(fig_dir, "ratio_heatmap", figure, plt)


def figure_ratio_by_type(fig_dir: str, inputs: Inputs, variants: List[Variant], results: Dict[str, VariantResult], plt: Any) -> None:
    """
    Mean ratio each matrix family receives, per variant.

    Where a rank-space policy shows its bias: it prices a rank at out + in, so
    on a group of mixed shapes it can compress a family harder than its score
    alone would justify.
    """
    types = present_matrix_types(inputs)
    rows: List[List[Any]] = []

    for variant in variants:
        ratio_map = results[variant.name].ratio_map

        for matrix_type in types:
            keys = [key for key in inputs.layers_str if matrix_type_of(key) == matrix_type]
            params = sum(inputs.param_count_map[key] for key in keys)
            removed = sum(inputs.param_count_map[key] * ratio_map.get(key, 0.0) for key in keys)
            rows.append([ variant.name, matrix_type, f"{removed / params:.6f}" if params else "" ])

    write_figure_csv(fig_dir, "ratio_by_type", [ "variant", "matrix_type", "mean_ratio" ], rows)

    if plt is None or not rows:
        return

    shown = variants[:8]
    width = 0.8 / max(1, len(shown))
    figure, axis = plt.subplots(figsize=(10, 5))

    for index, variant in enumerate(shown):
        values = [
            float(row[2]) for matrix_type in types
            for row in rows if row[0] == variant.name and row[1] == matrix_type and row[2]
        ]
        positions = [position + index * width for position in range(len(values))]
        axis.bar(positions, values, width=width, label=variant.name)

    axis.set_xticks([position + 0.4 for position in range(len(types))])
    axis.set_xticklabels(types, rotation=30, ha="right", fontsize=7)
    axis.set_ylabel("mean removal ratio")
    axis.set_title("Ratio per matrix family")
    axis.grid(alpha=0.3, axis="y")
    axis.legend(fontsize=7)
    save_figure(fig_dir, "ratio_by_type", figure, plt)


def figure_cap_binding(fig_dir: str, inputs: Inputs, variants: List[Variant], results: Dict[str, VariantResult], plt: Any) -> None:
    """
    How many matrices the per-matrix cap actually pins.

    A cap that binds nothing cannot change an allocation, which is what makes
    this the cheap way to choose the caps worth spending GPU time on.
    """
    rows: List[List[Any]] = []

    for variant in variants:
        ratio_map = results[variant.name].ratio_map
        cap_map = build_cap_map(
            inputs.layers_str,
            variant.config["max_ratio"],
            inputs.shapes,
            variant.config["min_rank_fraction"],
        )
        keys = [key for key in inputs.layers_str if key in ratio_map]
        assigned = [ratio_map[key] for key in keys]
        pinned = sum(1 for key in keys if ratio_map[key] >= cap_map[key] - RATIO_TOLERANCE)
        rows.append([
            variant.name,
            f"{variant.config['max_ratio']:.4f}",
            len(assigned),
            pinned,
            f"{pinned / len(assigned):.6f}" if assigned else "",
        ])

    write_figure_csv(fig_dir, "cap_binding", [ "variant", "max_ratio", "matrices", "pinned_at_cap", "fraction_pinned" ], rows)

    if plt is None or not rows:
        return

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(range(len(rows)), [float(row[4]) if row[4] else 0.0 for row in rows])
    axis.set_xticks(range(len(rows)))
    axis.set_xticklabels([row[0] for row in rows], rotation=45, ha="right", fontsize=6)
    axis.set_ylabel("fraction of matrices pinned at the cap")
    axis.set_title("Where --max_ratio binds")
    axis.grid(alpha=0.3, axis="y")
    save_figure(fig_dir, "cap_binding", figure, plt)


def figure_ratio_tail(
        fig_dir: str,
        inputs: Inputs,
        variants: List[Variant],
        results: Dict[str, VariantResult]
) -> None:
    """
    How much of the allocation sits in the aggressive region, and where it sits.

    The per-matrix peak is the wrong screen, and two runs at ratio 0.5 show why:
    `decoder` with `softmax_temp` reaches 0.900 on 31 matrices and measures 23.16,
    while `type` with `softmax_temp` reaches the same 0.900 on 11 and measures
    78.19. The first spreads its tail over 26 of 32 blocks and touches only q and
    k; the second concentrates it on blocks 0 to 2 across all seven families. What
    separates them is `max_block_ratio`, the fraction a whole block loses, which
    is why that column leads here and `peak` is kept only as context.
    """
    thresholds = ( 0.6, 0.7, 0.8, 0.85 )
    rows: List[List[Any]] = []
    doomed: List[Tuple[str, int, float]] = []
    screenable = selection_is_complete(inputs)

    for variant in variants:
        result = results[variant.name]
        ratio_map = result.ratio_map

        if not ratio_map:
            continue

        blocks = block_ratios(inputs, ratio_map)
        hot_layer, hot_ratio = peak_block(blocks)
        ranked = sorted(ratio_map.items(), key=lambda item: -item[1])
        top_layers = sorted({get_layer_idx_from_key(key) for key, _ in ranked[:8]})

        rows.append([
            variant.name,
            f"{hot_ratio:.6f}" if math.isfinite(hot_ratio) else "",
            hot_layer if hot_layer >= 0 else "",
            f"{blocks[0]:.6f}" if 0 in blocks else "",
            sum(1 for value in blocks.values() if value > BLOCK_RATIO_DANGER),
            f"{max(ratio_map.values()):.6f}",
            *[sum(1 for value in ratio_map.values() if value > threshold) for threshold in thresholds],
            ";".join(str(layer) for layer in top_layers),
        ])

        if screenable and math.isfinite(hot_ratio) and hot_ratio > BLOCK_RATIO_DANGER:
            doomed.append(( variant.label, hot_layer, hot_ratio ))

    write_figure_csv(
        fig_dir,
        "ratio_tail",
        [
            "variant", "max_block_ratio", "max_block_layer", "block0_ratio", "blocks_above_danger",
            "peak", *[f"above_{threshold}" for threshold in thresholds], "layers_of_top_8",
        ],
        rows,
    )

    if not screenable:
        print(
            "\n[REPORT] Not every matrix family is a target, so `max_block_ratio` is a fraction of "
            f"the selection and the {BLOCK_RATIO_DANGER:.2f} screen does not apply to this sweep"
        )
        return

    if not doomed:
        return

    print(
        f"\n[REPORT][WARNING] {len(doomed)} variant(s) push a decoder block past "
        f"{BLOCK_RATIO_DANGER:.2f} of its parameters:"
    )

    for label, layer, ratio in sorted(doomed, key=lambda item: -item[2]):
        print(f"  {label}   block {layer} loses {ratio:.4f}")

    print(
        "  every allocation measured past this point cost at least 3 perplexity against homogeneous;\n"
        "  staying under it rules out the depth failure only, not a bad split inside a block"
    )


def retained_rank_fraction(shape: Tuple[int, int], ratio: float) -> float:
    """
    Share of its full rank a matrix keeps at this ratio.

    A ratio is a share of parameters, and `rank = out * in * (1 - ratio) /
    (out + in)`, so the rank it leaves is `(1 - ratio) * max(out, in) /
    (out + in)` of `min(out, in)`. That conversion runs from 0.5 on a square
    matrix to nearly 1 on a very flat one, which is why the same ratio is a
    different amount of truncation depending on the shape.
    """
    out_features, in_features = shape
    return (1.0 - ratio) * max(out_features, in_features) / (out_features + in_features)


def figure_family_tail(
        fig_dir: str,
        inputs: Inputs,
        variants: List[Variant],
        results: Dict[str, VariantResult]
) -> None:
    """
    What each matrix family is asked to give up, and what rank that leaves it.

    The companion to `ratio_tail`, and the screen that replaces it wherever the
    block one goes blind. Under grouped-query attention the MLP carries most of
    a block -- 87% on Qwen2.5-7B against 67% on LLaMA-7B -- so the outer level
    fixes the block budget and every variant lands on the same depth profile:
    four Qwen runs spanning 12.0 to 48.0 perplexity share a `max_block_ratio` of
    0.2831 to the last digit. All of the freedom, and so all of the damage, has
    moved onto the family axis.

    The column that carries it is the retained rank fraction rather than the
    ratio, because `--max_ratio` is not shape invariant: a cap of 0.9 leaves a
    square matrix 5.0% of its rank, a 512x3584 projection 8.75% and an 18944x3584
    one 8.4%. Comparing families on ratio alone compares three different amounts
    of truncation.
    """
    families = sorted({matrix_type_of(key) for key in inputs.layers_str})
    rows: List[List[Any]] = []
    doomed: List[Tuple[str, str, float]] = []
    gqa = inputs.kv_sharing > 1

    for variant in variants:
        ratio_map = results[variant.name].ratio_map

        if not ratio_map:
            continue

        target = variant.config["compression_ratio"]

        for family in families:
            keys = [key for key in inputs.layers_str if matrix_type_of(key) == family and key in ratio_map]

            if not keys:
                continue

            ratios = [ratio_map[key] for key in keys]
            fractions = [retained_rank_fraction(inputs.shapes[key], ratio_map[key]) for key in keys]
            params = sum(inputs.param_count_map[key] for key in keys)
            weighted = sum(inputs.param_count_map[key] * ratio_map[key] for key in keys) / max(1, params)

            rows.append([
                variant.name,
                family,
                len(keys),
                inputs.head_partition.get(keys[0], ""),
                f"{weighted:.6f}",
                f"{max(ratios):.6f}",
                f"{weighted / target:.4f}" if target > 0 else "",
                f"{min(fractions):.6f}",
                f"{sum(fractions) / len(fractions):.6f}",
            ])

            is_screened_family = gqa and family in ( "self_attn.k_proj", "self_attn.v_proj" )

            if is_screened_family and min(fractions) < KV_RANK_FRACTION_DANGER:
                doomed.append(( variant.label, family, min(fractions) ))

    write_figure_csv(
        fig_dir,
        "family_tail",
        [
            "variant", "matrix_type", "matrices", "heads", "mean_ratio", "max_ratio",
            "ratio_over_target", "min_rank_fraction", "mean_rank_fraction",
        ],
        rows,
    )

    if not gqa:
        print(
            "\n[REPORT] Every query head owns its key and value, so the KV rank screen does "
            "not apply. On LLaMA-7B it would reject the best run at ratio 0.5, whose k_proj "
            "keeps 5% of its rank and still measures 18.89: under MHA that damage stays local "
            "to one head"
        )
        return

    if not doomed:
        return

    print(
        f"\n[REPORT][WARNING] {len(doomed)} variant/family pair(s) leave a key or value "
        f"projection under {KV_RANK_FRACTION_DANGER:.2f} of its rank, with "
        f"{inputs.kv_sharing} query heads reading each one:"
    )

    for label, family, fraction in sorted(doomed, key=lambda item: item[2]):
        print(f"  {label}   {family} keeps {fraction:.4f}")

    print(
        "  fitted to eight Qwen2.5-7B runs: every one at or below 0.141 measured at least\n"
        "  three times the homogeneous perplexity. Re-derive it per model, as with the block screen"
    )


def figure_map_distance(fig_dir: str, variants: List[Variant], results: Dict[str, VariantResult]) -> None:
    """
    How far apart two variants allocate, matrix by matrix.

    A sweep can hold two variants that produce the same allocation by different
    routes, and no amount of GPU time separates them. The peak of each is carried
    alongside, because two variants can be close on average and still sit on
    opposite sides of the ratio at which a matrix stops surviving truncation.

    There is no plot: the numbers are the point, and the pair list below them is
    what a stage file should be pruned against.
    """
    rows: List[List[Any]] = []
    duplicates: List[Tuple[str, str, float]] = []

    for index, first in enumerate(variants):
        for second in variants[index + 1:]:
            left = results[first.name].ratio_map
            right = results[second.name].ratio_map
            shared = sorted(set(left) & set(right))

            if not shared:
                continue

            deltas = [abs(left[key] - right[key]) for key in shared]
            mean_delta = sum(deltas) / len(deltas)
            max_delta = max(deltas)

            rows.append([
                first.name, second.name, len(shared),
                f"{mean_delta:.6f}", f"{max_delta:.6f}",
                f"{max(left.values()):.6f}", f"{max(right.values()):.6f}",
            ])

            if max_delta < DUPLICATE_MAP_MAX_DELTA:
                duplicates.append(( first.label, second.label, max_delta ))

    write_figure_csv(
        fig_dir,
        "map_distance",
        [ "variant_a", "variant_b", "matrices", "mean_abs_diff", "max_abs_diff", "peak_a", "peak_b" ],
        rows,
    )

    if not duplicates:
        return

    print(f"\n[REPORT][WARNING] {len(duplicates)} variant pair(s) allocate the same way:")

    for left, right, max_delta in sorted(duplicates, key=lambda item: item[2])[:MAX_REPORTED_DUPLICATES]:
        print(f"  {left}  ==  {right}   largest |delta ratio| on any matrix {max_delta:.4f}")

    if len(duplicates) > MAX_REPORTED_DUPLICATES:
        print(f"  ... and {len(duplicates) - MAX_REPORTED_DUPLICATES} more, see figures/map_distance.csv")

    print("  Running both of a pair pays twice for one experiment")


def figure_objectives(fig_dir: str, variants: List[Variant], results: Dict[str, VariantResult], ranks: Dict[str, Dict[str, float]], plt: Any) -> None:
    """
    Every variant priced under every objective.

    Reading it by row shows whether a variant is broadly good or only good at
    the objective its own score metric optimizes.
    """
    rows: List[List[Any]] = []

    for variant in variants:
        for objective in OBJECTIVES:
            value = results[variant.name].objectives[objective.name]
            best = min(
                (results[other.name].objectives[objective.name] for other in variants
                 if math.isfinite(results[other.name].objectives[objective.name])),
                default=float("nan"),
            )
            rows.append([
                variant.name,
                objective.name,
                f"{value:.6e}" if math.isfinite(value) else "",
                f"{ranks[objective.name][variant.name]:.0f}" if not math.isnan(ranks[objective.name][variant.name]) else "",
                f"{value / best:.6f}" if math.isfinite(value) and math.isfinite(best) and best > 0 else "",
                objective.circular_with,
            ])

    write_figure_csv(
        fig_dir,
        "objectives",
        [ "variant", "objective", "value", "rank", "value_vs_best", "circular_with" ],
        rows,
    )

    if plt is None or len(variants) < 2:
        return

    names = [objective.name for objective in OBJECTIVES]
    grid = [
        [ ranks[objective.name][variant.name] for objective in OBJECTIVES ]
        for variant in variants
    ]
    figure, axis = plt.subplots(figsize=(1.3 * len(names) + 4, 0.4 * len(variants) + 2))
    image = axis.imshow(grid, aspect="auto", cmap="RdYlGn_r")
    axis.set_xticks(range(len(names)))
    axis.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    axis.set_yticks(range(len(variants)))
    axis.set_yticklabels([variant.name for variant in variants], fontsize=6)
    axis.set_title("Rank per objective, 1 is best")
    figure.colorbar(image, ax=axis, label="rank")
    save_figure(fig_dir, "objectives", figure, plt)


def figure_oracle_gap(fig_dir: str, variants: List[Variant], results: Dict[str, VariantResult], oracles: Dict[str, Dict[str, float]], plt: Any) -> None:
    """How far each allocation sits above the best achievable value of an objective"""
    rows: List[List[Any]] = []

    for variant in variants:
        for objective in OBJECTIVES:
            value = results[variant.name].objectives[objective.name]
            oracle = oracles[variant.name][objective.name]
            gap = ""

            if math.isfinite(value) and math.isfinite(oracle) and oracle > 0.0:
                gap = f"{value / oracle:.6f}"

            rows.append([
                variant.name,
                objective.name,
                f"{value:.6e}" if math.isfinite(value) else "",
                f"{oracle:.6e}" if math.isfinite(oracle) else "",
                gap,
            ])

    write_figure_csv(fig_dir, "oracle_gap", [ "variant", "objective", "value", "oracle", "oracle_ratio" ], rows)

    if plt is None:
        return

    priced = [objective for objective in OBJECTIVES if objective.marginal is not None]

    if not priced:
        return

    width = 0.8 / len(priced)
    figure, axis = plt.subplots(figsize=(10, 5))

    for index, objective in enumerate(priced):
        gaps = []
        for variant in variants:
            row = next(row for row in rows if row[0] == variant.name and row[1] == objective.name)
            gaps.append(float(row[4]) if row[4] else 1.0)

        axis.bar([position + index * width for position in range(len(variants))], gaps, width=width, label=objective.name)

    axis.set_xticks([position + 0.4 for position in range(len(variants))])
    axis.set_xticklabels([variant.name for variant in variants], rotation=45, ha="right", fontsize=6)
    axis.set_yscale("log")
    axis.set_ylabel("value / lower bound (log)")
    axis.set_title("Distance to the best achievable value at the same budget and cap")
    axis.grid(alpha=0.3, axis="y")
    axis.legend(fontsize=7)
    save_figure(fig_dir, "oracle_gap", figure, plt)


def figure_dispersion(fig_dir: str, inputs: Inputs, variants: List[Variant], results: Dict[str, VariantResult], plt: Any) -> None:
    """
    How widely an allocation spreads its ratios.

    Comparing two policies at their default knobs compares their shape and their
    aggressiveness at once. Matching dispersion first is what isolates the shape,
    and reading it off this figure costs no GPU time.
    """
    swept = swept_keys(variants)
    rows: List[List[Any]] = []

    for variant in variants:
        ratios = torch.tensor(
            [results[variant.name].ratio_map.get(key, 0.0) for key in inputs.layers_str],
            dtype=torch.float64,
        )
        quantiles = torch.tensor([ 0.1, 0.9 ], dtype=torch.float64)
        low, high = torch.quantile(ratios, quantiles).tolist() if len(ratios) else ( float("nan"), float("nan") )
        rows.append([
            variant.name,
            *[variant.config[key] for key in swept],
            f"{float(ratios.std()):.6f}",
            f"{float(ratios.min()):.6f}",
            f"{float(ratios.max()):.6f}",
            f"{low:.6f}",
            f"{high:.6f}",
        ])

    write_figure_csv(
        fig_dir,
        "dispersion",
        [ "variant", *swept, "ratio_std", "ratio_min", "ratio_max", "ratio_p10", "ratio_p90" ],
        rows,
    )

    if plt is None or not rows:
        return

    figure, axis = plt.subplots(figsize=(10, 5))
    deviations = [float(row[len(swept) + 1]) for row in rows]

    # A single swept knob gives a curve worth reading; anything else is a
    # comparison of unordered configurations, so bars say more than a line
    if len(swept) == 1:
        pairs = sorted(zip([row[1] for row in rows], deviations), key=lambda item: str(item[0]))
        axis.plot([str(value) for value, _ in pairs], [deviation for _, deviation in pairs], marker="o")
        axis.set_xlabel(swept[0])
    else:
        axis.bar(range(len(rows)), deviations)
        axis.set_xticks(range(len(rows)))
        axis.set_xticklabels([row[0] for row in rows], rotation=45, ha="right", fontsize=6)

    axis.set_ylabel("standard deviation of the per-matrix ratio")
    axis.set_title("Allocation dispersion")
    axis.grid(alpha=0.3, axis="y")
    save_figure(fig_dir, "dispersion", figure, plt)


def write_figures(
        out_dir: str,
        inputs: Inputs,
        variants: List[Variant],
        results: Dict[str, VariantResult],
        ranks: Dict[str, Dict[str, float]],
        oracles: Dict[str, Dict[str, float]],
        render: bool
) -> None:
    """
    One tidy CSV per figure, plus a PNG preview when matplotlib is installed.

    The CSV is the deliverable: the thesis renders these through pgfplots so the
    figures carry its fonts, and matplotlib stays an optional convenience.
    """
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plt = load_pyplot() if render else None

    figure_scores_by_depth(fig_dir, inputs, variants, results, plt)
    figure_influence_by_depth(fig_dir, inputs, plt)
    figure_influence_vs_effrank(fig_dir, inputs, plt)
    figure_spectra(fig_dir, inputs, plt)

    figure_layer_ratios(fig_dir, inputs, variants, results, plt)
    figure_ratio_heatmap(fig_dir, inputs, variants, results, plt)
    figure_ratio_by_type(fig_dir, inputs, variants, results, plt)
    figure_cap_binding(fig_dir, inputs, variants, results, plt)

    figure_ratio_tail(fig_dir, inputs, variants, results)
    figure_family_tail(fig_dir, inputs, variants, results)
    figure_map_distance(fig_dir, variants, results)
    figure_objectives(fig_dir, variants, results, ranks, plt)
    figure_oracle_gap(fig_dir, variants, results, oracles, plt)
    figure_dispersion(fig_dir, inputs, variants, results, plt)

    print(f"[REPORT] Wrote figure data under {fig_dir}")


def format_table_row(cells: List[str], widths: List[int]) -> str:
    """Pad a row to the measured widths, the leading and trailing text columns left-aligned"""
    padded = [
        f"{cell:<{width}}" if index in ( 0, len(cells) - 1 ) else f"{cell:>{width}}"
        for index, ( cell, width ) in enumerate(zip(cells, widths))
    ]
    return "  ".join(padded).rstrip()


def render_summary_table(
        variants: List[Variant],
        inputs: Inputs,
        results: Dict[str, VariantResult],
        ranks: Dict[str, Dict[str, float]],
        mean_rank: Dict[str, float]
) -> None:
    """
    Print the variant comparison as one table, ordered by mean rank.

    Widths are measured from the cells instead of fixed, because a variant name
    is as long as the sweep makes it: a name overflowing a fixed field shifts
    every column to its right by its own excess, and no two rows line up.

    `worst block` carries the screen rather than the objectives, which rank the
    allocations backwards: it is the only column here that has been shown to
    order runs the same way a measured perplexity does.
    """
    axes = variants[0].axes if variants else ()
    rows: List[List[str]] = []
    header = [
        " / ".join(axes) if axes else "variant",
        "realized", "worst block", "mean rk",
        *(objective.name for objective in OBJECTIVES),
        "checks",
    ]

    ordered = sorted(
        variants,
        key=lambda item: mean_rank[item.name] if not math.isnan(mean_rank[item.name]) else math.inf,
    )

    for variant in ordered:
        result = results[variant.name]
        cells: List[str] = []

        for objective in OBJECTIVES:
            value = result.objectives[objective.name]
            rank = ranks[objective.name][variant.name]
            cell = "-"

            if math.isfinite(value) and not math.isnan(rank):
                cell = f"{value:.3e} ({rank:.0f})"

            cells.append(cell)

        hot_layer, hot_ratio = peak_block(block_ratios(inputs, result.ratio_map))
        block_cell = "-"

        if math.isfinite(hot_ratio):
            over = hot_ratio > BLOCK_RATIO_DANGER and selection_is_complete(inputs)
            block_cell = f"{hot_ratio:.3f} (L{hot_layer}){'!' if over else ''}"

        rows.append([
            variant.label,
            f"{result.realized_ratio:.4f}" if math.isfinite(result.realized_ratio) else "-",
            block_cell,
            f"{mean_rank[variant.name]:.2f}" if not math.isnan(mean_rank[variant.name]) else "-",
            *cells,
            "ok" if not result.checks else "; ".join(result.checks),
        ])

    widths = [max(len(row[column]) for row in ( header, *rows )) for column in range(len(header))]

    print()
    for row in ( header, *rows ):
        print(format_table_row(row, widths))


def main() -> None:
    args = parse_args()

    group_patterns: Dict[str, List[str]] = {}
    for group in args.group_patterns.split(";"):
        group_name, _, group_types = group.partition(":")
        group_patterns[group_name] = group_types.split(",")

    # Parsed before anything is read from disk, so a mistyped sweep fails at once
    variants = build_variants(args)
    inputs = load_inputs(args)

    print(f"\n[REPORT] Evaluating {len(variants)} variant(s)")

    results: Dict[str, VariantResult] = {}

    for variant in variants:
        # A cartesian product reaches combinations that do not exist, such as an
        # outer policy that needs per-block scores under a grouping that has
        # none. One of those must not take the rest of the sweep down with it
        try:
            results[variant.name] = evaluate_variant(inputs, variant, group_patterns)
        except (ValueError, NotImplementedError) as error:
            results[variant.name] = VariantResult(
                ratio_map={},
                score_map={},
                budget_log=f"rejected: {error}\n",
                realized_ratio=float("nan"),
                objectives={objective.name: float("nan") for objective in OBJECTIVES},
                target_removed=float("nan"),
                active_keys=[],
                checks=[f"rejected: {error}"],
                score_ratio_rho=float("nan"),
            )

    ranks = {
        objective.name: rank_variants({name: result.objectives[objective.name] for name, result in results.items()})
        for objective in OBJECTIVES
    }
    mean_rank = {name: mean_objective_rank(ranks, name) for name in results}
    oracles = compute_oracles(inputs, variants, results)

    render_summary_table(variants, inputs, results, ranks, mean_rank)

    print(
        "\n  Cells are the objective, with its rank across variants in parentheses and 1\n"
        "  being best, so a variant that wins one column and trails the rest is winning on\n"
        "  its own terms. Each objective and the score metric that optimizes it by\n"
        "  construction:",
    )
    for objective in OBJECTIVES:
        print(f"    {objective.name:<22} circular with {objective.circular_with}")

    report_importance(inputs)

    out_dir = args.out_dir or os.path.join(args.save_path, "allocation_reports", sanitize_model_name(args.model))
    write_reports(out_dir, inputs, variants, results, ranks, mean_rank, oracles)
    write_figures(out_dir, inputs, variants, results, ranks, oracles, render=args.plots)

    failed = [variant.name for variant in variants if results[variant.name].checks]

    if failed:
        print(f"\n[REPORT][WARNING] {len(failed)}/{len(variants)} variant(s) violated an allocation invariant")
        raise SystemExit(1)

    print(f"\n[REPORT] All {len(variants)} variant(s) satisfied every allocation invariant")


if __name__ == "__main__":
    main()
