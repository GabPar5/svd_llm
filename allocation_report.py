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
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

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


class Variant(NamedTuple):
    """One point of the sweep: a full allocation configuration and its name"""
    name: str
    config: Dict[str, Any]


class VariantResult(NamedTuple):
    ratio_map: Dict[str, float]
    score_map: Dict[str, float]
    budget_log: str
    realized_ratio: float
    predicted_loss: float
    checks: List[str]
    score_ratio_rho: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare compression-ratio allocations offline, from cached spectra",
    )

    parser.add_argument('--model', type=str, required=True, help='LLM the spectra were cached for')
    parser.add_argument('--save_path', type=str, default='./output', help='Root holding whitening_matrices/')
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
    parser.add_argument('--plots', action='store_true', help='Also render PNG previews, when matplotlib is installed')

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


def load_inputs(args: argparse.Namespace) -> Inputs:
    """Read the model config and the cached spectra, and reconcile the two"""
    version_str = "v2" if args.run_v2 else "v1"
    wm_dir = args.whitening_mat_path or whitening_dir(args.save_path, args.model, version_str)

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

        if swept:
            name = "__".join(f"{key}-{value}" for key, value in swept.items())
            # The name becomes a filename, and composite metrics carry a
            # separator no filename should hold
            name = re.sub(r"[^A-Za-z0-9.-]+", "_", name)

        variants.append(Variant(name=name, config=config))

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


def predicted_truncation_loss(inputs: Inputs, ratio_map: Dict[str, float]) -> float:
    """
    Total squared energy the allocation throws away, summed over all matrices.

    After whitening this is the theoretical activation reconstruction error, so
    it ranks allocations at equal budget without compressing anything. It is a
    proxy, not a perplexity: it ignores how errors compound across layers.
    """
    total = 0.0

    for key, ratio in ratio_map.items():
        spectrum = inputs.spectra.get(key)

        # A ratio of exactly 0.0 leaves the layer dense, no SVD and no loss
        if spectrum is None or ratio <= 0.0:
            continue

        out_features, in_features = inputs.shapes[key]
        rank = truncation_rank(out_features, in_features, ratio, len(spectrum))
        rescaled = spectrum.to(torch.float64) * math.sqrt(inputs.n_calibration_tokens)
        total += float(rescaled[rank:].pow(2).sum().item())

    return total


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

    out_of_bounds = [
        key for key, ratio in ratio_map.items()
        if ratio < -RATIO_TOLERANCE or ratio > config["max_ratio"] + RATIO_TOLERANCE
    ]
    if out_of_bounds:
        problems.append(f"{len(out_of_bounds)} ratios outside [0, {config['max_ratio']}], e.g. {out_of_bounds[0]}")

    grouping = build_allocation_groups(
        group_criterion=GroupBy(config["group_criterion"]),
        active_keys=budget.active_keys,
        score_map=score_map,
        group_patterns=group_patterns,
    )

    is_ratio_space = InnerAllocation(config["inner_allocation"]) in RATIO_SPACE_POLICIES
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
        if is_ratio_space and rho > RATIO_TOLERANCE:
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
    # carry nothing. Two cases are exempt: rank-space policies still favour
    # matrices that cost fewer parameters per rank, and a non-neutral outer
    # policy is still allocating by Block Influence, which this does not flatten
    is_neutral_outer = OuterAllocation(config["outer_allocation"]) is OuterAllocation.PARAM_SHARE

    if is_ratio_space and is_neutral_outer:
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
        predicted_loss=predicted_truncation_loss(inputs, ratio_map),
        checks=checks,
        score_ratio_rho=score_ratio_rho,
    )


def write_reports(
        out_dir: str,
        inputs: Inputs,
        variants: List[Variant],
        results: Dict[str, VariantResult],
        baseline_loss: float
) -> None:
    """One row per variant, per matrix and per decoder layer, ready for pgfplots"""
    os.makedirs(out_dir, exist_ok=True)
    budget_dir = os.path.join(out_dir, "budget")
    os.makedirs(budget_dir, exist_ok=True)

    summary_path = os.path.join(out_dir, "summary.csv")
    with open(summary_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "variant", *sorted(SWEEPABLE), "outer_allocation", "ratio_scope",
            "realized_ratio", "predicted_loss", "loss_vs_best",
            "min_ratio", "max_ratio_assigned", "mean_ratio", "ratio_std",
            "score_ratio_rho", "checks",
        ])

        for variant in variants:
            result = results[variant.name]
            ratios = torch.tensor(
                [result.ratio_map.get(key, 0.0) for key in inputs.layers_str],
                dtype=torch.float64,
            )

            writer.writerow([
                variant.name,
                *[variant.config[key] for key in sorted(SWEEPABLE)],
                variant.config["outer_allocation"],
                variant.config["ratio_scope"],
                f"{result.realized_ratio:.6f}",
                f"{result.predicted_loss:.6e}",
                f"{result.predicted_loss / baseline_loss:.6f}" if baseline_loss > 0 else "",
                f"{ratios.min().item():.6f}",
                f"{ratios.max().item():.6f}",
                f"{ratios.mean().item():.6f}",
                f"{ratios.std().item():.6f}",
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
            per_layer: Dict[int, List[str]] = defaultdict(list)

            for key in inputs.layers_str:
                per_layer[get_layer_idx_from_key(key)].append(key)

            for layer in sorted(per_layer):
                keys = per_layer[layer]
                params = sum(inputs.param_count_map[key] for key in keys)
                removed = sum(inputs.param_count_map[key] * result.ratio_map.get(key, 0.0) for key in keys)
                influence = inputs.importance.get(layer) if inputs.importance else None

                writer.writerow([
                    variant.name,
                    layer,
                    params,
                    f"{removed:.0f}",
                    f"{removed / params:.6f}" if params else "",
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


def render_plots(out_dir: str, inputs: Inputs, variants: List[Variant], results: Dict[str, VariantResult]) -> None:
    """PNG previews, when matplotlib is around. The CSV is the real deliverable"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "\n[REPORT] matplotlib is not installed, so no PNG was rendered. "
            "The CSV files feed pgfplots directly, which is what the thesis uses",
        )
        return

    figure, axis = plt.subplots(figsize=(9, 5))

    for variant in variants:
        result = results[variant.name]
        per_layer: Dict[int, List[str]] = defaultdict(list)

        for key in inputs.layers_str:
            per_layer[get_layer_idx_from_key(key)].append(key)

        layers = sorted(per_layer)
        ratios = [
            sum(inputs.param_count_map[key] * result.ratio_map.get(key, 0.0) for key in per_layer[layer])
            / sum(inputs.param_count_map[key] for key in per_layer[layer])
            for layer in layers
        ]
        axis.plot(layers, ratios, marker="o", markersize=3, label=variant.name)

    axis.set_xlabel("decoder layer")
    axis.set_ylabel("parameter-weighted removal ratio")
    axis.set_title("Allocation across decoder layers")
    axis.grid(alpha=0.3)

    if len(variants) <= 8:
        axis.legend(fontsize=7)

    path = os.path.join(out_dir, "layer_ratios.png")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)

    print(f"[REPORT] Wrote {path}")


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
                predicted_loss=float("nan"),
                checks=[f"rejected: {error}"],
                score_ratio_rho=float("nan"),
            )

    losses = [result.predicted_loss for result in results.values() if not math.isnan(result.predicted_loss)]
    baseline = min(losses) if losses else float("nan")

    print(f"\n{'variant':<48} {'realized':>9} {'pred. loss':>13} {'vs best':>8}  checks")
    for variant in variants:
        result = results[variant.name]
        relative = result.predicted_loss / baseline if baseline > 0 else float("nan")
        status = "ok" if not result.checks else "; ".join(result.checks)
        print(
            f"{variant.name:<48} {result.realized_ratio:>9.4f} "
            f"{result.predicted_loss:>13.4e} {relative:>8.4f}  {status}",
        )

    report_importance(inputs)

    out_dir = args.out_dir or os.path.join(args.save_path, "allocation_reports", sanitize_model_name(args.model))
    write_reports(out_dir, inputs, variants, results, baseline)

    if args.plots:
        render_plots(out_dir, inputs, variants, results)

    failed = [variant.name for variant in variants if results[variant.name].checks]

    if failed:
        print(f"\n[REPORT][WARNING] {len(failed)}/{len(variants)} variant(s) violated an allocation invariant")
        raise SystemExit(1)

    print(f"\n[REPORT] All {len(variants)} variant(s) satisfied every allocation invariant")


if __name__ == "__main__":
    main()
