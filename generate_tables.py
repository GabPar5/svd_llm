import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

PERPLEXITY_BENCHMARK_ORDER = [
    "wikitext",
    "c4",
]

LIKELIHOOD_BENCHMARK_ORDER = [
    "arc_easy",
    "hellaswag",
    "openbookqa",
    "piqa",
    "winogrande",
]

ACCURACY_BENCHMARKS = [
    "arc_easy",
    "hellaswag",
    "openbookqa",
    "piqa",
    "winogrande",
]

GENERATION_BENCHMARK_ORDER = [
    "gsm8k",
    "truthfulqa_gen",
]

BENCHMARK_ORDER = PERPLEXITY_BENCHMARK_ORDER + LIKELIHOOD_BENCHMARK_ORDER + GENERATION_BENCHMARK_ORDER

# Value columns as displayed: perplexity, likelihood, summary, generation
VALUE_COLUMNS = [
    *PERPLEXITY_BENCHMARK_ORDER,
    *LIKELIHOOD_BENCHMARK_ORDER,
    "avg_accuracy",
    *GENERATION_BENCHMARK_ORDER,
]

BENCHMARK_ALIASES = {
    "gsm8k": [ "gsm8k", "gsm8k_cot" ],
    "truthfulqa_gen": [ "truthfulqa_gen" ],
}

BENCHMARK_LABELS_MD = {
    "wikitext": "WikiText ↓",
    "c4": "C4 ↓",
    "arc_easy": "ARC-E ↑",
    "hellaswag": "HellaSwag ↑",
    "openbookqa": "OBQA ↑",
    "piqa": "PIQA ↑",
    "winogrande": "WinoG. ↑",
    "gsm8k": "GSM8K ↑",
    "truthfulqa_gen": "TruthfulQA ↑",
}

BENCHMARK_LABELS_LATEX = {
    "wikitext": r"WikiText $\downarrow$",
    "c4": r"C4 $\downarrow$",
    "arc_easy": r"ARC-E $\uparrow$",
    "hellaswag": r"HellaS. $\uparrow$",
    "openbookqa": r"OBQA $\uparrow$",
    "piqa": r"PIQA $\uparrow$",
    "winogrande": r"WinoG. $\uparrow$",
    "gsm8k": r"GSM8K $\uparrow$",
    "truthfulqa_gen": r"TruthfulQA $\uparrow$",
}

MATRIX_TOKEN_MAP = {
    "q": "q",
    "query": "q",
    "k": "k",
    "key": "k",
    "v": "v",
    "value": "v",
    "out": "out",
    "o": "out",
    "attn": "out",
    "attn_out": "out",
    "attention": "out",
    "attention_output": "out",
    "mlp": "mlp",
}

GROUPING_TOKENS = {
    "global": "global",
    "decoder": "decoder",
    "hierarchical": "hierarchical",
    "type": "matrix_type",
    "matrix": "matrix",
    "matrix_type": "matrix",
    "matrixtype": "matrix",
}

SCORING_TOKENS = {
    "truncation": "truncation_loss",
    "truncation_loss": "truncation_loss",
    "truncation_sq": "truncation_sq",
    "eff_rank": "eff_rank",
    "eff_rank_sq": "eff_rank_sq",
    "entropy": "entropy",
    "entropy_sq": "entropy_sq",
    "full_norm_tail_entropy": "full_norm_tail_entropy",
    "full_norm_sq_tail_entropy": "full_norm_sq_tail_entropy",
    "full_norm_tail_eff_rank": "full_norm_tail_eff_rank",
    "full_norm_sq_tail_eff_rank": "full_norm_sq_tail_eff_rank",
}

# Local halves a composite metric can fuse, and what it can fuse them with
COMPOSITE_LOCAL_TOKENS = (
    "truncation",
    "truncation_sq",
    "entropy",
    "entropy_sq",
    "eff_rank",
    "eff_rank_sq",
    "full_norm_tail_entropy",
    "full_norm_sq_tail_entropy",
    "full_norm_tail_eff_rank",
    "full_norm_sq_tail_eff_rank",
)
END_TO_END_TOKENS = ( "block_influence", )

# Both spellings resolve to the same label: the sidecar records the flag
# verbatim, while a filename cannot hold its separator and joins with "_".
# `norm|p` composites are left out because p is unbounded; those rows fall back
# to the sidecar, which is preferred anyway
SCORING_TOKENS.update({
    spelling: f"composite_{local}_{end_to_end}"
    for local in COMPOSITE_LOCAL_TOKENS
    for end_to_end in END_TO_END_TOKENS
    for spelling in ( f"composite|{local}|{end_to_end}", f"composite_{local}_{end_to_end}" )
})

SCHEME_TOKENS = {
    "het": "het",
    "heterogeneous": "het",
    "hom": "hom",
    "homogeneous": "hom",
}

# Sidecar written next to every result by main.py, see `save_run_config`
RUN_CONFIG_SUFFIX = ".config.json"

# Dimensions a stage gate compares runs along, as raw flag values so a resolved
# placeholder can be pasted straight into the next stage file. Most carry no
# filename token, which makes the sidecar the only place to read them from
HET_ONLY_DIMENSIONS = (
    "group_criterion",
    "score_metric",
    "inner_allocation",
    "outer_allocation",
    "max_ratio",
    "fusion_alpha",
    "offset",
    "softmax_temp",
    "outer_offset",
)

SHARED_DIMENSIONS = (
    "bypass_early_layers",
    "bypass_late_layers",
    "bypass_ratio",
    "ratio_scope",
    "seed",
)

GATE_DIMENSIONS = HET_ONLY_DIMENSIONS + SHARED_DIMENSIONS

# Compression targets, in the order the filename convention lists them
MATRIX_ARG_ORDER = (
    ("compress_att_q", "q"),
    ("compress_att_k", "k"),
    ("compress_att_v", "v"),
    ("compress_att_out", "out"),
    ("compress_mlp", "mlp"),
)

PREFERRED_METRICS = [
    "acc,none",
    "acc_norm,none",
]

GENERATION_METRICS = {
    "gsm8k": [
        "exact_match,strict-match",
        "exact_match,flexible-extract",
    ],
    "truthfulqa_gen": [
        "bleu_acc,none",
        "bleu_max,none",
        "rouge1_acc,none",
        "rouge1_max,none",
        "rougeL_acc,none",
        "rougeL_max,none",
    ],
}

def is_float_token(token: str) -> bool:
    try:
        float(token)
        return "." in token
    except ValueError:
        return False


def is_int_token(token: str) -> bool:
    return re.fullmatch(r"\d+", token) is not None


def safe_float(value: Any) -> Optional[float]:
    try:
        value = round(float(value), 2)
    except Exception:
        return None

    if math.isnan(value):
        return None

    return value


def fmt_accuracy(value: Any, decimals: int = 2) -> str:
    value = safe_float(value)
    if value is None:
        return "--"
    return f"{value:.{decimals}f}"


def fmt_ppl(value: Any, decimals: int = 2) -> str:
    value = safe_float(value)
    if value is None:
        return "--"
    return f"{value:.{decimals}f}"


def fmt_ratio(value: Any) -> str:
    value = safe_float(value)
    if value is None:
        return "--"
    return f"{int(round(value * 100))}\\%"


def fmt_ratio_md(value: Any) -> str:
    value = safe_float(value)
    if value is None:
        return "--"
    return f"{int(round(value * 100))}%"


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|")


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def latex_label_slug(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def normalize_filename_stem(path: Path) -> List[str]:
    stem = path.stem
    return stem.split("_")


# `--sequential_update` writes `_upd_<method>` into the run name, while
# `--update_taw_only` names its second checkpoint `_sequpd_<method>`
UPDATE_TOKEN_PATTERN = re.compile(r"_(?:sequpd|upd)_(lora|local_u)")


def update_method_from_name(run_name: str) -> str:
    match = UPDATE_TOKEN_PATTERN.search(run_name)
    return match.group(1) if match else ""


def base_run_name(run_name: str) -> str:
    """The run a sequential update started from, which is what pairs before with after"""
    return UPDATE_TOKEN_PATTERN.sub("", run_name)


def find_scoring(tokens: List[str], start_idx: int) -> Tuple[str, Optional[int], int]:
    """
    Parse a scoring token sequence starting at start_idx.

    Every key of SCORING_TOKENS is supported, matched longest-first because score
    names such as `eff_rank_sq` span several filename tokens.

    Returns:
        (scoring_name, scoring_start_idx, scoring_token_count)
    """
    for end in range(len(tokens), start_idx, -1):
        cand = "_".join(tokens[start_idx:end])
        if cand in SCORING_TOKENS:
            return SCORING_TOKENS[cand], start_idx, end - start_idx

    # A composite name embeds its local half ("composite_truncation_..."), which
    # the token-by-token fallback would happily mistake for the whole metric and
    # then read the bypass count off the wrong position. An unrecognized
    # composite is reported as such and left to the sidecar
    if start_idx < len(tokens) and tokens[start_idx] == "composite":
        return "unknown", None, 0

    # Fallback: scan token by token
    for i in range(start_idx, len(tokens)):
        tok = tokens[i]
        if tok in SCORING_TOKENS:
            mapped = SCORING_TOKENS[tok]
            count = 1
            if tok == "truncation" and i + 1 < len(tokens) and tokens[i + 1] == "loss":
                mapped = "truncation_loss"
                count = 2
            return mapped, i, count

    return "unknown", None, 0


def parse_filename(path: Path) -> Dict[str, Any]:
    """
    Supported filename styles:

    Original / uncompressed:
        Qwen_Qwen2.5_32B.json

    Heterogeneous compression:
        Qwen_Qwen2.5_32B_q_k_v_out_mlp_all_0.2_het_decoder_truncation_8_v2.json
        Qwen_Qwen2.5_32B_q_k_v_out_mlp_selected_0.2_het_decoder_truncation_8_v2.json

    Homogeneous compression:
        huggyllama_llama_7b_q_k_v_out_mlp_all_0.2_hom_8_v2.json
        huggyllama_llama_7b_q_k_v_out_mlp_selected_0.2_hom_8_v2.json
    """

    tokens = normalize_filename_stem(path)

    version = ""
    if tokens and re.fullmatch(r"v\d+", tokens[-1]):
        version = tokens.pop()

    has_ratio = any(is_float_token(tok) for tok in tokens)

    # Original model case
    if not has_ratio:
        model_name = "_".join(tokens)
        return {
            "file": path.name,
            "run_name": path.stem,
            "update_method": "",
            "checkpoint_path": "",
            "model": model_name,
            "is_original": True,
            "matrices": "none",
            "compression_ratio": 0.0,
            "scheme": "original",
            "grouping": "original",
            "scoring": "original",
            "bypassed_layers": 0,
            "filename_version": version,
        }

    ratio_idx = None
    compression_ratio = None

    for i, tok in enumerate(tokens):
        if is_float_token(tok):
            ratio_idx = i
            compression_ratio = float(tok)
            break

    matrices = []
    matrix_indices = []

    for i, tok in enumerate(tokens):
        mapped = MATRIX_TOKEN_MAP.get(tok)
        if mapped is not None:
            matrices.append(mapped)
            matrix_indices.append(i)

    seen = set()
    matrices = [m for m in matrices if not (m in seen or seen.add(m))]

    if matrix_indices:
        model_end = min(matrix_indices)
    elif ratio_idx is not None:
        model_end = ratio_idx
    else:
        model_end = len(tokens)

    model_name = "_".join(tokens[:model_end])

    scheme = "--"
    scheme_idx = None

    for i, tok in enumerate(tokens):
        if tok in SCHEME_TOKENS:
            scheme = SCHEME_TOKENS[tok]
            scheme_idx = i
            break

    grouping = "--"
    scoring = "--"
    bypassed_layers = 0

    if scheme == "het":
        if scheme_idx is not None and scheme_idx + 1 < len(tokens):
            next_tok = tokens[scheme_idx + 1]
            if next_tok in GROUPING_TOKENS:
                grouping = GROUPING_TOKENS[next_tok]
                score_start = scheme_idx + 2
            else:
                score_start = scheme_idx + 1

            scoring, scoring_idx, scoring_token_count = find_scoring(tokens, score_start)
            if scoring_idx is not None:
                bypass_idx = scoring_idx + scoring_token_count
                if bypass_idx < len(tokens) and is_int_token(tokens[bypass_idx]):
                    bypassed_layers = int(tokens[bypass_idx])

    elif scheme == "hom":
        grouping = "--"
        scoring = "--"

        if scheme_idx is not None:
            bypass_idx = scheme_idx + 1
            if bypass_idx < len(tokens) and is_int_token(tokens[bypass_idx]):
                bypassed_layers = int(tokens[bypass_idx])

    else:
        # Fallback for malformed names
        if scheme_idx is not None and scheme_idx + 1 < len(tokens):
            candidate = tokens[scheme_idx + 1]
            if candidate in GROUPING_TOKENS:
                grouping = GROUPING_TOKENS[candidate]
                score_start = scheme_idx + 2
            else:
                score_start = scheme_idx + 1

            scoring, scoring_idx, scoring_token_count = find_scoring(tokens, score_start)
            if scoring_idx is not None:
                bypass_idx = scoring_idx + scoring_token_count
                if bypass_idx < len(tokens) and is_int_token(tokens[bypass_idx]):
                    bypassed_layers = int(tokens[bypass_idx])

    return {
        "file": path.name,
        "run_name": path.stem,
        "update_method": update_method_from_name(path.stem),
        "checkpoint_path": "",
        "model": model_name,
        "is_original": False,
        "matrices": "+".join(matrices) if matrices else "unknown",
        "compression_ratio": compression_ratio,
        "scheme": scheme,
        "grouping": grouping,
        "scoring": scoring,
        "bypassed_layers": bypassed_layers,
        "filename_version": version,
    }


def is_run_config(path: Path) -> bool:
    return path.name.endswith(RUN_CONFIG_SUFFIX)


def load_run_config(result_path: Path) -> Optional[Dict[str, Any]]:
    """Read the sidecar written beside a result file, if the run produced one"""
    sidecar = result_path.with_name(f"{result_path.stem}{RUN_CONFIG_SUFFIX}")

    if not sidecar.is_file():
        return None

    try:
        with sidecar.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as error:
        print(f"[WARNING] Ignoring unreadable run config {sidecar}: {error}")
        return None


def sanitize_model_name(model_name: str) -> str:
    """Mirror of `src.utils.sanitize_model_name`, inlined to keep this script import-free"""
    return model_name.replace("/", "_").replace("-", "_")


def update_method_of(run_args: Dict[str, Any]) -> str:
    """The sequential-update method a run applied, empty when it applied none"""
    if not (run_args.get("sequential_update") or run_args.get("update_taw_only")):
        return ""

    return str(run_args.get("sequential_update_method") or "")


def checkpoint_path_for(run_args: Dict[str, Any], run_name: str) -> str:
    """
    The checkpoint this row describes, which is what fills a `__CKPT_*__`.

    An evaluation-only run was handed the path; a compression run wrote it under
    `<save_path>/models/<model>/<run_name>.pt`. `--update_taw_only` is neither:
    it writes a second checkpoint beside the one it loaded, and the row reports
    that one. A dense run wrote nothing at all.
    """
    if "use_compressed" in run_args and not run_args.get("use_compressed"):
        return ""

    given = run_args.get("compressed_model_path")

    if given:
        given = str(given)

        if not run_args.get("update_taw_only"):
            return given

        stem = given[:-3] if given.endswith(".pt") else given
        return f"{stem}_sequpd_{run_args.get('sequential_update_method')}.pt"

    save_path = run_args.get("save_path")
    model = run_args.get("model")

    if not save_path or not model or not run_name:
        return ""

    return str(Path(str(save_path)) / "models" / sanitize_model_name(str(model)) / f"{run_name}.pt")


def dimensions_from_allocation(allocation: Dict[str, Any]) -> Dict[str, Any]:
    """
    The dimensions the compression step recorded, policies and live knobs alike.

    These outrank the arguments: the sidecar records the knobs the policies
    actually read, so a knob passed to a policy that ignores it never shows up
    here as if it had been in effect.
    """
    knobs = { **(allocation.get("inner_knobs") or {}), **(allocation.get("outer_knobs") or {}) }

    recorded = {
        "inner_allocation": allocation.get("inner_allocation"),
        "outer_allocation": allocation.get("outer_allocation"),
        **knobs,
    }

    return {key: value for key, value in recorded.items() if value is not None}


def row_from_run_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the configuration columns of a row straight from the run sidecar.

    `parse_filename` has to infer these positionally from a name that cannot
    carry every dimension, so wherever the sidecar speaks it wins -- with one
    exception. A run that only evaluated, or only updated, an existing
    checkpoint was invoked without the flags that produced it, so its arguments
    describe that invocation rather than the compression it reports. There the
    filename, which is the checkpoint's own run name, stays authoritative and
    the `allocation` block merged in from the checkpoint fills the rest.
    """
    run_args = config.get("args")

    if not isinstance(run_args, dict):
        return {}

    allocation = config.get("allocation")
    allocation = allocation if isinstance(allocation, dict) else {}
    run_name = str(config.get("run_name") or "")

    row: Dict[str, Any] = {}

    # Left to the filename when the arguments do not carry it, which is the case
    # for a run that merely evaluated an already updated checkpoint
    for key, value in (
        ( "run_name", run_name ),
        ( "update_method", update_method_of(run_args) ),
        ( "checkpoint_path", checkpoint_path_for(run_args, run_name) ),
    ):
        if value:
            row[key] = value

    model = run_args.get("model")
    if model:
        row["model"] = sanitize_model_name(str(model))

    # Target against actual, the only way a run whose allocation drifted off its
    # budget stays visibly incomparable to one that hit it
    if allocation.get("realized_overall_ratio") is not None:
        row["realized_ratio"] = allocation["realized_overall_ratio"]

    row.update(dimensions_from_allocation(allocation))

    if run_args.get("compressed_model_path"):
        target_ratio = allocation.get("target_ratio")

        if target_ratio is not None:
            row["compression_ratio"] = round(float(target_ratio), 2)

        return row

    # Absent on runs predating the key, where the filename still decides
    if "use_compressed" in run_args:
        row["is_original"] = not run_args.get("use_compressed")

    if row.get("is_original"):
        # A dense run carries every compression default it never applied, so
        # reading its configuration off the arguments would invent a run
        row.update({
            "matrices": "none",
            "compression_ratio": 0.0,
            "scheme": "original",
            "grouping": "original",
            "scoring": "original",
            "bypassed_layers": 0,
        })

        return row

    matrices = [token for key, token in MATRIX_ARG_ORDER if run_args.get(key)]
    heterogeneous = bool(run_args.get("het"))

    # The bypass column has always meant "layers left out of redistribution",
    # so both ends are summed rather than reported separately
    bypassed = max(0, int(run_args.get("bypass_early_layers", -1) or -1))
    bypassed += max(0, int(run_args.get("bypass_late_layers", -1) or -1))

    row.update({
        "matrices": "+".join(matrices) if matrices else "unknown",
        "compression_ratio": round(float(run_args.get("compression_ratio", 0.0)), 2),
        "scheme": "het" if heterogeneous else "hom",
        "grouping": GROUPING_TOKENS.get(
            str(run_args.get("group_criterion", "")),
            str(run_args.get("group_criterion", "--")),
        ) if heterogeneous else "--",
        "scoring": SCORING_TOKENS.get(
            str(run_args.get("score_metric", "")),
            str(run_args.get("score_metric", "--")),
        ) if heterogeneous else "--",
        "bypassed_layers": bypassed,
        "filename_version": "v2" if run_args.get("run_v2") else "",
    })

    # Raw flag values, which is what a resolved placeholder has to be filled
    # with. A homogeneous run allocates nothing, so its heterogeneous defaults
    # would group it under a score it never computed
    dimensions = SHARED_DIMENSIONS + ( HET_ONLY_DIMENSIONS if heterogeneous else () )

    for dimension in dimensions:
        if dimension in run_args and dimension not in row:
            row[dimension] = run_args[dimension]

    for end in ( "bypass_early_layers", "bypass_late_layers" ):
        row[end] = max(0, int(row.get(end, -1) or -1))

    return row


def pick_accuracy_metric(task_result: Dict[str, Any]) -> Tuple[Optional[str], Optional[Any]]:
    for metric_name in PREFERRED_METRICS:
        if metric_name in task_result:
            return metric_name.replace(",none", ""), task_result[metric_name]
    return None, None


def clean_metric_name(metric_name: str) -> str:
    return metric_name.replace(",none", "").replace(",", "_")


def pick_generation_metric(benchmark: str, task_result: Dict[str, Any]) -> Tuple[Optional[str], Optional[Any]]:
    for metric_name in GENERATION_METRICS.get(benchmark, []):
        if metric_name in task_result:
            return clean_metric_name(metric_name), task_result[metric_name]

    for metric_name, value in task_result.items():
        if metric_name.endswith("_stderr,none") or metric_name in {"alias", "samples"}:
            continue
        if safe_float(value) is not None:
            return clean_metric_name(metric_name), value

    return None, None


def get_task_result(results: Dict[str, Any], benchmark: str) -> Optional[Dict[str, Any]]:
    for task_name in BENCHMARK_ALIASES.get(benchmark, [ benchmark ]):
        task_result = results.get(task_name)
        if task_result is not None:
            return task_result
    return None


def load_result(path: Path, prefer_lm_eval_model_name: bool = False) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # The filename stays the fallback for runs predating the sidecar
    row = parse_filename(path)

    run_config = load_run_config(path)
    if run_config is not None:
        row.update(row_from_run_config(run_config))

    if prefer_lm_eval_model_name:
        lm_eval_model = data.get("config", {}).get("model")
        if lm_eval_model:
            row["model"] = lm_eval_model.replace("/", "_")

    results = data.get("results", {})
    metric_used = {}
    acc_values = []

    for benchmark in BENCHMARK_ORDER:
        task_result = get_task_result(results, benchmark)

        if task_result is None:
            row[benchmark] = None
            metric_used[benchmark] = None
            continue

        if benchmark in PERPLEXITY_BENCHMARK_ORDER:
            row[benchmark] = task_result.get("token_perplexity,none")
            metric_used[benchmark] = "token_perplexity"
            continue

        if benchmark in GENERATION_BENCHMARK_ORDER:
            metric_name, value = pick_generation_metric(benchmark, task_result)
            row[benchmark] = value
            metric_used[benchmark] = metric_name
            continue

        metric_name, value = pick_accuracy_metric(task_result)
        row[benchmark] = value
        metric_used[benchmark] = metric_name

        value_float = safe_float(value)
        if benchmark in ACCURACY_BENCHMARKS and value_float is not None:
            acc_values.append(value_float)

    avg_accuracy = None
    if acc_values:
        avg_accuracy = sum(acc_values) / len(acc_values)

    row["avg_accuracy"] = avg_accuracy
    row["metric_used"] = metric_used

    return row


def sort_rows_hierarchical(row: Dict[str, Any]) -> Tuple:
    is_original = row.get("is_original", False)

    ratio = row.get("compression_ratio")
    ratio_sort = ratio if ratio is not None else 999.0

    return (
        row.get("model", ""),
        0 if is_original else 1,
        ratio_sort,
        row.get("bypassed_layers", 0),
        row.get("scheme", ""),
        row.get("grouping", ""),
        row.get("scoring", ""),
        row.get("matrices", ""),
        row.get("file", ""),
    )


def metric_value_for_display(row: Dict[str, Any], column: str) -> str:
    if column in PERPLEXITY_BENCHMARK_ORDER:
        return fmt_ppl(row.get(column))
    return fmt_accuracy(row.get(column))


def best_values(rows: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """
    Finds best benchmark values inside a local group.

    For the perplexity benchmarks, lower is better.
    For all other displayed benchmark values and average, higher is better.
    """

    best = {}

    for column in VALUE_COLUMNS:
        values = []

        for row in rows:
            value = safe_float(row.get(column))
            if value is not None:
                values.append(value)

        if not values:
            best[column] = None
        elif column in PERPLEXITY_BENCHMARK_ORDER:
            best[column] = min(values)
        else:
            best[column] = max(values)

    return best

def group_rows_for_model_by_bypass(
    rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[Any, Dict[Any, List[Dict[str, Any]]]]]:
    """Split the baseline rows out and group the rest by bypassed layers, then ratio"""
    original_rows = []
    grouped = defaultdict(lambda: defaultdict(list))

    for row in rows:
        if row.get("is_original", False):
            original_rows.append(row)
        else:
            grouped[row.get("bypassed_layers")][row.get("compression_ratio")].append(row)

    return sorted(original_rows, key=sort_rows_hierarchical), grouped


def best_values_non_original(rows: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    return best_values([r for r in rows if not r.get("is_original", False)])

def is_best(row: Dict[str, Any], benchmark: str, best: Dict[str, Optional[float]]) -> bool:
    value = safe_float(row.get(benchmark))
    best_value = best.get(benchmark)

    if value is None or best_value is None:
        return False

    return abs(value - best_value) < 1e-12


def markdown_faded(value: Any) -> str:
    return f'<span style="color: #888888;">{markdown_escape(value)}</span>'


def markdown_cell(value: str, highlight: bool) -> str:
    if highlight:
        return f"**{value}**"
    return value


def markdown_faded_cell(value: str, highlight: bool) -> str:
    return markdown_faded(value)


def latex_cell(value: str, highlight: bool) -> str:
    if highlight:
        return rf"\textbf{{{latex_escape(value)}}}"
    return latex_escape(value)


def latex_faded_cell(value: str, highlight: bool) -> str:
    return rf"\textcolor{{black!45}}{{{latex_escape(value)}}}"


def value_cells(
    row: Dict[str, Any],
    render: Callable[[str, bool], str],
    best: Optional[Dict[str, Optional[float]]] = None,
) -> List[str]:
    """
    Formatted value cells of one row, in table order.

    Without `best` nothing is highlighted, which is what baseline rows need.
    """
    cells = []

    for column in VALUE_COLUMNS:
        highlight = best is not None and is_best(row, column, best)
        cells.append(render(metric_value_for_display(row, column), highlight))

    return cells

def make_markdown_table_for_model(model_name: str, rows: List[Dict[str, Any]]) -> str:
    rows = sorted(rows, key=sort_rows_hierarchical)
    original_rows, grouped = group_rows_for_model_by_bypass(rows)

    headers = [
        "Ratio",
        "Grouping",
        "Scoring",
        "Scheme",
        "Matrices",
    ]

    headers += [BENCHMARK_LABELS_MD[b] for b in PERPLEXITY_BENCHMARK_ORDER]
    headers += [BENCHMARK_LABELS_MD[b] for b in LIKELIHOOD_BENCHMARK_ORDER]
    headers += [ "Average ↑" ]
    headers += [BENCHMARK_LABELS_MD[b] for b in GENERATION_BENCHMARK_ORDER]
    headers += [ "File" ]

    lines: List[str] = [ f"## {model_name}", "" ]

    for bypass in sorted(grouped.keys()):
        lines.append(f"### Bypassed layers: {bypass}")
        lines.append("")

        # Faded baseline row shown at the top of each bypass-specific table
        if original_rows:
            for row in original_rows:
                row_cells = [ markdown_faded("0%") ]
                row_cells += [ markdown_faded("--") ] * 4
                row_cells += value_cells(row, markdown_faded_cell)
                row_cells.append(markdown_faded(row.get("file", "--")))
                lines.append("| " + " | ".join(row_cells) + " |")

            lines.append("")

        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join([ "---" ] * len(headers)) + " |")

        for ratio in sorted(grouped[bypass].keys(), key=lambda x: -1 if x is None else x):
            local_rows = sorted(grouped[bypass][ratio], key=sort_rows_hierarchical)
            ratio_best = best_values_non_original(local_rows)

            first_ratio_row = True
            for row in local_rows:
                ratio_cell = ""
                if first_ratio_row:
                    ratio_cell = fmt_ratio_md(ratio)

                row_cells = [
                    ratio_cell,
                    row.get("grouping", "--"),
                    row.get("scoring", "--"),
                    row.get("scheme", "--"),
                    row.get("matrices", "--"),
                ]

                row_cells += value_cells(row, markdown_cell, ratio_best)
                row_cells.append(row.get("file", "--"))

                lines.append("| " + " | ".join(markdown_escape(v) for v in row_cells) + " |")
                first_ratio_row = False

        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def make_latex_table_for_model(
    model_name: str,
    rows: List[Dict[str, Any]],
    table_size: str = r"\scriptsize",
    use_adjustbox: bool = True,
    table_width: float = 1.6,
) -> str:
    rows = sorted(rows, key=sort_rows_hierarchical)
    original_rows, grouped = group_rows_for_model_by_bypass(rows)

    lines: List[str] = []

    for bypass in sorted(grouped.keys()):
        lines.append(r"\begin{table*}[t]")
        lines.append(r"\centering")
        lines.append(table_size)
        lines.append(r"\setlength{\tabcolsep}{3pt}")

        if use_adjustbox:
            lines.append(r"\begin{adjustbox}{width=" + str(table_width) + r"\textwidth,center}")

        ppl_benchmark_colspec = "r" * len(PERPLEXITY_BENCHMARK_ORDER)
        benchmark_colspec = "r" * len(LIKELIHOOD_BENCHMARK_ORDER)
        generation_colspec = "r" * len(GENERATION_BENCHMARK_ORDER)
        colspec = rf"l llll | {ppl_benchmark_colspec} {benchmark_colspec} r {generation_colspec}"
        lines.append(rf"\begin{{tabular}}{{{colspec}}}")
        lines.append(r"\toprule")

        ppl_start_col = 6
        ppl_end_col = ppl_start_col + len(PERPLEXITY_BENCHMARK_ORDER) - 1
        likelihood_start_col = ppl_end_col + 1
        likelihood_end_col = likelihood_start_col + len(LIKELIHOOD_BENCHMARK_ORDER) - 1
        summary_col = likelihood_end_col + 1
        generation_start_col = summary_col + 1
        generation_end_col = generation_start_col + len(GENERATION_BENCHMARK_ORDER) - 1
        total_col_count = generation_end_col

        lines.append(
            r"\multicolumn{1}{c}{Compression} & "
            r"\multicolumn{4}{c}{Configuration} & "
            rf"\multicolumn{{{len(PERPLEXITY_BENCHMARK_ORDER)}}}{{c}}{{Perplexity Benchmarks}} & "
            rf"\multicolumn{{{len(LIKELIHOOD_BENCHMARK_ORDER)}}}{{c}}{{Likelihood Benchmarks}} & "
            r"\multicolumn{1}{c}{Summary} & "
            rf"\multicolumn{{{len(GENERATION_BENCHMARK_ORDER)}}}{{c}}{{Generation Benchmarks}} \\",
        )

        lines.append(
            r"\cmidrule(lr){1-1}"
            r"\cmidrule(lr){2-5}"
            rf"\cmidrule(lr){{{ppl_start_col}-{ppl_end_col}}}"
            rf"\cmidrule(lr){{{likelihood_start_col}-{likelihood_end_col}}}"
            rf"\cmidrule(lr){{{summary_col}-{summary_col}}}"
            rf"\cmidrule(lr){{{generation_start_col}-{generation_end_col}}}",
        )

        header = [
            "Ratio",
            "Group",
            "Score",
            "Scheme",
            "Matrices",
        ]

        header += [BENCHMARK_LABELS_LATEX[b] for b in PERPLEXITY_BENCHMARK_ORDER]
        header += [BENCHMARK_LABELS_LATEX[b] for b in LIKELIHOOD_BENCHMARK_ORDER]
        header += [ r"Avg. $\uparrow$" ]
        header += [BENCHMARK_LABELS_LATEX[b] for b in GENERATION_BENCHMARK_ORDER]

        lines.append(" & ".join(header) + r" \\")
        lines.append(r"\midrule")

        # Faded baseline row shown at the top of each bypass-specific table
        if original_rows:
            for row in original_rows:
                cells = [ r"\textcolor{black!45}{0\%}" ]
                cells += [ r"\textcolor{black!45}{--}" ] * 4
                cells += value_cells(row, latex_faded_cell)

                lines.append(" & ".join(cells) + r" \\")
            lines.append(r"\midrule")

        sorted_ratios = sorted(grouped[bypass].keys(), key=lambda x: -1 if x is None else x)

        for ratio_idx, ratio in enumerate(sorted_ratios):
            local_rows = sorted(grouped[bypass][ratio], key=sort_rows_hierarchical)
            ratio_best = best_values_non_original(local_rows)

            first_ratio_row = True
            for row in local_rows:
                ratio_cell = ""
                if first_ratio_row:
                    ratio_cell = fmt_ratio(ratio)

                cells = [
                    ratio_cell,
                    latex_escape(row.get("grouping", "--")),
                    latex_escape(row.get("scoring", "--")),
                    latex_escape(row.get("scheme", "--")),
                    latex_escape(row.get("matrices", "--")),
                ]

                cells += value_cells(row, latex_cell, ratio_best)

                lines.append(" & ".join(cells) + r" \\")
                first_ratio_row = False

            if ratio_idx != len(sorted_ratios) - 1:
                lines.append(r"\cmidrule(lr){1-" + str(total_col_count) + r"}")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

        if use_adjustbox:
            lines.append(r"\end{adjustbox}")

        caption_model = latex_escape(model_name)
        label_model = latex_label_slug(model_name)

        lines.append(
            rf"\caption{{Zero-shot lm-eval results for {caption_model} (bypassed initial layers = {bypass}). "
            r"Rows are grouped by compression ratio. "
            r"Accuracy-style metrics are reported as percentages; \texttt{acc} is used when available and "
            r"\texttt{acc\_norm} otherwise. Average accuracy excludes generation benchmarks. "
            r"WikiText and C4 are reported as token perplexity, where lower is better. "
            r"Bold values indicate the best result within each compression ratio.}",
        )
        lines.append(rf"\label{{tab:lm-eval-hierarchical-{label_model}-b{bypass}}}")
        lines.append(r"\end{table*}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def make_metric_note(rows: List[Dict[str, Any]]) -> str:
    used = defaultdict(set)

    for row in rows:
        for benchmark, metric in row.get("metric_used", {}).items():
            if metric:
                used[benchmark].add(metric)

    parts = []
    for benchmark in BENCHMARK_ORDER:
        metrics = sorted(used.get(benchmark, []))
        if metrics:
            parts.append(f"`{benchmark}`: {', '.join(metrics)}")

    if not parts:
        return ""

    return "Metric used: " + "; ".join(parts) + "."


def build_markdown_report(rows_by_model: Dict[str, List[Dict[str, Any]]]) -> str:
    out = []

    out.append("# LM Eval Results")
    out.append("")
    out.append(
        "Hierarchical tables are grouped by compression ratio. "
        "Accuracy-style scores are percentages. `acc` is used when available; otherwise `acc_norm` is used. "
        "Average accuracy excludes generation benchmarks. "
        "`wikitext` and `c4` are token perplexity, where lower is better.",
    )
    out.append("")

    for model_name in sorted(rows_by_model):
        model_rows = rows_by_model[model_name]

        note = make_metric_note(model_rows)

        out.append(make_markdown_table_for_model(model_name, model_rows))
        out.append("")

        if note:
            out.append(note)
            out.append("")

    return "\n".join(out).rstrip() + "\n"


def build_latex_report(rows_by_model: Dict[str, List[Dict[str, Any]]], table_width: float = 1.6) -> str:
    out = []

    for model_name in sorted(rows_by_model):
        out.append(make_latex_table_for_model(model_name, rows_by_model[model_name], table_width = table_width))
        out.append("")

    return "\n".join(out).rstrip() + "\n"


def collect_rows(
    input_dir: Path,
    pattern: str,
    prefer_lm_eval_model_name: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    # Sidecars share the .json extension but describe a result, they are not one
    paths = sorted(p for p in input_dir.glob(pattern) if not is_run_config(p))

    if not paths:
        raise FileNotFoundError(f"No files matched {input_dir / pattern}")

    rows_by_model = defaultdict(list)

    for path in paths:
        row = load_result(path, prefer_lm_eval_model_name=prefer_lm_eval_model_name)
        rows_by_model[row["model"]].append(row)

    return rows_by_model


# ---------------------------------------------------------------------------
# Stage gates
#
# EXPERIMENTS.md builds the grid as one swept axis per stage around an otherwise
# fixed configuration, each stage ending in a gate whose answer fills the
# placeholders of the next stage file. Everything below turns collected runs into
# exactly those answers: one table per gate, and one summary holding the values
# to paste into the next stage
# ---------------------------------------------------------------------------

COMPOSITE_PREFIX = "composite|"

# The policy and grouping spellings the gates filter on, as `src.utils` spells
# them. This script stays import-free, so they are repeated rather than imported
DEFAULT_INNER_ALLOCATION = "waterfill"
DEFAULT_OUTER_ALLOCATION = "param_share"
BLOCK_GROUPINGS = ( "decoder", "hierarchical" )

# Block Influence is constant inside a decoder block, so a fused score cannot
# rank matrices under `decoder` or `hierarchical` and stage 6 needs a flat one
FLAT_GROUPINGS = ( "type", "global" )

# Stage 2 is the nine cells that decide the grouping and the two scores every
# later stage is held at. The families stages 2b and 2c add are deliberately kept
# out of it: they are promotion tests against this gate's own winner, and letting
# them into the table that elects it would make the promotion circular
STAGE2_SCORES = ( "truncation", "entropy", "eff_rank" )

# Stage 2b compares the squared scores against the amplitude ones they derive
# from, because its gate is a promotion test rather than a ranking of its own
SQUARED_SCORE_FAMILY = (
    "truncation",
    "truncation_sq",
    "entropy",
    "entropy_sq",
    "eff_rank",
    "eff_rank_sq",
)

# A realized ratio further than this from its target makes a run incomparable to
# one that hit its budget, which a gate has to say out loud rather than average in
RATIO_DRIFT_TOLERANCE = 0.005

# Every placeholder a stage file can carry, and the gate that answers it
PLACEHOLDER_SOURCES: Dict[str, str] = {
    "__CKPT_HOM_0.2__": "stage 1",
    "__CKPT_HOM_0.5__": "stage 1",
    "__BEST_GROUPING__": "stage 2",
    "__BEST_FLAT_GROUPING__": "stage 2",
    "__TOP1_SCORE__": "stage 2, promotable by 2b or 2c",
    "__TOP2_SCORE__": "stage 2",
    "__CKPT_BEST_SCORE_0.2__": "stage 2",
    "__CKPT_BEST_SCORE_0.5__": "stage 2",
    "__BEST_INNER__": "stage 3",
    "__CKPT_BEST_POLICY_0.2__": "stage 3",
    "__CKPT_BEST_POLICY_0.5__": "stage 3",
    "--max_ratio": "stage 4, into args/base_args.json",
    "__CKPT_BEST_BYPASS_0.2__": "stage 5",
    "__CKPT_BEST_COMPOSITE_0.2__": "stage 6",
    "__CKPT_HET_0.2__": "stages 2 to 6",
    "__CKPT_HET_0.5__": "stages 2 to 6",
}

PLACEHOLDER_SOURCES.update({
    f"__FINALIST{index}_{role}__": "stages 2 to 6"
    for index in ( 1, 2, 3 )
    for role in ( "GROUPING", "SCORE", "INNER" )
})

# Figures `allocation_report.py` writes that a gate reads back
OFFLINE_FIGURES = (
    "dispersion",
    "cap_binding",
    "influence_vs_effrank_rho",
    "ratio_by_type",
)


class Cell(NamedTuple):
    """One table cell. Builders emit plain text and leave markup to the renderer"""
    text: str
    bold: bool = False
    faded: bool = False


class Table(NamedTuple):
    """A table before it commits to a format, so one builder serves markdown and LaTeX"""
    title: str
    purpose: str
    headers: List[str]
    rows: List[List[Cell]]
    notes: List[str]


class OfflineStage(NamedTuple):
    """One `allocation_report.py --out_dir` directory, keyed by the stage it previews"""
    stage: str
    path: Path
    summary: List[Dict[str, str]]
    figures: Dict[str, List[Dict[str, str]]]


class GateContext(NamedTuple):
    """
    Everything a gate reads.

    `resolved` accumulates as the gates run in stage order, because most stages
    are defined relative to an earlier winner rather than in absolute terms.
    """
    rows: List[Dict[str, Any]]
    offline: Dict[str, OfflineStage]
    resolved: Dict[str, str]
    metric: str


class GateResult(NamedTuple):
    tables: List[Table]
    resolved: Dict[str, str]


class PivotRow(NamedTuple):
    """
    One comparison row of a gate: the same configuration at every ratio.

    Ranking happens per ratio and is averaged afterwards, which is what
    EXPERIMENTS.md asks for and what stops a ratio where half the arms are
    missing from deciding the gate on its own.
    """
    key: Tuple[str, ...]
    runs: Dict[float, Dict[str, Any]]
    values: Dict[float, Optional[float]]
    ranks: Dict[float, Optional[int]]
    mean_rank: Optional[float]


class GainRow(NamedTuple):
    """One configuration's heterogeneous and homogeneous arms, and the gap between them"""
    key: Tuple[str, ...]
    het: Optional[PivotRow]
    hom: Optional[PivotRow]
    gains: Dict[float, Optional[float]]
    mean_gain: Optional[float]


def as_float(value: Any) -> Optional[float]:
    """`safe_float` rounds to two decimals, which is too coarse to see a budget drift"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return None if math.isnan(number) else number


def mean_of(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def axis_text(value: Any) -> str:
    """
    One dimension value, as a table cell and as a join key.

    The offline CSVs hold every dimension as text while the sidecar holds native
    types, so both sides go through this before they are ever compared.
    """
    if value is None or value == "":
        return "--"

    if isinstance(value, bool):
        return str(value).lower()

    number = as_float(value)

    return f"{number:g}" if number is not None else str(value)


def dimension_text(name: str, value: Any) -> str:
    """`axis_text`, with the dimensions whose off state has two spellings folded together"""
    if name not in ( "bypass_early_layers", "bypass_late_layers" ):
        return axis_text(value)

    number = as_float(value)

    return str(max(0, int(number))) if number is not None else axis_text(value)


def fmt_realized(value: Any) -> str:
    number = as_float(value)
    return f"{number:.4f}" if number is not None else "--"


def gain_text(baseline: Optional[float], value: Optional[float]) -> str:
    """Perplexity a configuration saves over its homogeneous arm, positive being better"""
    if baseline is None or value is None:
        return "--"

    return f"{baseline - value:+.2f}"


def gate_metric_value(row: Dict[str, Any], metric: str) -> Optional[float]:
    """The number a gate ranks by: one perplexity column, or the mean of both"""
    if metric != "mean":
        return as_float(row.get(metric))

    values = [as_float(row.get(name)) for name in PERPLEXITY_BENCHMARK_ORDER]
    available = [value for value in values if value is not None]

    return mean_of(available)


def competition_ranks(values: Dict[Any, Optional[float]]) -> Dict[Any, Optional[int]]:
    """Ranks over one ratio, 1 being the lowest perplexity, ties sharing a rank"""
    ordered = sorted(value for value in values.values() if value is not None)

    return {
        key: ordered.index(value) + 1 if value is not None else None
        for key, value in values.items()
    }


def is_compression_run(row: Dict[str, Any]) -> bool:
    """A run that allocated something itself, as opposed to a dense or updated one"""
    return (
        not row.get("is_original")
        and not row.get("update_method")
        and row.get("scheme") in ( "het", "hom" )
    )


def bypasses_nothing(row: Dict[str, Any]) -> bool:
    return int(as_float(row.get("bypassed_layers")) or 0) == 0


def is_baseline_heterogeneous(row: Dict[str, Any]) -> bool:
    """A heterogeneous run at the stage 2 defaults, which most later gates sit on top of"""
    return (
        is_compression_run(row)
        and row.get("scheme") == "het"
        and bypasses_nothing(row)
        and not str(row.get("score_metric") or "").startswith(COMPOSITE_PREFIX)
    )


def gate_rows(
        rows: List[Dict[str, Any]],
        axes: Tuple[str, ...],
        select: Callable[[Dict[str, Any]], bool]
) -> Tuple[List[Dict[str, Any]], int]:
    """The runs a gate compares, and how many it had to drop for not recording an axis"""
    matched = [row for row in rows if select(row)]
    complete = [row for row in matched if all(axis in row for axis in axes)]

    return complete, len(matched) - len(complete)


def build_pivot(rows: List[Dict[str, Any]], axes: Tuple[str, ...], metric: str) -> List[PivotRow]:
    """
    Collapse runs into one row per configuration, ranked within each ratio.

    A configuration evaluated twice keeps whichever run produced a number, so a
    crashed or half-written result cannot displace a complete one. Two complete
    runs sharing every axis differ in something the gate is not looking at, and
    the first by run name wins so the table is at least reproducible;
    `confound_notes` is what makes that difference visible.
    """
    grouped: Dict[Tuple[str, ...], Dict[float, Dict[str, Any]]] = defaultdict(dict)

    for row in sorted(rows, key=lambda item: str(item.get("run_name") or item.get("file") or "")):
        ratio = as_float(row.get("compression_ratio"))

        if ratio is None:
            continue

        key = tuple(dimension_text(axis, row.get(axis)) for axis in axes)
        kept = grouped[key].get(ratio)

        replaces_kept = (
            kept is None
            or (gate_metric_value(kept, metric) is None and gate_metric_value(row, metric) is not None)
        )

        if replaces_kept:
            grouped[key][ratio] = row

    ratios = sorted({ratio for runs in grouped.values() for ratio in runs})

    values = {
        key: {
            ratio: gate_metric_value(runs[ratio], metric) if ratio in runs else None
            for ratio in ratios
        }
        for key, runs in grouped.items()
    }

    ranks_by_ratio = {
        ratio: competition_ranks({key: values[key][ratio] for key in grouped})
        for ratio in ratios
    }

    pivot: List[PivotRow] = []

    for key, runs in grouped.items():
        ranks = {ratio: ranks_by_ratio[ratio][key] for ratio in ratios}
        placed = [float(rank) for rank in ranks.values() if rank is not None]

        pivot.append(PivotRow(
            key=key,
            runs=runs,
            values=values[key],
            ranks=ranks,
            mean_rank=mean_of(placed),
        ))

    return sorted(pivot, key=lambda item: item.mean_rank if item.mean_rank is not None else math.inf)


def build_gain_rows(rows: List[Dict[str, Any]], axes: Tuple[str, ...], metric: str) -> List[GainRow]:
    """
    Pair each configuration's heterogeneous arm with its homogeneous one.

    The homogeneous arm is not padding: the gain over it at the same setting is
    the only quantity that separates a stage's mechanism from the budget itself.
    """
    het = {row.key: row for row in build_pivot([row for row in rows if row.get("scheme") == "het"], axes, metric)}
    hom = {row.key: row for row in build_pivot([row for row in rows if row.get("scheme") == "hom"], axes, metric)}

    ratios = sorted({
        ratio
        for pivot in ( het, hom )
        for row in pivot.values()
        for ratio in row.values
    })

    gain_rows: List[GainRow] = []

    for key in { **het, **hom }:
        het_row = het.get(key)
        hom_row = hom.get(key)
        gains: Dict[float, Optional[float]] = {}

        for ratio in ratios:
            het_value = het_row.values.get(ratio) if het_row is not None else None
            hom_value = hom_row.values.get(ratio) if hom_row is not None else None
            gains[ratio] = hom_value - het_value if het_value is not None and hom_value is not None else None

        realized = [gain for gain in gains.values() if gain is not None]

        gain_rows.append(GainRow(
            key=key,
            het=het_row,
            hom=hom_row,
            gains=gains,
            mean_gain=mean_of(realized),
        ))

    return sorted(gain_rows, key=lambda item: -item.mean_gain if item.mean_gain is not None else math.inf)


def homogeneous_baselines(rows: List[Dict[str, Any]], metric: str, bypassed: int = 0) -> Dict[float, Optional[float]]:
    """The homogeneous anchor at each ratio, the floor every heterogeneous stage is read against"""
    baselines: Dict[float, Optional[float]] = {}

    for row in rows:
        is_anchor = (
            is_compression_run(row)
            and row.get("scheme") == "hom"
            and int(as_float(row.get("bypassed_layers")) or 0) == bypassed
        )

        if not is_anchor:
            continue

        ratio = as_float(row.get("compression_ratio"))
        value = gate_metric_value(row, metric)

        if ratio is not None and value is not None:
            baselines[ratio] = value

    return baselines


def best_by(pivot: List[PivotRow], index: int) -> List[Tuple[str, float]]:
    """
    Mean rank per value of one axis, best first.

    A grouping holding several scores is judged by how those scores did on
    average, which is what "the grouping with the best mean rank" resolves to.
    """
    grouped: Dict[str, List[float]] = defaultdict(list)

    for row in pivot:
        if row.mean_rank is not None:
            grouped[row.key[index]].append(row.mean_rank)

    ranked = [( value, mean_of(ranks) ) for value, ranks in grouped.items()]

    return sorted(
        (( value, rank ) for value, rank in ranked if rank is not None),
        key=lambda item: item[1],
    )


def best_checkpoints(pivot: List[PivotRow], prefix: str) -> Dict[str, str]:
    """The rank-1 checkpoint at each ratio, which is what a `__CKPT_*__` role names"""
    resolved: Dict[str, str] = {}
    ratios = sorted({ratio for row in pivot for ratio in row.runs})

    for ratio in ratios:
        for row in pivot:
            if row.ranks.get(ratio) != 1:
                continue

            path = str(row.runs[ratio].get("checkpoint_path") or "")

            if path:
                resolved[f"{prefix}_{axis_text(ratio)}__"] = path

            break

    return resolved


def dominant_value(rows: List[Dict[str, Any]], dimension: str) -> Optional[str]:
    """Whichever value of a dimension the most runs share, ties broken by name for reproducibility"""
    counted = Counter(dimension_text(dimension, row[dimension]) for row in rows if dimension in row)

    if not counted:
        return None

    return min(counted.items(), key=lambda item: ( -item[1], item[0] ))[0]


def hold_at(
        rows: List[Dict[str, Any]],
        dimension: str,
        value: Optional[str]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Keep only the runs sitting at one value of a dimension the gate is not sweeping.

    A stage is run at one setting of everything it does not sweep, so another
    value inside a gate's selection belongs to a different stage that happens to
    share its axes: stage 4's cap sweep, for one, matches stage 3's table exactly
    apart from the cap. Left in, it decides the gate through a dimension the
    stage was never comparing.

    A run that does not record the dimension at all is kept, which is what lets a
    homogeneous arm survive a hold on an allocation dimension it never had.
    """
    if value is None:
        return rows, []

    kept = [
        row for row in rows
        if dimension not in row or dimension_text(dimension, row[dimension]) == value
    ]

    if len(kept) == len(rows):
        return rows, []

    dropped = sorted({dimension_text(dimension, row[dimension]) for row in rows if dimension in row} - { value })

    return kept, [
        f"held at `--{dimension} {value}`. {len(rows) - len(kept)} run(s) at "
        f"{', '.join(dropped)} are left to the stage that swept it",
    ]


def hold_dominant(rows: List[Dict[str, Any]], dimension: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """`hold_at` the value the selection itself is mostly made of"""
    return hold_at(rows, dimension, dominant_value(rows, dimension))


def confound_notes(rows: List[Dict[str, Any]], axes: Tuple[str, ...]) -> List[str]:
    """
    Dimensions that move inside a gate's selection without being one of its axes.

    Every stage sweeps one axis around a fixed configuration, so anything else
    moving means the table is pricing two changes at once.
    """
    notes: List[str] = []

    for dimension in GATE_DIMENSIONS:
        if dimension in axes:
            continue

        seen = {dimension_text(dimension, row[dimension]) for row in rows if dimension in row}

        if len(seen) > 1:
            notes.append(
                f"`{dimension}` varies inside this table ({', '.join(sorted(seen))}), "
                "so the comparison is confounded",
            )

    return notes


def drift_notes(pivot: List[PivotRow]) -> List[str]:
    """Runs whose realized removal missed the budget they are being compared at"""
    drifted: List[str] = []

    for row in pivot:
        for ratio, run in sorted(row.runs.items()):
            realized = as_float(run.get("realized_ratio"))

            if realized is not None and abs(realized - ratio) > RATIO_DRIFT_TOLERANCE:
                drifted.append(f"{' / '.join(row.key)} at {axis_text(ratio)} realized {realized:.4f}")

    if not drifted:
        return []

    return [f"realized removal drifted off target: {'; '.join(drifted)}"]


def skipped_note(skipped: int) -> List[str]:
    if not skipped:
        return []

    return [
        f"{skipped} matching run(s) left out for not recording an axis in their sidecar, "
        "which the filename cannot carry either",
    ]


def pivot_table(
        title: str,
        purpose: str,
        pivot: List[PivotRow],
        axis_headers: List[str],
        metric: str,
        baselines: Optional[Dict[float, Optional[float]]] = None,
        notes: Optional[List[str]] = None
) -> Table:
    """One gate table: the axes, the metric and its rank at every ratio, mean rank"""
    ratios = sorted({ratio for row in pivot for ratio in row.values})
    headers = list(axis_headers)

    for ratio in ratios:
        headers += [f"{metric}@{axis_text(ratio)}", f"rank@{axis_text(ratio)}"]

        if baselines:
            headers.append(f"gain@{axis_text(ratio)}")

    headers += [ "mean rank", "priced at" ]

    best_mean = min((row.mean_rank for row in pivot if row.mean_rank is not None), default=None)
    body: List[List[Cell]] = []
    coverage: Dict[str, int] = {}

    for row in pivot:
        is_winner = row.mean_rank is not None and row.mean_rank == best_mean
        cells = [Cell(text=value, bold=is_winner) for value in row.key]

        for ratio in ratios:
            value = row.values.get(ratio)
            rank = row.ranks.get(ratio)

            cells.append(Cell(text=fmt_ppl(value), bold=rank == 1))
            cells.append(Cell(text=str(rank) if rank is not None else "--"))

            if baselines:
                cells.append(Cell(text=gain_text(baselines.get(ratio), value)))

        priced = sum(1 for ratio in ratios if row.values.get(ratio) is not None)
        coverage[" / ".join(row.key)] = priced

        mean_rank = f"{row.mean_rank:.2f}" if row.mean_rank is not None else "--"
        cells.append(Cell(text=mean_rank, bold=is_winner))
        cells.append(Cell(text=f"{priced}/{len(ratios)}"))
        body.append(cells)

    extra_notes: List[str] = []

    if len(set(coverage.values())) > 1:
        extra_notes.append(
            "rows were not all priced at the same ratios, and a mean rank averaged over fewer of them is "
            "not comparable to one averaged over all: read `priced at` before the ordering",
        )

    if baselines:
        extra_notes.append("gain is homogeneous minus this row, so a positive number means heterogeneous won")

    return Table(
        title=title,
        purpose=purpose,
        headers=headers,
        rows=body,
        notes=[ *(notes or []), *extra_notes, *drift_notes(pivot) ],
    )


def gain_table(
        title: str,
        purpose: str,
        gain_rows: List[GainRow],
        axis_headers: List[str],
        metric: str,
        notes: Optional[List[str]] = None
) -> Table:
    """A paired table: both arms at every ratio, and the gap the stage is about"""
    ratios = sorted({
        ratio
        for row in gain_rows
        for pivot in ( row.het, row.hom )
        if pivot is not None
        for ratio in pivot.values
    })

    headers = list(axis_headers)

    for ratio in ratios:
        headers += [f"het@{axis_text(ratio)}", f"hom@{axis_text(ratio)}", f"gain@{axis_text(ratio)}"]

    headers.append("mean gain")

    best_gain = max((row.mean_gain for row in gain_rows if row.mean_gain is not None), default=None)
    body: List[List[Cell]] = []

    for row in gain_rows:
        is_winner = row.mean_gain is not None and row.mean_gain == best_gain
        cells = [Cell(text=value, bold=is_winner) for value in row.key]

        for ratio in ratios:
            het_value = row.het.values.get(ratio) if row.het is not None else None
            hom_value = row.hom.values.get(ratio) if row.hom is not None else None

            cells.append(Cell(text=fmt_ppl(het_value)))
            cells.append(Cell(text=fmt_ppl(hom_value)))
            cells.append(Cell(text=gain_text(hom_value, het_value)))

        mean_gain = f"{row.mean_gain:+.2f}" if row.mean_gain is not None else "--"
        cells.append(Cell(text=mean_gain, bold=is_winner))
        body.append(cells)

    return Table(
        title=title,
        purpose=purpose,
        headers=headers,
        rows=body,
        notes=[
            *(notes or []),
            "gain is hom minus het at the same setting, so a positive number means heterogeneous won",
        ],
    )


def aggregate_table(
        title: str,
        purpose: str,
        header: str,
        ranked: List[Tuple[str, float]],
        notes: Optional[List[str]] = None
) -> Table:
    """The per-axis aggregate a placeholder is read off, so the choice is auditable"""
    return Table(
        title=title,
        purpose=purpose,
        headers=[ header, "mean rank", "verdict" ],
        rows=[
            [
                Cell(text=value, bold=index == 0),
                Cell(text=f"{rank:.2f}", bold=index == 0),
                Cell(text="best" if index == 0 else ""),
            ]
            for index, ( value, rank ) in enumerate(ranked)
        ],
        notes=notes or [],
    )


# ---------------------------------------------------------------------------
# Offline previews, from allocation_report.py
# ---------------------------------------------------------------------------

def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(record) for record in csv.DictReader(handle)]
    except OSError as error:
        print(f"[WARNING] Ignoring unreadable {path}: {error}")
        return []


def load_offline_reports(allocation_dir: Path) -> Dict[str, OfflineStage]:
    """
    Read every `stage<N>/` directory `allocation_report.py --out_dir` wrote.

    The offline half of every gate is free, so a stage previewed but not yet run
    still reports what its allocation will look like.
    """
    if not allocation_dir.is_dir():
        print(f"[WARNING] No allocation reports under {allocation_dir}")
        return {}

    stages: Dict[str, OfflineStage] = {}

    for path in sorted(allocation_dir.iterdir()):
        match = re.fullmatch(r"stage(\d+[a-z]?)", path.name)

        if not path.is_dir() or match is None:
            continue

        stages[match.group(1)] = OfflineStage(
            stage=match.group(1),
            path=path,
            summary=read_csv_rows(path / "summary.csv"),
            figures={name: read_csv_rows(path / "figures" / f"{name}.csv") for name in OFFLINE_FIGURES},
        )

    return stages


def offline_summary_table(stage: OfflineStage) -> Optional[Table]:
    """What the offline preview said, before any GPU time was spent on the stage"""
    if not stage.summary:
        return None

    ranked = sorted(stage.summary, key=lambda record: as_float(record.get("mean_rank")) or math.inf)

    # A sweep over the budget names its variants identically at every ratio, so
    # the column has to be shown or the rows read as duplicates
    columns = ( "compression_ratio", "realized_ratio", "mean_rank", "ratio_std", "score_ratio_rho", "checks" )
    columns = tuple(column for column in columns if any(column in record for record in stage.summary))

    return Table(
        title=f"Offline preview of stage {stage.stage}",
        purpose=f"`allocation_report.py` summary from `{stage.path}`, ordered by offline mean rank",
        headers=[ "variant", *columns ],
        rows=[
            [
                Cell(text=record.get("variant", "--")),
                *[Cell(text=record.get(column) or "--") for column in columns],
            ]
            for record in ranked
        ],
        notes=[
            "these ranks price the allocation, not the model: a variant winning here and losing "
            "on perplexity bounds how far an offline proxy can substitute for evaluation",
        ],
    )


def offline_figure_table(
        stage: OfflineStage,
        figure: str,
        title: str,
        purpose: str,
        notes: Optional[List[str]] = None
) -> Optional[Table]:
    """One figure CSV, passed through as the table the gate needs to read it as"""
    records = stage.figures.get(figure) or []

    if not records:
        return None

    headers = list(records[0].keys())

    return Table(
        title=title,
        purpose=purpose,
        headers=headers,
        rows=[[Cell(text=record.get(header) or "--") for header in headers] for record in records],
        notes=notes or [],
    )


def offline_agreement_table(
        title: str,
        pivot: List[PivotRow],
        axes: Tuple[str, ...],
        stage: Optional[OfflineStage]
) -> Optional[Table]:
    """
    The offline ordering against the measured one, over the configurations both hold.

    EXPERIMENTS.md asks for this disagreement explicitly: it is what bounds how
    far the free offline preview can stand in for an hour of GPU per cell.
    """
    if stage is None or not stage.summary:
        return None

    offline_ranks: Dict[Tuple[str, ...], List[float]] = defaultdict(list)

    for record in stage.summary:
        if not all(axis in record for axis in axes):
            continue

        value = as_float(record.get("mean_rank"))

        if value is not None:
            offline_ranks[tuple(dimension_text(axis, record.get(axis)) for axis in axes)].append(value)

    shared = [row for row in pivot if row.key in offline_ranks and row.mean_rank is not None]

    if not shared:
        return None

    offline_means = {row.key: mean_of(offline_ranks[row.key]) for row in shared}
    offline_places = competition_ranks(offline_means)
    measured_places = competition_ranks({row.key: row.mean_rank for row in shared})

    rows: List[List[Cell]] = []

    for row in sorted(shared, key=lambda item: measured_places[item.key] or 0):
        offline_place = offline_places[row.key]
        measured_place = measured_places[row.key]
        agrees = offline_place == measured_place
        offline_mean = offline_means[row.key]

        rows.append([
            Cell(text=" / ".join(row.key)),
            Cell(text=f"{offline_mean:.2f}" if offline_mean is not None else "--"),
            Cell(text=str(offline_place) if offline_place is not None else "--"),
            Cell(text=f"{row.mean_rank:.2f}" if row.mean_rank is not None else "--"),
            Cell(text=str(measured_place) if measured_place is not None else "--"),
            Cell(text="same" if agrees else "differs", bold=not agrees),
        ])

    return Table(
        title=title,
        purpose=f"Offline mean rank from `{stage.path}` against the measured perplexity ordering",
        headers=[ "configuration", "offline mean rank", "offline place", "measured mean rank", "measured place", "order" ],
        rows=rows,
        notes=[
            "a row marked `differs` is a place where the offline objectives and the model disagree, "
            "which is a result to report rather than an error to fix",
        ],
    )


def offline_tables(context: GateContext, stage: str, figures: Tuple[Tuple[str, str, str], ...] = ()) -> List[Table]:
    """The offline preview of one stage, plus whichever of its figures the gate reads"""
    preview = context.offline.get(stage)

    if preview is None:
        return []

    tables = [table for table in [offline_summary_table(preview)] if table is not None]

    for figure, title, purpose in figures:
        table = offline_figure_table(preview, figure, title, purpose)

        if table is not None:
            tables.append(table)

    return tables


# ---------------------------------------------------------------------------
# The gates themselves, in the order EXPERIMENTS.md runs them
# ---------------------------------------------------------------------------

def dense_perplexity_notes(row: Dict[str, Any]) -> List[str]:
    """
    Stage 1's own check: wikitext and c4 must differ on the dense baseline.

    Identical values are the signature of the c4 task re-evaluating wikitext, in
    which case nothing downstream measures what it claims to.
    """
    wikitext = as_float(row.get("wikitext"))
    c4 = as_float(row.get("c4"))

    if wikitext is None or c4 is None:
        return ["the dense baseline is missing one of the two perplexities, so the c4 check cannot run"]

    if abs(wikitext - c4) < 1e-6:
        return [
            f"dense wikitext and c4 perplexity are identical ({wikitext:.4f}), the signature of the "
            "c4 task re-evaluating wikitext",
        ]

    return [f"dense wikitext {wikitext:.4f} differs from c4 {c4:.4f}, so the two tasks are distinct"]


def gate_stage1_anchors(context: GateContext) -> GateResult:
    """Stage 1: the dense floor and the homogeneous anchors every later table is read against"""
    def select(row: Dict[str, Any]) -> bool:
        if row.get("is_original"):
            return True

        return is_compression_run(row) and row.get("scheme") == "hom" and bypasses_nothing(row)

    rows = sorted((row for row in context.rows if select(row)), key=sort_rows_hierarchical)
    resolved: Dict[str, str] = {}
    notes: List[str] = []
    body: List[List[Cell]] = []

    for row in rows:
        dense = bool(row.get("is_original"))
        target = as_float(row.get("compression_ratio"))

        body.append([
            Cell(text="dense" if dense else "hom", faded=dense),
            Cell(text="--" if dense else axis_text(target), faded=dense),
            Cell(text=fmt_realized(row.get("realized_ratio")), faded=dense),
            *[Cell(text=fmt_ppl(row.get(name)), faded=dense) for name in PERPLEXITY_BENCHMARK_ORDER],
            Cell(text=str(row.get("checkpoint_path") or "--"), faded=dense),
        ])

        if dense:
            notes += dense_perplexity_notes(row)
            continue

        path = str(row.get("checkpoint_path") or "")

        if path and target is not None:
            resolved[f"__CKPT_HOM_{axis_text(target)}__"] = path

    table = Table(
        title="Stage 1 gate: anchors",
        purpose=(
            "The dense reference and the homogeneous runs. These two perplexities are the `hom` row "
            "of every table in chapter 4, and the floor the heterogeneous gain is measured from"
        ),
        headers=[ "run", "target", "realized", *PERPLEXITY_BENCHMARK_ORDER, "checkpoint" ],
        rows=body,
        notes=notes,
    )

    return GateResult(tables=[table], resolved=resolved)


def gate_stage2_score_grouping(context: GateContext) -> GateResult:
    """Stage 2 (RQ2): which grouping criterion and which spectral score make heterogeneity work"""
    axes = ( "group_criterion", "score_metric" )

    def select(row: Dict[str, Any]) -> bool:
        return (
            is_baseline_heterogeneous(row)
            and row.get("inner_allocation") == DEFAULT_INNER_ALLOCATION
            and row.get("outer_allocation") == DEFAULT_OUTER_ALLOCATION
            and row.get("score_metric") in STAGE2_SCORES
        )

    rows, skipped = gate_rows(context.rows, axes, select)
    rows, held = hold_dominant(rows, "max_ratio")
    pivot = build_pivot(rows, axes, context.metric)

    tables = [pivot_table(
        title="Stage 2 gate: score x grouping",
        purpose=(
            "`inner_allocation` fixed at `waterfill`, so grouping and score are the only axes. "
            "Ranked within each ratio, then averaged, never on a single objective"
        ),
        pivot=pivot,
        axis_headers=[ "group_criterion", "score_metric" ],
        metric=context.metric,
        baselines=homogeneous_baselines(context.rows, context.metric),
        notes=[ *skipped_note(skipped), *held, *confound_notes(rows, axes) ],
    )]

    resolved: Dict[str, str] = {}
    groupings = best_by(pivot, 0)

    if not groupings:
        return GateResult(tables=tables, resolved=resolved)

    best_grouping = groupings[0][0]
    resolved["__BEST_GROUPING__"] = best_grouping
    flat = [value for value, _ in groupings if value in FLAT_GROUPINGS]

    if flat:
        resolved["__BEST_FLAT_GROUPING__"] = flat[0]

    tables.append(aggregate_table(
        title="Stage 2 gate: grouping aggregate",
        purpose="Mean of the row mean ranks inside each grouping, which is what decides `__BEST_GROUPING__`",
        header="group_criterion",
        ranked=groupings,
        notes=[
            "`__BEST_FLAT_GROUPING__` is the better of `type` and `global`, never `decoder`: "
            "Block Influence is constant inside a decoder block, so stage 6 would collapse to homogeneous",
        ],
    ))

    within = [row for row in pivot if row.key[0] == best_grouping]
    scores = best_by(within, 1)

    for placeholder, index in ( ( "__TOP1_SCORE__", 0 ), ( "__TOP2_SCORE__", 1 ) ):
        if len(scores) > index:
            resolved[placeholder] = scores[index][0]

    if scores:
        tables.append(aggregate_table(
            title=f"Stage 2 gate: scores within {best_grouping}",
            purpose="Averaged over ratios, which is what decides `__TOP1_SCORE__` and `__TOP2_SCORE__`",
            header="score_metric",
            ranked=scores,
        ))

    resolved.update(best_checkpoints(pivot, "__CKPT_BEST_SCORE"))

    agreement = offline_agreement_table("Stage 2 gate: offline against measured", pivot, axes, context.offline.get("2"))

    if agreement is not None:
        tables.append(agreement)

    tables += offline_tables(context, "2")

    return GateResult(tables=tables, resolved=resolved)


def score_family_gate(
        context: GateContext,
        title: str,
        purpose: str,
        belongs: Callable[[str], bool],
        offline_stage: str
) -> GateResult:
    """
    Stages 2b and 2c: a score family measured against the incumbent `__TOP1_SCORE__`.

    The incumbent is pulled into the same table on purpose, since both gates are
    promotion tests rather than rankings of a family on its own.
    """
    axes = ( "score_metric", )
    grouping = context.resolved.get("__BEST_GROUPING__")
    incumbent = context.resolved.get("__TOP1_SCORE__", "")

    def select(row: Dict[str, Any]) -> bool:
        score = str(row.get("score_metric") or "")

        return (
            is_baseline_heterogeneous(row)
            and row.get("inner_allocation") == DEFAULT_INNER_ALLOCATION
            and ( grouping is None or row.get("group_criterion") == grouping )
            and ( belongs(score) or score == incumbent )
        )

    rows, skipped = gate_rows(context.rows, axes, select)
    rows, held = hold_dominant(rows, "max_ratio")
    pivot = build_pivot(rows, axes, context.metric)
    resolved: Dict[str, str] = {}
    notes = [ *skipped_note(skipped), *held, *confound_notes(rows, axes) ]

    if grouping is not None:
        notes.append(f"held at `--group_criterion {grouping}`, as stage 2 resolved it")

    winner = pivot[0].key[0] if pivot else ""

    if winner and not incumbent:
        notes.append("stage 2 has not resolved `__TOP1_SCORE__` yet, so nothing is promoted from here")
    elif winner and winner != incumbent:
        resolved["__TOP1_SCORE__"] = winner
        resolved.update(best_checkpoints(pivot, "__CKPT_BEST_SCORE"))
        notes.append(f"`{winner}` beats the incumbent `{incumbent}` on mean rank and is promoted to `__TOP1_SCORE__`")
    elif winner:
        notes.append(f"the incumbent `{incumbent}` holds, so `__TOP1_SCORE__` is unchanged")

    tables = [pivot_table(
        title=title,
        purpose=purpose,
        pivot=pivot,
        axis_headers=[ "score_metric" ],
        metric=context.metric,
        baselines=homogeneous_baselines(context.rows, context.metric),
        notes=notes,
    )]

    tables += offline_tables(context, offline_stage)

    return GateResult(tables=tables, resolved=resolved)


def gate_stage2b_squared(context: GateContext) -> GateResult:
    """Stage 2b: whether weighting by energy rather than amplitude changes the ranking"""
    return score_family_gate(
        context,
        title="Stage 2b gate: squared spectra",
        purpose=(
            "Each `_sq` score against the amplitude score it derives from. Compare the ratio maps in "
            "the offline preview too: a pair agreeing to three decimals is one experiment, not two"
        ),
        belongs=lambda score: score in SQUARED_SCORE_FAMILY,
        offline_stage="2b",
    )


def gate_stage2c_schatten(context: GateContext) -> GateResult:
    """Stage 2c: the Schatten p-norm family, which fills thesis 3.1.1.2"""
    return score_family_gate(
        context,
        title="Stage 2c gate: Schatten p-norms",
        purpose=(
            "`norm|p` spans genuinely different signals: the nuclear norm of the truncated tail, its "
            "largest singular value, its smallest. `norm|-inf` is the most fragile quantity of the family"
        ),
        belongs=lambda score: score.startswith("norm|"),
        offline_stage="2c",
    )


def gate_stage3_policies(context: GateContext) -> GateResult:
    """Stage 3 (RQ3): whether the policy spending a group budget matters apart from the score"""
    grouping = context.resolved.get("__BEST_GROUPING__")
    top_scores = [context.resolved.get(name) for name in ( "__TOP1_SCORE__", "__TOP2_SCORE__" )]
    top_scores = [score for score in top_scores if score]

    inner_axes = ( "inner_allocation", "score_metric" )

    def select_inner(row: Dict[str, Any]) -> bool:
        return (
            is_baseline_heterogeneous(row)
            and row.get("outer_allocation") == DEFAULT_OUTER_ALLOCATION
            and ( grouping is None or row.get("group_criterion") == grouping )
            and ( not top_scores or row.get("score_metric") in top_scores )
        )

    inner_rows, inner_skipped = gate_rows(context.rows, inner_axes, select_inner)
    inner_rows, inner_held = hold_dominant(inner_rows, "max_ratio")
    inner_pivot = build_pivot(inner_rows, inner_axes, context.metric)

    tables = [pivot_table(
        title="Stage 3 gate: inner allocation policies",
        purpose=(
            "The four inner policies at the grouping and scores stage 2 chose. Their knobs must first "
            "be matched on ratio dispersion, or this table prices shape and aggressiveness at once"
        ),
        pivot=inner_pivot,
        axis_headers=[ "inner_allocation", "score_metric" ],
        metric=context.metric,
        baselines=homogeneous_baselines(context.rows, context.metric),
        notes=[ *skipped_note(inner_skipped), *inner_held, *confound_notes(inner_rows, inner_axes) ],
    )]

    # The score stays an axis rather than being fixed: the ablation is only
    # controlled when the two groupings are compared at the same score
    outer_axes = ( "group_criterion", "outer_allocation", "inner_allocation", "score_metric" )

    def select_outer(row: Dict[str, Any]) -> bool:
        return is_baseline_heterogeneous(row) and row.get("group_criterion") in BLOCK_GROUPINGS

    outer_rows, outer_skipped = gate_rows(context.rows, outer_axes, select_outer)
    outer_rows, outer_held = hold_dominant(outer_rows, "max_ratio")
    outer_pivot = build_pivot(outer_rows, outer_axes, context.metric)

    tables.append(pivot_table(
        title="Stage 3 gate: the outer level",
        purpose=(
            "`decoder` + `param_share` against `hierarchical` + `waterfill`. The two criteria bucket "
            "matrices identically and differ only in whether Block Influence may move budget between "
            "blocks, which makes this the thesis contribution's own controlled test"
        ),
        pivot=outer_pivot,
        axis_headers=[ "group_criterion", "outer_allocation", "inner_allocation", "score_metric" ],
        metric=context.metric,
        baselines=homogeneous_baselines(context.rows, context.metric),
        notes=[
            *skipped_note(outer_skipped),
            *outer_held,
            "report this apart from the inner-policy comparison: it answers a different question",
        ],
    ))

    resolved: Dict[str, str] = {}
    policies = best_by(inner_pivot, 0)

    if policies:
        resolved["__BEST_INNER__"] = policies[0][0]
        resolved.update(best_checkpoints(inner_pivot, "__CKPT_BEST_POLICY"))
        tables.append(aggregate_table(
            title="Stage 3 gate: inner policy aggregate",
            purpose="Averaged over the scores and ratios it was run at, which is what decides `__BEST_INNER__`",
            header="inner_allocation",
            ranked=policies,
        ))

    tables += offline_tables(
        context,
        "3",
        figures=(
            (
                "dispersion",
                "Stage 3 preview: ratio dispersion",
                "Match `--offset`, `--softmax_temp` and `--outer_offset` so the policies spread "
                "their ratios comparably, then set them in `args/base_args.json`",
            ),
            (
                "ratio_by_type",
                "Stage 3 preview: mean ratio per matrix family",
                "Where a rank-space policy's bias against mixed shapes becomes visible, thesis 3.3.5",
            ),
        ),
    )

    return GateResult(tables=tables, resolved=resolved)


def gate_stage4_cap(context: GateContext) -> GateResult:
    """Stage 4: `--max_ratio`, which Swift-SVD reports as first-order rather than a guard rail"""
    axes = ( "max_ratio", )
    grouping = context.resolved.get("__BEST_GROUPING__")
    score = context.resolved.get("__TOP1_SCORE__")
    inner = context.resolved.get("__BEST_INNER__")

    def select(row: Dict[str, Any]) -> bool:
        return (
            is_baseline_heterogeneous(row)
            and ( grouping is None or row.get("group_criterion") == grouping )
            and ( score is None or row.get("score_metric") == score )
            and ( inner is None or row.get("inner_allocation") == inner )
        )

    rows, skipped = gate_rows(context.rows, axes, select)
    notes = list(skipped_note(skipped))

    # Whatever stage 2 and 3 have not decided yet still has to be held somewhere,
    # or the cap column prices the cap together with the configuration around it
    for dimension, value in (
        ( "group_criterion", grouping ),
        ( "score_metric", score ),
        ( "inner_allocation", inner ),
    ):
        if value is None:
            rows, held = hold_dominant(rows, dimension)
            notes += held

    pivot = build_pivot(rows, axes, context.metric)
    resolved: Dict[str, str] = {}
    notes += confound_notes(rows, axes)

    if pivot:
        resolved["--max_ratio"] = pivot[0].key[0]
        notes.append(
            f"cap `{pivot[0].key[0]}` wins on mean rank. If it is below the 0.9 default, thesis 3.3 has "
            "to call the cap a first-order hyperparameter, and stages 5, 6 and 8 need it in `args/base_args.json`",
        )

    tables = [pivot_table(
        title="Stage 4 gate: the per-matrix cap",
        purpose="How far a single matrix may be compressed, at the configuration stages 2 and 3 chose",
        pivot=pivot,
        axis_headers=[ "max_ratio" ],
        metric=context.metric,
        baselines=homogeneous_baselines(context.rows, context.metric),
        notes=notes,
    )]

    tables += offline_tables(
        context,
        "4",
        figures=(
            (
                "cap_binding",
                "Stage 4 preview: how many matrices each cap pins",
                "A cap pinning nothing cannot change an allocation, so it never needed a run",
            ),
        ),
    )

    return GateResult(tables=tables, resolved=resolved)


def gate_stage5_bypass(context: GateContext) -> GateResult:
    """Stage 5 (RQ4): whether exempting outer blocks beats compressing everything"""
    axes = ( "bypass_early_layers", "bypass_late_layers" )

    def select(row: Dict[str, Any]) -> bool:
        return is_compression_run(row) and not str(row.get("score_metric") or "").startswith(COMPOSITE_PREFIX)

    rows, skipped = gate_rows(context.rows, axes, select)
    notes = list(skipped_note(skipped))

    # The bypass-0 reference sits among every other stage's runs while this
    # stage's own runs are few, so the configuration to hold everything at is
    # read off the bypassed arm and then imposed, not taken from the majority
    bypassed = [row for row in rows if row.get("scheme") == "het" and not bypasses_nothing(row)]

    for dimension in ( "max_ratio", "group_criterion", "score_metric", "inner_allocation" ):
        rows, held = hold_at(rows, dimension, dominant_value(bypassed or rows, dimension))
        notes += held

    gain_rows = build_gain_rows(rows, axes, context.metric)
    resolved: Dict[str, str] = {}

    baseline_gain = next((row.mean_gain for row in gain_rows if row.key == ( "0", "0" )), None)
    bypassing = [row for row in gain_rows if row.key != ( "0", "0" ) and row.mean_gain is not None]

    if baseline_gain is not None and bypassing:
        best_gain = bypassing[0].mean_gain or 0.0
        verdict = "holds"

        if best_gain < baseline_gain:
            verdict = "shrinks"
        elif best_gain > baseline_gain:
            verdict = "grows"

        notes.append(
            f"the heterogeneous gain at bypass 0 is {baseline_gain:+.2f} and the best bypassed setting "
            f"reaches {best_gain:+.2f}, so the gain {verdict} as blocks are exempted. A shrinking gain means "
            "both mechanisms are competing for the same redundancy, which is the second half of RQ4",
        )

    for row in bypassing:
        path = str(row.het.runs[0.2].get("checkpoint_path") or "") if row.het is not None and 0.2 in row.het.runs else ""

        if path:
            resolved["__CKPT_BEST_BYPASS_0.2__"] = path
            break

    tables = [gain_table(
        title="Stage 5 gate: bypassing outer blocks",
        purpose=(
            "Both arms at every bypass setting. The homogeneous arm is the point of the stage: only the "
            "gain over it at the same setting separates bypassing from heterogeneity"
        ),
        gain_rows=gain_rows,
        axis_headers=[ "bypass_early", "bypass_late" ],
        metric=context.metric,
        notes=notes,
    )]

    tables += offline_tables(context, "5")

    return GateResult(tables=tables, resolved=resolved)


def gate_stage6_composite(context: GateContext) -> GateResult:
    """Stage 6: whether fusing a spectral score with Block Influence beats either alone"""
    axes = ( "score_metric", "fusion_alpha" )

    def select(row: Dict[str, Any]) -> bool:
        return (
            is_compression_run(row)
            and row.get("scheme") == "het"
            and bypasses_nothing(row)
            and str(row.get("score_metric") or "").startswith(COMPOSITE_PREFIX)
        )

    rows, skipped = gate_rows(context.rows, axes, select)
    rows, held = hold_dominant(rows, "max_ratio")
    pivot = build_pivot(rows, axes, context.metric)
    notes = [
        *skipped_note(skipped),
        *held,
        *confound_notes(rows, axes),
        "`influence_tail` in the offline objective panel is circular with a composite score, exactly as "
        "`frobenius_tail` is with `truncation`, so a win on that column is not evidence",
    ]

    grouping = context.resolved.get("__BEST_FLAT_GROUPING__")
    groupings_used = {str(row.get("group_criterion")) for row in rows}

    if groupings_used & set(BLOCK_GROUPINGS):
        notes.append(
            "a run here used a per-block grouping, where Block Influence is constant inside a group and "
            "the fused allocation collapses to exactly homogeneous",
        )
    elif grouping is not None:
        notes.append(f"held at `--group_criterion {grouping}`, as stage 2 resolved it")

    tables = [pivot_table(
        title="Stage 6 gate: composite scores",
        purpose="The scalar counterpart of the hierarchical allocator: one per-matrix score fused with per-block Block Influence",
        pivot=pivot,
        axis_headers=[ "score_metric", "fusion_alpha" ],
        metric=context.metric,
        baselines=homogeneous_baselines(context.rows, context.metric),
        notes=notes,
    )]

    resolved: Dict[str, str] = {}
    resolved.update(best_checkpoints(pivot, "__CKPT_BEST_COMPOSITE"))
    # Only 0.2 has a role in stage 9, and an extra ratio would invent a placeholder
    resolved = {key: value for key, value in resolved.items() if key in PLACEHOLDER_SOURCES}

    rho_stage = context.offline.get("0") or context.offline.get("6")
    rho_table = None

    if rho_stage is not None:
        rho_table = offline_figure_table(
            rho_stage,
            "influence_vs_effrank_rho",
            "Stage 6 gate: Block Influence against effective rank",
            "The gate on this whole stage, from stage 0",
            notes=[
                "Swift-SVD reports this correlation as negative, which is what makes the two signals "
                "complementary. A positive rho means they agree rather than complement, the `beta^alpha` "
                "convention pushes both the same way, and these runs measure something other than intended",
            ],
        )

    if rho_table is not None:
        tables.insert(0, rho_table)

    tables += offline_tables(context, "6")

    return GateResult(tables=tables, resolved=resolved)


def gate_stage7_finalists(context: GateContext) -> GateResult:
    """Stage 7: the three configurations worth spending another model on"""
    axes = ( "group_criterion", "score_metric", "inner_allocation" )
    rows, skipped = gate_rows(context.rows, axes, is_baseline_heterogeneous)
    rows, held = hold_dominant(rows, "max_ratio")
    pivot = build_pivot(rows, axes, context.metric)
    finalists = pivot[:3]

    resolved: Dict[str, str] = {}

    for index, row in enumerate(finalists, start=1):
        for role, position in ( ( "GROUPING", 0 ), ( "SCORE", 1 ), ( "INNER", 2 ) ):
            resolved[f"__FINALIST{index}_{role}__"] = row.key[position]

    if finalists:
        for ratio, run in sorted(finalists[0].runs.items()):
            path = str(run.get("checkpoint_path") or "")
            placeholder = f"__CKPT_HET_{axis_text(ratio)}__"

            if path and placeholder in PLACEHOLDER_SOURCES:
                resolved[placeholder] = path

    tables = [pivot_table(
        title="Stage 7 gate: finalists",
        purpose=(
            "Every heterogeneous configuration collected so far, ranked together. The top three fill the "
            "`__FINALIST*__` placeholders, and the first also answers `__CKPT_HET_*__` for stage 10"
        ),
        pivot=pivot,
        axis_headers=[ "group_criterion", "score_metric", "inner_allocation" ],
        metric=context.metric,
        baselines=homogeneous_baselines(context.rows, context.metric),
        notes=[
            *skipped_note(skipped),
            *held,
            *confound_notes(rows, axes),
            "the Spearman sign and the score-versus-depth shape are model properties, so rerun stage 0 "
            "against each new model before trusting that these three transfer",
        ],
    )]

    return GateResult(tables=tables, resolved=resolved)


def gate_stage8_curve(context: GateContext) -> GateResult:
    """Stage 8 (RQ1): how the heterogeneous gain depends on the target ratio"""
    axes = ( "group_criterion", "score_metric", "inner_allocation" )
    rows, _ = gate_rows(context.rows, axes, is_baseline_heterogeneous)
    rows, held = hold_dominant(rows, "max_ratio")
    pivot = build_pivot(rows, axes, context.metric)
    baselines = homogeneous_baselines(context.rows, context.metric)
    ratios = sorted(set(baselines) | {ratio for row in pivot for ratio in row.runs})

    body: List[List[Cell]] = []
    gains: List[float] = []

    for ratio in ratios:
        priced = [row for row in pivot if row.values.get(ratio) is not None]
        best = min(priced, key=lambda row: row.values[ratio] or math.inf) if priced else None
        het_value = best.values.get(ratio) if best is not None else None
        hom_value = baselines.get(ratio)
        gain = hom_value - het_value if het_value is not None and hom_value is not None else None

        if gain is not None:
            gains.append(gain)

        body.append([
            Cell(text=axis_text(ratio)),
            Cell(text=fmt_ppl(hom_value)),
            Cell(text=fmt_ppl(het_value)),
            Cell(text=gain_text(hom_value, het_value), bold=gain is not None and gain > 0),
            Cell(text=" / ".join(best.key) if best is not None else "--"),
        ])

    notes = [
        *held,
        "the best heterogeneous configuration is picked per ratio, so the column may change configuration "
        "between rows, which is itself the answer if it does",
    ]

    if len(gains) > 1:
        trend = "widens" if gains[-1] > gains[0] else "narrows"
        notes.append(
            f"the gain goes from {gains[0]:+.2f} at the lowest ratio to {gains[-1]:+.2f} at the highest, so it "
            f"{trend} with the budget. At the high end read this together with stage 4 and `cap_binding.csv`",
        )

    tables = [Table(
        title="Stage 8 gate: the ratio curve",
        purpose="Both arms across every collected budget, which is the shape RQ1 asks about rather than a slope",
        headers=[ "ratio", f"hom {context.metric}", f"best het {context.metric}", "gain", "configuration" ],
        rows=body,
        notes=notes,
    )]

    tables += offline_tables(context, "8")

    return GateResult(tables=tables, resolved={})


def gate_stage9_roster(context: GateContext) -> GateResult:
    """Stage 9 allocates nothing: its gate is filling nine checkpoint paths"""
    roles = tuple(
        ( placeholder, source )
        for placeholder, source in PLACEHOLDER_SOURCES.items()
        if placeholder.startswith("__CKPT_")
    )

    evaluated = {
        str(row.get("run_name") or ""): row
        for row in context.rows
        if row.get("avg_accuracy") is not None
    }

    body: List[List[Cell]] = []

    for placeholder, source in roles:
        path = context.resolved.get(placeholder, "")
        stem = Path(path).stem if path else ""

        body.append([
            Cell(text=placeholder),
            Cell(text=path or "not resolved yet", bold=bool(path)),
            Cell(text=source),
            Cell(text="yes" if path and Path(path).is_file() else "no"),
            Cell(text="yes" if stem in evaluated else "no"),
        ])

    table = Table(
        title="Stage 9 gate: checkpoint roster",
        purpose="Every `__CKPT_*__` role, the path that fills it, and whether the full suite has run on it yet",
        headers=[ "placeholder", "checkpoint", "comes from", "on disk", "full suite done" ],
        rows=body,
        notes=[
            "`mathqa` is worth one attempt and is expected to fail, since `datasets` no longer permits "
            "custom loading scripts. Report the failure rather than working around it",
        ],
    )

    return GateResult(tables=[table], resolved={})


def gate_stage10_lora(context: GateContext) -> GateResult:
    """Stage 10 (RQ5): whether heterogeneous allocation is a better starting point for fine-tuning"""
    before = {str(row.get("run_name") or ""): row for row in context.rows if not row.get("update_method")}
    updated = [row for row in context.rows if row.get("update_method")]

    body: List[List[Cell]] = []
    # Keyed by phase, ratio and scheme, so the het-minus-hom gap can be read
    # before the update and after it at the same budget
    measured: Dict[Tuple[str, float, str], float] = {}

    for row in sorted(updated, key=sort_rows_hierarchical):
        origin = before.get(base_run_name(str(row.get("run_name") or "")))
        ratio = as_float(row.get("compression_ratio"))
        after_value = gate_metric_value(row, context.metric)
        before_value = gate_metric_value(origin, context.metric) if origin is not None else None
        scheme = str((origin if origin is not None else row).get("scheme") or "--")

        body.append([
            Cell(text=scheme),
            Cell(text=axis_text(ratio)),
            Cell(text=str(row.get("update_method") or "--")),
            Cell(text=fmt_ppl(before_value)),
            Cell(text=fmt_ppl(after_value)),
            Cell(text=gain_text(before_value, after_value)),
        ])

        if ratio is None:
            continue

        for phase, value in ( ( "before", before_value ), ( "after", after_value ) ):
            if value is not None:
                measured[( phase, ratio, scheme )] = value

    notes = [
        "recovered is before minus after, so a positive number means the sequential update lowered perplexity",
        "read the het-minus-hom gap before the update against the same gap after it. A gap that closes means "
        "the update recovers what a bad allocation lost, and RQ5 is answered negatively",
    ]

    for ratio in sorted({key[1] for key in measured}):
        gaps: Dict[str, float] = {}

        for phase in ( "before", "after" ):
            het = measured.get(( phase, ratio, "het" ))
            hom = measured.get(( phase, ratio, "hom" ))

            if het is not None and hom is not None:
                gaps[phase] = hom - het

        if len(gaps) < 2:
            continue

        verdict = "closes" if gaps["after"] < gaps["before"] else "holds or widens"
        notes.append(
            f"at ratio {axis_text(ratio)} the het-minus-hom gap goes from {gaps['before']:+.2f} before the "
            f"update to {gaps['after']:+.2f} after it, so it {verdict}",
        )

    table = Table(
        title="Stage 10 gate: sequential update",
        purpose="Each updated checkpoint against the run it started from, paired through the run name",
        headers=[ "scheme", "ratio", "method", f"before {context.metric}", f"after {context.metric}", "recovered" ],
        rows=body,
        notes=notes,
    )

    return GateResult(tables=[table], resolved={})


GATES: Tuple[Callable[[GateContext], GateResult], ...] = (
    gate_stage1_anchors,
    gate_stage2_score_grouping,
    gate_stage2b_squared,
    gate_stage2c_schatten,
    gate_stage3_policies,
    gate_stage4_cap,
    gate_stage5_bypass,
    gate_stage6_composite,
    gate_stage7_finalists,
    gate_stage8_curve,
    gate_stage9_roster,
    gate_stage10_lora,
)


def placeholder_table(resolved: Dict[str, str]) -> Table:
    """
    The one table this report exists for: what to paste into the next stage file.

    Unresolved rows are listed too, so a gate still waiting on runs reads as
    waiting rather than as absent.
    """
    body: List[List[Cell]] = []

    for placeholder, source in PLACEHOLDER_SOURCES.items():
        value = resolved.get(placeholder, "")

        body.append([
            Cell(text=placeholder),
            Cell(text=value or "not resolved yet", bold=bool(value)),
            Cell(text=source),
            Cell(text="ready" if value else "waiting on runs"),
        ])

    for placeholder in sorted(set(resolved) - set(PLACEHOLDER_SOURCES)):
        body.append([
            Cell(text=placeholder),
            Cell(text=resolved[placeholder], bold=True),
            Cell(text="resolved by a gate, no stage file asks for it"),
            Cell(text="ready"),
        ])

    return Table(
        title="Placeholders",
        purpose="Every value a stage file waits on, and what the collected runs resolve it to",
        headers=[ "placeholder", "value", "resolved by", "status" ],
        rows=body,
        notes=[
            "`run_experiments.py` refuses to start while a placeholder is unresolved, so a `waiting on runs` "
            "row is a stage that cannot run yet rather than a defaulted one",
        ],
    )


def gate_tables_for_model(
        rows: List[Dict[str, Any]],
        offline: Dict[str, OfflineStage],
        metric: str
) -> List[Table]:
    """Run every gate in stage order, so a later one can read what an earlier one resolved"""
    resolved: Dict[str, str] = {}
    tables: List[Table] = []

    for gate in GATES:
        result = gate(GateContext(rows=rows, offline=offline, resolved=resolved, metric=metric))
        tables += result.tables
        resolved.update(result.resolved)

    return [ placeholder_table(resolved), *tables ]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def markdown_gate_cell(cell: Cell) -> str:
    if cell.faded:
        return markdown_faded(cell.text)

    text = markdown_escape(cell.text)

    return f"**{text}**" if cell.bold else text


def sentence_case(text: str) -> str:
    return text[:1].upper() + text[1:]


def latex_gate_cell(cell: Cell) -> str:
    if cell.faded:
        return latex_faded_cell(cell.text, False)

    return latex_cell(cell.text, cell.bold)


def render_table_markdown(table: Table) -> List[str]:
    lines = [ f"### {table.title}", "" ]

    if table.purpose:
        lines += [ table.purpose, "" ]

    if not table.rows:
        lines += [ "_No runs collected for this gate yet_", "" ]
    else:
        lines.append("| " + " | ".join(markdown_escape(header) for header in table.headers) + " |")
        lines.append("| " + " | ".join([ "---" ] * len(table.headers)) + " |")

        for row in table.rows:
            lines.append("| " + " | ".join(markdown_gate_cell(cell) for cell in row) + " |")

        lines.append("")

    for note in table.notes:
        lines += [ f"> {note}", "" ]

    return lines


def render_table_latex(table: Table) -> List[str]:
    if not table.rows:
        return [ f"% {table.title}: no runs collected yet", "" ]

    colspec = "l" + "r" * (len(table.headers) - 1)

    # Markdown gives every note its own block, so a note is written as a bare
    # clause. A caption strings them into prose and has to supply what that
    # leaves out: the separators and the leading capital
    prose = [sentence_case(part) for part in ( table.purpose, *table.notes ) if part]
    caption = ". ".join([ table.title, *prose ]) + "."

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{adjustbox}{max width=\textwidth,center}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & ".join(latex_escape(header) for header in table.headers) + r" \\",
        r"\midrule",
    ]

    for row in table.rows:
        lines.append(" & ".join(latex_gate_cell(cell) for cell in row) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{tab:gate-{latex_label_slug(table.title)}}}",
        r"\end{table*}",
        "",
    ]

    return lines


GATE_REPORT_INTRO = (
    "One table per gate of the staged grid in `EXPERIMENTS.md`. Perplexity columns are ranked within "
    "each ratio and averaged across ratios, never compared as raw numbers between budgets. The first "
    "table is the only one needed to move to the next stage: it holds every placeholder and its value."
)


def build_gate_report_markdown(
        rows_by_model: Dict[str, List[Dict[str, Any]]],
        offline: Dict[str, OfflineStage],
        metric: str
) -> str:
    out = [ "# Experiment grid gates", "", GATE_REPORT_INTRO, "" ]

    for model_name in sorted(rows_by_model):
        out += [ f"## {model_name}", "" ]

        for table in gate_tables_for_model(rows_by_model[model_name], offline, metric):
            out += render_table_markdown(table)

    return "\n".join(out).rstrip() + "\n"


def build_gate_report_latex(
        rows_by_model: Dict[str, List[Dict[str, Any]]],
        offline: Dict[str, OfflineStage],
        metric: str
) -> str:
    out: List[str] = []

    for model_name in sorted(rows_by_model):
        out += [ f"% ==== {model_name} ====", "" ]

        for table in gate_tables_for_model(rows_by_model[model_name], offline, metric):
            out += render_table_latex(table)

    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate hierarchical markdown or LaTeX tables from lm-eval JSON result files",
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing lm-eval JSON files",
    )

    parser.add_argument(
        "-p",
        "--pattern",
        default="*.json",
        help="Glob pattern for result files. Default: *.json",
    )

    parser.add_argument(
        "-w",
        "--table_width",
        type=float,
        default=1.6,
        help="Table width for adjustbox",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("lm_eval_report.md"),
        help="Output file. Default: lm_eval_report.md",
    )

    parser.add_argument(
        "-f",
        "--format",
        choices=[ "markdown", "latex" ],
        default="markdown",
        help="Output format. Default: markdown",
    )

    parser.add_argument(
        "--prefer-lm-eval-model-name",
        action="store_true",
        help=(
            "Use the model name stored inside the JSON config instead of the filename-derived model name. "
            "By default, the filename-derived model name is used"
        ),
    )

    parser.add_argument(
        "-r",
        "--report",
        choices=[ "benchmarks", "gates", "both" ],
        default="benchmarks",
        help=(
            "benchmarks: the per-model result tables. gates: one table per stage gate of EXPERIMENTS.md, "
            "with the placeholder values the next stage needs. Default: benchmarks"
        ),
    )

    parser.add_argument(
        "--allocation_dir",
        type=Path,
        default=None,
        help=(
            "Root holding the allocation_report.py output directories, usually "
            "./output/allocation_reports. Every stage<N>/ inside it is attached to its gate"
        ),
    )

    parser.add_argument(
        "--gate_metric",
        choices=[ *PERPLEXITY_BENCHMARK_ORDER, "mean" ],
        default="wikitext",
        help="Perplexity a gate ranks its variants by, or the mean of both. Default: wikitext",
    )

    args = parser.parse_args()

    rows_by_model = collect_rows(
        args.input_dir,
        args.pattern,
        prefer_lm_eval_model_name=args.prefer_lm_eval_model_name,
    )

    offline: Dict[str, OfflineStage] = {}
    if args.allocation_dir is not None:
        offline = load_offline_reports(args.allocation_dir)
        print(f"Read {len(offline)} offline stage preview(s) from {args.allocation_dir}")

    sections: List[str] = []

    if args.report in ( "benchmarks", "both" ):
        if args.format == "markdown":
            sections.append(build_markdown_report(rows_by_model))
        else:
            sections.append(build_latex_report(rows_by_model, table_width = float(args.table_width)))

    if args.report in ( "gates", "both" ):
        if args.format == "markdown":
            sections.append(build_gate_report_markdown(rows_by_model, offline, args.gate_metric))
        else:
            sections.append(build_gate_report_latex(rows_by_model, offline, args.gate_metric))

    args.output.write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
