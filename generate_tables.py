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
    "truncation_rel": "truncation_rel",
    "truncation_sq_rel": "truncation_sq_rel",
    "eff_rank_rel": "eff_rank_rel",
    "eff_rank_sq_rel": "eff_rank_sq_rel",
    "entropy_rel": "entropy_rel",
    "entropy_sq_rel": "entropy_sq_rel",
    "full_norm_tail_entropy": "full_norm_tail_entropy",
    "full_norm_sq_tail_entropy": "full_norm_sq_tail_entropy",
    "full_norm_tail_eff_rank": "full_norm_tail_eff_rank",
    "full_norm_sq_tail_eff_rank": "full_norm_sq_tail_eff_rank",
}

COMPOSITE_PREFIX = "composite|"

# Local halves a composite metric can fuse, and what it can fuse them with
COMPOSITE_LOCAL_TOKENS = (
    "truncation",
    "truncation_sq",
    "entropy",
    "entropy_sq",
    "eff_rank",
    "eff_rank_sq",
    "truncation_rel",
    "truncation_sq_rel",
    "entropy_rel",
    "entropy_sq_rel",
    "eff_rank_rel",
    "eff_rank_sq_rel",
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
    for spelling in ( f"{COMPOSITE_PREFIX}{local}|{end_to_end}", f"composite_{local}_{end_to_end}" )
})

SCHEME_TOKENS = {
    "het": "het",
    "heterogeneous": "het",
    "hom": "hom",
    "homogeneous": "hom",
}

# The policy and grouping spellings the gates filter on, as `src.utils` spells
# them. This script stays import-free, so they are repeated rather than imported
DEFAULT_INNER_ALLOCATION = "waterfill"
DEFAULT_OUTER_ALLOCATION = "param_share"

# `--group_criterion`, as the flag takes it. `GROUPING_TOKENS` maps to the table
# label instead, and the two are a different string for `type`
GROUPING_FLAG_VALUES = {
    "global": "global",
    "decoder": "decoder",
    "hierarchical": "hierarchical",
    "matrix_type": "type",
    "matrix": "type",
}

# `_<inner>` and `_out<outer>`, the two suffix tokens carrying a name rather than
# a number. `swift_pool` and the rest span two tokens, so they are matched by
# joining ahead
INNER_ALLOCATION_TOKENS = ( "waterfill", "drank_lagrangian", "swift_pool", "softmax_temp" )
OUTER_ALLOCATION_TOKENS = ( "param_share", "waterfill" )
POLICY_TOKEN_SPAN = 2

# What a numeric suffix token spells, and the value `build_run_name` leaves the
# token out at. A dimension with no token is a run that never moved it, which is
# what makes a name parseable back into flag values
NAME_TOKEN_DEFAULTS = (
    ( "max_ratio", "cap", 0.9 ),
    ( "min_rank_fraction", "mrf", 0.0 ),
    ( "seed", "seed", 6363 ),
    ( "bypass_ratio", "bypr", 0.0 ),
    ( "fusion_alpha", "fa", 0.5 ),
    ( "offset", "off", 1.5 ),
    ( "softmax_temp", "temp", 1.0 ),
    ( "outer_offset", "ooff", 1.5 ),
)

# Dimensions whose token is the bare flag rather than a value, so they are
# matched exactly instead of by numeric suffix
NAME_FLAG_TOKENS = (
    ( "head_block_svd", "hb" ),
)

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


def latex_column_label(column: str) -> str:
    return r"Avg. $\uparrow$" if column == "avg_accuracy" else BENCHMARK_LABELS_LATEX[column]


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


# `build_run_name` keeps the bare integer for an early-only bypass, so both
# spellings have to parse or a `byp0-1` run loses everything past its score
BYPASS_RANGE_PATTERN = re.compile(r"byp(\d+)-(\d+)")


def parse_bypass_token(token: str) -> Optional[Tuple[int, int]]:
    """The two ends a bypass token names, or None when the token is not one"""
    match = BYPASS_RANGE_PATTERN.fullmatch(token)

    if match is not None:
        return int(match.group(1)), int(match.group(2))

    return ( int(token), 0 ) if is_int_token(token) else None


def score_metric_from_tokens(joined: str) -> str:
    """
    The `--score_metric` value a filename's scoring tokens spell.

    `find_scoring` answers with the table label, which is what a row is displayed
    under; a gate filters on the flag value instead, and a resolved placeholder
    is pasted back into a stage file as one
    """
    for end_to_end in END_TO_END_TOKENS:
        prefix, suffix = "composite_", f"_{end_to_end}"

        if joined.startswith(prefix) and joined.endswith(suffix):
            return f"{COMPOSITE_PREFIX}{joined[len(prefix):-len(suffix)]}|{end_to_end}"

    return "truncation" if joined == "truncation_loss" else joined


def numeric_suffix(token: str, prefix: str) -> Optional[float]:
    """The number a `<prefix><value>` token carries, or None when it is not one"""
    if not token.startswith(prefix):
        return None

    try:
        return float(token[len(prefix):])
    except ValueError:
        return None


def inner_allocation_at(tokens: List[str], start: int) -> Tuple[Optional[str], int]:
    """The inner policy the tokens at `start` spell, and how many they took"""
    for end in range(min(len(tokens), start + POLICY_TOKEN_SPAN), start, -1):
        candidate = "_".join(tokens[start:end])

        if candidate in INNER_ALLOCATION_TOKENS:
            return candidate, end - start

    return None, 1


def suffix_dimensions(tokens: List[str]) -> Dict[str, Any]:
    """
    The policies and knobs `build_run_name` appends after the bypass token.

    They are read by pattern rather than by position: a token is emitted only for
    a dimension that left its default, so which of them are present, and in what
    order, differs from run to run
    """
    found: Dict[str, Any] = {}
    index = 0

    while index < len(tokens):
        token = tokens[index]
        step = 1
        outer = next((name for name in OUTER_ALLOCATION_TOKENS if token == f"out{name}"), None)
        inner, inner_span = inner_allocation_at(tokens, index)

        flag = next((name for name, marker in NAME_FLAG_TOKENS if token == marker), None)

        if outer is not None:
            found["outer_allocation"] = outer
        elif inner is not None:
            found["inner_allocation"] = inner
            step = inner_span
        elif flag is not None:
            found[flag] = True
        else:
            for name, prefix, _ in NAME_TOKEN_DEFAULTS:
                value = numeric_suffix(token, prefix)

                if value is not None:
                    found[name] = value
                    break

        index += step

    return found


def run_name_dimensions(
        tokens: List[str],
        scheme: str,
        ratio_idx: Optional[int],
        suffix_start: int,
        group_criterion: Optional[str],
        score_metric: Optional[str],
        bypass: Optional[Tuple[int, int]]
) -> Dict[str, Any]:
    """
    The raw flag values the run name encodes, which is what a gate compares on.

    The sidecar outranks these wherever it speaks, but a run that only evaluated
    an existing checkpoint was invoked without the flags that produced it: its
    sidecar records the evaluation, and the name is the only thing left that
    still describes the compression.
    """
    if scheme not in ( "het", "hom" ):
        return {}

    heterogeneous = scheme == "het"
    early, late = bypass if bypass is not None else ( 0, 0 )

    dimensions: Dict[str, Any] = {
        "bypass_early_layers": early,
        "bypass_late_layers": late,
    }

    if ratio_idx is not None and ratio_idx > 0 and tokens[ratio_idx - 1] not in MATRIX_TOKEN_MAP:
        dimensions["ratio_scope"] = tokens[ratio_idx - 1]

    from_suffix = suffix_dimensions(tokens[suffix_start:])

    if heterogeneous:
        dimensions["inner_allocation"] = from_suffix.get("inner_allocation", DEFAULT_INNER_ALLOCATION)
        dimensions["outer_allocation"] = from_suffix.get("outer_allocation", DEFAULT_OUTER_ALLOCATION)

        if group_criterion is not None:
            dimensions["group_criterion"] = group_criterion

        if score_metric is not None:
            dimensions["score_metric"] = score_metric

    # A homogeneous run allocates nothing, so the knobs a policy reads would
    # group it under a decision it never took
    for name, _, default in NAME_TOKEN_DEFAULTS:
        if name in HET_ONLY_DIMENSIONS and not heterogeneous:
            continue

        dimensions[name] = from_suffix.get(name, default)

    for name, _ in NAME_FLAG_TOKENS:
        dimensions[name] = from_suffix.get(name, False)

    return dimensions


def parse_filename(path: Path) -> Dict[str, Any]:
    """
    Supported filename styles:

    Original / uncompressed:
        Qwen_Qwen2.5_32B.json

    Heterogeneous compression:
        Qwen_Qwen2.5_32B_q_k_v_out_mlp_all_0.2_het_decoder_truncation_8_v2.json
        Qwen_Qwen2.5_32B_q_k_v_out_mlp_selected_0.2_het_decoder_truncation_8_v2.json
        Qwen_Qwen2.5_32B_q_k_v_out_mlp_all_0.2_het_decoder_truncation_byp2-4_softmax_temp_cap0.8_v2.json

    Homogeneous compression:
        huggyllama_llama_7b_q_k_v_out_mlp_all_0.2_hom_8_v2.json
        huggyllama_llama_7b_q_k_v_out_mlp_selected_0.2_hom_8_v2.json

    Alongside the display columns the tables are grouped by, the raw flag values
    the name encodes come back too, so a run whose sidecar describes a later
    evaluation rather than its own compression is still placed by every gate
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
    group_criterion = None
    score_metric = None
    bypass = None
    suffix_start = len(tokens)
    scheme_end = scheme_idx + 1 if scheme_idx is not None else len(tokens)
    names_an_allocation = scheme != "hom" and scheme_end < len(tokens)

    # A malformed name is read like a heterogeneous one: the tokens it does carry
    # still place the grouping and the score
    if names_an_allocation:
        score_start = scheme_end

        if tokens[score_start] in GROUPING_TOKENS:
            grouping = GROUPING_TOKENS[tokens[score_start]]
            group_criterion = GROUPING_FLAG_VALUES.get(grouping)
            score_start += 1

        scoring, scoring_idx, scoring_token_count = find_scoring(tokens, score_start)

        if scoring_idx is not None:
            suffix_start = scoring_idx + scoring_token_count
            score_metric = score_metric_from_tokens("_".join(tokens[scoring_idx:suffix_start]))

    elif scheme == "hom":
        suffix_start = scheme_end

    if suffix_start < len(tokens):
        bypass = parse_bypass_token(tokens[suffix_start])

    if bypass is not None:
        suffix_start += 1

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
        "bypassed_layers": sum(bypass) if bypass is not None else 0,
        "filename_version": version,
        **run_name_dimensions(
            tokens=tokens,
            scheme=scheme,
            ratio_idx=ratio_idx,
            suffix_start=suffix_start,
            group_criterion=group_criterion,
            score_metric=score_metric,
            bypass=bypass,
        ),
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

    # Two runs from different machines carry different whitening caches, and on
    # LLaMA-7B that is worth more perplexity than most of what a gate compares
    if run_args.get("save_path"):
        row["environment"] = str(run_args["save_path"])

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


def present_columns(rows: List[Dict[str, Any]], columns: List[str]) -> List[str]:
    """
    The columns at least one row fills, in table order.

    The screening stages evaluate wikitext alone and the full suite only runs at
    the end, so a table built from the full column list carries a dead column
    for every benchmark that stage has not reached yet.
    """
    return [
        column for column in columns
        if any(safe_float(row.get(column)) is not None for row in rows)
    ]


def value_cells(
    row: Dict[str, Any],
    render: Callable[[str, bool], str],
    best: Optional[Dict[str, Optional[float]]] = None,
    columns: Optional[List[str]] = None,
) -> List[str]:
    """
    Formatted value cells of one row, in table order.

    Without `best` nothing is highlighted, which is what baseline rows need.
    """
    cells = []

    for column in columns if columns is not None else VALUE_COLUMNS:
        highlight = best is not None and is_best(row, column, best)
        cells.append(render(metric_value_for_display(row, column), highlight))

    return cells

def make_markdown_table_for_model(model_name: str, rows: List[Dict[str, Any]]) -> str:
    rows = sorted(rows, key=sort_rows_hierarchical)
    original_rows, grouped = group_rows_for_model_by_bypass(rows)

    perplexity = present_columns(rows, PERPLEXITY_BENCHMARK_ORDER)
    likelihood = present_columns(rows, LIKELIHOOD_BENCHMARK_ORDER)
    generation = present_columns(rows, GENERATION_BENCHMARK_ORDER)
    summary = present_columns(rows, [ "avg_accuracy" ])
    columns = [ *perplexity, *likelihood, *summary, *generation ]

    headers = [
        "Ratio",
        "Grouping",
        "Scoring",
        "Scheme",
        "Matrices",
    ]

    headers += [BENCHMARK_LABELS_MD[b] for b in perplexity]
    headers += [BENCHMARK_LABELS_MD[b] for b in likelihood]
    headers += [ "Average ↑" ] * len(summary)
    headers += [BENCHMARK_LABELS_MD[b] for b in generation]
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
                row_cells += value_cells(row, markdown_faded_cell, columns=columns)
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

                row_cells += value_cells(row, markdown_cell, ratio_best, columns=columns)
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

    # A group with no measured column is dropped entirely: an empty multicolumn
    # and a reversed cmidrule span are both invalid LaTeX
    column_groups = [
        ( title, present_columns(rows, members) )
        for title, members in (
            ( "Perplexity Benchmarks", PERPLEXITY_BENCHMARK_ORDER ),
            ( "Likelihood Benchmarks", LIKELIHOOD_BENCHMARK_ORDER ),
            ( "Summary", [ "avg_accuracy" ] ),
            ( "Generation Benchmarks", GENERATION_BENCHMARK_ORDER ),
        )
    ]
    column_groups = [( title, members ) for title, members in column_groups if members]
    columns = [column for _, members in column_groups for column in members]

    lines: List[str] = []

    for bypass in sorted(grouped.keys()):
        lines.append(r"\begin{table*}[t]")
        lines.append(r"\centering")
        lines.append(table_size)
        lines.append(r"\setlength{\tabcolsep}{3pt}")

        if use_adjustbox:
            lines.append(r"\begin{adjustbox}{width=" + str(table_width) + r"\textwidth,center}")

        value_colspec = " ".join("r" * len(members) for _, members in column_groups)
        lines.append(rf"\begin{{tabular}}{{l llll | {value_colspec}}}")
        lines.append(r"\toprule")

        # The configuration block occupies columns 1 to 5, so the value groups
        # start at 6 and each one begins where the previous ended
        rules = [ r"\cmidrule(lr){1-1}", r"\cmidrule(lr){2-5}" ]
        group_start = 6

        for _, members in column_groups:
            group_end = group_start + len(members) - 1
            rules.append(rf"\cmidrule(lr){{{group_start}-{group_end}}}")
            group_start = group_end + 1

        total_col_count = group_start - 1

        lines.append(
            " & ".join([
                r"\multicolumn{1}{c}{Compression}",
                r"\multicolumn{4}{c}{Configuration}",
                *[rf"\multicolumn{{{len(members)}}}{{c}}{{{title}}}" for title, members in column_groups],
            ]) + r" \\",
        )

        lines.append("".join(rules))

        header = [
            "Ratio",
            "Group",
            "Score",
            "Scheme",
            "Matrices",
        ]

        header += [latex_column_label(column) for column in columns]

        lines.append(" & ".join(header) + r" \\")
        lines.append(r"\midrule")

        # Faded baseline row shown at the top of each bypass-specific table
        if original_rows:
            for row in original_rows:
                cells = [ r"\textcolor{black!45}{0\%}" ]
                cells += [ r"\textcolor{black!45}{--}" ] * 4
                cells += value_cells(row, latex_faded_cell, columns=columns)

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

                cells += value_cells(row, latex_cell, ratio_best, columns=columns)

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

# Rows spanning less than this fraction of the best value at a ratio are not
# separated by it, so the ranking there carries no information
RESOLUTION_SPREAD = 0.01

# Every placeholder a stage file can carry, and the gate that answers it
PLACEHOLDER_SOURCES: Dict[str, str] = {
    "__CKPT_HOM_0.2__": "stage 1",
    "__CKPT_HOM_0.5__": "stage 1",
    "__BEST_GROUPING__": "stage 2, re-resolved by 3c",
    "__BEST_FLAT_GROUPING__": "stage 2",
    "__TOP1_SCORE__": "stage 2, promotable by 2b or 2c",
    "__TOP2_SCORE__": "stage 2",
    "__CKPT_BEST_SCORE_0.2__": "stage 2",
    "__CKPT_BEST_SCORE_0.5__": "stage 2",
    "__BEST_OUTER__": "stage 3c",
    "__CKPT_BEST_OUTER_0.2__": "stage 3c",
    "__CKPT_BEST_OUTER_0.5__": "stage 3c",
    "__BEST_INNER__": "stage 4",
    "__CKPT_BEST_POLICY_0.2__": "stage 4",
    "__CKPT_BEST_POLICY_0.5__": "stage 4",
    "--max_ratio": "stage 3, into args/base_args.json",
    "__BEST_OUTER_OFFSET__": "stage 4c, into args/base_args.json",
    "__BEST_BYPASS_EARLY__": "stage 5",
    "__BEST_BYPASS_LATE__": "stage 5",
    "__CKPT_BEST_BYPASS_0.2__": "stage 5",
    "__CKPT_BEST_COMPOSITE_0.2__": "stage 6",
    "__CKPT_HET_0.2__": "stages 2 to 6",
    "__CKPT_HET_0.5__": "stages 2 to 6",
}

PLACEHOLDER_SOURCES.update({
    f"__FINALIST{index}_{role}__": "stages 2 to 6"
    for index in ( 1, 2, 3 )
    for role in ( "GROUPING", "SCORE", "INNER", "OUTER" )
})

# The shape-invariant sibling of the finalist score, which stage 7c substitutes
# wherever it pairs the score fix with another one. Derived rather than ranked:
# it is whatever `__FINALIST1_SCORE__` becomes once its spectrum-length ceiling
# is divided out, so the two can never name different families of score
PLACEHOLDER_SOURCES["__FINALIST1_SCORE_REL__"] = "stage 7, from __FINALIST1_SCORE__"

# Stage 7d carries onto a second grouped-query model only what stage 7c elects,
# so these are answered by the fix runs rather than by the LLaMA grid
PLACEHOLDER_SOURCES.update({
    f"__GQA_WINNER{index}_{role}__": "stage 7c"
    for index in ( 1, 2 )
    for role in ( "GROUPING", "SCORE", "INNER", "OUTER" )
})

# Suffix that turns a spectral score into its shape-invariant counterpart, and
# the scores that have one. A `norm|p` or a composite has no `_rel` spelling, so
# a finalist naming one leaves `__FINALIST1_SCORE_REL__` unresolved on purpose
RELATIVE_SCORE_SUFFIX = "_rel"
RELATIVE_SCORE_BASES = (
    "truncation",
    "truncation_sq",
    "entropy",
    "entropy_sq",
    "eff_rank",
    "eff_rank_sq",
)

# In rank order, because stages 2b and 2c re-resolve both of them together: a
# squared score or a Schatten norm may take either place, which is why no stage
# file past 2c names a score literally
SCORE_PLACEHOLDERS = ( "__TOP1_SCORE__", "__TOP2_SCORE__" )

# Figures `allocation_report.py` writes that a gate reads back
OFFLINE_FIGURES = (
    "dispersion",
    "cap_binding",
    "influence_vs_effrank_rho",
    "ratio_by_type",
    "ratio_tail",
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


class Resolution(NamedTuple):
    """
    A placeholder's value, and how many candidates the gate chose it from.

    A gate whose deciding axis held one entrant still resolves to a real value
    and reads as decided, which is how `__BEST_INNER__` came out of a table
    holding only `waterfill`. The count is what separates a decision from a
    default, and it is the only thing distinguishing the two in the report.

    `candidates` is None where no choice was involved at all: the homogeneous
    anchor of a ratio is the only run that could have filled its role
    """
    value: str
    candidates: Optional[int]


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
    resolved: Dict[str, Resolution]
    metric: str


class GateResult(NamedTuple):
    tables: List[Table]
    resolved: Dict[str, Resolution]


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


def anchor_matrices(rows: List[Dict[str, Any]]) -> Optional[str]:
    """
    The family selection the grid is run at, and so the only `hom` arm anchoring it.

    Stage 6b sweeps that selection on homogeneous runs alone, so its rows answer
    every filter asking for `hom` and, left in, decide the baseline -- and with it
    the gain every other gate reports. The heterogeneous runs are what the grid is
    made of, so the family they share is the one to read the anchors at.
    """
    heterogeneous = [row for row in rows if is_compression_run(row) and row.get("scheme") == "het"]
    compressed = heterogeneous or [row for row in rows if is_compression_run(row)]

    return dominant_value(compressed, "matrices")


def is_anchor_family(row: Dict[str, Any], family: Optional[str]) -> bool:
    """Whether a run compresses the selection the grid is anchored on"""
    return family is None or axis_text(row.get("matrices")) == family


def build_gain_rows(rows: List[Dict[str, Any]], axes: Tuple[str, ...], metric: str) -> List[GainRow]:
    """
    Pair each configuration's heterogeneous arm with its homogeneous one.

    The homogeneous arm is not padding: the gain over it at the same setting is
    the only quantity that separates a stage's mechanism from the budget itself.
    """
    family = anchor_matrices(rows)
    het = {row.key: row for row in build_pivot([row for row in rows if row.get("scheme") == "het"], axes, metric)}
    hom_rows = [row for row in rows if row.get("scheme") == "hom" and is_anchor_family(row, family)]
    hom = {row.key: row for row in build_pivot(hom_rows, axes, metric)}

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
    family = anchor_matrices(rows)
    baselines: Dict[float, Optional[float]] = {}

    for row in rows:
        is_anchor = (
            is_compression_run(row)
            and row.get("scheme") == "hom"
            and is_anchor_family(row, family)
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


def resolved_value(resolved: Dict[str, Resolution], placeholder: str) -> Optional[str]:
    """The value a gate resolved, or None while no gate has answered yet"""
    resolution = resolved.get(placeholder)
    return resolution.value if resolution is not None else None


def best_checkpoints(pivot: List[PivotRow], prefix: str) -> Dict[str, Resolution]:
    """The rank-1 checkpoint at each ratio, which is what a `__CKPT_*__` role names"""
    resolved: Dict[str, Resolution] = {}
    ratios = sorted({ratio for row in pivot for ratio in row.runs})

    for ratio in ratios:
        # Only the rows this ratio actually priced were in the running for it
        competing = sum(1 for row in pivot if row.values.get(ratio) is not None)

        for row in pivot:
            if row.ranks.get(ratio) != 1:
                continue

            path = str(row.runs[ratio].get("checkpoint_path") or "")

            if path:
                resolved[f"{prefix}_{axis_text(ratio)}__"] = Resolution(value=path, candidates=competing)

            break

    return resolved


def dominant_of(values: List[str]) -> str:
    """The most common of a list, ties broken by name so a report is reproducible"""
    counted = Counter(values)
    return min(counted.items(), key=lambda item: ( -item[1], item[0] ))[0]


def policy_interaction_table(pivots: Dict[str, List[PivotRow]]) -> Table:
    """
    The inner-policy ranking side by side across the grouping arms.

    The point of running more than one arm is that the ranking is not expected to
    agree, and a reader should not have to diff two panels to find out. Where the
    orders differ, policy and grouping cannot be chosen independently, which is
    itself an answer to RQ3 rather than an obstacle to one.
    """
    arms = sorted(pivots)
    ranked = {arm: best_by(pivots[arm], 0) for arm in arms}
    policies = sorted({policy for order in ranked.values() for policy, _ in order})
    rows: List[List[Cell]] = []

    for policy in policies:
        cells = [Cell(text=f"`{policy}`")]

        for arm in arms:
            order = ranked[arm]
            place = next((index for index, ( name, _ ) in enumerate(order) if name == policy), None)
            mean = next((rank for name, rank in order if name == policy), None)
            text = "-" if place is None else f"{place + 1} ({mean:.2f})"
            cells.append(Cell(text=text, bold=place == 0))

        rows.append(cells)

    winners = {order[0][0] for order in ranked.values() if order}
    notes = ["cells are the policy's place under that grouping, with its mean rank in parentheses"]

    if len(winners) > 1:
        notes.append(
            "the arms disagree on the winner, so `__BEST_INNER__` is only valid alongside "
            "`__BEST_GROUPING__`: report the interaction rather than a single policy"
        )
    else:
        notes.append("the arms agree on the winner, so the policy choice survives a change of grouping")

    return Table(
        title="Stage 4 gate: policy ranking across grouping arms",
        purpose=(
            "Whether the inner policy can be chosen independently of the grouping. `drank_lagrangian` "
            "prices a rank at `out + in`, so it is expected to move budget between matrix families "
            "wherever a group mixes shapes and to be inert wherever every shape in a group is equal"
        ),
        headers=[ "inner_allocation", *(f"place under {arm}" for arm in arms) ],
        rows=rows,
        notes=notes,
    )


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
    share its axes: stage 3's cap sweep, for one, matches stage 4's table exactly
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


def hold_shared(
        rows: List[Dict[str, Any]],
        dimension: str,
        across: str
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Drop values of `dimension` that not every value of `across` was run at.

    A mean rank over cells one arm has and another does not is not an ablation:
    the arm with the wider or better-behaved set wins on its extra cells rather
    than on the factor under test. Intersecting first is what makes an aggregate
    over arms mean "the same experiment, one factor changed".
    """
    per_arm: Dict[str, set] = defaultdict(set)

    for row in rows:
        if dimension in row and across in row:
            per_arm[dimension_text(across, row[across])].add(dimension_text(dimension, row[dimension]))

    if len(per_arm) < 2:
        return rows, []

    shared = set.intersection(*per_arm.values())
    kept = [
        row for row in rows
        if dimension not in row or dimension_text(dimension, row[dimension]) in shared
    ]

    if len(kept) == len(rows):
        return rows, []

    dropped = sorted(set.union(*per_arm.values()) - shared)

    if not shared:
        return rows, [
            f"no value of `{dimension}` was run under every `{across}`, so this table is not a "
            f"controlled comparison: {', '.join(dropped)} each appear under one arm only"
        ]

    return kept, [
        f"restricted to the `{dimension}` values every `{across}` was run at "
        f"({', '.join(sorted(shared))}); {', '.join(dropped)} ran under one arm only and would let "
        f"it win on cells the others never had"
    ]


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


def environment_notes(runs: List[Dict[str, Any]]) -> List[str]:
    """
    Runs a table mixes that were not produced on the same machine.

    The whitening cache is an input to the allocation, not an output of it, so a
    run whose cache came from a different machine is a different experiment. On
    LLaMA-7B the two collected sets differ by 0.08 perplexity at ratio 0.2 and
    0.14 at 0.5 on allocations that agree to four decimals, which is larger than
    the entire spread the stage 2 grid was ranked on.
    """
    environments = sorted({str(run["environment"]) for run in runs if run.get("environment")})

    if len(environments) < 2:
        return []

    return [
        f"this table mixes {len(environments)} run environments ({', '.join(environments)}), whose "
        "whitening caches are not the same input: compare within one before comparing across",
    ]


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


def resolution_notes(pivot: List[PivotRow], ratios: List[float]) -> List[str]:
    """
    Ratios whose rows are too close together for the ranking to mean anything.

    A rank is only worth reading where the configurations it separates actually
    differ. On LLaMA-7B at ratio 0.2 the nine stage 2 cells spanned 0.03
    perplexity with three exact ties, and averaging that rank with the one from
    0.5 handed half the decision to a quantity this design cannot resolve
    """
    notes: List[str] = []

    for ratio in ratios:
        priced = [row.values[ratio] for row in pivot if row.values.get(ratio) is not None]

        if len(priced) < 2 or min(priced) <= 0.0:
            continue

        spread = max(priced) - min(priced)
        relative = spread / min(priced)

        if relative >= RESOLUTION_SPREAD:
            continue

        notes.append(
            f"the {len(priced)} rows at {axis_text(ratio)} span {spread:.2f} ({relative:.1%} of the best "
            "value), so the ranking at this ratio is not resolvable by this design and the mean rank is "
            "decided by the other ratios",
        )

    return notes


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

    extra_notes = resolution_notes(pivot, ratios)
    extra_notes += environment_notes([run for row in pivot for run in row.runs.values()])

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

    paired_runs = [
        run
        for row in gain_rows
        for pivot in ( row.het, row.hom )
        if pivot is not None
        for run in pivot.runs.values()
    ]

    return Table(
        title=title,
        purpose=purpose,
        headers=headers,
        rows=body,
        notes=[
            *(notes or []),
            *environment_notes(paired_runs),
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
        # A stage may be previewed more than once, under a suffixed directory
        # such as `stage4_knobs`, and both still belong to stage 4
        match = re.fullmatch(r"stage(\d+[a-z]?)(?:[_-].*)?", path.name)

        # Sorted order puts the unsuffixed directory first, and that one is the
        # canonical preview of its stage
        if not path.is_dir() or match is None or match.group(1) in stages:
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

def dense_perplexity_notes(row: Dict[str, Any], rows: List[Dict[str, Any]]) -> List[str]:
    """
    Stage 1's own check: wikitext and c4 must differ on the dense baseline.

    Identical values are the signature of the c4 task re-evaluating wikitext, in
    which case nothing downstream measures what it claims to. The screening
    stages deliberately evaluate wikitext alone, so a corpus with no c4 anywhere
    is waiting for the full suite rather than broken, and only a corpus that has
    c4 elsewhere but not here is missing something.
    """
    wikitext = as_float(row.get("wikitext"))
    c4 = as_float(row.get("c4"))

    if wikitext is None:
        return ["the dense baseline has no wikitext perplexity, so nothing below is measured against a floor"]

    if c4 is None:
        if not any(as_float(other.get("c4")) is not None for other in rows):
            return [
                "c4 was not evaluated in these runs, which is expected while screening: the check that "
                "wikitext and c4 differ runs when the full suite does",
            ]

        return ["other runs carry a c4 perplexity but the dense baseline does not, so the check cannot run"]

    if abs(wikitext - c4) < 1e-6:
        return [
            f"dense wikitext and c4 perplexity are identical ({wikitext:.4f}), the signature of the "
            "c4 task re-evaluating wikitext",
        ]

    return [f"dense wikitext {wikitext:.4f} differs from c4 {c4:.4f}, so the two tasks are distinct"]


def gate_stage1_anchors(context: GateContext) -> GateResult:
    """Stage 1: the dense floor and the homogeneous anchors every later table is read against"""
    family = anchor_matrices(context.rows)

    def select(row: Dict[str, Any]) -> bool:
        if row.get("is_original"):
            return True

        return (
            is_compression_run(row)
            and row.get("scheme") == "hom"
            and bypasses_nothing(row)
            and is_anchor_family(row, family)
        )

    rows = sorted((row for row in context.rows if select(row)), key=sort_rows_hierarchical)
    resolved: Dict[str, Resolution] = {}
    notes: List[str] = []
    body: List[List[Cell]] = []

    if family is not None:
        notes.append(
            f"held at the `{family}` selection the grid is run at, so stage 6b's narrower families "
            "cannot stand in for an anchor a heterogeneous run is measured against",
        )

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
            notes += dense_perplexity_notes(row, context.rows)
            continue

        path = str(row.get("checkpoint_path") or "")

        if path and target is not None:
            # One homogeneous run per ratio, so nothing competed for the role
            resolved[f"__CKPT_HOM_{axis_text(target)}__"] = Resolution(value=path, candidates=None)

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

    resolved: Dict[str, Resolution] = {}
    groupings = best_by(pivot, 0)

    if not groupings:
        return GateResult(tables=tables, resolved=resolved)

    best_grouping = groupings[0][0]
    resolved["__BEST_GROUPING__"] = Resolution(value=best_grouping, candidates=len(groupings))
    flat = [value for value, _ in groupings if value in FLAT_GROUPINGS]

    if flat:
        resolved["__BEST_FLAT_GROUPING__"] = Resolution(value=flat[0], candidates=len(flat))

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
            resolved[placeholder] = Resolution(value=scores[index][0], candidates=len(scores))

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
    Stages 2b and 2c: a score family measured against the incumbent scores.

    Both incumbents are pulled into the same table on purpose. These gates are
    promotion tests rather than rankings of a family on its own, and every stage
    from 3 on holds its scores as placeholders precisely so that either of them
    can still move here.
    """
    axes = ( "score_metric", )
    grouping = resolved_value(context.resolved, "__BEST_GROUPING__")
    incumbents = [resolved_value(context.resolved, name) for name in SCORE_PLACEHOLDERS]
    incumbents = [score for score in incumbents if score]

    def select(row: Dict[str, Any]) -> bool:
        score = str(row.get("score_metric") or "")

        return (
            is_baseline_heterogeneous(row)
            and row.get("inner_allocation") == DEFAULT_INNER_ALLOCATION
            and ( grouping is None or row.get("group_criterion") == grouping )
            and ( belongs(score) or score in incumbents )
        )

    rows, skipped = gate_rows(context.rows, axes, select)
    rows, held = hold_dominant(rows, "max_ratio")
    pivot = build_pivot(rows, axes, context.metric)
    resolved: Dict[str, Resolution] = {}
    notes = [ *skipped_note(skipped), *held, *confound_notes(rows, axes) ]

    if grouping is not None:
        notes.append(f"held at `--group_criterion {grouping}`, as stage 2 resolved it")

    ranked = [row.key[0] for row in pivot if row.mean_rank is not None]

    if ranked and not incumbents:
        notes.append("stage 2 has not resolved the scores yet, so nothing is promoted from here")
    elif ranked:
        for index, placeholder in enumerate(SCORE_PLACEHOLDERS):
            if index >= len(ranked) or ranked[index] == resolved_value(context.resolved, placeholder):
                continue

            resolved[placeholder] = Resolution(value=ranked[index], candidates=len(ranked))
            notes.append(
                f"`{ranked[index]}` takes `{placeholder}` from "
                f"`{resolved_value(context.resolved, placeholder)}` on mean rank",
            )

        if not resolved:
            notes.append(f"the incumbents hold, so {' and '.join(SCORE_PLACEHOLDERS)} are unchanged")

        # The checkpoint role follows whichever run now holds the top score
        if SCORE_PLACEHOLDERS[0] in resolved:
            resolved.update(best_checkpoints(pivot, "__CKPT_BEST_SCORE"))

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


def gate_stage3c_outer(context: GateContext) -> GateResult:
    """
    Stage 3c: the outer level, which is the thesis contribution's own test.

    This gate owns `__BEST_GROUPING__` from here on. Stage 2 can only nominate
    among the flat criteria, so leaving the placeholder with it would hold every
    later stage at `decoder` however far ahead `hierarchical` finished — the
    outer level is not on stage 2's ballot.
    """
    # The score stays an axis rather than being fixed: the ablation is only
    # controlled when the two groupings are compared at the same score
    axes = ( "group_criterion", "outer_allocation", "inner_allocation", "score_metric" )

    def select(row: Dict[str, Any]) -> bool:
        return is_baseline_heterogeneous(row) and row.get("group_criterion") in BLOCK_GROUPINGS

    rows, skipped = gate_rows(context.rows, axes, select)
    rows, held = hold_dominant(rows, "max_ratio")

    # The inner policy has to be held, or the aggregate below compares an arm
    # carrying every policy against one carrying a single policy and reads the
    # difference as the outer level. `decoder` has the whole stage-4 block behind
    # it, including two allocations that cost eleven and twenty perplexity
    rows, inner_held = hold_dominant(rows, "inner_allocation")
    rows, score_held = hold_shared(rows, "score_metric", "group_criterion")
    pivot = build_pivot(rows, axes, context.metric)

    tables = [pivot_table(
        title="Stage 3c gate: the outer level",
        purpose=(
            "`decoder` + `param_share` against `hierarchical` + `waterfill`. The two criteria bucket "
            "matrices identically and differ only in whether Block Influence may move budget between "
            "blocks, which makes this the thesis contribution's own controlled test"
        ),
        pivot=pivot,
        axis_headers=[ "group_criterion", "outer_allocation", "inner_allocation", "score_metric" ],
        metric=context.metric,
        baselines=homogeneous_baselines(context.rows, context.metric),
        notes=[
            *skipped_note(skipped),
            *held,
            *inner_held,
            *score_held,
            "read `figures/ratio_tail.csv` beside this: the mechanism is whether `max_block_ratio` "
            "leaves layer 0, and a win without that movement is a win for something else",
        ],
    )]

    resolved: Dict[str, Resolution] = {}
    levels = best_by(pivot, 0)

    if levels:
        winner = levels[0][0]
        resolved["__BEST_GROUPING__"] = Resolution(value=winner, candidates=len(levels))
        resolved.update(best_checkpoints(pivot, "__CKPT_BEST_OUTER"))

        # The outer policy travels with the criterion that won, because
        # `hierarchical` + `param_share` reproduces `decoder` exactly and pairing
        # the winner with the wrong policy would silently undo the stage
        outer = [row.key[1] for row in pivot if row.key[0] == winner and row.mean_rank is not None]

        if outer:
            resolved["__BEST_OUTER__"] = Resolution(
                value=dominant_of(outer),
                candidates=len(set(outer)),
            )

        tables.append(aggregate_table(
            title="Stage 3c gate: grouping aggregate",
            purpose=(
                "Averaged over the scores, inner policies and ratios each criterion was run at, which "
                "is what re-resolves `__BEST_GROUPING__` and picks `__BEST_OUTER__` alongside it"
            ),
            header="group_criterion",
            ranked=levels,
        ))

    tables += offline_tables(
        context,
        "3c",
        figures=(
            (
                "ratio_tail",
                "Stage 3c preview: where the block tail sits",
                "`max_block_ratio` and its layer. The outer level earns its place by moving the worst "
                "block off layer 0, and this is where that is visible before any GPU time is spent",
            ),
        ),
    )

    return GateResult(tables=tables, resolved=resolved)


def gate_stage4_policies(context: GateContext) -> GateResult:
    """Stage 4 (RQ3): whether the policy spending a group budget matters apart from the score"""
    top_scores = [resolved_value(context.resolved, name) for name in ( "__TOP1_SCORE__", "__TOP2_SCORE__" )]
    top_scores = [score for score in top_scores if score]

    inner_axes = ( "inner_allocation", "score_metric" )

    def select_arm(arm: str) -> Callable[[Dict[str, Any]], bool]:
        def select(row: Dict[str, Any]) -> bool:
            return (
                is_baseline_heterogeneous(row)
                and row.get("group_criterion") == arm
                and ( not top_scores or row.get("score_metric") in top_scores )
            )

        return select

    # One panel per grouping arm. Pooling them would rank the policies through
    # the grouping instead of within it, and the two do not commute:
    # `drank_lagrangian` prices a rank at `out + in`, so it reallocates between
    # matrix families wherever a group mixes shapes and is inert wherever it
    # does not
    arms = sorted({
        dimension_text("group_criterion", row["group_criterion"])
        for row in context.rows
        if is_baseline_heterogeneous(row) and row.get("group_criterion")
    })

    tables: List[Table] = []
    pivots: Dict[str, List[PivotRow]] = {}

    for arm in arms:
        arm_rows, arm_skipped = gate_rows(context.rows, inner_axes, select_arm(arm))
        arm_rows, arm_held = hold_dominant(arm_rows, "max_ratio")
        arm_pivot = build_pivot(arm_rows, inner_axes, context.metric)

        if not arm_pivot:
            continue

        pivots[arm] = arm_pivot
        tables.append(pivot_table(
            title=f"Stage 4 gate: inner allocation policies under `{arm}`",
            purpose=(
                f"The inner policies at `{arm}`, with the outer level whatever that criterion carries. "
                "Their aggressiveness cannot be matched — only `--softmax_temp` is a live knob — so "
                "report the dispersion beside the result rather than implying it was controlled"
            ),
            pivot=arm_pivot,
            axis_headers=[ "inner_allocation", "score_metric" ],
            metric=context.metric,
            baselines=homogeneous_baselines(context.rows, context.metric),
            notes=[ *skipped_note(arm_skipped), *arm_held, *confound_notes(arm_rows, inner_axes) ],
        ))

    resolved: Dict[str, Resolution] = {}
    grouping = resolved_value(context.resolved, "__BEST_GROUPING__")
    deciding = pivots.get(grouping or "", [])

    if deciding:
        policies = best_by(deciding, 0)

        if policies:
            resolved["__BEST_INNER__"] = Resolution(value=policies[0][0], candidates=len(policies))
            resolved.update(best_checkpoints(deciding, "__CKPT_BEST_POLICY"))
            tables.append(aggregate_table(
                title="Stage 4 gate: inner policy aggregate",
                purpose=(
                    f"Averaged over the scores and ratios each policy was run at, under `{grouping}` "
                    "alone. `__BEST_INNER__` is read from the arm that will actually be used, because "
                    "the policy ranking flips between arms"
                ),
                header="inner_allocation",
                ranked=policies,
            ))

    if len(pivots) > 1:
        tables.append(policy_interaction_table(pivots))

    tables += offline_tables(
        context,
        "4",
        figures=(
            (
                "dispersion",
                "Stage 4 preview: ratio dispersion",
                "Compare how aggressively each policy spreads its ratios. `--offset` moves none of "
                "them and only `--softmax_temp` is live, so where two cannot be matched, report the "
                "dispersion beside the result rather than implying it was controlled",
            ),
            (
                "ratio_by_type",
                "Stage 4 preview: mean ratio per matrix family",
                "Where a rank-space policy's bias against mixed shapes becomes visible, thesis 3.3.5",
            ),
        ),
    )

    return GateResult(tables=tables, resolved=resolved)


def gate_stage3_cap(context: GateContext) -> GateResult:
    """Stage 3: `--max_ratio`, which Swift-SVD reports as first-order rather than a guard rail"""
    axes = ( "max_ratio", )
    grouping = resolved_value(context.resolved, "__BEST_GROUPING__")
    score = resolved_value(context.resolved, "__TOP1_SCORE__")
    inner = resolved_value(context.resolved, "__BEST_INNER__")

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
    resolved: Dict[str, Resolution] = {}
    notes += confound_notes(rows, axes)

    if pivot:
        resolved["--max_ratio"] = Resolution(value=pivot[0].key[0], candidates=len(pivot))
        notes.append(
            f"cap `{pivot[0].key[0]}` wins on mean rank. If it is below the 0.9 default, thesis 3.3 has "
            "to call the cap a first-order hyperparameter, and stages 5, 6 and 8 need it in `args/base_args.json`",
        )

    tables = [pivot_table(
        title="Stage 3 gate: the per-matrix cap",
        purpose="How far a single matrix may be compressed, at the configuration stages 2 and 3 chose",
        pivot=pivot,
        axis_headers=[ "max_ratio" ],
        metric=context.metric,
        baselines=homogeneous_baselines(context.rows, context.metric),
        notes=notes,
    )]

    tables += offline_tables(
        context,
        "3",
        figures=(
            (
                "cap_binding",
                "Stage 3 preview: how many matrices each cap pins",
                "A cap pinning nothing cannot change an allocation, so it never needed a run",
            ),
        ),
    )

    return GateResult(tables=tables, resolved=resolved)


def varying_labels(keys: List[Tuple[str, ...]], dimensions: Tuple[str, ...]) -> Dict[Tuple[str, ...], str]:
    """
    Name each configuration by the dimensions that actually differ between them.

    A replicate table is keyed on every dimension so that two runs are only
    pooled when they truly match, but printing all of them makes the row
    unreadable and hides the one or two that identify it.
    """
    # Compared within one scheme, not across: a heterogeneous key differs from a
    # homogeneous one on every allocation dimension simply because the second has
    # none, and listing all of them says nothing about which row this is
    scheme = dimensions.index("scheme") if "scheme" in dimensions else 0
    families: Dict[str, List[Tuple[str, ...]]] = defaultdict(list)

    for key in keys:
        families[key[scheme]].append(key)

    labels: Dict[Tuple[str, ...], str] = {}

    for family, members in families.items():
        varying = [
            index for index in range(len(dimensions))
            if index != scheme and len({key[index] for key in members}) > 1
        ]

        for key in members:
            parts = [key[index] for index in varying if key[index] not in ( "", "--", "none", "None" )]
            labels[key] = " / ".join([ family, *parts ])

    return labels


def gate_stage1b_replicates(context: GateContext) -> GateResult:
    """
    Stage 1b: how far apart two runs of the same configuration land.

    The pipeline is deterministic given its whitening cache, so the spread this
    reports is the spread of the calibration draw. It resolves nothing and is
    read by every later table: a difference smaller than this is not a result.
    """
    # Two runs are replicates only if every dimension except the seed agrees.
    # Keying on fewer than all of them lets a swept axis masquerade as noise: on
    # the pilot corpus, keying on the grouping and score alone reported the cap
    # sweep's six-perplexity range as the resolution of the grid
    identity = ( "scheme", "matrices", *(d for d in GATE_DIMENSIONS if d != "seed") )
    rows = [row for row in context.rows if is_compression_run(row)]
    grouped: Dict[Tuple[str, ...], Dict[float, List[Tuple[str, float]]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        value = as_float(row.get(context.metric))
        ratio = as_float(row.get("compression_ratio"))

        if value is None or ratio is None:
            continue

        key = tuple(dimension_text(axis, row.get(axis)) for axis in identity)
        grouped[key][ratio].append(( dimension_text("seed", row.get("seed")), value ))

    body: List[List[Cell]] = []
    widest = 0.0
    labels = varying_labels(sorted(grouped), identity)

    for key in sorted(grouped):
        for ratio in sorted(grouped[key]):
            seeded = grouped[key][ratio]
            values = [value for _, value in seeded]

            # Two runs of one configuration at one seed are the same run written
            # twice, not a replicate, so the seeds have to actually differ
            if len(values) < 2 or len({seed for seed, _ in seeded}) < 2:
                continue

            spread = max(values) - min(values)
            widest = max(widest, spread)
            body.append([
                Cell(text=labels[key]),
                Cell(text=axis_text(ratio)),
                Cell(text=str(len({seed for seed, _ in seeded}))),
                Cell(text=fmt_ppl(min(values))),
                Cell(text=fmt_ppl(max(values))),
                Cell(text=f"{spread:.3f}", bold=True),
            ])

    notes = [
        "a row needs two seeds of one otherwise identical configuration, so an empty table means the "
        "replicates have not been run, or that they reused the shared whitening directory and collapsed "
        "onto the same allocation",
    ]

    if body:
        notes.append(
            f"the widest spread is {widest:.3f}: treat any later difference below it as unresolved, and "
            "read the low-spread notes on the other gates against this number rather than against 1%",
        )

    tables = [Table(
        title="Stage 1b gate: the replicate floor",
        purpose=(
            "The same configuration run at more than one seed, which is the resolution of every "
            "comparison in the grid. Each seed needs its own whitening directory or it reports zero"
        ),
        headers=[ "configuration", "ratio", "seeds", "best", "worst", "spread" ],
        rows=body,
        notes=notes,
    )]

    return GateResult(tables=tables, resolved={})


def gate_stage4c_outer_offset(context: GateContext) -> GateResult:
    """Stage 4c: how hard the outer level is allowed to reweight depth"""
    axes = ( "outer_offset", )
    score = resolved_value(context.resolved, "__TOP1_SCORE__")
    inner = resolved_value(context.resolved, "__BEST_INNER__")

    def select(row: Dict[str, Any]) -> bool:
        return (
            is_baseline_heterogeneous(row)
            and row.get("outer_allocation") != DEFAULT_OUTER_ALLOCATION
            and ( score is None or row.get("score_metric") == score )
            and ( inner is None or row.get("inner_allocation") == inner )
        )

    rows, skipped = gate_rows(context.rows, axes, select)
    rows, held = hold_dominant(rows, "max_ratio")
    pivot = build_pivot(rows, axes, context.metric)

    tables = [pivot_table(
        title="Stage 4c gate: the outer offset ladder",
        purpose=(
            "The knob that sets how far Block Influence may move budget between blocks. It is not "
            "monotone in danger: the default sits near a minimum of the worst block ratio, and the "
            "largest values converge back onto `param_share`"
        ),
        pivot=pivot,
        axis_headers=[ "outer_offset" ],
        metric=context.metric,
        baselines=homogeneous_baselines(context.rows, context.metric),
        notes=[
            *skipped_note(skipped),
            *held,
            *confound_notes(rows, axes),
            "check the optimum against `max_block_ratio` from the offline ladder, and check whether it "
            "is the same at both ratios: if it is not, the knob is budget-dependent",
        ],
    )]

    resolved: Dict[str, Resolution] = {}

    if pivot:
        resolved["__BEST_OUTER_OFFSET__"] = Resolution(value=pivot[0].key[0], candidates=len(pivot))

    tables += offline_tables(context, "4c")

    return GateResult(tables=tables, resolved=resolved)


def gate_stage4d_temperature(context: GateContext) -> GateResult:
    """Stage 4d: the one live inner-policy knob, under both depth regimes"""
    axes = ( "group_criterion", "softmax_temp" )

    def select(row: Dict[str, Any]) -> bool:
        return is_baseline_heterogeneous(row) and row.get("inner_allocation") == "softmax_temp"

    rows, skipped = gate_rows(context.rows, axes, select)
    rows, held = hold_dominant(rows, "max_ratio")

    # The stage runs one score, so a second one inside the selection belongs to
    # stage 4 and would price the temperature together with the score
    rows, score_held = hold_at(rows, "score_metric", resolved_value(context.resolved, "__TOP1_SCORE__"))
    pivot = build_pivot(rows, axes, context.metric)

    tables = [pivot_table(
        title="Stage 4d gate: the temperature ladder",
        purpose=(
            "`--softmax_temp` under a grouping that flattens depth and one that reweights it. The "
            "optimum is not expected to agree between the two, because the outer level has already "
            "spent the depth budget in the second"
        ),
        pivot=pivot,
        axis_headers=[ "group_criterion", "softmax_temp" ],
        metric=context.metric,
        baselines=homogeneous_baselines(context.rows, context.metric),
        notes=[
            *skipped_note(skipped),
            *held,
            *score_held,
            *confound_notes(rows, axes),
            "a temperature the offline screen rejected should be absent here; if it is present, it was "
            "run against the screen and the row is a finding about the policy rather than a candidate",
        ],
    )]

    tables += offline_tables(context, "4d")

    return GateResult(tables=tables, resolved={})


def gate_stage6b_family_budget(context: GateContext) -> GateResult:
    """Stage 6b: which matrix families the budget should land on, selection by selection"""
    axes = ( "matrices", )

    def select(row: Dict[str, Any]) -> bool:
        return is_compression_run(row) and row.get("scheme") == "hom" and bypasses_nothing(row)

    rows, skipped = gate_rows(context.rows, axes, select)
    rows, held = hold_dominant(rows, "ratio_scope")
    pivot = build_pivot(rows, axes, context.metric)

    tables = [pivot_table(
        title="Stage 6b gate: the family budget",
        purpose=(
            "Homogeneous runs whose only difference is which families are compressed, with "
            "`--ratio_scope all` holding the global removed parameters at the target. The score and the "
            "policy are absent on purpose, so the family axis moves alone"
        ),
        pivot=pivot,
        axis_headers=[ "matrices" ],
        metric=context.metric,
        baselines=homogeneous_baselines(context.rows, context.metric),
        notes=[
            *skipped_note(skipped),
            *held,
            *confound_notes(rows, axes),
            "the all-families row is the stage 1 anchor, so a selection beating it means the budget is "
            "better spent narrowly; `max_block_ratio` is a fraction of the selection here and its "
            "threshold does not apply",
        ],
    )]

    tables += offline_tables(context, "6b")

    return GateResult(tables=tables, resolved={})


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
    resolved: Dict[str, Resolution] = {}

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

    if bypassing:
        # Sorted by mean gain, so the setting itself comes from the head even
        # when its checkpoint has already been cleaned up
        for placeholder, position in ( ( "__BEST_BYPASS_EARLY__", 0 ), ( "__BEST_BYPASS_LATE__", 1 ) ):
            resolved[placeholder] = Resolution(value=bypassing[0].key[position], candidates=len(bypassing))

    for row in bypassing:
        path = str(row.het.runs[0.2].get("checkpoint_path") or "") if row.het is not None and 0.2 in row.het.runs else ""

        if path:
            resolved["__CKPT_BEST_BYPASS_0.2__"] = Resolution(value=path, candidates=len(bypassing))
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

    grouping = resolved_value(context.resolved, "__BEST_FLAT_GROUPING__")
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

    resolved: Dict[str, Resolution] = {}
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
    # `outer_allocation` is an axis rather than a held dimension because a finalist
    # naming `hierarchical` is only reproducible with the outer policy that won
    # beside it: `hierarchical` + `param_share` is `decoder` under another name
    axes = ( "group_criterion", "score_metric", "inner_allocation", "outer_allocation" )
    rows, skipped = gate_rows(context.rows, axes, is_baseline_heterogeneous)
    rows, held = hold_dominant(rows, "max_ratio")
    pivot = build_pivot(rows, axes, context.metric)
    finalists = pivot[:3]

    resolved: Dict[str, Resolution] = {}

    for index, row in enumerate(finalists, start=1):
        for role, position in ( ( "GROUPING", 0 ), ( "SCORE", 1 ), ( "INNER", 2 ), ( "OUTER", 3 ) ):
            candidates = len({other.key[position] for other in pivot})
            resolved[f"__FINALIST{index}_{role}__"] = Resolution(value=row.key[position], candidates=candidates)

    if finalists:
        for ratio, run in sorted(finalists[0].runs.items()):
            path = str(run.get("checkpoint_path") or "")
            placeholder = f"__CKPT_HET_{axis_text(ratio)}__"

            if path and placeholder in PLACEHOLDER_SOURCES:
                resolved[placeholder] = Resolution(value=path, candidates=len(pivot))

        # Derived, not ranked: stage 7c pairs the score fix with the other two,
        # and pairing it with anything but the finalist's own score would change
        # two things at once
        winner = str(finalists[0].key[1])

        if winner in RELATIVE_SCORE_BASES:
            resolved["__FINALIST1_SCORE_REL__"] = Resolution(
                value=f"{winner}{RELATIVE_SCORE_SUFFIX}",
                candidates=len(pivot),
            )

    tables = [pivot_table(
        title="Stage 7 gate: finalists",
        purpose=(
            "Every heterogeneous configuration collected so far, ranked together. The top three fill the "
            "`__FINALIST*__` placeholders, and the first also answers `__CKPT_HET_*__` for stage 10"
        ),
        pivot=pivot,
        axis_headers=[ "group_criterion", "score_metric", "inner_allocation", "outer_allocation" ],
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


def carries_gqa_fix(row: Dict[str, Any]) -> bool:
    """
    Whether a run uses one of the three grouped-query repairs.

    Self-selecting on the fix rather than on the model, so the gate stays empty
    on a multi-head grid instead of answering stage 7d out of runs that never
    had the defect to repair
    """
    score = str(row.get("score_metric") or "")
    floor = row.get("min_rank_fraction")

    return (
        score.endswith(RELATIVE_SCORE_SUFFIX)
        or bool(row.get("head_block_svd"))
        or ( floor is not None and float(floor) > 0.0 )
    )


def gate_stage7c_gqa_fixes(context: GateContext) -> GateResult:
    """Stage 7c: which of the three grouped-query repairs earns its place"""
    # The fixes are the axes; the allocation around them is held, because stage
    # 7c varies what repairs a configuration rather than which configuration it is
    axes = ( "score_metric", "min_rank_fraction", "head_block_svd" )
    rows, skipped = gate_rows(context.rows, axes, carries_gqa_fix)
    notes = list(skipped_note(skipped))

    for dimension in ( "group_criterion", "inner_allocation", "outer_allocation", "max_ratio" ):
        rows, held = hold_dominant(rows, dimension)
        notes += held

    pivot = build_pivot(rows, axes, context.metric)
    winners = pivot[:2]
    resolved: Dict[str, Resolution] = {}
    notes += confound_notes(rows, axes)

    for index, row in enumerate(winners, start=1):
        sample = next(iter(row.runs.values()), {})
        resolved[f"__GQA_WINNER{index}_SCORE__"] = Resolution(
            value=row.key[0],
            candidates=len({other.key[0] for other in pivot}),
        )

        # Held rather than ranked, so they are read off the winning run instead
        # of a pivot key that does not carry them
        for role, dimension in (
            ( "GROUPING", "group_criterion" ),
            ( "INNER", "inner_allocation" ),
            ( "OUTER", "outer_allocation" ),
        ):
            value = sample.get(dimension)

            if value is not None:
                resolved[f"__GQA_WINNER{index}_{role}__"] = Resolution(value=str(value), candidates=1)

    if pivot:
        notes.append(
            "a repair that does not beat the homogeneous anchor of its own model is a negative result and "
            "stays off. `--head_block_svd` and `--min_rank_fraction` are off by default, so reporting one "
            "as unhelpful costs nothing downstream",
        )
        notes.append(
            "the `__GQA_WINNER*__` placeholders name the allocation only. If a winning row sets "
            "`min_rank_fraction` or `head_block_svd`, add that flag to stage 7d by hand when resolving",
        )

    tables = [pivot_table(
        title="Stage 7c gate: the grouped-query repairs",
        purpose=(
            "Every run carrying a shape-invariant score, a shape-aware rank floor or a head-block "
            "factorization, ranked against the homogeneous anchors of the same model. The top two fill "
            "the `__GQA_WINNER*__` placeholders that stage 7d carries onto a second sharing factor"
        ),
        pivot=pivot,
        axis_headers=[ "score_metric", "min_rank_fraction", "head_block_svd" ],
        metric=context.metric,
        baselines=homogeneous_baselines(context.rows, context.metric),
        notes=notes,
    )]

    tables += offline_tables(context, "7c")

    return GateResult(tables=tables, resolved=resolved)


def gate_stage8_curve(context: GateContext) -> GateResult:
    """Stage 8 (RQ1): how the heterogeneous gain depends on the target ratio"""
    axes = ( "group_criterion", "score_metric", "inner_allocation" )
    rows, _ = gate_rows(context.rows, axes, is_baseline_heterogeneous)
    notes: List[str] = []

    # Stages 3c and 4c swept these around the same allocation, and the sweeps
    # only exist at the two screening budgets. Without holding them, a ratio the
    # sweep happened to cover is represented by whichever arm's name sorts first
    # rather than by the arm its neighbours are traced at, which puts a kink in
    # the curve at exactly the budgets the rest of the grid is measured on
    for dimension in ( "max_ratio", "outer_offset", "softmax_temp" ):
        rows, held = hold_dominant(rows, dimension)
        notes += held

    # Carried in the key so the table names the arm it is traced at, and so a
    # value that escapes the hold above appears as its own row instead of
    # silently standing in for the held one
    # `outer_allocation` belongs here for the reason stage 7 gives: `hierarchical`
    # with `param_share` is `decoder` under another name, so a row naming the
    # grouping without it does not identify a configuration
    labelled = axes + ( "outer_allocation", "outer_offset", "softmax_temp" )
    pivot = build_pivot(rows, labelled, context.metric)
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

    notes += [
        "the best heterogeneous configuration is picked per ratio, so the column may change configuration "
        "between rows, which is itself the answer if it does",
        *confound_notes(rows, labelled),
    ]

    if len(gains) > 1:
        trend = "widens" if gains[-1] > gains[0] else "narrows"
        notes.append(
            f"the gain goes from {gains[0]:+.2f} at the lowest ratio to {gains[-1]:+.2f} at the highest, so it "
            f"{trend} with the budget. At the high end read this together with stage 3 and `cap_binding.csv`",
        )

    tables = [Table(
        title="Stage 8 gate: the ratio curve",
        purpose="Both arms across every collected budget, which is the shape RQ1 asks about rather than a slope",
        headers=[
            "ratio",
            f"hom {context.metric}",
            f"best het {context.metric}",
            "gain",
            "grouping / score / inner / outer / outer_offset / softmax_temp",
        ],
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
        path = resolved_value(context.resolved, placeholder) or ""
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
    gate_stage1b_replicates,
    gate_stage2_score_grouping,
    gate_stage2b_squared,
    gate_stage2c_schatten,
    gate_stage3_cap,
    gate_stage3c_outer,
    gate_stage4_policies,
    gate_stage4c_outer_offset,
    gate_stage4d_temperature,
    gate_stage5_bypass,
    gate_stage6_composite,
    gate_stage6b_family_budget,
    gate_stage7_finalists,
    gate_stage7c_gqa_fixes,
    gate_stage8_curve,
    gate_stage9_roster,
    gate_stage10_lora,
)


def resolution_status(resolution: Optional[Resolution]) -> str:
    """
    Whether a value was decided or merely defaulted.

    A gate whose deciding axis held one entrant produces a real value that reads
    as settled. Saying how many candidates it beat is what stops a stage from
    being carried forward on the strength of the only run that had happened yet
    """
    if resolution is None:
        return "waiting on runs"

    if resolution.candidates is None:
        return "ready"

    if resolution.candidates <= 1:
        return "provisional (1 candidate)"

    return f"ready ({resolution.candidates} candidates)"


def placeholder_table(resolved: Dict[str, Resolution]) -> Table:
    """
    The one table this report exists for: what to paste into the next stage file.

    Unresolved rows are listed too, so a gate still waiting on runs reads as
    waiting rather than as absent.
    """
    body: List[List[Cell]] = []

    for placeholder, source in PLACEHOLDER_SOURCES.items():
        resolution = resolved.get(placeholder)

        body.append([
            Cell(text=placeholder),
            Cell(text=resolution.value if resolution is not None else "not resolved yet", bold=resolution is not None),
            Cell(text=source),
            Cell(text=resolution_status(resolution)),
        ])

    for placeholder in sorted(set(resolved) - set(PLACEHOLDER_SOURCES)):
        body.append([
            Cell(text=placeholder),
            Cell(text=resolved[placeholder].value, bold=True),
            Cell(text="resolved by a gate, no stage file asks for it"),
            Cell(text=resolution_status(resolved[placeholder])),
        ])

    return Table(
        title="Placeholders",
        purpose="Every value a stage file waits on, and what the collected runs resolve it to",
        headers=[ "placeholder", "value", "resolved by", "status" ],
        rows=body,
        notes=[
            "`run_experiments.py` refuses to start while a placeholder is unresolved, so a `waiting on runs` "
            "row is a stage that cannot run yet rather than a defaulted one",
            "a `provisional` row was decided by a gate whose deciding axis held a single candidate, so it "
            "reports the only run that has happened rather than a winner: rerun the stage before trusting it",
        ],
    )


def gate_tables_for_model(
        rows: List[Dict[str, Any]],
        offline: Dict[str, OfflineStage],
        metric: str
) -> List[Table]:
    """Run every gate in stage order, so a later one can read what an earlier one resolved"""
    resolved: Dict[str, Resolution] = {}
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
