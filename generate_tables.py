#!/usr/bin/env python3

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


LIKELIHOOD_BENCHMARK_ORDER = [
    "wikitext",
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

BENCHMARK_ORDER = LIKELIHOOD_BENCHMARK_ORDER + GENERATION_BENCHMARK_ORDER

BENCHMARK_ALIASES = {
    "gsm8k": ["gsm8k", "gsm8k_cot"],
    "truthfulqa_gen": ["truthfulqa_gen"],
}

BENCHMARK_LABELS_MD = {
    "wikitext": "WikiText ppl ↓",
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
    "type": "matrix_type",
    "matrix": "matrix",
    "matrix_type": "matrix",
    "matrixtype": "matrix",
}

SCORING_TOKENS = {
    "truncation": "truncation_loss",
    "truncation_loss": "truncation_loss",
    "entropy": "entropy",
}

SCHEME_TOKENS = {
    "het": "het",
    "heterogeneous": "het",
    "hom": "hom",
    "homogeneous": "hom",
}

DENOMINATOR_TOKENS = {
    "all": "all",
    "selected": "selected",
}

PREFERRED_METRICS = [
    "acc_norm,none",
    "acc,none",
]

GENERATION_METRICS = {
    "gsm8k": [
        "exact_match,strict-match",
        "exact_match,none",
        "acc,none",
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

def compressed_rows_only(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if not row.get("is_original", False)]

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
        value = float(value)
    except Exception:
        return None

    if math.isnan(value):
        return None

    return value


def fmt_accuracy(value: Any, decimals: int = 2) -> str:
    value = safe_float(value)
    if value is None:
        return "--"
    return f"{100.0 * value:.{decimals}f}"


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


def find_scoring(tokens: List[str]) -> Tuple[str, Optional[int], int]:
    """
    Returns:
        scoring_name, scoring_start_idx, scoring_token_count

    Handles both:
        truncation
        truncation_loss

    If the filename is split by underscores, then "truncation_loss"
    appears as ["truncation", "loss"].
    """

    for i, tok in enumerate(tokens):
        if tok == "truncation":
            if i + 1 < len(tokens) and tokens[i + 1] == "loss":
                return "truncation_loss", i, 2
            return "truncation_loss", i, 1

        if tok == "entropy":
            return "entropy", i, 1

        if tok == "truncation_loss":
            return "truncation_loss", i, 1

    return "unknown", None, 0


def parse_filename(path: Path) -> Dict[str, Any]:
    """
    Supported filename styles:

    Original / uncompressed:
        Qwen_Qwen2.5_32B.json

    Heterogeneous compression with denominator token:
        Qwen_Qwen2.5_32B_q_k_v_out_mlp_all_0.2_het_decoder_truncation_8_v2.json
        Qwen_Qwen2.5_32B_q_k_v_out_mlp_selected_0.2_het_decoder_truncation_8_v2.json

    Homogeneous compression with denominator token:
        huggyllama_llama_7b_q_k_v_out_mlp_all_0.2_hom_8_v2.json
        huggyllama_llama_7b_q_k_v_out_mlp_selected_0.2_hom_8_v2.json

    Older format without denominator token is also supported:
        Qwen_Qwen2.5_32B_q_k_v_out_mlp_0.2_het_decoder_truncation_8_v2.json

    Meaning of denominator:
        all:
            the whole parameter count was used as denominator for the compression ratio.

        selected:
            only the parameter count of the selected matrix types was used as denominator.
    """

    tokens = normalize_filename_stem(path)

    version = ""
    if tokens and re.fullmatch(r"v\d+", tokens[-1]):
        version = tokens.pop()

    has_ratio = any(is_float_token(tok) for tok in tokens)

    # Original model case.
    if not has_ratio:
        model_name = "_".join(tokens)
        return {
            "file": path.name,
            "model": model_name,
            "is_original": True,
            "matrices": "none",
            "compression_denominator": "original",
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

    compression_denominator = "--"

    # New format: denominator token appears after matrix tokens and before ratio.
    # Example:
    #   ..._q_k_v_out_mlp_all_0.2_...
    #   ..._q_k_v_out_mlp_selected_0.2_...
    if ratio_idx is not None and ratio_idx - 1 >= 0:
        maybe_denominator = tokens[ratio_idx - 1]
        if maybe_denominator in DENOMINATOR_TOKENS:
            compression_denominator = DENOMINATOR_TOKENS[maybe_denominator]
        # If it wasn't catched during compression, this case = "all" denominator always
        if all(matrix in matrices for matrix in ["q","k","v","out","mlp"]):
            compression_denominator = "all"

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
        # Heterogeneous format:
        # ... matrices denominator ratio het grouping scoring bypass version
        for tok in tokens:
            if tok in GROUPING_TOKENS:
                grouping = GROUPING_TOKENS[tok]

        scoring, scoring_idx, scoring_token_count = find_scoring(tokens)

        if scoring_idx is not None:
            bypass_idx = scoring_idx + scoring_token_count
            if bypass_idx < len(tokens) and is_int_token(tokens[bypass_idx]):
                bypassed_layers = int(tokens[bypass_idx])

    elif scheme == "hom":
        # Homogeneous format:
        # ... matrices denominator ratio hom bypass version
        grouping = "--"
        scoring = "--"

        if scheme_idx is not None:
            bypass_idx = scheme_idx + 1
            if bypass_idx < len(tokens) and is_int_token(tokens[bypass_idx]):
                bypassed_layers = int(tokens[bypass_idx])

    else:
        # Fallback for older or malformed names.
        grouping = "--"
        scoring = "--"

        for tok in tokens:
            if tok in GROUPING_TOKENS:
                grouping = GROUPING_TOKENS[tok]

        scoring, scoring_idx, scoring_token_count = find_scoring(tokens)

        if scoring_idx is not None:
            bypass_idx = scoring_idx + scoring_token_count
            if bypass_idx < len(tokens) and is_int_token(tokens[bypass_idx]):
                bypassed_layers = int(tokens[bypass_idx])

    return {
        "file": path.name,
        "model": model_name,
        "is_original": False,
        "matrices": "+".join(matrices) if matrices else "unknown",
        "compression_denominator": compression_denominator,
        "compression_ratio": compression_ratio,
        "scheme": scheme,
        "grouping": grouping,
        "scoring": scoring,
        "bypassed_layers": bypassed_layers,
        "filename_version": version,
    }


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
    for task_name in BENCHMARK_ALIASES.get(benchmark, [benchmark]):
        task_result = results.get(task_name)
        if task_result is not None:
            return task_result
    return None


def load_result(path: Path, prefer_lm_eval_model_name: bool = False) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    row = parse_filename(path)

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

        if benchmark == "wikitext":
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

    row["avg_accuracy"] = sum(acc_values) / len(acc_values) if acc_values else None
    row["metric_used"] = metric_used

    return row


def sort_rows_hierarchical(row: Dict[str, Any]):
    is_original = row.get("is_original", False)

    ratio = row.get("compression_ratio")
    ratio_sort = ratio if ratio is not None else 999.0

    return (
        row.get("model", ""),
        0 if is_original else 1,
        ratio_sort,
        row.get("bypassed_layers", 0),
        row.get("scheme", ""),
        row.get("compression_denominator", ""),
        row.get("grouping", ""),
        row.get("scoring", ""),
        row.get("matrices", ""),
        row.get("file", ""),
    )


def group_rows_for_model(rows: List[Dict[str, Any]]):
    """
    Returns:
        original_rows, grouped_rows

    grouped_rows:
        compression ratio
          bypassed layers
            rows
    """

    original_rows = []
    grouped = defaultdict(lambda: defaultdict(list))

    for row in rows:
        if row.get("is_original", False):
            original_rows.append(row)
        else:
            grouped[row.get("compression_ratio")][row.get("bypassed_layers")].append(row)

    return sorted(original_rows, key=sort_rows_hierarchical), grouped


def metric_value_for_display(row: Dict[str, Any], benchmark: str) -> str:
    if benchmark == "wikitext":
        return fmt_ppl(row.get(benchmark))
    return fmt_accuracy(row.get(benchmark))


def best_values(rows: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """
    Finds best benchmark values inside a local group.

    For WikiText, lower is better.
    For all other displayed benchmark values and average, higher is better.
    """

    best = {}

    for benchmark in BENCHMARK_ORDER + ["avg_accuracy"]:
        values = []

        for row in rows:
            value = safe_float(row.get(benchmark))
            if value is not None:
                values.append(value)

        if not values:
            best[benchmark] = None
        elif benchmark == "wikitext":
            best[benchmark] = min(values)
        else:
            best[benchmark] = max(values)

    return best


def is_best(row: Dict[str, Any], benchmark: str, best: Dict[str, Optional[float]]) -> bool:
    value = safe_float(row.get(benchmark))
    best_value = best.get(benchmark)

    if value is None or best_value is None:
        return False

    return abs(value - best_value) < 1e-12


def method_label(row: Dict[str, Any]) -> str:
    return f"{row['grouping']} / {row['scoring']} / {row['scheme']} / {row['matrices']}"

def markdown_faded(value: Any) -> str:
    return f'<span style="color: #888888;">{markdown_escape(value)}</span>'

def make_markdown_table_for_model(model_name: str, rows: List[Dict[str, Any]]) -> str:
    rows = sorted(rows, key=sort_rows_hierarchical)
    table_best = best_values(compressed_rows_only(rows))
    original_rows, grouped = group_rows_for_model(rows)

    headers = [
        "Ratio",
        "Bypass",
        "Grouping",
        "Scoring",
        "Scheme",
        "Denom.",
        "Matrices",
    ]

    headers += [BENCHMARK_LABELS_MD[b] for b in LIKELIHOOD_BENCHMARK_ORDER]
    headers += ["Average ↑"]
    headers += [BENCHMARK_LABELS_MD[b] for b in GENERATION_BENCHMARK_ORDER]
    headers += ["File"]

    lines = []
    lines.append(f"## {model_name}")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    if original_rows:
        for row in original_rows:
            row_cells = [
                markdown_faded("0%"),
                markdown_faded("--"),
                markdown_faded("--"),
                markdown_faded("--"),
                markdown_faded("--"),
                markdown_faded("--"),
                markdown_faded("--"),
            ]

            for benchmark in LIKELIHOOD_BENCHMARK_ORDER:
                value = metric_value_for_display(row, benchmark)
                row_cells.append(markdown_faded(value))

            avg_value = fmt_accuracy(row.get("avg_accuracy"))
            row_cells.append(markdown_faded(avg_value))

            for benchmark in GENERATION_BENCHMARK_ORDER:
                value = metric_value_for_display(row, benchmark)
                row_cells.append(markdown_faded(value))

            row_cells.append(markdown_faded(row.get("file", "--")))

            lines.append("| " + " | ".join(row_cells) + " |")

    for ratio in sorted(grouped.keys(), key=lambda x: -1 if x is None else x):
        bypass_groups = grouped[ratio]

        first_ratio_row = True

        for bypass in sorted(bypass_groups.keys()):
            local_rows = sorted(bypass_groups[bypass], key=sort_rows_hierarchical)

            first_bypass_row = True

            for row in local_rows:
                row_cells = []

                row_cells.append(fmt_ratio_md(ratio) if first_ratio_row else "")
                row_cells.append(str(bypass) if first_bypass_row else "")
                row_cells.append(row.get("grouping", "--"))
                row_cells.append(row.get("scoring", "--"))
                row_cells.append(row.get("scheme", "--"))
                row_cells.append(row.get("compression_denominator", "--"))
                row_cells.append(row.get("matrices", "--"))

                for benchmark in LIKELIHOOD_BENCHMARK_ORDER:
                    value = metric_value_for_display(row, benchmark)
                    if is_best(row, benchmark, table_best):
                        value = f"**{value}**"
                    row_cells.append(value)

                avg_value = fmt_accuracy(row.get("avg_accuracy"))
                if is_best(row, "avg_accuracy", table_best):
                    avg_value = f"**{avg_value}**"
                row_cells.append(avg_value)

                for benchmark in GENERATION_BENCHMARK_ORDER:
                    value = metric_value_for_display(row, benchmark)
                    if is_best(row, benchmark, table_best):
                        value = f"**{value}**"
                    row_cells.append(value)

                row_cells.append(row.get("file", "--"))

                lines.append("| " + " | ".join(markdown_escape(v) for v in row_cells) + " |")

                first_ratio_row = False
                first_bypass_row = False

    return "\n".join(lines)


def latex_metric_cell(row: Dict[str, Any], benchmark: str, best: Dict[str, Optional[float]]) -> str:
    value = metric_value_for_display(row, benchmark)

    if is_best(row, benchmark, best):
        return rf"\textbf{{{latex_escape(value)}}}"

    return latex_escape(value)


def latex_average_cell(row: Dict[str, Any], best: Dict[str, Optional[float]]) -> str:
    value = fmt_accuracy(row.get("avg_accuracy"))

    if is_best(row, "avg_accuracy", best):
        return rf"\textbf{{{latex_escape(value)}}}"

    return latex_escape(value)


def make_latex_table_for_model(
    model_name: str,
    rows: List[Dict[str, Any]],
    table_size: str = r"\scriptsize",
    use_adjustbox: bool = True,
    table_width: float = 1.6
) -> str:
    rows = sorted(rows, key=sort_rows_hierarchical)
    table_best = best_values(compressed_rows_only(rows))
    original_rows, grouped = group_rows_for_model(rows)

    lines = []

    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(table_size)
    lines.append(r"\setlength{\tabcolsep}{3pt}")

    if use_adjustbox:
        adjustbox_str = r"\begin{adjustbox}{width=" + str(table_width) + r"\textwidth,center}"
        lines.append(adjustbox_str)

    benchmark_colspec = "r" * len(LIKELIHOOD_BENCHMARK_ORDER)
    generation_colspec = "r" * len(GENERATION_BENCHMARK_ORDER)
    colspec = rf"ll lllll | {benchmark_colspec} r {generation_colspec}"
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")

    config_end_col = 7
    likelihood_start_col = config_end_col + 1
    likelihood_end_col = likelihood_start_col + len(LIKELIHOOD_BENCHMARK_ORDER) - 1
    summary_col = likelihood_end_col + 1
    generation_start_col = summary_col + 1
    generation_end_col = generation_start_col + len(GENERATION_BENCHMARK_ORDER) - 1
    total_col_count = generation_end_col

    lines.append(
        r"\multicolumn{2}{c}{Compression} & "
        r"\multicolumn{5}{c}{Configuration} & "
        rf"\multicolumn{{{len(LIKELIHOOD_BENCHMARK_ORDER)}}}{{c}}{{Likelihood Benchmarks}} & "
        r"\multicolumn{1}{c}{Summary} & "
        rf"\multicolumn{{{len(GENERATION_BENCHMARK_ORDER)}}}{{c}}{{Generation Benchmarks}} \\"
    )

    lines.append(
        r"\cmidrule(lr){1-2}"
        r"\cmidrule(lr){3-7}"
        rf"\cmidrule(lr){{{likelihood_start_col}-{likelihood_end_col}}}"
        rf"\cmidrule(lr){{{summary_col}-{summary_col}}}"
        rf"\cmidrule(lr){{{generation_start_col}-{generation_end_col}}}"
    )

    header = [
        "Ratio",
        "Byp.",
        "Group",
        "Score",
        "Scheme",
        "Denom.",
        "Matrices",
    ]

    header += [BENCHMARK_LABELS_LATEX[b] for b in LIKELIHOOD_BENCHMARK_ORDER]
    header += [r"Avg. $\uparrow$"]
    header += [BENCHMARK_LABELS_LATEX[b] for b in GENERATION_BENCHMARK_ORDER]

    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    # Semi-transparent original-model row.
    # Requires: \usepackage[table]{xcolor}
    if original_rows:
        for row in original_rows:
            cells = [
                r"\textcolor{black!45}{0\%}",
                r"\textcolor{black!45}{--}",
                r"\textcolor{black!45}{--}",
                r"\textcolor{black!45}{--}",
                r"\textcolor{black!45}{--}",
                r"\textcolor{black!45}{--}",
                r"\textcolor{black!45}{--}",
            ]

            for benchmark in LIKELIHOOD_BENCHMARK_ORDER:
                value = latex_escape(metric_value_for_display(row, benchmark))
                cells.append(rf"\textcolor{{black!45}}{{{value}}}")

            avg_value = latex_escape(fmt_accuracy(row.get("avg_accuracy")))
            cells.append(rf"\textcolor{{black!45}}{{{avg_value}}}")

            for benchmark in GENERATION_BENCHMARK_ORDER:
                value = latex_escape(metric_value_for_display(row, benchmark))
                cells.append(rf"\textcolor{{black!45}}{{{value}}}")

            lines.append(" & ".join(cells) + r" \\")

        if grouped:
            lines.append(r"\midrule")

    sorted_ratios = sorted(grouped.keys(), key=lambda x: -1 if x is None else x)

    for ratio_idx, ratio in enumerate(sorted_ratios):
        bypass_groups = grouped[ratio]

        ratio_rows = []
        for bypass in bypass_groups:
            ratio_rows.extend(bypass_groups[bypass])

        ratio_row_count = len(ratio_rows)
        ratio_printed = False

        for bypass_idx, bypass in enumerate(sorted(bypass_groups.keys())):
            local_rows = sorted(bypass_groups[bypass], key=sort_rows_hierarchical)
            bypass_row_count = len(local_rows)
            bypass_printed = False

            for row in local_rows:
                cells = []

                if not ratio_printed:
                    cells.append(rf"\multirow{{{ratio_row_count}}}{{*}}{{{fmt_ratio(ratio)}}}")
                    ratio_printed = True
                else:
                    cells.append("")

                if not bypass_printed:
                    cells.append(rf"\multirow{{{bypass_row_count}}}{{*}}{{{latex_escape(bypass)}}}")
                    bypass_printed = True
                else:
                    cells.append("")

                cells.append(latex_escape(row.get("grouping", "--")))
                cells.append(latex_escape(row.get("scoring", "--")))
                cells.append(latex_escape(row.get("scheme", "--")))
                cells.append(latex_escape(row.get("compression_denominator", "--")))
                cells.append(latex_escape(row.get("matrices", "--")))

                for benchmark in LIKELIHOOD_BENCHMARK_ORDER:
                    cells.append(latex_metric_cell(row, benchmark, table_best))

                cells.append(latex_average_cell(row, table_best))

                for benchmark in GENERATION_BENCHMARK_ORDER:
                    cells.append(latex_metric_cell(row, benchmark, table_best))

                lines.append(" & ".join(cells) + r" \\")

            if bypass_idx != len(bypass_groups) - 1:
                lines.append(rf"\cmidrule(lr){{2-{total_col_count}}}")

        if ratio_idx != len(sorted_ratios) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    if use_adjustbox:
        lines.append(r"\end{adjustbox}")

    caption_model = latex_escape(model_name)
    label_model = latex_label_slug(model_name)

    lines.append(
        rf"\caption{{Hierarchical zero-shot lm-eval results for {caption_model}. "
        r"Rows are grouped by compression ratio and bypassed initial layers. "
        r"Accuracy-style metrics are reported as percentages; \texttt{acc\_norm} is used when available and "
        r"\texttt{acc} otherwise. Average accuracy excludes generation benchmarks. "
        r"WikiText is reported as token perplexity, where lower is better. "
        r"Bold values indicate the best result in the table for each metric.}"
    )
    lines.append(rf"\label{{tab:lm-eval-hierarchical-{label_model}}}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


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
        "Hierarchical tables are grouped by compression ratio and bypassed initial layers. "
        "Accuracy-style scores are percentages. `acc_norm` is used when available; otherwise `acc` is used. "
        "Average accuracy excludes generation benchmarks. "
        "`wikitext` is token perplexity, where lower is better."
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

    out.append(r"% Required packages:")
    out.append(r"% \usepackage{booktabs}")
    out.append(r"% \usepackage{multirow}")
    out.append(r"% \usepackage{graphicx}")
    out.append(r"% \usepackage[table]{xcolor}")
    out.append("")

    for model_name in sorted(rows_by_model):
        out.append(make_latex_table_for_model(model_name, rows_by_model[model_name], table_width = table_width))
        out.append("")

    return "\n".join(out).rstrip() + "\n"


def collect_rows(
    input_dir: Path,
    pattern: str,
    prefer_lm_eval_model_name: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    paths = sorted(input_dir.glob(pattern))

    if not paths:
        raise FileNotFoundError(f"No files matched {input_dir / pattern}")

    rows_by_model = defaultdict(list)

    for path in paths:
        row = load_result(path, prefer_lm_eval_model_name=prefer_lm_eval_model_name)
        rows_by_model[row["model"]].append(row)

    return rows_by_model


def main():
    parser = argparse.ArgumentParser(
        description="Generate hierarchical markdown or LaTeX tables from lm-eval JSON result files."
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing lm-eval JSON files.",
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
        choices=["markdown", "latex"],
        default="markdown",
        help="Output format. Default: markdown",
    )

    parser.add_argument(
        "--prefer-lm-eval-model-name",
        action="store_true",
        help=(
            "Use the model name stored inside the JSON config instead of the filename-derived model name. "
            "By default, the filename-derived model name is used."
        ),
    )

    args = parser.parse_args()

    rows_by_model = collect_rows(
        args.input_dir,
        args.pattern,
        prefer_lm_eval_model_name=args.prefer_lm_eval_model_name,
    )

    if args.format == "markdown":
        report = build_markdown_report(rows_by_model)
    elif args.format == "latex":
        report = build_latex_report(rows_by_model, table_width = float(args.table_width))
    else:
        raise ValueError(f"Unsupported format: {args.format}")

    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
