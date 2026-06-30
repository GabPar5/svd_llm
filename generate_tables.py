import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PERPLEXITY_BENCHMARK_ORDER = [
    "wikitext",
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

BENCHMARK_ALIASES = {
    "gsm8k": ["gsm8k", "gsm8k_cot"],
    "truthfulqa_gen": ["truthfulqa_gen"],
}

BENCHMARK_LABELS_MD = {
    "wikitext": "WikiText ↓",
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
    "truncation_sq": "truncation_sq",
    "eff_rank": "eff_rank",
    "eff_rank_sq": "eff_rank_sq",
    "entropy": "entropy",
    "entropy_sq": "entropy_sq",
}

SCORE_ORDER = {
    "truncation_loss": 0,
    "truncation_sq": 1,
    "eff_rank": 2,
    "eff_rank_sq": 3,
    "entropy": 4,
    "entropy_sq": 5,
    "unknown": 999,
    "original": -1,
    "--": 998,
}

SCHEME_TOKENS = {
    "het": "het",
    "heterogeneous": "het",
    "hom": "hom",
    "homogeneous": "hom",
}

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


def find_scoring(tokens: List[str], start_idx: int) -> Tuple[str, Optional[int], int]:
    """
    Parse a scoring token sequence starting at start_idx.

    Supports:
      truncation
      truncation_sq
      truncation_loss
      eff_rank
      eff_rank_sq
      entropy
      entropy_sq

    Returns:
        (scoring_name, scoring_start_idx, scoring_token_count)
    """

    candidates = [
        "truncation_loss",
        "truncation_sq",
        "eff_rank_sq",
        "entropy_sq",
        "eff_rank",
        "truncation",
        "entropy",
    ]

    for end in range(len(tokens), start_idx, -1):
        cand = "_".join(tokens[start_idx:end])
        if cand in SCORING_TOKENS:
            return SCORING_TOKENS[cand], start_idx, end - start_idx

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

    # Original model case.
    if not has_ratio:
        model_name = "_".join(tokens)
        return {
            "file": path.name,
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
        # Fallback for malformed names.
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

def group_rows_for_model_by_bypass(rows: List[Dict[str, Any]]):
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


def method_label(row: Dict[str, Any]) -> str:
    return f"{row['grouping']} / {row['scoring']} / {row['scheme']} / {row['matrices']}"

def markdown_faded(value: Any) -> str:
    return f'<span style="color: #888888;">{markdown_escape(value)}</span>'

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
    headers += ["Average ↑"]
    headers += [BENCHMARK_LABELS_MD[b] for b in GENERATION_BENCHMARK_ORDER]
    headers += ["File"]

    lines: List[str] = [f"## {model_name}", ""]

    for bypass in sorted(grouped.keys()):
        lines.append(f"### Bypassed layers: {bypass}")
        lines.append("")

        # Faded baseline row shown at the top of each bypass-specific table.
        if original_rows:
            for row in original_rows:
                row_cells = [
                    markdown_faded("0%"),
                    markdown_faded("--"),
                    markdown_faded("--"),
                    markdown_faded("--"),
                    markdown_faded("--"),
                ]

                for benchmark in PERPLEXITY_BENCHMARK_ORDER:
                    row_cells.append(markdown_faded(metric_value_for_display(row, benchmark)))

                for benchmark in LIKELIHOOD_BENCHMARK_ORDER:
                    row_cells.append(markdown_faded(metric_value_for_display(row, benchmark)))

                row_cells.append(markdown_faded(fmt_accuracy(row.get("avg_accuracy"))))

                for benchmark in GENERATION_BENCHMARK_ORDER:
                    row_cells.append(markdown_faded(metric_value_for_display(row, benchmark)))

                row_cells.append(markdown_faded(row.get("file", "--")))
                lines.append("| " + " | ".join(row_cells) + " |")

            lines.append("")

        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for ratio in sorted(grouped[bypass].keys(), key=lambda x: -1 if x is None else x):
            local_rows = sorted(grouped[bypass][ratio], key=sort_rows_hierarchical)
            ratio_best = best_values_non_original(local_rows)

            first_ratio_row = True
            for row in local_rows:
                row_cells = [
                    fmt_ratio_md(ratio) if first_ratio_row else "",
                    row.get("grouping", "--"),
                    row.get("scoring", "--"),
                    row.get("scheme", "--"),
                    row.get("matrices", "--"),
                ]

                for benchmark in PERPLEXITY_BENCHMARK_ORDER:
                    value = metric_value_for_display(row, benchmark)
                    if is_best(row, benchmark, ratio_best):
                        value = f"**{value}**"
                    row_cells.append(value)

                for benchmark in LIKELIHOOD_BENCHMARK_ORDER:
                    value = metric_value_for_display(row, benchmark)
                    if is_best(row, benchmark, ratio_best):
                        value = f"**{value}**"
                    row_cells.append(value)

                avg_value = fmt_accuracy(row.get("avg_accuracy"))
                if is_best(row, "avg_accuracy", ratio_best):
                    avg_value = f"**{avg_value}**"
                row_cells.append(avg_value)

                for benchmark in GENERATION_BENCHMARK_ORDER:
                    value = metric_value_for_display(row, benchmark)
                    if is_best(row, benchmark, ratio_best):
                        value = f"**{value}**"
                    row_cells.append(value)

                row_cells.append(row.get("file", "--"))

                lines.append("| " + " | ".join(markdown_escape(v) for v in row_cells) + " |")
                first_ratio_row = False

        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


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
            rf"\multicolumn{{{len(GENERATION_BENCHMARK_ORDER)}}}{{c}}{{Generation Benchmarks}} \\"
        )

        lines.append(
            r"\cmidrule(lr){1-1}"
            r"\cmidrule(lr){2-5}"
            rf"\cmidrule(lr){{{ppl_start_col}-{ppl_end_col}}}"
            rf"\cmidrule(lr){{{likelihood_start_col}-{likelihood_end_col}}}"
            rf"\cmidrule(lr){{{summary_col}-{summary_col}}}"
            rf"\cmidrule(lr){{{generation_start_col}-{generation_end_col}}}"
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
        header += [r"Avg. $\uparrow$"]
        header += [BENCHMARK_LABELS_LATEX[b] for b in GENERATION_BENCHMARK_ORDER]

        lines.append(" & ".join(header) + r" \\")
        lines.append(r"\midrule")

        # Faded baseline row shown at the top of each bypass-specific table.
        if original_rows:
            for row in original_rows:
                cells = [
                    r"\textcolor{black!45}{0\%}",
                    r"\textcolor{black!45}{--}",
                    r"\textcolor{black!45}{--}",
                    r"\textcolor{black!45}{--}",
                    r"\textcolor{black!45}{--}",
                ]

                for benchmark in PERPLEXITY_BENCHMARK_ORDER:
                    value = latex_escape(metric_value_for_display(row, benchmark))
                    cells.append(rf"\textcolor{{black!45}}{{{value}}}")

                for benchmark in LIKELIHOOD_BENCHMARK_ORDER:
                    value = latex_escape(metric_value_for_display(row, benchmark))
                    cells.append(rf"\textcolor{{black!45}}{{{value}}}")

                avg_value = latex_escape(fmt_accuracy(row.get("avg_accuracy")))
                cells.append(rf"\textcolor{{black!45}}{{{avg_value}}}")

                for benchmark in GENERATION_BENCHMARK_ORDER:
                    value = latex_escape(metric_value_for_display(row, benchmark))
                    cells.append(rf"\textcolor{{black!45}}{{{value}}}")

                lines.append(" & ".join(cells) + r" \\")
            lines.append(r"\midrule")

        sorted_ratios = sorted(grouped[bypass].keys(), key=lambda x: -1 if x is None else x)

        for ratio_idx, ratio in enumerate(sorted_ratios):
            local_rows = sorted(grouped[bypass][ratio], key=sort_rows_hierarchical)
            ratio_best = best_values_non_original(local_rows)

            first_ratio_row = True
            for row in local_rows:
                cells = [
                    fmt_ratio(ratio) if first_ratio_row else "",
                    latex_escape(row.get("grouping", "--")),
                    latex_escape(row.get("scoring", "--")),
                    latex_escape(row.get("scheme", "--")),
                    latex_escape(row.get("matrices", "--")),
                ]

                for benchmark in PERPLEXITY_BENCHMARK_ORDER:
                    cells.append(latex_metric_cell(row, benchmark, ratio_best))

                for benchmark in LIKELIHOOD_BENCHMARK_ORDER:
                    cells.append(latex_metric_cell(row, benchmark, ratio_best))

                cells.append(latex_average_cell(row, ratio_best))

                for benchmark in GENERATION_BENCHMARK_ORDER:
                    cells.append(latex_metric_cell(row, benchmark, ratio_best))

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
            r"WikiText is reported as token perplexity, where lower is better. "
            r"Bold values indicate the best result within each compression ratio.}"
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
