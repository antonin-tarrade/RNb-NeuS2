"""Generate LaTeX tables from evaluation metrics."""

import json
import re
from pathlib import Path

import numpy as np

from ..config import Config


METHODS_ORDER = [
    "meshroom",
    "realityscan",
    "colmap",
    "gomvs_marigold",
    "gomvs_omnidata",
    "gomvs_unimsps",
    "pgsr",
    "rnbneus2",
]

DOWNSCALES = ["d1", "d2", "d4"]

# Patterns to extract (base_method, downscale) from directory names
_PARSE_RULES = [
    (re.compile(r"^meshroom$"), "meshroom", "d1"),
    (re.compile(r"^realityscan$"), "realityscan", "d1"),
    (re.compile(r"^colmap_(d[124])_refin$"), "colmap", None),
    (re.compile(r"^pgsr_(d[124])_refin$"), "pgsr", None),
    (re.compile(r"^gomvs_marigold_(d[124])_refin$"), "gomvs_marigold", None),
    (re.compile(r"^gomvs_omnidata_(d[124])_refin$"), "gomvs_omnidata", None),
    (re.compile(r"^gomvs_unimsps_(d[124])_refin$"), "gomvs_unimsps", None),
    (re.compile(r"^rnbneus2-unimsps-(d[124])-refin_pct90$"), "rnbneus2", None),
]


def parse_method_name(method_name: str) -> tuple[str, str] | None:
    """Parse a method directory name into (base_method, downscale).

    Returns None if the method doesn't match any known pattern.
    """
    for pattern, base, fixed_ds in _PARSE_RULES:
        m = pattern.match(method_name)
        if m:
            ds = fixed_ds if fixed_ds else m.group(1)
            return base, ds
    return None


def _collect_chamfer_data(config: Config) -> dict:
    """Collect chamfer values for all object/method combos.

    Returns:
        {object_name: {(base, ds): chamfer_value}}
    """
    data = {}
    for obj in config.dataset.objects:
        eval_root = config.get_eval_root(obj)
        if not eval_root.exists():
            continue

        obj_data = {}
        for results_raw in eval_root.rglob("results_raw"):
            if not results_raw.is_dir() or "Groundtruth" in results_raw.parts:
                continue
            method_dir = results_raw.parent
            method_name = str(method_dir.relative_to(eval_root))

            parsed = parse_method_name(method_name)
            if parsed is None:
                continue

            metrics_path = config.get_eval_dir(obj, method_name) / "metrics.json"
            if not metrics_path.exists():
                continue

            with open(metrics_path) as f:
                metrics = json.load(f)

            obj_data[parsed] = metrics.get("chamfer")

        data[obj] = obj_data
    return data


def _compute_aggregated_means(config: Config, data: dict) -> dict:
    """Compute mean CD by aggregating raw distances across objects.

    Returns:
        {(base, ds): mean_chamfer}
    """
    all_keys = set()
    for obj_data in data.values():
        all_keys.update(obj_data.keys())

    means = {}
    for key in all_keys:
        base, ds = key
        all_d2g = []
        all_g2d = []

        for obj in config.dataset.objects:
            if key not in data.get(obj, {}):
                continue

            eval_root = config.get_eval_root(obj)
            for results_raw in eval_root.rglob("results_raw"):
                if not results_raw.is_dir() or "Groundtruth" in results_raw.parts:
                    continue
                method_dir = results_raw.parent
                method_name = str(method_dir.relative_to(eval_root))
                parsed = parse_method_name(method_name)
                if parsed != key:
                    continue

                dist_dir = config.get_eval_dir(obj, method_name) / "distances"
                d2g_path = dist_dir / "data2gt_dist.npy"
                g2d_path = dist_dir / "gt2data_dist.npy"

                if d2g_path.exists() and g2d_path.exists():
                    d2g = np.load(d2g_path)
                    g2d = np.load(g2d_path)
                    if config.evaluation.max_dist is not None:
                        d2g = d2g[d2g < config.evaluation.max_dist]
                        g2d = g2d[g2d < config.evaluation.max_dist]
                    all_d2g.append(d2g)
                    all_g2d.append(g2d)
                break

        if all_d2g and all_g2d:
            concat_d2g = np.concatenate(all_d2g)
            concat_g2d = np.concatenate(all_g2d)
            means[key] = (float(np.mean(concat_d2g)) + float(np.mean(concat_g2d))) / 2

    return means


def generate_chamfer_table(
    config: Config,
    decimals: int = 3,
    include_mean: bool = True,
) -> str:
    """Generate a LaTeX table comparing Chamfer Distance across methods and downscales.

    Args:
        config: Pipeline configuration.
        decimals: Number of decimal places.
        include_mean: Include aggregated Mean row.

    Returns:
        LaTeX string for the table.
    """
    data = _collect_chamfer_data(config)
    means = _compute_aggregated_means(config, data) if include_mean else {}

    n_methods = len(METHODS_ORDER)
    n_cols = 1 + n_methods * 3  # object col + 3 per method

    col_spec = "l" + "".join("rrr" for _ in METHODS_ORDER)

    fmt = f"{{:.{decimals}f}}"

    def _format_val(val):
        if val is None:
            return "-"
        return fmt.format(val)

    # Find best/second-best per row for highlighting
    def _get_row_values(obj_data):
        vals = []
        for base in METHODS_ORDER:
            for ds in DOWNSCALES:
                v = obj_data.get((base, ds))
                if v is not None:
                    vals.append(v)
        return vals

    def _rank_row(obj_data):
        vals = sorted(set(v for v in _get_row_values(obj_data) if v is not None))
        best = vals[0] if len(vals) >= 1 else None
        second = vals[1] if len(vals) >= 2 else None
        return best, second

    def _cell(val, best, second):
        s = _format_val(val)
        if val is None:
            return s
        if best is not None and val == best:
            return f"\\textbf{{{s}}}"
        if second is not None and val == second:
            return f"\\underline{{{s}}}"
        return s

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\footnotesize")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header row 1: method names
    header1_parts = [""]
    for base in METHODS_ORDER:
        display = base.replace("_", "\\_")
        header1_parts.append(f"\\multicolumn{{3}}{{c}}{{{display}}}")
    lines.append(" & ".join(header1_parts) + " \\\\")

    # Cmidrules under each method group
    cmidrules = []
    for i, _ in enumerate(METHODS_ORDER):
        start = 2 + i * 3
        end = start + 2
        cmidrules.append(f"\\cmidrule(lr){{{start}-{end}}}")
    lines.append(" ".join(cmidrules))

    # Header row 2: d1/d2/d4
    header2_parts = ["Object"]
    for _ in METHODS_ORDER:
        header2_parts.extend(["d1", "d2", "d4"])
    lines.append(" & ".join(header2_parts) + " \\\\")
    lines.append("\\midrule")

    # Data rows
    for obj in config.dataset.objects:
        obj_data = data.get(obj, {})
        best, second = _rank_row(obj_data)

        row_parts = [obj.replace("_", "\\_")]
        for base in METHODS_ORDER:
            for ds in DOWNSCALES:
                val = obj_data.get((base, ds))
                row_parts.append(_cell(val, best, second))
        lines.append(" & ".join(row_parts) + " \\\\")

    # Mean row
    if include_mean and means:
        lines.append("\\midrule")
        best_mean, second_mean = None, None
        mean_vals = sorted(set(v for v in means.values()))
        if len(mean_vals) >= 1:
            best_mean = mean_vals[0]
        if len(mean_vals) >= 2:
            second_mean = mean_vals[1]

        row_parts = ["\\textit{Mean}"]
        for base in METHODS_ORDER:
            for ds in DOWNSCALES:
                val = means.get((base, ds))
                row_parts.append(_cell(val, best_mean, second_mean))
        lines.append(" & ".join(row_parts) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")  # close resizebox
    lines.append("\\caption{Chamfer Distance (mm) across methods and downscale levels. "
                  "\\textbf{Bold}: best per row, \\underline{underline}: second best. "
                  "Mean is computed by aggregating raw distances across objects.}")
    lines.append("\\label{tab:chamfer}")
    lines.append("\\end{table}")

    return "\n".join(lines)
