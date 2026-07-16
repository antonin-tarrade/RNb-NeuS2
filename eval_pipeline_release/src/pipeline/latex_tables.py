"""Generate LaTeX tables from evaluation metrics.

Supports two table layouts:
  - Resolution-based  (when config.dataset.resolutions is non-empty):
      rows = methods, columns = resolutions, cells = Chamfer Distance
  - Downscale-based   (legacy, when resolutions is empty):
      rows = objects,  columns = method × downscale
"""

import json
import re
from pathlib import Path

import numpy as np

from ..config import Config


# ── Legacy constants (downscale-based layout) ─────────────────────────────────

METHODS_ORDER = [
    "meshroom", "realityscan", "colmap",
    "gomvs_marigold", "gomvs_omnidata", "gomvs_unimsps",
    "pgsr", "rnbneus2",
]

DOWNSCALES = ["d1", "d2", "d4"]

_PARSE_RULES = [
    (re.compile(r"^meshroom$"),                          "meshroom",       "d1"),
    (re.compile(r"^realityscan$"),                       "realityscan",    "d1"),
    (re.compile(r"^colmap_(d[124])_refin$"),             "colmap",         None),
    (re.compile(r"^pgsr_(d[124])_refin$"),               "pgsr",           None),
    (re.compile(r"^gomvs_marigold_(d[124])_refin$"),     "gomvs_marigold", None),
    (re.compile(r"^gomvs_omnidata_(d[124])_refin$"),     "gomvs_omnidata", None),
    (re.compile(r"^gomvs_unimsps_(d[124])_refin$"),      "gomvs_unimsps",  None),
    (re.compile(r"^rnbneus2-unimsps-(d[124])-refin_pct90$"), "rnbneus2",   None),
]


def parse_method_name(method_name: str) -> tuple[str, str] | None:
    """Parse a legacy method directory name into (base_method, downscale)."""
    for pattern, base, fixed_ds in _PARSE_RULES:
        m = pattern.match(method_name)
        if m:
            ds = fixed_ds if fixed_ds else m.group(1)
            return base, ds
    return None


# ── Resolution-based table ─────────────────────────────────────────────────────

def _collect_chamfer_resolution(config: Config) -> dict:
    """Collect Chamfer values for all (object, method, resolution) combos.

    Returns:
        {object_name: {(method, resolution): chamfer_value}}
    """
    data: dict[str, dict] = {}
    for obj in config.dataset.objects:
        obj_data: dict[tuple, float] = {}
        for method in config.dataset.methods:
            for resolution in config.dataset.resolutions:
                method_res = f"{method}/{resolution}"
                eval_dir = config.get_eval_dir(obj, method_res)
                metrics_path = eval_dir / "metrics.json"
                if not metrics_path.exists():
                    continue
                with open(metrics_path) as f:
                    metrics = json.load(f)
                val = metrics.get("chamfer")
                if val is not None:
                    obj_data[(method, resolution)] = float(val)
        data[obj] = obj_data
    return data


def _generate_resolution_table(
    config: Config,
    decimals: int = 3,
    include_mean: bool = True,
) -> str:
    """Generate LaTeX table: rows=methods, columns=resolutions.

    Best/second-best highlighted per column (best method at each resolution).
    """
    methods     = config.dataset.methods
    resolutions = config.dataset.resolutions
    objects     = config.dataset.objects

    data = _collect_chamfer_resolution(config)

    fmt = f"{{:.{decimals}f}}"

    def _fmt(val):
        return fmt.format(val) if val is not None else "--"

    def _cell(val, best, second):
        s = _fmt(val)
        if val is None:
            return s
        if best is not None and abs(val - best) < 1e-9:
            return f"\\textbf{{{s}}}"
        if second is not None and abs(val - second) < 1e-9:
            return f"\\underline{{{s}}}"
        return s

    # Sanitise names for LaTeX
    def _tex(s):
        return s.replace("_", "\\_")

    n_res   = len(resolutions)
    n_meth  = len(methods)
    col_spec = "l" + "r" * n_res

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\footnotesize")
    if n_res > 4:
        lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header
    header = ["Méthode"] + [f"{r}" for r in resolutions]
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")

    # One section per object (usually just one)
    for obj_idx, obj in enumerate(objects):
        if len(objects) > 1:
            lines.append(
                f"\\multicolumn{{{1 + n_res}}}{{l}}"
                f"{{\\textit{{{_tex(obj)}}}}} \\\\"
            )
            lines.append("\\midrule")

        obj_data = data.get(obj, {})

        # Per-column best/second-best (best method for each resolution)
        col_sorted: dict[str, list] = {}
        for r in resolutions:
            vals = sorted(
                [(m, obj_data.get((m, r))) for m in methods if obj_data.get((m, r)) is not None],
                key=lambda x: x[1]
            )
            col_sorted[r] = [v[1] for v in vals]

        def _best(r):
            vs = col_sorted[r]
            return vs[0] if len(vs) >= 1 else None

        def _second(r):
            vs = col_sorted[r]
            return vs[1] if len(vs) >= 2 else None

        for method in methods:
            row = [_tex(method)]
            for r in resolutions:
                val = obj_data.get((method, r))
                row.append(_cell(val, _best(r), _second(r)))
            lines.append(" & ".join(row) + " \\\\")

    # Mean row across objects (skip if only one object — identical to data row)
    if include_mean and len(objects) > 1:
        lines.append("\\midrule")

        # Aggregate across objects per (method, resolution)
        for method in methods:
            row_vals: dict[str, float | None] = {}
            for r in resolutions:
                vals = [
                    data[o][(method, r)]
                    for o in objects
                    if (method, r) in data.get(o, {})
                ]
                row_vals[r] = float(np.mean(vals)) if vals else None

            # Per-column best/second-best in the mean row
            mean_col_sorted: dict[str, list] = {}
            for r in resolutions:
                col_means = []
                for m2 in methods:
                    vs = [data[o].get((m2, r)) for o in objects
                          if (m2, r) in data.get(o, {})]
                    if vs:
                        col_means.append(float(np.mean(vs)))
                mean_col_sorted[r] = sorted(col_means)

            row = [f"\\textit{{Moy.~{_tex(method)}}}"]
            for r in resolutions:
                val = row_vals.get(r)
                ms = mean_col_sorted[r]
                b  = ms[0] if len(ms) >= 1 else None
                s  = ms[1] if len(ms) >= 2 else None
                row.append(_cell(val, b, s))
            lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if n_res > 4:
        lines.append("}")   # close resizebox
    lines.append(
        "\\caption{Distance de Chamfer (mm) selon la méthode et la résolution. "
        "\\textbf{Gras}~: meilleure méthode par résolution, "
        "\\underline{souligné}~: deuxième meilleure.}"
    )
    lines.append("\\label{tab:chamfer_resolution}")
    lines.append("\\end{table}")

    return "\n".join(lines)


# ── Legacy downscale-based table ───────────────────────────────────────────────

def _collect_chamfer_data(config: Config) -> dict:
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
    all_keys = set()
    for obj_data in data.values():
        all_keys.update(obj_data.keys())

    means = {}
    for key in all_keys:
        all_d2g, all_g2d = [], []
        for obj in config.dataset.objects:
            if key not in data.get(obj, {}):
                continue
            eval_root = config.get_eval_root(obj)
            for results_raw in eval_root.rglob("results_raw"):
                if not results_raw.is_dir() or "Groundtruth" in results_raw.parts:
                    continue
                method_dir = results_raw.parent
                method_name = str(method_dir.relative_to(eval_root))
                if parse_method_name(method_name) != key:
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
            cd2g = np.concatenate(all_d2g)
            cg2d = np.concatenate(all_g2d)
            means[key] = (float(np.mean(cd2g)) + float(np.mean(cg2d))) / 2
    return means


def _generate_downscale_table(config: Config, decimals: int = 3, include_mean: bool = True) -> str:
    """Legacy table: rows=objects, columns=method×downscale."""
    data  = _collect_chamfer_data(config)
    means = _compute_aggregated_means(config, data) if include_mean else {}

    col_spec = "l" + "".join("rrr" for _ in METHODS_ORDER)
    fmt = f"{{:.{decimals}f}}"

    def _fmt(val):
        return fmt.format(val) if val is not None else "-"

    def _rank_row(obj_data):
        vals = sorted(set(
            v for base in METHODS_ORDER for ds in DOWNSCALES
            if (v := obj_data.get((base, ds))) is not None
        ))
        best   = vals[0] if len(vals) >= 1 else None
        second = vals[1] if len(vals) >= 2 else None
        return best, second

    def _cell(val, best, second):
        s = _fmt(val)
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

    header1 = [""]
    for base in METHODS_ORDER:
        header1.append(f"\\multicolumn{{3}}{{c}}{{{base.replace('_', chr(92)+'_')}}}")
    lines.append(" & ".join(header1) + " \\\\")

    cmidrules = []
    for i in range(len(METHODS_ORDER)):
        start = 2 + i * 3
        cmidrules.append(f"\\cmidrule(lr){{{start}-{start+2}}}")
    lines.append(" ".join(cmidrules))

    header2 = ["Object"] + ["d1", "d2", "d4"] * len(METHODS_ORDER)
    lines.append(" & ".join(header2) + " \\\\")
    lines.append("\\midrule")

    for obj in config.dataset.objects:
        obj_data = data.get(obj, {})
        best, second = _rank_row(obj_data)
        row = [obj.replace("_", "\\_")]
        for base in METHODS_ORDER:
            for ds in DOWNSCALES:
                row.append(_cell(obj_data.get((base, ds)), best, second))
        lines.append(" & ".join(row) + " \\\\")

    if include_mean and means:
        lines.append("\\midrule")
        mean_vals = sorted(set(means.values()))
        bm = mean_vals[0] if mean_vals else None
        sm = mean_vals[1] if len(mean_vals) >= 2 else None
        row = ["\\textit{Mean}"]
        for base in METHODS_ORDER:
            for ds in DOWNSCALES:
                row.append(_cell(means.get((base, ds)), bm, sm))
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\caption{Chamfer Distance (mm). \\textbf{Bold}: best, \\underline{underline}: second best.}")
    lines.append("\\label{tab:chamfer}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ── Public entry point ─────────────────────────────────────────────────────────

def generate_chamfer_table(
    config: Config,
    decimals: int = 3,
    include_mean: bool = True,
) -> str:
    """Generate LaTeX Chamfer Distance table.

    Automatically dispatches to the resolution-based layout when
    config.dataset.resolutions is non-empty, otherwise uses the legacy
    method×downscale layout.
    """
    if config.dataset.resolutions:
        return _generate_resolution_table(config, decimals=decimals, include_mean=include_mean)
    return _generate_downscale_table(config, decimals=decimals, include_mean=include_mean)