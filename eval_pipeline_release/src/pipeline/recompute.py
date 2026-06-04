"""Recompute metrics from saved distances without re-running evaluation."""

import json
from pathlib import Path

import numpy as np

from ..config import Config
from ..core.metrics import compute_metrics


def recompute_metrics(
    config: Config,
    object_name: str,
    method_name: str,
) -> dict | None:
    """Recompute metrics.json and curves from saved distance arrays.

    Uses config.evaluation.max_dist and config.evaluation.fscore_thresholds.

    Returns:
        Dict with metrics, or None if distances don't exist.
    """
    eval_dir = config.get_eval_dir(object_name, method_name)
    distances_dir = eval_dir / "distances"

    d2g_path = distances_dir / "data2gt_dist.npy"
    g2d_path = distances_dir / "gt2data_dist.npy"

    if not d2g_path.exists() or not g2d_path.exists():
        return None

    print(f"  Recomputing: {object_name}/{method_name}")

    dist_data2gt = np.load(d2g_path)
    dist_gt2data = np.load(g2d_path)

    metrics = compute_metrics(
        dist_data2gt,
        dist_gt2data,
        config.evaluation.fscore_thresholds,
        config.evaluation.max_dist,
    )

    gt_points_path = distances_dir / "gt_points.npy"
    data_points_path = distances_dir / "data_points.npy"
    if gt_points_path.exists():
        metrics["n_gt_points"] = len(np.load(gt_points_path))
    if data_points_path.exists():
        metrics["n_data_points"] = len(np.load(data_points_path))
    metrics["seed"] = 42

    metrics_path = eval_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    curves_dir = eval_dir / "curves"
    curves_dir.mkdir(exist_ok=True)
    np.save(curves_dir / "thresholds.npy", np.array(metrics["thresholds"]))
    np.save(curves_dir / "precision.npy", np.array(metrics["precision"]))
    np.save(curves_dir / "recall.npy", np.array(metrics["recall"]))
    np.save(curves_dir / "fscore.npy", np.array(metrics["fscore"]))

    print(f"    chamfer={metrics['chamfer']:.4f} (max_dist={config.evaluation.max_dist})")
    return metrics
