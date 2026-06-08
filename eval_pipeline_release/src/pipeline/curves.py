"""Compute dense precision/recall/F-score curves from saved distances."""

from pathlib import Path

import numpy as np

from ..config import Config
from ..core.metrics import compute_fscore_curve


def compute_curves(
    config: Config,
    object_name: str,
    method_name: str,
    n_thresholds: int = 100,
    max_threshold: float = 1.5,
    force: bool = False,
) -> dict | None:
    """Compute dense P/R/F curves from pre-computed distances.

    Requires evaluate to have run first (distances/*.npy must exist).

    Args:
        config: Pipeline configuration.
        object_name: Object name.
        method_name: Method name.
        n_thresholds: Number of thresholds to sample.
        max_threshold: Maximum threshold value (mm).
        force: Overwrite existing curves.

    Returns:
        Dict with thresholds/precision/recall/fscore arrays, or None if skipped.
    """
    eval_dir = config.get_eval_dir(object_name, method_name)
    distances_dir = eval_dir / "distances"
    cleaned_mesh = config.get_cleaned_mesh_path(object_name, method_name)

    if not cleaned_mesh.exists():
        return None

    d2g_path = distances_dir / "data2gt_dist.npy"
    g2d_path = distances_dir / "gt2data_dist.npy"

    if not d2g_path.exists() or not g2d_path.exists():
        return None

    out_dir = eval_dir / "curves"
    marker = out_dir / "thresholds.npy"
    if marker.exists() and not force:
        existing_t = np.load(marker)
        if len(existing_t) >= n_thresholds:
            return None

    print(f"  Computing dense curves: {object_name}/{method_name}")

    dist_data2gt = np.load(d2g_path)
    dist_gt2data = np.load(g2d_path)

    thresholds = np.linspace(0, max_threshold, n_thresholds)
    result = compute_fscore_curve(dist_data2gt, dist_gt2data, thresholds, max_dist=config.evaluation.max_dist)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "thresholds.npy", result["thresholds"])
    np.save(out_dir / "precision.npy", result["precision"])
    np.save(out_dir / "recall.npy", result["recall"])
    np.save(out_dir / "fscore.npy", result["fscore"])

    print(f"    {n_thresholds} thresholds [0, {max_threshold}] -> {out_dir}")
    return result
