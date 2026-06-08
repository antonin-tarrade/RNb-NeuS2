"""I/O utilities for evaluation results."""

import json
from pathlib import Path

import numpy as np


def load_metrics(eval_dir: str | Path) -> dict:
    """Load metrics.json from an evaluation directory.

    Args:
        eval_dir: Path to evaluation directory.

    Returns:
        Dictionary with metrics.
    """
    metrics_path = Path(eval_dir) / "metrics.json"
    with open(metrics_path) as f:
        return json.load(f)


def save_metrics(eval_dir: str | Path, metrics: dict) -> None:
    """Save metrics to metrics.json.

    Args:
        eval_dir: Path to evaluation directory.
        metrics: Metrics dictionary.
    """
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = eval_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


def load_distances(eval_dir: str | Path) -> dict:
    """Load all distance arrays from an evaluation directory.

    Args:
        eval_dir: Path to evaluation directory.

    Returns:
        Dictionary with:
            - gt2data_dist: Distances from GT to data
            - gt2data_idx: Indices of nearest data for each GT
            - data2gt_dist: Distances from data to GT
            - data2gt_idx: Indices of nearest GT for each data
            - gt_points: GT point cloud
            - data_points: Data point cloud
    """
    dist_dir = Path(eval_dir) / "distances"

    return {
        "gt2data_dist": np.load(dist_dir / "gt2data_dist.npy"),
        "gt2data_idx": np.load(dist_dir / "gt2data_idx.npy"),
        "data2gt_dist": np.load(dist_dir / "data2gt_dist.npy"),
        "data2gt_idx": np.load(dist_dir / "data2gt_idx.npy"),
        "gt_points": np.load(dist_dir / "gt_points.npy"),
        "data_points": np.load(dist_dir / "data_points.npy"),
    }


def save_distances(
    eval_dir: str | Path,
    gt2data_dist: np.ndarray,
    gt2data_idx: np.ndarray,
    data2gt_dist: np.ndarray,
    data2gt_idx: np.ndarray,
    gt_points: np.ndarray,
    data_points: np.ndarray,
) -> None:
    """Save distance arrays to evaluation directory.

    Args:
        eval_dir: Path to evaluation directory.
        gt2data_dist: Distances from GT to data.
        gt2data_idx: Indices of nearest data for each GT.
        data2gt_dist: Distances from data to GT.
        data2gt_idx: Indices of nearest GT for each data.
        gt_points: GT point cloud.
        data_points: Data point cloud.
    """
    dist_dir = Path(eval_dir) / "distances"
    dist_dir.mkdir(parents=True, exist_ok=True)

    np.save(dist_dir / "gt2data_dist.npy", gt2data_dist)
    np.save(dist_dir / "gt2data_idx.npy", gt2data_idx)
    np.save(dist_dir / "data2gt_dist.npy", data2gt_dist)
    np.save(dist_dir / "data2gt_idx.npy", data2gt_idx)
    np.save(dist_dir / "gt_points.npy", gt_points)
    np.save(dist_dir / "data_points.npy", data_points)


def load_curves(eval_dir: str | Path) -> dict:
    """Load F-score curve arrays from an evaluation directory.

    Args:
        eval_dir: Path to evaluation directory.

    Returns:
        Dictionary with thresholds, precision, recall, fscore arrays.
    """
    curves_dir = Path(eval_dir) / "curves"

    return {
        "thresholds": np.load(curves_dir / "thresholds.npy"),
        "precision": np.load(curves_dir / "precision.npy"),
        "recall": np.load(curves_dir / "recall.npy"),
        "fscore": np.load(curves_dir / "fscore.npy"),
    }


def save_curves(
    eval_dir: str | Path,
    thresholds: np.ndarray,
    precision: np.ndarray,
    recall: np.ndarray,
    fscore: np.ndarray,
) -> None:
    """Save F-score curve arrays.

    Args:
        eval_dir: Path to evaluation directory.
        thresholds: Threshold values.
        precision: Precision at each threshold.
        recall: Recall at each threshold.
        fscore: F-score at each threshold.
    """
    curves_dir = Path(eval_dir) / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    np.save(curves_dir / "thresholds.npy", thresholds)
    np.save(curves_dir / "precision.npy", precision)
    np.save(curves_dir / "recall.npy", recall)
    np.save(curves_dir / "fscore.npy", fscore)
