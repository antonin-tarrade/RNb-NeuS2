"""Global evaluation: compute distances, Chamfer, and F-score metrics."""

import json
from pathlib import Path

import numpy as np

from ..config import Config
from ..core.mesh import load_mesh
from ..core.sampling import upsample_mesh, downsample_pcd
from ..core.metrics import compute_distances, compute_metrics
import cv2


def evaluate(
    config: Config,
    object_name: str,
    method_name: str,
    force: bool = False,
    seed: int = 42,
) -> dict:
    """Evaluate a reconstructed mesh against ground truth.

    Computes bidirectional distances and stores:
        - distances/gt2data_dist.npy, gt2data_idx.npy
        - distances/data2gt_dist.npy, data2gt_idx.npy
        - distances/gt_points.npy, data_points.npy
        - curves/thresholds.npy, precision.npy, recall.npy, fscore.npy
        - metrics.json

    Args:
        config: Pipeline configuration.
        object_name: Object name.
        method_name: Method name.
        force: Overwrite existing output.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with computed metrics.
    """
    gt_dir = config.get_gt_dir(object_name)
    eval_dir = config.get_eval_dir(object_name, method_name)
    cleaned_mesh_path = config.get_cleaned_mesh_path(object_name, method_name)

    gt_pcd_path = gt_dir / "gt_pcd.npy"
    metrics_path = eval_dir / "metrics.json"

    if metrics_path.exists() and not force:
        print(f"  Metrics exist, loading: {metrics_path}")
        with open(metrics_path) as f:
            return json.load(f)

    print(f"Evaluating: {object_name}/{method_name}")

    if not gt_pcd_path.exists():
        raise FileNotFoundError(f"GT point cloud not found: {gt_pcd_path}")
    if not cleaned_mesh_path.exists():
        raise FileNotFoundError(f"Cleaned mesh not found: {cleaned_mesh_path}")

    gt_pcd = np.load(gt_pcd_path)
    print(f"  Loaded GT point cloud: {len(gt_pcd)} points")

    data_mesh = load_mesh(cleaned_mesh_path)
    print(f"  Loaded cleaned mesh: {len(data_mesh.vertices)} vertices")

    density = config.evaluation.downsample_density

    print(f"  Upsampling data mesh (density={density})")
    data_pcd = upsample_mesh(data_mesh.vertices, data_mesh.faces, density)
    print(f"  Upsampled to {len(data_pcd)} points")

    gt_min_z = gt_pcd[:, 2].min()
    z_mask = data_pcd[:, 2] > gt_min_z
    data_pcd = data_pcd[z_mask]
    print(f"  After z-filter: {len(data_pcd)} points")

    nan_mask = ~np.isnan(data_pcd).any(axis=1)
    data_pcd = data_pcd[nan_mask]

    print(f"  Downsampling data point cloud")
    data_down = downsample_pcd(data_pcd, density, shuffle=True, seed=seed)
    print(f"  Downsampled to {len(data_down)} points")

    print(f"  Computing distances")
    dist_data2gt, idx_data2gt = compute_distances(data_down, gt_pcd)
    dist_gt2data, idx_gt2data = compute_distances(gt_pcd, data_down)

    # Apply excluded mask if it exists
    excluded_path = gt_dir / "challenges" / "excluded.npy"
    if excluded_path.exists():
        excluded = np.load(excluded_path)
        gt_keep = ~excluded
        data_keep = gt_keep[idx_data2gt]
        n_gt_excl = excluded.sum()
        n_data_excl = (~data_keep).sum()
        print(f"  Applying excluded mask: {n_gt_excl:,} GT points, {n_data_excl:,} data points excluded")
        dist_gt2data_filtered = dist_gt2data[gt_keep]
        dist_data2gt_filtered = dist_data2gt[data_keep]
    else:
        dist_gt2data_filtered = dist_gt2data
        dist_data2gt_filtered = dist_data2gt

    print(f"  Computing metrics")
    metrics = compute_metrics(
        dist_data2gt_filtered,
        dist_gt2data_filtered,
        config.evaluation.fscore_thresholds,
        config.evaluation.max_dist,
    )
    metrics["seed"] = seed
    metrics["n_gt_points"] = len(gt_pcd)
    metrics["n_data_points"] = len(data_down)

    eval_dir.mkdir(parents=True, exist_ok=True)
    distances_dir = eval_dir / "distances"
    curves_dir = eval_dir / "curves"
    distances_dir.mkdir(exist_ok=True)
    curves_dir.mkdir(exist_ok=True)

    np.save(distances_dir / "gt2data_dist.npy", dist_gt2data)
    np.save(distances_dir / "gt2data_idx.npy", idx_gt2data)
    np.save(distances_dir / "data2gt_dist.npy", dist_data2gt)
    np.save(distances_dir / "data2gt_idx.npy", idx_data2gt)
    np.save(distances_dir / "gt_points.npy", gt_pcd)
    np.save(distances_dir / "data_points.npy", data_down)

    np.save(curves_dir / "thresholds.npy", np.array(metrics["thresholds"]))
    np.save(curves_dir / "precision.npy", np.array(metrics["precision"]))
    np.save(curves_dir / "recall.npy", np.array(metrics["recall"]))
    np.save(curves_dir / "fscore.npy", np.array(metrics["fscore"]))

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Chamfer: {metrics['chamfer']:.4f}")
    # Find closest threshold to 1.0 for display
    thresholds = metrics["thresholds"]
    if 1.0 in thresholds:
        fscore_1 = metrics["fscore"][thresholds.index(1.0)]
    else:
        # Find closest threshold
        closest_idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - 1.0))
        fscore_1 = metrics["fscore"][closest_idx]
    print(f"  F-score@1.0: {fscore_1:.4f}")
    print(f"  Saved results to {eval_dir}")

    # --- MAE Normal Evaluation (additive, non-blocking) ---
    try:
        normals_metrics = evaluate_normals(config, object_name, method_name, force)
        if normals_metrics:
            metrics.update(normals_metrics)
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
    except Exception as e:
        print(f"  Normals MAE: failed ({e}), continuing without normals metrics")

    return metrics


def evaluate_normals(
    config: Config,
    object_name: str,
    method_name: str,
    force: bool = False,
) -> dict | None:
    """Evaluate normal maps for a method against GT.

    Returns normals metrics dict, or None if normals eval not applicable.
    """
    # Lazy import: only import if normals evaluation is enabled
    from ..core.normals import load_normal_map, compute_mae, render_normals_from_mesh
    
    if config.normals is None or not config.normals.enabled:
        return None

    pose_source = config.get_normals_pose_source(method_name)
    downscale = config.get_normals_downscale(method_name)
    gt_normals_dir = config.get_gt_normals_dir(object_name, pose_source, downscale)

    gt_normals_path = gt_normals_dir / "normals"
    if not gt_normals_path.exists():
        print(f"  Normals MAE: skipping (GT normals not found at {gt_normals_path})")
        return None

    gt_normal_files = sorted(gt_normals_path.glob("*.png"))
    if not gt_normal_files:
        print(f"  Normals MAE: skipping (no GT normal PNGs in {gt_normals_path})")
        return None

    method_dir = config.get_method_dir(object_name, method_name)
    normals_eval_dir = method_dir / "normals_eval"
    per_view_dir = normals_eval_dir / "per_view"

    # Check if already computed
    if per_view_dir.exists() and not force:
        existing = list(per_view_dir.glob("*.npy"))
        if existing:
            print(f"  Normals MAE: loading existing ({len(existing)} views)")
            all_errors = []
            for npy_path in sorted(existing):
                err_map = np.load(npy_path)
                mask = err_map > 0
                if mask.any():
                    all_errors.append(err_map[mask])
            if all_errors:
                all_errors = np.concatenate(all_errors)
                return _aggregate_mae(all_errors, len(existing), pose_source, downscale)
            return None

    # Get method normals
    precomputed_dir = config.get_method_normals_dir(object_name, method_name)

    if precomputed_dir is not None and precomputed_dir.exists():
        method_normals_path = precomputed_dir
        print(f"  Normals MAE: using pre-computed normals from {method_normals_path}")
    else:
        cleaned_mesh_path = config.get_cleaned_mesh_path(object_name, method_name)
        if not cleaned_mesh_path.exists():
            print(f"  Normals MAE: skipping (no cleaned mesh and no pre-computed normals)")
            return None

        sfm_path = config.resolve_normals_sfm_path(object_name, pose_source, downscale)
        if not sfm_path.exists():
            print(f"  Normals MAE: skipping (sfm not found: {sfm_path})")
            return None

        rendered_dir = normals_eval_dir / "rendered"
        print(f"  Normals MAE: rendering method normals from {cleaned_mesh_path.name}")

        method_mesh = load_mesh(cleaned_mesh_path)
        render_normals_from_mesh(
            method_mesh, sfm_path, rendered_dir,
            chunk_size=config.normals.rendering.chunk_size,
            samples=config.normals.rendering.samples,
            force=force,
        )
        method_normals_path = rendered_dir / "normals"

    # Compute per-view MAE
    per_view_dir.mkdir(parents=True, exist_ok=True)
    gt_masks_path = gt_normals_dir / "masks"

    all_errors = []
    n_views = 0

    for gt_file in gt_normal_files:
        view_id = gt_file.stem
        method_file = method_normals_path / f"{view_id}.png"

        if not method_file.exists():
            continue

        normals_gt, mask_gt = load_normal_map(gt_file)

        gt_mask_file = gt_masks_path / f"{view_id}.png"
        if gt_mask_file.exists():
            mask_gt_from_file = cv2.imread(str(gt_mask_file), cv2.IMREAD_GRAYSCALE) > 127
            mask_gt = mask_gt & mask_gt_from_file

        normals_pred, mask_pred = load_normal_map(method_file)

        result = compute_mae(normals_gt, normals_pred, mask_gt, mask_pred)

        np.save(per_view_dir / f"{view_id}.npy", result["angular_error_map"])

        if result["n_valid_pixels"] > 0:
            valid = result["angular_error_map"] > 0
            all_errors.append(result["angular_error_map"][valid])
            n_views += 1

    if not all_errors:
        print(f"  Normals MAE: no valid views found")
        return None

    all_errors = np.concatenate(all_errors)
    metrics = _aggregate_mae(all_errors, n_views, pose_source, downscale)

    print(f"  Normals MAE: {metrics['normals_mae_mean']:.2f}° mean, "
          f"{metrics['normals_mae_median']:.2f}° median ({n_views} views)")

    return metrics


def _aggregate_mae(errors: np.ndarray, n_views: int, pose_source: str, downscale: str) -> dict:
    """Aggregate angular errors into summary metrics."""
    return {
        "normals_mae_mean": float(np.mean(errors)),
        "normals_mae_median": float(np.median(errors)),
        "normals_pct_below_5": float(np.mean(errors < 5.0)),
        "normals_pct_below_10": float(np.mean(errors < 10.0)),
        "normals_pct_below_20": float(np.mean(errors < 20.0)),
        "normals_n_views": n_views,
        "normals_pose_source": pose_source,
        "normals_downscale": downscale,
    }
