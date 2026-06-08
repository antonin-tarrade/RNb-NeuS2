"""Visualization pipeline: render meshes with metric coloring."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ..config import Config
from ..core.camera import load_cameras_auto
from ..core.mesh import load_mesh
from ..core.colormap import (
    color_uniform,
    color_by_accuracy,
    color_by_completeness,
    color_by_visibility,
    color_by_curvature,
)
from ..core.rendering import (
    render_views,
    add_colorbar,
    check_pyrender,
    PYRENDER_AVAILABLE,
)
from PIL import Image
import trimesh


def points_to_mesh(points: np.ndarray, colors: np.ndarray, point_size: float = 0.5) -> trimesh.Trimesh:
    """Convert colored point cloud to a mesh of small spheres for rendering.

    For efficiency, creates billboards (camera-facing quads) instead of spheres.
    Actually, creates a simple point cloud mesh that pyrender can render.
    """
    # Create a PointCloud that can be rendered
    cloud = trimesh.PointCloud(points)
    cloud.colors = colors
    return cloud


def visualize_method(
    config: Config,
    object_name: str,
    method_name: str,
    metrics: list[str],
    view_indices: list[int] | None = None,
    scale: float | None = None,
    crop: bool | None = None,
    crop_margin: int | None = None,
    cmap_overrides: dict | None = None,
    max_dist: float | None = None,
    force: bool = False,
    decimation_target: int = 10_000_000,
    exclude_mode: str | None = None,
) -> dict[str, list[Path]]:
    """Generate all visualization renders for a method.

    Metric assignment:
    - GT mesh only: visibility, curvature (GT attributes)
    - Method mesh only: accuracy, completeness (distance metrics)
    - Both: uniform (plain gray render)

    Args:
        config: Pipeline configuration
        object_name: Object name
        method_name: Method name
        metrics: List of metrics to visualize
        view_indices: View indices to render (None = use config default)
        scale: Resolution scale factor (None = use config)
        crop: Crop to bbox (None = use config)
        crop_margin: Margin around bbox (None = use config)
        cmap_overrides: Override colormaps per metric
        max_dist: Max distance for error coloring (None = use config)
        force: Overwrite existing files
        decimation_target: Max number of faces for visualization (decimate if exceeded)
        exclude_mode: How to handle excluded regions: 'none', 'gray', 'remove' (None = use config)

    Returns:
        Dict mapping metric name to list of generated image paths
    """
    check_pyrender()

    # Metric categories
    GT_ONLY_METRICS = {"visibility", "curvature"}
    METHOD_ONLY_METRICS = {"accuracy", "completeness"}
    BOTH_METRICS = {"uniform"}

    # Get paths
    method_dir = config.get_method_dir(object_name, method_name)
    cleaned_mesh_path = config.get_cleaned_mesh_path(object_name, method_name)
    cameras_path = config.get_cameras_path(object_name, method_name)
    eval_dir = config.get_eval_dir(object_name, method_name)
    gt_dir = config.get_gt_dir(object_name)

    if exclude_mode is None:
        exclude_mode = config.visualization.exclude_mode

    vis_suffix = "visualizations_remove" if exclude_mode == "remove" else "visualizations"
    vis_dir = method_dir / vis_suffix
    gt_vis_dir = gt_dir / vis_suffix

    # Use config defaults if not specified
    if view_indices is None:
        view_indices = config.visualization.default_views
    if scale is None:
        scale = config.visualization.scale
    if crop is None:
        crop = config.visualization.crop
    if crop_margin is None:
        crop_margin = config.visualization.crop_margin
    if max_dist is None:
        max_dist = config.visualization.max_dist

    decimation_target = decimation_target or config.visualization.decimation_target
    min_component_size = config.visualization.min_component_size

    colormaps = config.visualization.colormaps.copy()
    if cmap_overrides:
        colormaps.update(cmap_overrides)

    # Load cameras
    if not cameras_path.exists():
        raise FileNotFoundError(f"Cameras not found: {cameras_path}")
    cameras = load_cameras_auto(cameras_path)

    # Load meshes (only load method mesh if needed by requested metrics)
    needs_method_mesh = bool(set(metrics) & (METHOD_ONLY_METRICS | BOTH_METRICS))
    method_mesh = None
    if needs_method_mesh and cleaned_mesh_path.exists():
        method_mesh = load_mesh(cleaned_mesh_path)
        if len(method_mesh.faces) > decimation_target:
            print(f"  Decimating method mesh: {len(method_mesh.faces)} -> {decimation_target} faces")
            method_mesh = method_mesh.simplify_quadric_decimation(face_count=decimation_target)

    gt_mesh_path = config.get_gt_mesh_path(object_name, cleaned=True)
    if not gt_mesh_path.exists():
        gt_mesh_path = config.get_gt_mesh_path(object_name, cleaned=False)
    gt_mesh = load_mesh(gt_mesh_path) if gt_mesh_path.exists() else None
    gt_mesh_original_vertices = gt_mesh.vertices.copy() if gt_mesh is not None else None
    if gt_mesh is not None and len(gt_mesh.faces) > decimation_target:
        print(f"  Decimating GT mesh: {len(gt_mesh.faces)} -> {decimation_target} faces")
        gt_mesh = gt_mesh.simplify_quadric_decimation(face_count=decimation_target)

    # Load GT point cloud (needed for attribute mapping to decimated meshes)
    gt_pcd_path = gt_dir / "gt_pcd.npy"
    gt_pcd = np.load(gt_pcd_path) if gt_pcd_path.exists() else None

    # Load and apply exclusion mask according to exclude_mode
    excluded_npy_path = gt_dir / "challenges" / "excluded.npy"
    gt_excluded_mask = None
    method_excluded_mask = None

    if exclude_mode != "none" and excluded_npy_path.exists() and gt_pcd is not None:
        print(f"  Loading exclusion mask from {excluded_npy_path.name} (mode={exclude_mode})")
        excluded_pcd = np.load(excluded_npy_path)

        if len(excluded_pcd) != len(gt_pcd):
            print(f"  WARNING: excluded.npy ({len(excluded_pcd)}) != gt_pcd ({len(gt_pcd)}), skipping exclusion mask")
        elif exclude_mode == "remove":
            from ..core.mesh import filter_mesh_by_vertex_mask
            # Remove excluded faces from meshes
            if gt_mesh is not None:
                tree = cKDTree(gt_pcd)
                _, indices = tree.query(gt_mesh.vertices, k=1)
                keep_mask = ~excluded_pcd[indices].astype(bool)
                n_before = len(gt_mesh.vertices)
                gt_mesh = filter_mesh_by_vertex_mask(gt_mesh, keep_mask)
                print(f"  Removed excluded from GT mesh: {n_before} -> {len(gt_mesh.vertices)} vertices")
            if method_mesh is not None:
                tree = cKDTree(gt_pcd)
                _, indices = tree.query(method_mesh.vertices, k=1)
                keep_mask = ~excluded_pcd[indices].astype(bool)
                n_before = len(method_mesh.vertices)
                method_mesh = filter_mesh_by_vertex_mask(method_mesh, keep_mask)
                print(f"  Removed excluded from method mesh: {n_before} -> {len(method_mesh.vertices)} vertices")
        else:
            # mode == "gray": pass masks to colormap functions
            if gt_mesh is not None:
                tree = cKDTree(gt_pcd)
                _, indices = tree.query(gt_mesh.vertices, k=1)
                gt_excluded_mask = excluded_pcd[indices]
            if method_mesh is not None:
                tree = cKDTree(gt_pcd)
                _, indices = tree.query(method_mesh.vertices, k=1)
                method_excluded_mask = excluded_pcd[indices]

    print(f"Visualizing: {object_name}/{method_name} (exclude_mode={exclude_mode})")
    if method_mesh:
        print(f"  Method mesh: {len(method_mesh.vertices)} vertices")
    if gt_mesh:
        print(f"  GT mesh: {len(gt_mesh.vertices)} vertices")
    print(f"  Cameras: {len(cameras['K'])} views")
    print(f"  Metrics: {metrics}")

    results = {}

    # Compute uniform crop bboxes from GT mesh first (largest mesh = most inclusive bbox)
    crop_bboxes = None
    gt_uniform_rendered = False
    bbox_path = gt_vis_dir / "bbox.json"

    if crop and bbox_path.exists() and not force:
        with open(bbox_path) as f:
            raw = json.load(f)
        crop_bboxes = {int(k): tuple(v) for k, v in raw.items()}
        print(f"  Loaded crop bboxes from {bbox_path} ({len(crop_bboxes)} views)")

    if crop and crop_bboxes is None and gt_mesh is not None:
        gt_output = gt_vis_dir / "uniform"
        gt_output.mkdir(parents=True, exist_ok=True)

        print("  Computing uniform crop bboxes from GT mesh...")
        colored_gt = color_uniform(gt_mesh, excluded_mask=gt_excluded_mask)
        _, crop_bboxes = render_views(
            colored_gt, cameras, view_indices, gt_output,
            scale=scale, crop=True, crop_margin=crop_margin, crop_bboxes=None,
            lighting_mode="uniform", min_component_size=min_component_size
        )
        print(f"    Computed bboxes for {len(crop_bboxes)} views")

        with open(bbox_path, "w") as f:
            json.dump({str(k): [int(x) for x in v] for k, v in crop_bboxes.items()}, f, indent=2)
        print(f"    Saved bboxes to {bbox_path}")

        results["uniform_gt"] = list(gt_output.glob("view_*.png"))
        gt_uniform_rendered = True

    for metric in metrics:
        # Determine which mesh to use and output directory
        target_excluded_mask = None
        if metric in GT_ONLY_METRICS:
            if gt_mesh is None:
                print(f"  Skipping {metric}: GT mesh not found")
                continue
            target_mesh = gt_mesh
            target_excluded_mask = gt_excluded_mask
            output_dir = gt_vis_dir / metric
        elif metric in METHOD_ONLY_METRICS:
            if method_mesh is None:
                print(f"  Skipping {metric}: method mesh not found")
                continue
            target_mesh = method_mesh
            target_excluded_mask = method_excluded_mask
            output_dir = vis_dir / metric
        elif metric in BOTH_METRICS:
            pass  # Handled specially below
        elif metric == "mae":
            pass  # Handled specially below
        else:
            print(f"  Unknown metric: {metric}")
            continue

        # Special handling for "uniform" - render both meshes
        if metric == "uniform":
            # Render method mesh
            if method_mesh is not None:
                method_output = vis_dir / "uniform"
                if method_output.exists() and not force:
                    existing = list(method_output.glob("view_*.png"))
                    if existing:
                        print(f"  uniform (method): already exists, skipping")
                        results["uniform_method"] = existing
                    else:
                        results["uniform_method"] = []
                else:
                    print(f"  Generating uniform (method) visualizations...")
                    colored = color_uniform(method_mesh, excluded_mask=method_excluded_mask)
                    paths, _ = render_views(colored, cameras, view_indices, method_output,
                                        scale=scale, crop=crop, crop_margin=crop_margin,
                                        crop_bboxes=crop_bboxes, lighting_mode="uniform",
                                        min_component_size=min_component_size)
                    results["uniform_method"] = paths
                    print(f"    Generated {len(paths)} images")

            # GT uniform - check if already rendered during bbox computation
            if gt_mesh is not None and not gt_uniform_rendered:
                gt_output = gt_vis_dir / "uniform"
                if gt_output.exists() and not force:
                    existing = list(gt_output.glob("view_*.png"))
                    if existing:
                        print(f"  uniform (GT): already exists, skipping")
                        results["uniform_gt"] = existing
                    else:
                        results["uniform_gt"] = []
                else:
                    print(f"  Generating uniform (GT) visualizations...")
                    colored = color_uniform(gt_mesh, excluded_mask=gt_excluded_mask)
                    paths, computed_bboxes = render_views(colored, cameras, view_indices, gt_output,
                                        scale=scale, crop=crop, crop_margin=crop_margin,
                                        crop_bboxes=crop_bboxes, lighting_mode="uniform",
                                        min_component_size=min_component_size)
                    results["uniform_gt"] = paths
                    if crop_bboxes is None and crop:
                        crop_bboxes = computed_bboxes
                    print(f"    Generated {len(paths)} images")
            elif gt_uniform_rendered:
                print(f"  uniform (GT): already rendered for bbox computation")
            continue

        # Image-space MAE heatmap — no mesh/pyrender needed
        if metric == "mae":
            normals_eval_dir = method_dir / "normals_eval" / "per_view"
            if not normals_eval_dir.exists():
                print(f"  Skipping mae: normals_eval/per_view not found")
                continue

            output_dir = vis_dir / "mae"
            if output_dir.exists() and not force:
                existing = list(output_dir.glob("view_*.png"))
                if existing:
                    print(f"  mae: already exists, skipping")
                    results["mae"] = existing
                    continue

            output_dir.mkdir(parents=True, exist_ok=True)
            import matplotlib.cm as cm

            mae_cmap_name = colormaps.get("mae", "jet")
            mae_vmax = 45.0
            if config.normals and config.normals.visualization:
                mae_cmap_name = config.normals.visualization.cmap
                mae_vmax = config.normals.visualization.vmax

            colormap_fn = cm.get_cmap(mae_cmap_name)
            paths_out = []

            # Use view indices to select npy files
            npy_files = sorted(normals_eval_dir.glob("*.npy"))
            view_list = view_indices if view_indices else list(range(len(npy_files)))

            for idx in view_list:
                if idx >= len(npy_files):
                    continue

                error_map = np.load(npy_files[idx])
                mask = error_map > 0

                # Normalize and apply colormap
                normalized = np.clip(error_map / mae_vmax, 0.0, 1.0)
                colored = (colormap_fn(normalized)[:, :, :3] * 255).astype(np.uint8)
                colored[~mask] = [128, 128, 128]

                # Crop using same bboxes if available
                if crop and crop_bboxes and idx in crop_bboxes:
                    x1, y1, x2, y2 = crop_bboxes[idx]
                    colored = colored[y1:y2, x1:x2]

                out_path = output_dir / f"view_{idx:04d}.png"
                Image.fromarray(colored).save(out_path)
                paths_out.append(out_path)

            # Generate colorbar
            if paths_out:
                dummy = np.full((100, 100, 3), 255, dtype=np.uint8)
                with_cb = add_colorbar(dummy, 0.0, mae_vmax, mae_cmap_name, "Angular Error (°)")
                colorbar_img = with_cb[:, 100:]
                cb_path = output_dir / "colorbar.png"
                Image.fromarray(colorbar_img).save(cb_path)

            results["mae"] = paths_out
            print(f"    Generated {len(paths_out)} MAE heatmaps")
            continue

        # Check if already done
        if output_dir.exists() and not force:
            existing = list(output_dir.glob("view_*.png"))
            if existing:
                print(f"  {metric}: already exists, skipping")
                results[metric] = existing
                continue

        print(f"  Generating {metric} visualizations...")

        # Color the mesh based on metric
        colorbar_params = None

        if metric == "accuracy":
            dist_path = eval_dir / "distances" / "data2gt_dist.npy"
            points_path = eval_dir / "distances" / "data_points.npy"
            if not dist_path.exists():
                print(f"    Skipping {metric}: {dist_path} not found")
                continue
            dist = np.load(dist_path)
            # Map distances from sampled points to mesh vertices using KDTree
            if points_path.exists() and method_mesh is not None:
                points = np.load(points_path)
                if len(dist) == len(points):
                    print(f"    Mapping {len(points)} sampled points to {len(method_mesh.vertices)} mesh vertices")
                    tree = cKDTree(points)
                    _, indices = tree.query(method_mesh.vertices, k=1)
                    vertex_dist = dist[indices]
                    colored_mesh, colorbar_params = color_by_accuracy(
                        method_mesh, vertex_dist, max_dist=max_dist, 
                        cmap=colormaps.get("accuracy", "jet"),
                        excluded_mask=method_excluded_mask
                    )
                else:
                    print(f"    Skipping {metric}: points/dist size mismatch")
                    continue
            elif len(dist) == len(target_mesh.vertices):
                colored_mesh, colorbar_params = color_by_accuracy(
                    target_mesh, dist, max_dist=max_dist, 
                    cmap=colormaps.get("accuracy", "jet"),
                    excluded_mask=target_excluded_mask
                )
            else:
                print(f"    Skipping {metric}: distance array size mismatch ({len(dist)} vs {len(target_mesh.vertices)})")
                continue

        elif metric == "completeness":
            dist_path = eval_dir / "distances" / "gt2data_dist.npy"
            points_path = eval_dir / "distances" / "gt_points.npy"
            if not dist_path.exists():
                print(f"    Skipping {metric}: {dist_path} not found")
                continue
            dist = np.load(dist_path)
            # Map distances from sampled GT points to GT mesh vertices using KDTree
            if points_path.exists() and gt_mesh is not None:
                points = np.load(points_path)
                if len(dist) == len(points):
                    print(f"    Mapping {len(points)} sampled GT points to {len(gt_mesh.vertices)} GT mesh vertices")
                    tree = cKDTree(points)
                    _, indices = tree.query(gt_mesh.vertices, k=1)
                    vertex_dist = dist[indices]
                    colored_mesh, colorbar_params = color_by_completeness(
                        gt_mesh, vertex_dist, max_dist=max_dist, 
                        cmap=colormaps.get("completeness", "jet"),
                        excluded_mask=gt_excluded_mask
                    )
                else:
                    print(f"    Skipping {metric}: points/dist size mismatch")
                    continue
            elif gt_mesh is not None and len(dist) == len(gt_mesh.vertices):
                colored_mesh, colorbar_params = color_by_completeness(
                    gt_mesh, dist, max_dist=max_dist, 
                    cmap=colormaps.get("completeness", "jet"),
                    excluded_mask=gt_excluded_mask
                )
            else:
                if gt_mesh is None:
                    print(f"    Skipping {metric}: GT mesh not found")
                else:
                    print(f"    Skipping {metric}: distance array size mismatch ({len(dist)} vs {len(gt_mesh.vertices)})")
                continue
            output_dir = vis_dir / metric  # Still save under method's vis dir

        elif metric == "visibility":
            vis_path = gt_dir / "attributes" / "visibility_count.npy"
            if not vis_path.exists():
                print(f"    Skipping {metric}: {vis_path} not found")
                continue
            vis_count = np.load(vis_path)
            if len(vis_count) != len(target_mesh.vertices):
                # Find the right source points for KNN mapping
                source_pts = None
                if gt_pcd is not None and len(vis_count) == len(gt_pcd):
                    source_pts = gt_pcd
                elif gt_mesh_original_vertices is not None and len(vis_count) == len(gt_mesh_original_vertices):
                    source_pts = gt_mesh_original_vertices
                if source_pts is not None:
                    print(f"    Mapping {len(vis_count)} attribute values to {len(target_mesh.vertices)} mesh vertices via KNN")
                    tree = cKDTree(source_pts)
                    _, indices = tree.query(target_mesh.vertices, k=1)
                    vis_count = vis_count[indices]

            if len(vis_count) != len(target_mesh.vertices):
                print(f"    Skipping {metric}: visibility array size mismatch ({len(vis_count)} vs {len(target_mesh.vertices)})")
                continue
            colored_mesh, colorbar_params = color_by_visibility(
                target_mesh, vis_count, cmap=colormaps.get("visibility", "viridis"),
                excluded_mask=target_excluded_mask
            )

        elif metric == "curvature":
            curv_path = gt_dir / "attributes" / "curvature_values.npy"
            if not curv_path.exists():
                print(f"    Skipping {metric}: {curv_path} not found")
                continue
            curvature = np.load(curv_path)
            if len(curvature) != len(target_mesh.vertices):
                source_pts = None
                if gt_pcd is not None and len(curvature) == len(gt_pcd):
                    source_pts = gt_pcd
                elif gt_mesh_original_vertices is not None and len(curvature) == len(gt_mesh_original_vertices):
                    source_pts = gt_mesh_original_vertices
                if source_pts is not None:
                    print(f"    Mapping {len(curvature)} attribute values to {len(target_mesh.vertices)} mesh vertices via KNN")
                    tree = cKDTree(source_pts)
                    _, indices = tree.query(target_mesh.vertices, k=1)
                    curvature = curvature[indices]

            if len(curvature) != len(target_mesh.vertices):
                print(f"    Skipping {metric}: curvature array size mismatch ({len(curvature)} vs {len(target_mesh.vertices)})")
                continue
            colored_mesh, colorbar_params = color_by_curvature(
                target_mesh, curvature, cmap=colormaps.get("curvature", "coolwarm"),
                excluded_mask=target_excluded_mask
            )

        # Render views
        paths, _ = render_views(
            colored_mesh,
            cameras,
            view_indices,
            output_dir,
            scale=scale,
            crop=crop,
            crop_margin=crop_margin,
            crop_bboxes=crop_bboxes,
            min_component_size=min_component_size
        )

        # Generate and save colorbar if needed
        if colorbar_params is not None:
            dummy = np.full((100, 100, 3), 255, dtype=np.uint8)
            with_cb = add_colorbar(
                dummy,
                colorbar_params["vmin"],
                colorbar_params["vmax"],
                colorbar_params["cmap"],
                colorbar_params["label"],
            )
            colorbar_img = with_cb[:, 100:]
            cb_path = output_dir / "colorbar.png"
            Image.fromarray(colorbar_img).save(cb_path)
            print(f"    Saved colorbar to {cb_path}")

        results[metric] = paths
        print(f"    Generated {len(paths)} images")

    return results
