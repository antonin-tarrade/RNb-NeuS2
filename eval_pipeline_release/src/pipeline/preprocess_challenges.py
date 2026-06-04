"""Generate challenge masks from colored PLY meshes and 2D binary masks."""

from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from ..config import Config
from ..core.camera import load_cameras_auto, project_points
from ..core.mesh import load_mesh
from ..core.visibility import ray_visibility_check


COLOR_THRESHOLDS = {
    "red": {"R_min": 200, "G_max": 100, "B_max": 100},
    "yellow": {"R_min": 200, "G_min": 200, "B_max": 100},
    "green": {"R_max": 100, "G_min": 200, "B_max": 100},
}


def _detect_red_vertices(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return boolean mask of vertices with red vertex colors."""
    if not hasattr(mesh.visual, "vertex_colors") or mesh.visual.vertex_colors is None:
        return np.zeros(len(mesh.vertices), dtype=bool)

    colors = np.asarray(mesh.visual.vertex_colors)[:, :3]
    t = COLOR_THRESHOLDS["red"]
    return (colors[:, 0] > t["R_min"]) & (colors[:, 1] < t["G_max"]) & (colors[:, 2] < t["B_max"])


def _transfer_mesh_mask_to_pcd(
    mesh_vertices: np.ndarray,
    vertex_mask: np.ndarray,
    gt_pcd: np.ndarray,
    max_dist: float,
) -> np.ndarray:
    """Transfer a boolean mask from mesh vertices to GT point cloud via nearest neighbor.

    Builds KDTree on ALL mesh vertices (not just marked ones) for efficiency,
    then checks if the nearest vertex is marked.

    Args:
        mesh_vertices: Mesh vertices (M, 3).
        vertex_mask: Boolean mask on mesh vertices (M,).
        gt_pcd: GT point cloud (N, 3).
        max_dist: Maximum distance for transfer.

    Returns:
        Boolean mask on gt_pcd (N,).
    """
    if not vertex_mask.any():
        return np.zeros(len(gt_pcd), dtype=bool)

    tree = cKDTree(mesh_vertices)
    dists, indices = tree.query(gt_pcd, k=1, workers=-1)
    return (dists < max_dist) & vertex_mask[indices]


def _process_ply(
    ply_path: Path,
    gt_pcd: np.ndarray,
    max_dist: float,
) -> np.ndarray:
    """Extract red vertices from a colored PLY and transfer to GT point cloud."""
    mesh = trimesh.load(str(ply_path), process=False, force="mesh")
    print(f"    Loaded {ply_path.name}: {len(mesh.vertices):,} vertices")

    red_mask = _detect_red_vertices(mesh)
    n_red = red_mask.sum()
    print(f"    Red vertices: {n_red:,} / {len(mesh.vertices):,} ({100*n_red/len(mesh.vertices):.1f}%)")

    if n_red == 0:
        print(f"    WARNING: no red vertices found")
        return np.zeros(len(gt_pcd), dtype=bool)

    return _transfer_mesh_mask_to_pcd(mesh.vertices, red_mask, gt_pcd, max_dist)


def _process_png(
    png_path: Path,
    gt_mesh: trimesh.Trimesh,
    gt_pcd: np.ndarray,
    cameras: dict,
    max_dist: float,
) -> np.ndarray:
    """Project a 2D binary mask onto GT mesh vertices, then transfer to GT point cloud."""
    view_id = png_path.stem.rsplit("_", 1)[0]

    view_ids = cameras.get("view_ids", [])
    view_idx = None
    for i, vid in enumerate(view_ids):
        if str(vid) == view_id:
            view_idx = i
            break

    if view_idx is None:
        print(f"    WARNING: viewId {view_id} not found in sfm.json, skipping")
        return np.zeros(len(gt_pcd), dtype=bool)

    mask_img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        print(f"    WARNING: could not load {png_path}")
        return np.zeros(len(gt_pcd), dtype=bool)
    mask_bin = mask_img > 127

    print(f"    Mask {png_path.name}: viewId={view_id} (idx={view_idx}), {mask_bin.sum():,} masked pixels")

    P = cameras["P"][view_idx]
    camera_center = cameras["camera_centers"][view_idx]
    vertices = gt_mesh.vertices

    projected, depth_valid = project_points(vertices, P)
    h, w = mask_bin.shape
    x = projected[:, 0]
    y = projected[:, 1]

    in_bounds = depth_valid & (x >= 0) & (x < w) & (y >= 0) & (y < h) & ~np.isnan(x) & ~np.isnan(y)

    in_mask = np.zeros(len(vertices), dtype=bool)
    valid_idx = np.where(in_bounds)[0]
    if len(valid_idx) > 0:
        xi = x[valid_idx].astype(int)
        yi = y[valid_idx].astype(int)
        in_mask[valid_idx] = mask_bin[yi, xi]

    visible = ray_visibility_check(vertices, gt_mesh, camera_center)
    vertex_mask = in_mask & visible

    n_marked = vertex_mask.sum()
    print(f"    Marked {n_marked:,} mesh vertices (in mask + visible)")

    return _transfer_mesh_mask_to_pcd(gt_mesh.vertices, vertex_mask, gt_pcd, max_dist)


def preprocess_challenges(
    config: Config,
    object_name: str,
    force: bool = False,
) -> None:
    """Generate challenge masks from raw sources in challenges_raw/.

    Scans Groundtruth/challenges_raw/ for PLY and PNG files.
    Produces boolean .npy masks in Groundtruth/challenges/.

    Args:
        config: Pipeline configuration.
        object_name: Object name.
        force: Overwrite existing masks.
    """
    gt_dir = config.get_gt_dir(object_name)
    raw_dir = gt_dir / "challenges_raw"
    out_dir = gt_dir / "challenges"
    gt_pcd_path = gt_dir / "gt_pcd.npy"

    if not raw_dir.exists():
        return

    if not gt_pcd_path.exists():
        raise FileNotFoundError(f"gt_pcd.npy not found: {gt_pcd_path}. Run preprocess-gt first.")

    gt_pcd = np.load(gt_pcd_path)
    max_dist = config.evaluation.downsample_density * 2

    ply_files = list(raw_dir.glob("*.ply"))
    png_files = list(raw_dir.glob("*.png"))

    if not ply_files and not png_files:
        return

    print(f"Processing challenges for {object_name}")
    print(f"  Sources: {len(ply_files)} PLY, {len(png_files)} PNG")
    print(f"  GT point cloud: {len(gt_pcd):,} points")

    gt_mesh = None
    cameras = None

    categories: dict[str, np.ndarray] = {}

    for ply_path in sorted(ply_files):
        stem = ply_path.stem
        if stem.endswith("_excluded"):
            category = "excluded"
        elif "_Artec_" in stem:
            category = "excluded"
        elif "_CT_" in stem:
            category = "excluded"
        else:
            parts = stem.split("_", 1)
            if len(parts) < 2:
                print(f"  Skipping {ply_path.name}: no category in name")
                continue
            category = parts[1]

        print(f"  [{category}] Processing {ply_path.name}")
        mask = _process_ply(ply_path, gt_pcd, max_dist)

        if category in categories:
            categories[category] |= mask
        else:
            categories[category] = mask

    for png_path in sorted(png_files):
        parts = png_path.stem.rsplit("_", 1)
        if len(parts) < 2:
            print(f"  Skipping {png_path.name}: no category in name (expected {{viewId}}_{{category}}.png)")
            continue
        category = parts[-1]

        if gt_mesh is None:
            gt_mesh_path = config.get_gt_mesh_path(object_name, cleaned=True)
            if not gt_mesh_path.exists():
                gt_mesh_path = config.get_gt_mesh_path(object_name, cleaned=False)
            gt_mesh = load_mesh(gt_mesh_path)
            print(f"  Loaded GT mesh: {len(gt_mesh.vertices):,} vertices")

        if cameras is None:
            cameras_path = config.get_cameras_path(object_name)
            cameras = load_cameras_auto(cameras_path)
            print(f"  Loaded {len(cameras['K'])} cameras")

        print(f"  [{category}] Processing {png_path.name}")
        mask = _process_png(png_path, gt_mesh, gt_pcd, cameras, max_dist)

        if category in categories:
            categories[category] |= mask
        else:
            categories[category] = mask

    out_dir.mkdir(parents=True, exist_ok=True)
    for category, mask in categories.items():
        out_path = out_dir / f"{category}.npy"
        if out_path.exists() and not force:
            print(f"  {category}.npy exists, skipping (use --force)")
            continue
        np.save(out_path, mask)
        n = mask.sum()
        print(f"  Saved {category}.npy: {n:,} / {len(gt_pcd):,} points ({100*n/len(gt_pcd):.1f}%)")

    print(f"  Done: {list(categories.keys())}")
