"""Visibility computation for point clouds using ray tracing."""

from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image
from tqdm import tqdm

from .camera import get_camera_centers, project_points


def load_mask(mask_path: str | Path) -> np.ndarray:
    """Load a binary mask image.

    Args:
        mask_path: Path to mask image (PNG).

    Returns:
        Binary mask array, shape (H, W).
    """
    img = Image.open(mask_path)
    mask = np.array(img)
    if mask.ndim == 3:
        mask = mask[..., 0]
    # Handle both uint8 (0/255) and boolean masks
    if mask.dtype == bool:
        return mask
    return mask > 127


def create_dilation_kernel(radius: int) -> np.ndarray:
    """Create a circular dilation kernel.

    Args:
        radius: Dilation radius in pixels.

    Returns:
        Binary kernel array.
    """
    diameter = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))


def ray_visibility_check(
    points: np.ndarray,
    mesh: trimesh.Trimesh,
    camera_center: np.ndarray,
    ray_offset_factor: float = 0.01 / 100,
) -> np.ndarray:
    """Check visibility of points from a camera using ray tracing.

    Args:
        points: 3D points to check, shape (N, 3).
        mesh: Mesh for occlusion testing.
        camera_center: Camera center position, shape (3,).
        ray_offset_factor: Factor for ray origin offset to avoid self-intersection.

    Returns:
        Boolean array, True if point is visible (not occluded), shape (N,).
    """
    ray_directions = camera_center - points
    ray_norms = np.linalg.norm(ray_directions, axis=-1, keepdims=True)

    # Use per-point offset based on individual ray length (not global max)
    # This prevents numerical issues for points at varying distances
    ray_offsets = ray_offset_factor * ray_norms
    ray_directions_normalized = ray_directions / np.maximum(ray_norms, 1e-10)
    ray_origins = points + ray_offsets * ray_directions_normalized

    hit = mesh.ray.intersects_any(ray_origins, ray_directions_normalized)

    return ~hit


def check_points_in_bounds(
    points: np.ndarray,
    P: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Check if 3D points project inside image bounds.

    Args:
        points: 3D points, shape (N, 3).
        P: Projection matrix (4x4).
        image_shape: (height, width) of the image.

    Returns:
        Boolean array, True if point projects inside image bounds, shape (N,).
    """
    projected, depth_valid = project_points(points, P)

    h, w = image_shape
    x = projected[:, 0]
    y = projected[:, 1]

    in_bounds = depth_valid & (x >= 0) & (x < w) & (y >= 0) & (y < h) & ~np.isnan(x) & ~np.isnan(y)

    return in_bounds


def check_points_in_mask(
    points: np.ndarray,
    P: np.ndarray,
    mask: np.ndarray,
    dilation_radius: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Check if 3D points project inside a 2D mask.

    Args:
        points: 3D points, shape (N, 3).
        P: Projection matrix (4x4).
        mask: Binary mask, shape (H, W).
        dilation_radius: Optional mask dilation radius.

    Returns:
        Tuple of:
            - in_bounds: Boolean array, True if point is inside image bounds, shape (N,).
            - in_mask: Boolean array, True if point projects inside mask, shape (N,).
    """
    if dilation_radius is not None and dilation_radius > 0:
        kernel = create_dilation_kernel(dilation_radius)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    projected, depth_valid = project_points(points, P)

    h, w = mask.shape
    x = projected[:, 0]
    y = projected[:, 1]

    # Check bounds (also handles NaN from invalid depth)
    in_bounds = depth_valid & (x >= 0) & (x < w) & (y >= 0) & (y < h) & ~np.isnan(x) & ~np.isnan(y)

    in_mask = np.zeros(len(points), dtype=bool)
    valid_idx = np.where(in_bounds)[0]

    if len(valid_idx) > 0:
        xi = x[valid_idx].astype(int)
        yi = y[valid_idx].astype(int)
        in_mask[valid_idx] = mask[yi, xi]

    return in_bounds, in_mask


def compute_visibility_count(
    points: np.ndarray,
    mesh: trimesh.Trimesh,
    cameras: dict,
    masks_dir: str | Path,
    dilation_radius: int = 0,
    show_progress: bool = True,
    use_masks: bool = True,
) -> np.ndarray:
    """Compute per-point visibility count across all cameras.

    A point is visible from a camera if:
    1. It projects inside the (dilated) mask (if use_masks=True) or image bounds
    2. It is not occluded by the mesh

    Args:
        points: 3D points, shape (N, 3).
        mesh: Mesh for occlusion testing.
        cameras: Camera dict with 'P', 'Rt' arrays.
        masks_dir: Directory containing mask images (000.png, 001.png, ...).
        dilation_radius: Mask dilation radius in pixels.
        show_progress: Show progress bar.
        use_masks: Whether to use 2D masks for filtering.

    Returns:
        Visibility count per point, shape (N,), dtype int.
    """
    masks_dir = Path(masks_dir)
    n_views = len(cameras["P"])
    camera_centers = get_camera_centers(cameras["Rt"])

    visibility_count = np.zeros(len(points), dtype=np.int32)

    iterator = range(n_views)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing visibility")

    # Use mask_names from cameras dict if available (SfMData format)
    mask_names = cameras.get("mask_names")

    for view_idx in iterator:
        mask_path = None
        if mask_names is not None:
            mask_path = masks_dir / mask_names[view_idx]
        else:
            # Fallback: try sequential naming conventions
            for pattern in [f"{view_idx:03d}.png", f"V{view_idx:02d}.png"]:
                p = masks_dir / pattern
                if p.exists():
                    mask_path = p
                    break
        
        if not mask_path or not mask_path.exists():
            continue

        mask = load_mask(mask_path)
        P = cameras["P"][view_idx]
        camera_center = camera_centers[view_idx]

        if use_masks:
            _, in_view = check_points_in_mask(points, P, mask, dilation_radius)
        else:
            in_view = check_points_in_bounds(points, P, mask.shape)

        visible_from_cam = ray_visibility_check(points, mesh, camera_center)

        visible = in_view & visible_from_cam
        visibility_count += visible.astype(np.int32)

    return visibility_count


def filter_by_visibility(
    vertices: np.ndarray,
    mesh: trimesh.Trimesh,
    cameras: dict,
    masks_dir: str | Path,
    dilation_radius: int = 12,
    show_progress: bool = True,
    use_masks: bool = True,
) -> np.ndarray:
    """Create a vertex mask keeping only vertices visible from at least one view.

    Uses strict mask filtering if use_masks=True: if a vertex projects inside a view's 
    image bounds but outside the mask, it is removed.
    Otherwise (use_masks=False), it only checks for occlusion and camera bounds.

    Args:
        vertices: Mesh vertices, shape (N, 3).
        mesh: Mesh for occlusion testing.
        cameras: Camera dict.
        masks_dir: Masks directory.
        dilation_radius: Mask dilation radius.
        show_progress: Show progress bar.
        use_masks: Whether to use 2D masks for filtering.

    Returns:
        Boolean mask, True for visible vertices, shape (N,).
    """
    masks_dir = Path(masks_dir)
    n_views = len(cameras["P"])
    camera_centers = get_camera_centers(cameras["Rt"])
    n_vertices = len(vertices)

    # Pre-load and dilate all masks
    masks = []
    image_shapes = []
    kernel = create_dilation_kernel(dilation_radius) if dilation_radius > 0 else None

    mask_names = cameras.get("mask_names")
    for view_idx in range(n_views):
        mask_path = None
        if mask_names is not None:
            mask_path = masks_dir / mask_names[view_idx]
        else:
            # Try multiple naming conventions
            for pattern in [f"{view_idx:03d}.png", f"V{view_idx:02d}.png"]:
                p = masks_dir / pattern
                if p.exists():
                    mask_path = p
                    break
        
        if mask_path and mask_path.exists():
            mask = load_mask(mask_path)
            image_shapes.append(mask.shape)
            if use_masks:
                if kernel is not None:
                    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
                masks.append(mask)
            else:
                masks.append(None)
        else:
            masks.append(None)
            image_shapes.append(None)

    # Pass 1: Find vertices inside at least one image's bounds
    if show_progress:
        print("  Pass 1: Finding vertices inside image bounds...")
    outside_all_images = np.ones(n_vertices, dtype=bool)

    for view_idx in range(n_views):
        if image_shapes[view_idx] is None:
            continue
        P = cameras["P"][view_idx]
        in_bounds = check_points_in_bounds(vertices, P, image_shapes[view_idx])
        outside_all_images[in_bounds] = False

    inside_any_image = ~outside_all_images
    if show_progress:
        print(f"    Vertices inside at least one image: {inside_any_image.sum()}/{n_vertices}")

    # Pass 2: Strict mask filtering
    # For each view, if vertex is inside bounds but outside mask, remove it
    if show_progress:
        print(f"  Pass 2: Mask filtering (use_masks={use_masks})...")
    keep_mask = inside_any_image.copy()

    if use_masks:
        iterator = range(n_views)
        if show_progress:
            iterator = tqdm(iterator, desc="  Filtering by masks")

        for view_idx in iterator:
            if masks[view_idx] is None:
                continue

            P = cameras["P"][view_idx]
            in_bounds, in_mask = check_points_in_mask(vertices, P, masks[view_idx], dilation_radius=0)

            # Keep if: (outside this view's bounds) OR (inside this view's mask)
            keep_this_view = ~in_bounds | in_mask
            keep_mask &= keep_this_view
    else:
        if show_progress:
            print("    Skipping mask filtering as use_masks=False")

    if show_progress:
        print(f"    Vertices after mask filtering: {keep_mask.sum()}/{n_vertices}")

    # Pass 3: Ray tracing for occlusion
    # Keep vertices visible (not occluded) from at least one view
    if show_progress:
        print("  Pass 3: Ray tracing for occlusion...")

    # Pre-compute which vertices are in mask for each view (needed for ray tracing)
    in_view_all_views = np.zeros((n_views, n_vertices), dtype=bool)
    for view_idx in range(n_views):
        if image_shapes[view_idx] is None:
            continue
        P = cameras["P"][view_idx]
        if use_masks and masks[view_idx] is not None:
            _, in_view = check_points_in_mask(vertices, P, masks[view_idx], dilation_radius=0)
        else:
            in_view = check_points_in_bounds(vertices, P, image_shapes[view_idx])
        in_view_all_views[view_idx] = in_view

    # Compute ray visibility for all views
    hits_all_views = np.zeros((n_views, n_vertices), dtype=bool)

    iterator = range(n_views)
    if show_progress:
        iterator = tqdm(iterator, desc="  Ray tracing")

    for view_idx in iterator:
        if image_shapes[view_idx] is None:
            continue
        camera_center = camera_centers[view_idx]
        visible_from_cam = ray_visibility_check(vertices, mesh, camera_center)
        hits_all_views[view_idx] = ~visible_from_cam

    # A vertex is visible if: not hit (not occluded) AND in mask (or bounds) for that view
    visible_per_view = ~hits_all_views & in_view_all_views
    visible_any = visible_per_view.any(axis=0)

    # Final mask: passed mask filtering AND visible from at least one view
    final_mask = keep_mask & visible_any

    if show_progress:
        print(f"    Final visible vertices: {final_mask.sum()}/{n_vertices}")

    return final_mask
