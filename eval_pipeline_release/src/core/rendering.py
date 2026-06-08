"""Offscreen mesh rendering with pyrender."""

import os
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import cv2
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pathlib import Path
from PIL import Image
from tqdm import tqdm

try:
    import pyrender
    PYRENDER_AVAILABLE = True
except ImportError:
    PYRENDER_AVAILABLE = False


def check_pyrender():
    """Raise ImportError if pyrender is not available."""
    if not PYRENDER_AVAILABLE:
        raise ImportError(
            "pyrender required for visualization. "
            "Install with: pip install pyrender PyOpenGL"
        )


class OffscreenRenderer:
    """Pyrender-based offscreen mesh renderer."""

    def __init__(self, width: int, height: int, K: np.ndarray, znear: float = 0.1, zfar: float = 10000.0):
        """Initialize offscreen renderer with camera intrinsics.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            K: Camera intrinsic matrix (4x4)
            znear: Near clipping plane distance
            zfar: Far clipping plane distance
        """
        check_pyrender()

        if K.shape != (4, 4):
            raise ValueError(f"K must be 4x4, got shape {K.shape}")
        if not np.isfinite(K).all():
            raise ValueError("Camera intrinsics contain NaN or inf values")

        self.width = width
        self.height = height

        # Extract intrinsics from K matrix
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        # Create pyrender camera with appropriate clip planes
        self.camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=znear, zfar=zfar)

        # Create renderer
        self.renderer = pyrender.OffscreenRenderer(width, height)

    def render_mesh(self, mesh: trimesh.Trimesh, Rt: np.ndarray, lighting_mode: str = "colored") -> np.ndarray:
        """Render a colored mesh from a given camera pose.

        Args:
            mesh: Trimesh with vertex_colors set
            Rt: Extrinsic matrix [R|t] (4x4), world-to-camera
            lighting_mode: "uniform" for surface detail (directional), "colored" for vivid colors (ambient)

        Returns:
            RGB image as numpy array (H, W, 3), dtype uint8
        """
        if Rt.shape != (4, 4):
            raise ValueError(f"Rt must be 4x4, got shape {Rt.shape}")

        # Lighting setup depends on mode
        if lighting_mode == "uniform":
            # Low ambient + strong directional for surface detail contrast
            scene = pyrender.Scene(
                bg_color=[1.0, 1.0, 1.0, 1.0],
                ambient_light=[0.3, 0.3, 0.3],
            )
        else:
            # High ambient for vivid vertex colors
            scene = pyrender.Scene(
                bg_color=[1.0, 1.0, 1.0, 1.0],
                ambient_light=[0.6, 0.6, 0.6],
            )

        # Convert trimesh to pyrender mesh - preserve vertex colors
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        scene.add(pyrender_mesh)

        # Camera pose: pyrender expects camera-to-world, but we have world-to-camera (Rt)
        # So we need to invert: pose = Rt^-1
        try:
            camera_pose = np.linalg.inv(Rt)
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Extrinsic matrix is singular: {e}") from e

        # pyrender uses OpenGL convention: camera looks down -Z
        # AliceVision/OpenCV uses +Z, so we need to flip Y and Z axes
        flip = np.diag([1, -1, -1, 1]).astype(np.float32)
        camera_pose = camera_pose @ flip

        scene.add(self.camera, pose=camera_pose)

        # Add directional light
        if lighting_mode == "uniform":
            # Strong frontal light for surface detail
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        else:
            # Moderate fill light for colored renders
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.5)
        scene.add(light, pose=camera_pose)

        # Render
        color, _ = self.renderer.render(scene)

        return color

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        """Release GPU/context resources."""
        self.renderer.delete()


def add_colorbar(
    image: np.ndarray,
    vmin: float,
    vmax: float,
    cmap: str,
    label: str,
    position: str = "right",
) -> np.ndarray:
    """Add matplotlib colorbar to an image.

    Args:
        image: RGB image (H, W, 3)
        vmin, vmax: Colorbar range
        cmap: Matplotlib colormap name
        label: Colorbar label with units
        position: "right" or "bottom"

    Returns:
        Image with colorbar added (H, W', 3) or (H', W, 3)
    """
    h, w = image.shape[:2]

    # Create figure with image and colorbar
    if position == "right":
        fig_width = w / 100 + 0.8  # Extra space for colorbar
        fig_height = h / 100
        fig = Figure(figsize=(fig_width, fig_height), dpi=100)

        # Image axes
        ax_img = fig.add_axes([0, 0, w / 100 / fig_width, 1])
        ax_img.imshow(image)
        ax_img.axis("off")

        # Colorbar axes
        ax_cb = fig.add_axes([w / 100 / fig_width + 0.02, 0.1, 0.05, 0.8])
    else:  # bottom
        fig_width = w / 100
        fig_height = h / 100 + 0.6
        fig = Figure(figsize=(fig_width, fig_height), dpi=100)

        # Image axes
        ax_img = fig.add_axes([0, 0.6 / fig_height, 1, h / 100 / fig_height])
        ax_img.imshow(image)
        ax_img.axis("off")

        # Colorbar axes
        ax_cb = fig.add_axes([0.1, 0.15, 0.8, 0.08])

    # Create colorbar
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    orientation = "vertical" if position == "right" else "horizontal"
    cb = fig.colorbar(sm, cax=ax_cb, orientation=orientation)
    cb.set_label(label)

    # Render to numpy array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    result = np.asarray(buf)[:, :, :3].copy()

    plt.close(fig)

    return result


def get_content_bbox(
    image: np.ndarray,
    margin: int = 20,
    background: tuple[int, int, int] = (255, 255, 255),
    min_component_size: int = 100,
) -> tuple[int, int, int, int]:
    """Get bounding box of non-background content.

    Args:
        image: RGB image (H, W, 3)
        margin: Pixels to add around bounding box
        background: Background color to detect content against
        min_component_size: Minimum number of pixels for a connected component to be kept

    Returns:
        (rmin, rmax, cmin, cmax) bounding box coordinates
    """
    bg = np.array(background, dtype=np.uint8)
    mask = ~np.all(image == bg, axis=-1)

    # Filter out noise components (small disconnected pixel groups)
    if min_component_size > 0:
        mask_uint8 = mask.astype(np.uint8)
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8)
        
        # Keep only large components
        new_mask = np.zeros_like(mask)
        for i in range(1, n_labels):  # Skip background component 0
            if stats[i, cv2.CC_STAT_AREA] >= min_component_size:
                new_mask[labels == i] = True
        mask = new_mask

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        h, w = image.shape[:2]
        return (0, h - 1, 0, w - 1)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    h, w = image.shape[:2]
    rmin = max(0, rmin - margin)
    rmax = min(h - 1, rmax + margin)
    cmin = max(0, cmin - margin)
    cmax = min(w - 1, cmax + margin)

    return (rmin, rmax, cmin, cmax)


def crop_to_bbox(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Crop image to specified bounding box.

    Args:
        image: RGB image (H, W, 3)
        bbox: (rmin, rmax, cmin, cmax) bounding box

    Returns:
        Cropped image
    """
    rmin, rmax, cmin, cmax = bbox
    return image[rmin:rmax + 1, cmin:cmax + 1].copy()


def crop_to_content(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    margin: int = 20,
    background: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Crop image to bounding box of non-background content.

    Args:
        image: RGB image (H, W, 3)
        mask: Optional mask (if None, detect from background color)
        margin: Pixels to add around bounding box
        background: Background color to detect content against

    Returns:
        Cropped image
    """
    if mask is None:
        # Detect content as non-background pixels
        bg = np.array(background, dtype=np.uint8)
        mask = ~np.all(image == bg, axis=-1)

    # Find bounding box
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        # No content found, return original
        return image

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Add margin
    h, w = image.shape[:2]
    rmin = max(0, rmin - margin)
    rmax = min(h - 1, rmax + margin)
    cmin = max(0, cmin - margin)
    cmax = min(w - 1, cmax + margin)

    return image[rmin:rmax + 1, cmin:cmax + 1].copy()


def render_views(
    mesh: trimesh.Trimesh,
    cameras: dict,
    view_indices: list[int] | None,
    output_dir: Path,
    scale: float = 1.0,
    crop: bool = False,
    crop_margin: int = 20,
    crop_bboxes: dict[int, tuple[int, int, int, int]] | None = None,
    lighting_mode: str = "colored",
    min_component_size: int = 100,
    max_resolution: int | None = 1000,
) -> tuple[list[Path], dict[int, tuple[int, int, int, int]]]:
    """Batch render multiple views.

    Args:
        mesh: Colored trimesh to render
        cameras: Camera dict with K, Rt arrays (from load_cameras_auto)
        view_indices: Which views to render (None = all)
        output_dir: Output directory for PNGs
        scale: Resolution scale factor
        crop: If True, crop to object bounding box
        crop_margin: Margin around bounding box
        crop_bboxes: Pre-computed bboxes per view index for uniform cropping
        lighting_mode: "uniform" for surface detail, "colored" for vivid colors
        min_component_size: Minimum component size for cropping noise removal

    Returns:
        (saved_paths, computed_bboxes) - paths and bboxes computed from this render
    """
    check_pyrender()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_views = len(cameras["K"])
    if view_indices is None:
        view_indices = list(range(n_views))

    saved_paths = []
    computed_bboxes = {}

    # Get render dimensions
    K_ref = cameras["K"][view_indices[0]]
    if "image_width" in cameras and "image_height" in cameras:
        width = int(cameras["image_width"] * scale)
        height = int(cameras["image_height"] * scale)
        print(f"  Render resolution: {width}x{height} (from sfm image_width/height, scale={scale})")
    else:
        cx, cy = K_ref[0, 2], K_ref[1, 2]
        width = int(cx * 2 * scale)
        height = int(cy * 2 * scale)
        print(f"  Render resolution: {width}x{height} (from cx*2/cy*2, scale={scale})")

    # Scale intrinsics
    K_scaled = K_ref.copy()
    K_scaled[0, 0] *= scale  # fx
    K_scaled[1, 1] *= scale  # fy
    K_scaled[0, 2] *= scale  # cx
    K_scaled[1, 2] *= scale  # cy

    # Compute znear/zfar from mesh bounds and camera positions
    mesh_center = mesh.centroid
    mesh_radius = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]) / 2

    # Find min/max distance from any camera to mesh center
    min_dist = float('inf')
    max_dist = 0
    for idx in view_indices:
        Rt = cameras["Rt"][idx]
        R, t = Rt[:3, :3], Rt[:3, 3]
        cam_pos = -R.T @ t
        dist = np.linalg.norm(cam_pos - mesh_center)
        min_dist = min(min_dist, dist)
        max_dist = max(max_dist, dist)

    # Set clip planes with margin
    znear = max(0.1, min_dist - mesh_radius * 2)
    zfar = max_dist + mesh_radius * 2

    with OffscreenRenderer(width, height, K_scaled, znear=znear, zfar=zfar) as renderer:
        for idx in tqdm(view_indices, desc="Rendering views"):
            Rt = cameras["Rt"][idx]

            image = renderer.render_mesh(mesh, Rt, lighting_mode=lighting_mode)

            if crop:
                if crop_bboxes is not None and idx in crop_bboxes:
                    # Use pre-computed bbox for uniform cropping
                    image = crop_to_bbox(image, crop_bboxes[idx])
                else:
                    # Compute bbox and crop
                    bbox = get_content_bbox(image, margin=crop_margin, min_component_size=min_component_size)
                    computed_bboxes[idx] = bbox
                    image = crop_to_bbox(image, bbox)

            # Resize if needed
            if max_resolution is not None:
                h, w = image.shape[:2]
                if max(h, w) > max_resolution:
                    ratio = max_resolution / max(h, w)
                    new_w, new_h = int(w * ratio), int(h * ratio)
                    image = np.array(Image.fromarray(image).resize((new_w, new_h), Image.LANCZOS))

            # Save
            output_path = output_dir / f"view_{idx:03d}.png"
            Image.fromarray(image).save(output_path)
            saved_paths.append(output_path)

    return saved_paths, computed_bboxes
