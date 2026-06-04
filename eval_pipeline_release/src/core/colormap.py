"""Mesh coloring by metric values."""

import numpy as np
import trimesh
import matplotlib.pyplot as plt


def color_mesh_by_metric(
    mesh: trimesh.Trimesh,
    values: np.ndarray,
    cmap: str = "jet",
    vmin: float | None = None,
    vmax: float | None = None,
    clip: bool = True,
    excluded_mask: np.ndarray | None = None,
) -> trimesh.Trimesh:
    """Return mesh with vertex_colors set according to values.

    Args:
        mesh: Input mesh
        values: Metric values per vertex, shape (N_vertices,)
        cmap: Matplotlib colormap name
        vmin, vmax: Color range (auto-computed if None)
        clip: Clip outliers to [vmin, vmax]
        excluded_mask: Boolean mask of excluded vertices (True = excluded)

    Returns:
        New trimesh with vertex_colors set
    """
    values = np.asarray(values, dtype=np.float32)

    if vmin is None:
        vmin = float(np.nanmin(values))
    if vmax is None:
        vmax = float(np.nanmax(values))

    if clip:
        values = np.clip(values, vmin, vmax)

    # Normalize to [0, 1]
    if vmax > vmin:
        normalized = (values - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(values)

    # Apply colormap
    colormap = plt.get_cmap(cmap)
    colors = colormap(normalized)  # Returns (N, 4) RGBA in [0, 1]
    colors_uint8 = (colors * 255).astype(np.uint8)

    # Override colors for excluded vertices (light gray)
    if excluded_mask is not None:
        excluded_mask = np.asarray(excluded_mask, dtype=bool)
        if len(excluded_mask) == len(colors_uint8):
            colors_uint8[excluded_mask] = [200, 200, 200, 255]

    # Create new mesh with colors
    colored_mesh = mesh.copy()
    colored_mesh.visual.vertex_colors = colors_uint8

    return colored_mesh


def color_uniform(
    mesh: trimesh.Trimesh,
    color: tuple[int, int, int] = (200, 200, 200),
    excluded_mask: np.ndarray | None = None,
) -> trimesh.Trimesh:
    """Apply uniform color to mesh (no colorbar needed).

    Args:
        mesh: Input mesh
        color: RGB tuple (0-255)
        excluded_mask: Boolean mask of excluded vertices

    Returns:
        Mesh with uniform vertex_colors
    """
    colored_mesh = mesh.copy()
    n_vertices = len(mesh.vertices)
    colors = np.full((n_vertices, 4), [color[0], color[1], color[2], 255], dtype=np.uint8)
    
    # We could technically color excluded differently here, but "uniform" 
    # is usually already gray (200,200,200). 
    # If the user wants excluded to be distinct, we could use a different gray.
    # For now, let's keep it consistent with the user's request "they remain in light gray (like uniform)".
    
    colored_mesh.visual.vertex_colors = colors
    return colored_mesh


def color_by_accuracy(
    mesh: trimesh.Trimesh,
    dist_data2gt: np.ndarray,
    max_dist: float = 5.0,
    cmap: str = "jet",
    excluded_mask: np.ndarray | None = None,
) -> tuple[trimesh.Trimesh, dict]:
    """Color mesh by data→GT distance (accuracy error)."""
    colored = color_mesh_by_metric(
        mesh, dist_data2gt, cmap=cmap, vmin=0.0, vmax=max_dist, excluded_mask=excluded_mask
    )
    params = {
        "vmin": 0.0,
        "vmax": max_dist,
        "cmap": cmap,
        "label": "Accuracy [mm]",
    }
    return colored, params


def color_by_completeness(
    mesh: trimesh.Trimesh,
    dist_gt2data: np.ndarray,
    max_dist: float = 5.0,
    cmap: str = "jet",
    excluded_mask: np.ndarray | None = None,
) -> tuple[trimesh.Trimesh, dict]:
    """Color GT mesh by GT→data distance (completeness error)."""
    colored = color_mesh_by_metric(
        mesh, dist_gt2data, cmap=cmap, vmin=0.0, vmax=max_dist, excluded_mask=excluded_mask
    )
    params = {
        "vmin": 0.0,
        "vmax": max_dist,
        "cmap": cmap,
        "label": "Completeness [mm]",
    }
    return colored, params


def color_by_visibility(
    mesh: trimesh.Trimesh,
    visibility_count: np.ndarray,
    cmap: str = "viridis",
    excluded_mask: np.ndarray | None = None,
) -> tuple[trimesh.Trimesh, dict]:
    """Color mesh by visibility count (number of views)."""
    vmin = 0
    vmax = int(np.max(visibility_count)) if len(visibility_count) > 0 else 0
    colored = color_mesh_by_metric(
        mesh, visibility_count, cmap=cmap, vmin=vmin, vmax=vmax, excluded_mask=excluded_mask
    )
    params = {
        "vmin": vmin,
        "vmax": vmax,
        "cmap": cmap,
        "label": "Visibility [views]",
    }
    return colored, params


def color_by_curvature(
    mesh: trimesh.Trimesh,
    curvature: np.ndarray,
    cmap: str = "coolwarm",
    symmetric: bool = True,
    percentile: float = 99.0,
    excluded_mask: np.ndarray | None = None,
) -> tuple[trimesh.Trimesh, dict]:
    """Color mesh by curvature values."""
    if symmetric:
        vmax = float(np.percentile(np.abs(curvature), percentile)) if len(curvature) > 0 else 0
        vmin = -vmax
    else:
        vmin = float(np.percentile(curvature, 100 - percentile)) if len(curvature) > 0 else 0
        vmax = float(np.percentile(curvature, percentile)) if len(curvature) > 0 else 0

    colored = color_mesh_by_metric(
        mesh, curvature, cmap=cmap, vmin=vmin, vmax=vmax, excluded_mask=excluded_mask
    )
    params = {
        "vmin": vmin,
        "vmax": vmax,
        "cmap": cmap,
        "label": "Curvature",
    }
    return colored, params
