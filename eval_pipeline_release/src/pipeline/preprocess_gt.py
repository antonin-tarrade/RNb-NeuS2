"""Ground truth preprocessing: sampling, curvature, and visibility computation."""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..config import Config
from ..core.camera import load_cameras_auto
from ..core.mesh import load_mesh, compute_vertex_curvature
from ..core.sampling import upsample_mesh, downsample_pcd
from ..core.visibility import compute_visibility_count


def compute_gt_curvature(mesh, gt_pcd: np.ndarray, radius: float | None = None) -> np.ndarray:
    """Compute curvature values for GT point cloud.

    Uses vertex curvature from the mesh and transfers to nearest sampled points.

    Args:
        mesh: GT mesh.
        gt_pcd: Sampled GT point cloud, shape (N, 3).
        radius: Spatial radius for curvature estimation.

    Returns:
        Curvature values, shape (N,).
    """
    vertex_curvature = compute_vertex_curvature(mesh, radius=radius)

    from ..core.metrics import compute_distances
    _, nearest_vertex_idx = compute_distances(gt_pcd, mesh.vertices)

    return vertex_curvature[nearest_vertex_idx].astype(np.float32)


def preprocess_gt(
    config: Config,
    object_name: str,
    compute_curvature: bool = True,
    compute_visibility: bool = True,
    clean_gt: bool = True,
    force: bool = False,
) -> None:
    """Preprocess ground truth: clean mesh, sample point cloud and compute attributes.

    Creates:
        - Groundtruth/gt_cleaned.ply (cleaned GT mesh with visibility filtering)
        - Groundtruth/gt_pcd.npy (N, 3)
        - Groundtruth/attributes/curvature_values.npy (N,) float
        - Groundtruth/attributes/visibility_count.npy (N,) int

    Args:
        config: Pipeline configuration.
        object_name: Name of the object to process.
        compute_curvature: Whether to compute curvature values.
        compute_visibility: Whether to compute visibility counts.
        clean_gt: Whether to clean the GT mesh by visibility.
        force: Overwrite existing files.
    """
    gt_dir = config.get_gt_dir(object_name)
    # Search for multiple meshes if they exist (e.g. 25_CT_plant_p*.ply)
    mesh_path_default = config.get_gt_mesh_path(object_name, cleaned=False)
    
    # Check if there are multiple parts (e.g. p1, p2, ...)
    parts = sorted(list(mesh_path_default.parent.glob(f"{object_name.split('_')[0]}_*_p*.ply")))
    if not parts:
        parts = [mesh_path_default] if mesh_path_default.exists() else []

    if not parts:
        raise FileNotFoundError(f"No GT meshes found for {object_name} in {mesh_path_default.parent}")

    cleaned_mesh_path = config.get_gt_mesh_path(object_name, cleaned=True)
    gt_pcd_path = gt_dir / "gt_pcd.npy"
    attributes_dir = gt_dir / "attributes"

    print(f"Preprocessing GT for {object_name}")
    print(f"  Found {len(parts)} mesh parts")

    # Step 1: Clean GT mesh by visibility if requested
    if clean_gt and (not cleaned_mesh_path.exists() or force):
        print(f"  Cleaning GT mesh by visibility (use_masks={config.cleanup.use_masks})...")
        from ..core.mesh import save_mesh, filter_mesh_by_vertex_mask
        from ..core.visibility import filter_by_visibility
        import trimesh
        
        cameras_path = config.get_cameras_path(object_name)
        masks_dir = config.get_masks_dir(object_name)
        
        if not cameras_path.exists():
            print(f"  Warning: cameras.npz not found, skipping GT cleaning")
            clean_gt = False
        elif not masks_dir.exists() and config.cleanup.use_masks:
            print(f"  Warning: masks dir not found, skipping GT cleaning")
            clean_gt = False
        else:
            # Load and merge all parts
            merged_mesh = None
            for p in parts:
                print(f"    Loading {p.name}...")
                m = load_mesh(p)
                if merged_mesh is None:
                    merged_mesh = m
                else:
                    merged_mesh = trimesh.util.concatenate([merged_mesh, m])

            cameras = load_cameras_auto(cameras_path)

            print(f"  Raw merged GT mesh: {len(merged_mesh.vertices)} vertices, {len(merged_mesh.faces)} faces")
            
            # Get visibility mask
            visible_mask = filter_by_visibility(
                merged_mesh.vertices,
                merged_mesh,
                cameras,
                masks_dir,
                config.cleanup.dilation_radius,
                show_progress=True,
                use_masks=config.cleanup.use_masks,
            )
            
            # Filter mesh by mask
            n_visible = np.sum(visible_mask)
            if n_visible == 0:
                # Coordinate system mismatch - use raw mesh instead
                print(f"  WARNING: All vertices filtered (likely coordinate system mismatch)")
                print(f"  Using raw GT mesh")
                cleaned_mesh = merged_mesh
            else:
                cleaned_mesh = filter_mesh_by_vertex_mask(merged_mesh, visible_mask)
            
            print(f"  Cleaned GT mesh: {len(cleaned_mesh.vertices)} vertices, {len(cleaned_mesh.faces)} faces")
            
            gt_dir.mkdir(parents=True, exist_ok=True)
            save_mesh(cleaned_mesh, cleaned_mesh_path)
            print(f"  Saved cleaned GT mesh")
    
    # Use cleaned mesh if available, otherwise raw mesh
    if cleaned_mesh_path.exists():
        mesh_to_use = cleaned_mesh_path
        print(f"  Using cleaned GT mesh")
    else:
        # If not cleaned, we still need to merge parts for sampling
        import trimesh
        merged_mesh = None
        for p in parts:
            m = load_mesh(p)
            if merged_mesh is None:
                merged_mesh = m
            else:
                merged_mesh = trimesh.util.concatenate([merged_mesh, m])
        mesh = merged_mesh
        print(f"  Using raw merged GT mesh")
    
    if cleaned_mesh_path.exists():
        mesh = load_mesh(cleaned_mesh_path)
    
    print(f"  Final mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    if gt_pcd_path.exists() and not force:
        print(f"  Loading existing gt_pcd.npy")
        gt_pcd = np.load(gt_pcd_path)
    else:
        print(f"  Upsampling mesh (density={config.evaluation.downsample_density})")
        pcd = upsample_mesh(
            mesh.vertices,
            mesh.faces,
            config.evaluation.downsample_density,
        )
        print(f"  Upsampled to {len(pcd)} points")

        print(f"  Downsampling point cloud")
        gt_pcd = downsample_pcd(pcd, config.evaluation.downsample_density)
        print(f"  Downsampled to {len(gt_pcd)} points")

        gt_dir.mkdir(parents=True, exist_ok=True)
        np.save(gt_pcd_path, gt_pcd)
        print(f"  Saved gt_pcd.npy")

    attributes_dir.mkdir(parents=True, exist_ok=True)

    if compute_curvature:
        curvature_path = attributes_dir / "curvature_values.npy"
        if curvature_path.exists() and not force:
            print(f"  Curvature already exists, skipping")
        else:
            print(f"  Computing curvature values (radius={config.evaluation.curvature_radius})")
            curvature = compute_gt_curvature(mesh, gt_pcd, radius=config.evaluation.curvature_radius)
            np.save(curvature_path, curvature)
            print(f"  Saved curvature_values.npy")

    if compute_visibility:
        visibility_path = attributes_dir / "visibility_count.npy"
        if visibility_path.exists() and not force:
            print(f"  Visibility already exists, skipping")
        else:
            cameras_path = config.get_cameras_path(object_name)
            masks_dir = config.get_masks_dir(object_name)

            if not cameras_path.exists():
                print(f"  Warning: cameras.npz not found, skipping visibility")
            elif not masks_dir.exists() and config.cleanup.use_masks:
                print(f"  Warning: masks dir not found, skipping visibility")
            else:
                print(f"  Computing visibility counts")
                cameras = load_cameras_auto(cameras_path)
                visibility = compute_visibility_count(
                    gt_pcd, mesh, cameras, masks_dir, show_progress=True,
                    use_masks=config.cleanup.use_masks
                )
                np.save(visibility_path, visibility)
                print(f"  Saved visibility_count.npy (range: {visibility.min()}-{visibility.max()})")

    print(f"  Done preprocessing {object_name}")
