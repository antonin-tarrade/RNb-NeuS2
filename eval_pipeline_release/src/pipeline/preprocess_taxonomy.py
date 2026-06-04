"""Generate taxonomy masks from taxonomy.json, colored PLY meshes and 2D masks."""

import json
from pathlib import Path

import numpy as np

from ..config import Config
from .preprocess_challenges import _process_ply, _process_png, _transfer_mesh_mask_to_pcd
from ..core.camera import load_cameras_auto
from ..core.mesh import load_mesh

OBJECT_PREFIX_MAP = {
    "01": "01_rock",
    "02": "02_vase",
    "03": "03_hammer_head",
    "04": "04_candle",
    "05": "05_pillow",
    "06": "06_can",
    "07": "07_glass",
    "08": "08_boule",
    "09": "09_jam_pot",
    "10": "10_colander",
    "11": "11_ecorce",
    "12": "12_assiette",
    "13": "13_paw_pad",
    "14": "14_moka",
    "15": "15_lego",
    "16": "16_surgery_scissors",
    "17": "17_knife",
    "18": "18_spikes_ball",
    "19": "19_straw_bob",
    "21": "21_sneakers",
    "22": "22_surgery_drill",
    "23": "23_whisk",
    "24": "24_warrior_head",
    "25": "25_plant",
    "26": "26_grid",
    "27": "27_bowl",
    "28": "28_spine",
}


def _resolve_object_name(json_name: str) -> str:
    """Resolve taxonomy.json object name to eval object name.

    Handles variants like '11_bark' -> '11_ecorce', '12_plate' -> '12_assiette'.
    """
    prefix = json_name.split("_")[0]
    return OBJECT_PREFIX_MAP.get(prefix, json_name)


def preprocess_taxonomy(
    config: Config,
    taxonomy_dir: Path,
    object_name: str = None,
    force: bool = False,
) -> None:
    """Generate taxonomy challenge masks from taxonomy.json and associated files.

    For each category/object:
      - mode 1: all GT points marked (minus excluded)
      - mode 2: extract red vertices from PLY or project PNG masks (minus excluded)

    Args:
        config: Pipeline configuration.
        taxonomy_dir: Directory containing taxonomy.json + PLY/PNG files.
        object_name: Process only this object (None = all).
        force: Overwrite existing masks.
    """
    taxonomy_path = taxonomy_dir / "taxonomy.json"
    if not taxonomy_path.exists():
        raise FileNotFoundError(f"taxonomy.json not found: {taxonomy_path}")

    with open(taxonomy_path) as f:
        taxonomy = json.load(f)

    objects = [object_name] if object_name else config.dataset.objects
    max_dist = config.evaluation.downsample_density * 2

    for obj in objects:
        gt_dir = config.get_gt_dir(obj)
        gt_pcd_path = gt_dir / "gt_pcd.npy"

        if not gt_pcd_path.exists():
            continue

        gt_pcd = np.load(gt_pcd_path)
        out_dir = gt_dir / "challenges"
        out_dir.mkdir(parents=True, exist_ok=True)

        excluded_path = out_dir / "excluded.npy"
        excluded = np.load(excluded_path) if excluded_path.exists() else np.zeros(len(gt_pcd), dtype=bool)

        gt_mesh = None
        cameras = None

        for category, entries in taxonomy.items():
            if not entries:
                continue

            matching_entry = None
            for json_name, mode in entries.items():
                if _resolve_object_name(json_name) == obj:
                    matching_entry = (json_name, mode)
                    break

            if matching_entry is None:
                continue

            json_name, mode = matching_entry
            out_path = out_dir / f"{category}.npy"

            if out_path.exists() and not force:
                print(f"  {obj}/{category}.npy exists, skipping")
                continue

            print(f"  {obj} [{category}] mode={mode}")

            if mode == 1:
                mask = ~excluded
                print(f"    Whole object (minus excluded): {mask.sum():,} / {len(gt_pcd):,}")
            elif mode == 2:
                mask = np.zeros(len(gt_pcd), dtype=bool)
                prefix = obj.split("_")[0]

                ply_path = taxonomy_dir / f"{prefix}_{category}.ply"
                if ply_path.exists():
                    print(f"    PLY: {ply_path.name}")
                    mask |= _process_ply(ply_path, gt_pcd, max_dist)

                png_files = sorted(taxonomy_dir.glob("*.png"))
                obj_pngs = []
                for png in png_files:
                    parts = png.stem.rsplit("_", 1)
                    if len(parts) == 2 and parts[1] == category:
                        obj_pngs.append(png)
                    elif len(parts) == 1:
                        obj_pngs.append(png)

                if obj_pngs and not ply_path.exists():
                    if gt_mesh is None:
                        gt_mesh_path = config.get_gt_mesh_path(obj, cleaned=True)
                        if not gt_mesh_path.exists():
                            gt_mesh_path = config.get_gt_mesh_path(obj, cleaned=False)
                        gt_mesh = load_mesh(gt_mesh_path)
                        print(f"    Loaded GT mesh: {len(gt_mesh.vertices):,} vertices")

                    if cameras is None:
                        cameras_path = config.get_cameras_path(obj)
                        cameras = load_cameras_auto(cameras_path)

                    for png in obj_pngs:
                        print(f"    PNG: {png.name}")
                        mask |= _process_png(png, gt_mesh, gt_pcd, cameras, max_dist)

                mask &= ~excluded
                print(f"    Marked (minus excluded): {mask.sum():,} / {len(gt_pcd):,}")
            else:
                print(f"    Unknown mode {mode}, skipping")
                continue

            np.save(out_path, mask)
            print(f"    Saved {category}.npy")

    print("Done")
