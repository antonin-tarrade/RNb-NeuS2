import numpy as np
import mcubes
import sys
import os
import argparse
import time
from abc import ABC, abstractmethod
import torch

class Method(ABC):
    def __init__(self, snapshot_path, transform=None, testbed=None):
        # Always set transform first, regardless of testbed
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

        # If testbed provided, reuse it instead of loading
        if testbed is not None:
            self.testbed = testbed
            return

        # Otherwise, load snapshot normally
        try:
            import pyngp as ngp
            ##TODO : look for build folder relatively to this script
        except ImportError:
            print("Error: pyngp not found.")
            sys.exit(1)
        self.testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
        print(f"Loading snapshot from {snapshot_path}...")
        if not os.path.exists(snapshot_path):
            print(f"Error: Snapshot file {snapshot_path} not found.")
            return
        self.testbed.load_snapshot(snapshot_path)

    @abstractmethod
    def run(self):
        pass

class ExtractionMethod(Method):


    @property
    def method_label(self) -> str:
        """Identity string for filenames. Overridden by ChunkedExtractor."""
        return self.__class__.__name__

    def _get_world_transform(self):
        """Extract the nerf→world transform from the loaded dataset."""
        ds = self.testbed.nerf.training.dataset
        nerf_scale  = float(ds.scale)
        nerf_offset = np.array(ds.offset, dtype=np.float32)

        # Try to get n2w_s/n2w_t from the training JSON
        n2w_s = 1.0
        n2w_t = np.zeros(3, dtype=np.float32)

        if self.transform is not None:
            # First, try direct n2w_s and n2w_t (old format)
            if "n2w_s" in self.transform:
                n2w_s = self.transform.get("n2w_s", n2w_s)
                n2w_t = np.array(self.transform.get("n2w_t", n2w_t), dtype=np.float32)
            # Otherwise, extract from n2w matrix (new format)
            elif "n2w" in self.transform:
                n2w_matrix = np.array(self.transform["n2w"], dtype=np.float32)
                # Extract scale from diagonal (assuming uniform scaling)
                n2w_s = float(n2w_matrix[0, 0])
                # Extract translation from last column (first 3 elements)
                n2w_t = n2w_matrix[:3, 3].astype(np.float32)
                print(f"Extracted from n2w matrix: scale={n2w_s}, translation={n2w_t}")
        else:
            print("  WARNING: transform not provided, "
                  "n2w_s=1 (mesh will be in [-1,1] not metric scale)")

        return {
            "nerf_scale":  nerf_scale,
            "nerf_offset": nerf_offset,
            "n2w_s":       n2w_s,
            "n2w_t":       n2w_t,
        }

    def _apply_world_transform(self, vertices: np.ndarray) -> np.ndarray:
        t = self._get_world_transform()
        p = (vertices - t["nerf_offset"]) / t["nerf_scale"]
        return (t["n2w_s"] * p + t["n2w_t"]).astype(np.float32)

    def _apply_world_transform_normals(self, normals: np.ndarray) -> np.ndarray:
        t = self._get_world_transform()
        n = normals * t["n2w_s"]
        norms = np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        return (n / norms).astype(np.float32)


    def save_o3d(self, vertices, triangles, colors=None, normals=None,
                 output_path="mesh.obj"):
        import open3d as o3d

        vertices = self._apply_world_transform(vertices)
        if normals is not None:
            normals = - self._apply_world_transform_normals(normals) # Inverted to match GT

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices  = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.orient_triangles()
        if colors is not None:
            mesh.vertex_colors = o3d.utility.Vector3dVector(
                np.clip(colors, 0, 1))
        if normals is not None:
            mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
        o3d.io.write_triangle_mesh(output_path, mesh)
        print(f"Saved {output_path}")

    @abstractmethod
    def extract(self, testbed, resolution, aabb, threshold):
        pass

    def run(self, resolution=256, threshold=0.0, name=None,
            output_path=None, count_time=False):
        if output_path is None:
            name = name or "mesh"
            output_path = f"{name}_{resolution}_{self.method_label}.obj"

        aabb = self.testbed.render_aabb
        start_time = time.time()
        vertices, triangles, colors, normals, n_verts, n_faces = self.extract(
            self.testbed, resolution, aabb, threshold)
        elapsed = time.time() - start_time
        if count_time:
            print(f"Extraction took {elapsed:.2f}s. Verts: {n_verts}, Faces: {n_faces}")
        self.save_o3d(vertices, triangles, colors, normals, output_path)
        return elapsed, n_verts, n_faces



class NativeMethod(ExtractionMethod):

    @abstractmethod
    def call_testbed(self, testbed, resolution, aabb, threshold):
        pass

    def extract(self, testbed, resolution, aabb, threshold):
        mesh_data = self.call_testbed(testbed, resolution, aabb, threshold)
        V, F = mesh_data['V'], mesh_data['F']
        return V, F, mesh_data.get('C'), mesh_data.get('N'), len(V), len(F)


class ExternalMethod(ExtractionMethod):

    @abstractmethod
    def extract_vt(self,sdf,res,aabb,threshold):
        pass

    @abstractmethod
    def preprocess_density(self, sdf):
        pass

    @abstractmethod
    def convert_indices(self, vertices, resolution, aabb):
        pass

    def get_density(self, testbed, resolution, aabb):
        return testbed.get_density_grid(resolution, resolution, resolution, aabb)

    def extract(self, testbed, resolution, aabb, threshold):
        sdf = self.get_density(testbed, resolution, aabb)
        modified_sdf = self.preprocess_density(sdf)
        vertices, triangles = self.extract_vt(modified_sdf, resolution, aabb, threshold)
        corrected_vertices = self.convert_indices(vertices, resolution, aabb)
        print("Querying vertex colors and normals...")
        data = testbed.query_vertex_colors_and_normals(corrected_vertices.astype(np.float32))
        return corrected_vertices, triangles, data["colors"], data["normals"], len(corrected_vertices), len(triangles)


class CudaMC(NativeMethod):
    def call_testbed(self, testbed, resolution, aabb, threshold):
        print(f"Mode: Internal CUDA Marching Cubes...")
        return testbed.compute_marching_cubes_mesh(resolution, resolution, resolution, aabb, threshold)

class ChunkedCudaMC(NativeMethod):
    def call_testbed(self, testbed, resolution, aabb, threshold):
        print(f"Mode: Chunked CUDA Marching Cubes...")
        return testbed.compute_marching_cubes_mesh_chunked(resolution, resolution, resolution, aabb, threshold)

class PyMCubes(ExternalMethod):

    def preprocess_density(self, sdf):
        return np.transpose(sdf, (2, 1, 0))

    def extract_vt(self, sdf, resolution, aabb, threshold):
        print(f"Mode: PyMCubes...")

        vertices, triangles = mcubes.marching_cubes(sdf, threshold)
        return vertices.astype(np.float32), triangles.astype(np.int32)

    def convert_indices(self, vertices, resolution, aabb):
        res_scale = (aabb.max - aabb.min) / resolution
        return vertices * res_scale + aabb.min

class FlexiCubes(ExternalMethod):

    def __init__(self, snapshot_path, transform=None, testbed=None):
        from kaolin.ops.conversions import FlexiCubes
        import torch
        super().__init__(snapshot_path, transform, testbed)
        self.device = torch.device("cuda")
        self.fc = FlexiCubes(device=self.device)

    def preprocess_density(self, sdf):

        # Match NeRF [Z, Y, X] to FlexiCubes [X, Y, Z]
        density_np = np.transpose(sdf, (2, 1, 0)).copy()

        return (torch.from_numpy(density_np).float().to(self.device).reshape(-1))

    def extract_vt(self, sdf, resolution, aabb, threshold):


        # Flexicube resolution is number of voxels, which is one less than number of grid points
        fc_res = resolution - 1
        x_nx3, cube_fx8 = self.fc.construct_voxel_grid(fc_res)
        x_nx3 = x_nx3.to(self.device)

        verts, faces, _ = self.fc(x_nx3, sdf, cube_fx8, fc_res)
        faces_np = faces.detach().cpu().numpy().astype(np.int32)
        return verts.detach().cpu().numpy().astype(np.float32), faces_np

    def convert_indices(self, vertices, resolution, aabb):
        # [-0.5, 0.5] → [0, 1] → world space via AABB
        v01 = vertices + 0.5
        aabb_min = np.array(aabb.min, dtype=np.float32)
        aabb_max = np.array(aabb.max, dtype=np.float32)
        return v01 * (aabb_max - aabb_min) + aabb_min


class DualContouring(ExternalMethod):

    def __init__(self, snapshot_path, transform=None, testbed=None):
        super().__init__(snapshot_path, transform, testbed)
        try:
            from diso import DiffDMC  # DISO provides both DMC and DC
        except ImportError:
            raise ImportError("pip install diso")
        import torch
        self.device = torch.device("cuda")
        self.dmc = DiffDMC(dtype=torch.float32).to(self.device)

    def preprocess_density(self, sdf):

        # Get density grid [Z,Y,X] → transpose to [X,Y,Z]
        density_np = np.transpose(sdf, (2, 1, 0)).copy()

        return torch.from_numpy(density_np).float().to(self.device)


    def extract_vt(self, sdf, resolution, aabb, threshold):

        verts, faces = self.dmc(sdf)
        faces_np = faces.detach().cpu().numpy().astype(np.int32)
        return verts.detach().cpu().numpy().astype(np.float32), faces_np

    def convert_indices(self, vertices, resolution, aabb):
        # DISO outputs [0, 1] normalized to its input grid → map to world space
        aabb_min = np.array(aabb.min, dtype=np.float32)
        aabb_max = np.array(aabb.max, dtype=np.float32)
        return vertices * (aabb_max - aabb_min) + aabb_min


# ════════════════════════════════════════════════════════════════════════════
#  Seam-aware stitching (replaces the old global vertex weld)
#
#  Key idea: only vertices inside the inter-chunk overlap slabs can ever be
#  duplicated. So instead of hashing every vertex through a Python dict, we:
#    1. Compute the overlap intervals per axis once (pure geometry).
#    2. Mark seam candidates with a vectorized box test.
#    3. Merge only those candidates with a KD-tree radius query + connected
#       components (tolerance-based -> immune to float drift between chunks,
#       which exact-key hashing is not).
#    4. Deduplicate faces with a vectorized np.unique on sorted index triplets.
# ════════════════════════════════════════════════════════════════════════════

def _axis_chunk_ranges(n_points: int, cs: int, stride: int) -> list:
    """Grid-point ranges [a, b) of each chunk along one axis,
    including the border slide-back. Mirrors the loop logic exactly."""
    import math
    n = max(1, math.ceil((n_points - 1) / stride))
    ranges = []
    for i in range(n):
        a = i * stride
        b = a + cs
        if b > n_points:
            b = n_points
            a = max(0, b - cs)
        ranges.append((a, b))
    return ranges


def _axis_overlap_intervals(ranges: list) -> list:
    """World-index intervals where consecutive chunks overlap."""
    out = []
    for (a1, b1), (a2, b2) in zip(ranges, ranges[1:]):
        lo, hi = max(a1, a2), min(b1, b2)
        if lo < hi:
            out.append((lo, hi))
    return out


def _seam_candidate_mask(verts: np.ndarray,
                         overlap_world: list) -> np.ndarray:
    """Boolean mask of vertices lying inside any padded overlap slab.

    overlap_world: per-axis list of (world_lo, world_hi) intervals.
    """
    mask = np.zeros(len(verts), dtype=bool)
    for axis in range(3):
        for lo, hi in overlap_world[axis]:
            mask |= (verts[:, axis] >= lo) & (verts[:, axis] <= hi)
    return mask


def _merge_seam_vertices(verts: np.ndarray, seam_mask: np.ndarray,
                         eps: float) -> np.ndarray:
    """Index remap merging seam vertices closer than eps.

    Returns remap array of length len(verts): remap[i] = canonical index.
    Non-seam vertices map to themselves.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    remap = np.arange(len(verts), dtype=np.int64)
    seam_idx = np.flatnonzero(seam_mask)
    if len(seam_idx) < 2:
        return remap

    tree = cKDTree(verts[seam_idx])
    pairs = tree.query_pairs(r=eps, output_type="ndarray")
    if len(pairs) == 0:
        return remap

    n = len(seam_idx)
    adj = coo_matrix(
        (np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])),
        shape=(n, n))
    _, labels = connected_components(adj, directed=False)

    # Canonical = first occurrence of each component.
    # seam_idx is ascending, so the first occurrence has the lowest global id.
    _, first_pos = np.unique(labels, return_index=True)
    canonical_global = seam_idx[first_pos]
    remap[seam_idx] = canonical_global[labels]
    return remap


def _stitch_chunks(verts: np.ndarray, faces: np.ndarray,
                   colors: np.ndarray, normals: np.ndarray,
                   overlap_world: list, eps: float) -> tuple:
    """Merge duplicated seam vertices, drop degenerate/duplicate faces,
    compact the vertex buffer. Fully vectorized except the KD-tree query."""

    seam_mask = _seam_candidate_mask(verts, overlap_world)
    n_seam = int(seam_mask.sum())
    print(f"  Stitching: {n_seam:,}/{len(verts):,} seam candidates ...",
          end=" ", flush=True)

    remap = _merge_seam_vertices(verts, seam_mask, eps)
    faces = remap[faces]

    # Degenerate faces (collapsed by the merge)
    keep = ((faces[:, 0] != faces[:, 1]) &
            (faces[:, 1] != faces[:, 2]) &
            (faces[:, 0] != faces[:, 2]))
    faces = faces[keep]

    # Duplicate faces (same cell extracted by two overlapping chunks).
    # Sort indices row-wise so winding permutations collide, keep first
    # occurrence to preserve the original winding.
    if len(faces) > 0:
        key = np.sort(faces, axis=1)
        _, uniq = np.unique(key, axis=0, return_index=True)
        faces = faces[np.sort(uniq)]

    # Compact: drop now-unreferenced duplicate vertices
    used = np.unique(faces)
    new_id = np.full(len(verts), -1, dtype=np.int64)
    new_id[used] = np.arange(len(used), dtype=np.int64)
    faces = new_id[faces].astype(np.int32)
    verts = verts[used]
    colors = colors[used] if colors is not None else None
    normals = normals[used] if normals is not None else None

    print(f"→ {len(verts):,} verts after stitch")
    return verts, faces, colors, normals


class ChunkedExtractor(ExtractionMethod):

    # Cells of overlap between adjacent chunks. 1 is required so that dual
    # methods (FlexiCubes, DualContouring) can emit the faces that cross a
    # chunk boundary: their vertices live INSIDE cells and their faces connect
    # NEIGHBORING cells, so with zero shared cells neither side can build the
    # seam-crossing faces -> holes along every chunk border.
    OVERLAP = 1

    def __init__(self, inner: "ExternalMethod"):
        if not isinstance(inner, ExternalMethod):
            raise TypeError(
                f"ChunkedExtractor requires an ExternalMethod instance, "
                f"got {type(inner).__name__}"
            )
        self.inner     = inner
        self.testbed   = inner.testbed   # shared reference, not a copy
        self.transform = inner.transform

    @property
    def method_label(self) -> str:
        return f"{self.inner.__class__.__name__}_chunked"

    def __repr__(self) -> str:
        return (f"ChunkedExtractor(inner={self.inner.__class__.__name__})")

    def extract(self, testbed, resolution, aabb, threshold):
        import pyngp as ngp
        import math

        # ── Geometry of the full grid ─────────────────────────────────────
        aabb_min = np.array(aabb.min, dtype=np.float32)  # [3] world origin
        aabb_max = np.array(aabb.max, dtype=np.float32)
        diag     = aabb_max - aabb_min                    # [3] world extents

        sz, sy, sx = resolution, resolution, resolution

        # World size of one voxel along each axis (X, Y, Z in world order).
        step = np.array(
            [diag[0] / sx, diag[1] / sy, diag[2] / sz],
            dtype=np.float32
        )

        # ── Chunk grid: cs grid points per chunk, OVERLAP shared cells ────
        cs = 128                              # fixed chunk size (API requirement)
        stride = cs - 1 - self.OVERLAP        # grid-point stride between chunks

        ranges_x = _axis_chunk_ranges(sx, cs, stride)
        ranges_y = _axis_chunk_ranges(sy, cs, stride)
        ranges_z = _axis_chunk_ranges(sz, cs, stride)
        nx, ny, nz = len(ranges_x), len(ranges_y), len(ranges_z)

        # Pre-compute padded world-space overlap slabs per axis (the only
        # regions where duplicated geometry can exist).
        overlap_world = []
        for axis, ranges in enumerate((ranges_x, ranges_y, ranges_z)):
            slabs = []
            for lo, hi in _axis_overlap_intervals(ranges):
                pad = 1.0 * step[axis]  # one voxel of safety on each side
                slabs.append((aabb_min[axis] + lo * step[axis] - pad,
                              aabb_min[axis] + hi * step[axis] + pad))
            overlap_world.append(slabs)

        # Merge tolerance: far below the minimum feature distance of any
        # extraction method, far above accumulated float error across chunks.
        weld_eps = float(step.min()) * 1e-3

        print(f"  [{self.__repr__()}]  "
              f"grid={resolution}x{resolution}x{resolution}  →  "
              f"{nx}×{ny}×{nz} chunks  (chunk_res={cs}, overlap={self.OVERLAP} cell)")

        all_verts: list[np.ndarray] = []
        all_faces: list[np.ndarray] = []
        all_colors: list[np.ndarray] = []
        all_normals: list[np.ndarray] = []
        vert_offset: int = 0

        # ── Iterate chunks ────────────────────────────────────────────────
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):

                    x0, x1 = ranges_x[ix]
                    y0, y1 = ranges_y[iy]
                    z0, z1 = ranges_z[iz]

                    cx = x1 - x0
                    cy = y1 - y0
                    cz = z1 - z0

                    # ── World AABB for this chunk ─────────────────────────
                    chunk_min = aabb_min + np.array([x0, y0, z0], dtype=np.float32) * step
                    chunk_max = aabb_min + np.array([x1, y1, z1], dtype=np.float32) * step

                    chunk_aabb = ngp.BoundingBox(
                        chunk_min.tolist(), chunk_max.tolist())

                    print(f"    ({ix:02d},{iy:02d},{iz:02d})  "
                          f"chunk_res={cx}  "
                          f"Mode: {self.inner.__class__.__name__}...", end=" ", flush=True)


                    # Fetch chunk density grid directly from testbed (guaranteed size 128)
                    chunk_sdf = testbed.get_density_grid(cx, cy, cz, chunk_aabb)

                    # Run the inner pipeline sequence
                    modified_chunk_sdf = self.inner.preprocess_density(chunk_sdf)

                    v, f = self.inner.extract_vt(
                        modified_chunk_sdf, cs,
                        chunk_aabb, threshold)

                    if len(v) == 0:
                        print("empty")
                        continue

                    v = self.inner.convert_indices(v, cs, chunk_aabb)


                    # ── Query colors and normals at chunk vertices ─────
                    print(f"(querying {len(v):,} colors/normals)", end=" ", flush=True)
                    query_data = testbed.query_vertex_colors_and_normals(v.astype(np.float32))
                    c = query_data["colors"].astype(np.float32)
                    n = query_data["normals"].astype(np.float32)

                    all_verts.append(v)
                    all_faces.append(f + vert_offset)
                    all_colors.append(c)
                    all_normals.append(n)

                    vert_offset += len(v)
                    print(f"-> {len(v):,} verts")


        # ── Stitch: seam-only merge + face dedup ─────────────────────────
        if not all_verts:
            return (np.zeros((0, 3), dtype=np.float32),
                    np.zeros((0, 3), dtype=np.int32),
                    np.zeros((0, 3), dtype=np.float32),
                    np.zeros((0, 3), dtype=np.float32), 0, 0)

        verts = np.concatenate(all_verts, axis=0)
        faces = np.concatenate(all_faces, axis=0).astype(np.int64)
        colors = np.concatenate(all_colors, axis=0)
        normals = np.concatenate(all_normals, axis=0)

        verts, faces, colors, normals = _stitch_chunks(
            verts, faces, colors, normals, overlap_world, weld_eps)

        return verts, faces, colors, normals, len(verts), len(faces)

class ComparisonMethod(Method):

    MAX_RES = {
        "CudaMC": 512, "ChunkedCudaMC": float('inf'),
        "PyMCubes": 512, "FlexiCubes": 256, "DualContouring": 512,
        "ChunkedExtractor": float('inf'),
    }

    def __init__(self, snapshot_path, transform, methods=None,
                 resolutions=None, base_name="mesh",
                 chunked=False, force=False, also_chunked=False):
        super().__init__(snapshot_path, transform)
        self.snapshot_path = snapshot_path
        self.base_name     = base_name
        self.methods       = methods or ["cuda", "chunkedCuda", "pymcubes", "flexicubes"]
        self.resolutions   = resolutions or [128, 256, 512]
        self.chunked       = chunked
        self.force         = force
        self.also_chunked  = also_chunked  # run both normal + chunked in compare mode

    def _max_res_for(self, method) -> float:
        cls_name = method.__class__.__name__
        if cls_name == "ChunkedExtractor":
            return float('inf')
        return self.MAX_RES.get(cls_name, 9999)

    def run(self, threshold=0.0, output_dir="comparison_results", count_time=False):
        import pathlib, gc

        object_root = pathlib.Path(output_dir) / self.base_name
        object_root.mkdir(parents=True, exist_ok=True)

        summary = self._load_existing_summary(object_root)

        for mode in self.methods:
            # Build variant list: always include the base mode.
            # If --also_chunked, append a chunked variant for external methods.
            variants = [(mode, self.chunked)]
            if self.also_chunked and mode in EXTERNAL_MAP and not self.chunked:
                variants.append((mode, True))  # add chunked twin

            for mode_name, is_chunked in variants:
                try:
                    method = _build_method(
                        mode_name, self.snapshot_path, self.transform,
                        is_chunked, testbed=self.testbed)
                except ValueError as e:
                    print(f"Skipping {mode_name} (chunked={is_chunked}): {e}")
                    continue

                label = (method.method_label
                         if hasattr(method, "method_label")
                         else method.__class__.__name__)

                method_dir = object_root / label
                method_dir.mkdir(exist_ok=True)
                summary.setdefault(label, {})

                limit = self._max_res_for(method)

                for res in self.resolutions:
                    if res > limit:
                        print(f"SKIPPED: {label} @ {res} exceeds limit {limit}")
                        summary[label][res] = {"status": "SKIPPED"}
                        continue

                    out = method_dir / f"{self.base_name}_{res}_{label}.obj"

                    if out.exists() and not self.force:
                        print(f"EXISTS (skip): {out.name}  — use --force / -f to overwrite")
                        if res not in summary[label]:
                            summary[label][res] = {
                                "status": "existing", "verts": "?", "faces": "?", "time": "?"}
                        continue

                    try:
                        elapsed, v, f = method.run(
                            resolution=res, threshold=threshold,
                            output_path=str(out), count_time=count_time)
                        summary[label][res] = {
                            "status": "success", "time": elapsed, "verts": v, "faces": f}
                    except Exception as e:
                        print(f"Error {label} @ {res}: {e}")
                        summary[label][res] = {"status": "failed", "error": str(e)}
                    finally:
                        self._cleanup_gpu()
                        gc.collect()

        self._save_summary(summary, object_root)

    def _load_existing_summary(self, object_root) -> dict:
        import pathlib
        summary_file = object_root / f"{self.base_name}_comparison_summary.txt"
        if not summary_file.exists():
            return {}

        summary = {}
        with open(summary_file) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("=") or line.startswith("-") \
                        or line.startswith("COMP") or line.startswith("Meth"):
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) < 6:
                    continue
                method, res_str, status = parts[0], parts[1], parts[2]
                try:
                    res = int(res_str)
                except ValueError:
                    continue
                summary.setdefault(method, {})
                if status == "OK":
                    try:
                        summary[method][res] = {
                            "status": "existing",
                            "time":  float(parts[3]) if parts[3] != "N/A" else "?",
                            "verts": int(parts[4].replace(",", "")) if parts[4] != "N/A" else "?",
                            "faces": int(parts[5].replace(",", "")) if parts[5] != "N/A" else "?",
                        }
                    except (ValueError, IndexError):
                        summary[method][res] = {"status": "existing"}
                else:
                    summary[method][res] = {"status": "existing"}
        return summary

    def _save_summary(self, results, output_path):
        summary_file = output_path / f"{self.base_name}_comparison_summary.txt"
        with open(summary_file, "w") as fh:
            fh.write(f"COMPARISON FOR OBJECT: {self.base_name}\n")
            fh.write("=" * 80 + "\n")
            fh.write(f"{'Method':<24} | {'Res':<5} | {'Status':<8} | "
                     f"{'Time(s)':<9} | {'Vertices':<11} | {'Faces':<11}\n")
            fh.write("-" * 80 + "\n")
            for method, ress in results.items():
                for res, d in ress.items():
                    st = d.get("status", "?")
                    if st in ("success", "existing"):
                        t  = f"{d['time']:.2f}" if isinstance(d.get("time"), float) else str(d.get("time", "?"))
                        v  = f"{d['verts']:,}"   if isinstance(d.get("verts"), int)  else str(d.get("verts", "?"))
                        fc = f"{d['faces']:,}"   if isinstance(d.get("faces"), int)  else str(d.get("faces", "?"))
                        tag = "OK" if st == "success" else "OK(old)"
                        fh.write(f"{method:<24} | {res:<5} | {tag:<8} | "
                                 f"{t:<9} | {v:<11} | {fc:<11}\n")
                    else:
                        fh.write(f"{method:<24} | {res:<5} | "
                                 f"{st:<8} | {'N/A':<9} | {'N/A':<11} | {'N/A':<11}\n")
        print(f"\nSummary saved to {summary_file}")

    def _cleanup_gpu(self):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass


EXTERNAL_MAP = {"pymcubes": PyMCubes, "flexicubes": FlexiCubes, "dc": DualContouring}
NATIVE_MAP   = {"cuda": CudaMC, "chunkedCuda": ChunkedCudaMC}

def _build_method(mode, snapshot_path, transform, chunked, testbed=None):
    if mode in NATIVE_MAP:
        if chunked:
            print(f"WARNING: --chunked ignored for native '{mode}' (use chunkedCuda).")
        return NATIVE_MAP[mode](snapshot_path, transform, testbed=testbed)

    if mode not in EXTERNAL_MAP:
        raise ValueError(f"Unknown mode: {mode}. "
                         f"Choices: {list(NATIVE_MAP)+list(EXTERNAL_MAP)}")

    inner = EXTERNAL_MAP[mode](snapshot_path, transform, testbed=testbed)
    return ChunkedExtractor(inner) if chunked else inner


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract mesh from NeRF")
    parser.add_argument("snapshot_path", type=str)
    parser.add_argument("--transform", type=str, default=None,
                        help="Path to JSON file containing world transform")
    parser.add_argument("--mode", type=str,
                        choices=["cuda", "pymcubes", "chunkedCuda", "flexicubes", "dc", "compare"],
                        default="cuda")
    parser.add_argument("--res",    type=int,   default=256)
    parser.add_argument("--thresh", type=float, default=0)
    parser.add_argument("--name",   type=str,   default="mesh",
                        help="Base name for output files")
    parser.add_argument("--count_time", action="store_true")
    parser.add_argument("--compare_methods", type=str,
                        default="cuda,chunkedCuda,pymcubes,flexicubes,dc")
    parser.add_argument("--compare_res",  type=str, default="128,256,512")
    parser.add_argument("--compare_out",  type=str, default="comparison_results")
    parser.add_argument("--force", "-f",  action="store_true",
                        help="Overwrite existing results instead of skipping them")
    parser.add_argument("--chunked", action="store_true",
                        help="Use chunked extraction (external methods only). Chunk size fixed at 128.")
    parser.add_argument("--also_chunked", action="store_true",
                        help="In compare mode, also run a chunked variant of each external method "
                             "alongside the normal one (e.g. PyMCubes + PyMCubes_chunked).")

    args = parser.parse_args()

    transform = None
    if args.transform is not None:
        with open(args.transform, "r") as f:
            import json
            transform = json.load(f)

    if args.mode == "compare":
        methods     = [m.strip() for m in args.compare_methods.split(",")]
        resolutions = [int(r) for r in args.compare_res.split(",")]
        ComparisonMethod(
            args.snapshot_path, transform,
            methods=methods, resolutions=resolutions,
            base_name=args.name,
            chunked=args.chunked,
            force=args.force,
            also_chunked=args.also_chunked,
        ).run(threshold=args.thresh, output_dir=args.compare_out,
              count_time=args.count_time)
    else:
        method = _build_method(args.mode, args.snapshot_path, transform, args.chunked)
        method.run(resolution=args.res, threshold=args.thresh,
                   name=args.name, count_time=args.count_time)