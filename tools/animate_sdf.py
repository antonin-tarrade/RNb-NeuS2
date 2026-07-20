"""
animate_sdf.py
==============
Visualizes the Signed Distance Field from a NeuS2 / RNb-NeuS2 snapshot as
a video. Companion to animate_marching_cubes.py (same inputs, same dark theme).

TWO MODES — pick with --mode:

  slices  A plane sweeps through the SDF volume showing scalar values as a
          diverging heatmap (blue = positive/outside, red = negative/inside,
          white = the surface at threshold). The zero-level contour is drawn
          as a bold white line on every slice. Ends with the extracted surface
          revealed so you can connect "this is what the SDF looks like" to
          "this is the mesh marching cubes extracts from it."

  shells  Multiple isosurfaces at different SDF threshold levels are revealed
          one by one — outermost shell first (large positive SDF = far outside
          the object), working inward to the surface and beyond (negative SDF
          = inside the object). Each shell is color-coded by its distance value.
          Ends with a 360° turntable spin of the complete "onion."

  both    Runs slices first, then shells, in one video.

INPUT : NeuS2 / RNb-NeuS2 snapshot (.msgpack)
OUTPUT: only the .mp4 video — no mesh files written.

REQUIREMENTS: numpy, matplotlib, imageio, imageio-ffmpeg, mcubes, pyngp

USAGE:
    python animate_sdf.py snapshot.msgpack --mode shells --res 40 --out_video sdf.mp4
    python animate_sdf.py snapshot.msgpack --mode slices --res 48 --axis 2
    python animate_sdf.py snapshot.msgpack --mode both   --res 32

TIPS:
  - --res 32-48 is the sweet spot: low enough that individual shells/slices are
    clearly visible, high enough that the shape is recognisable.
  - For slices, higher --res looks better (more detail in the heatmap).
  - For shells, lower --res is faster and visually clearer (fewer noisy faces).
  - --n_shells 7 is a good default; use 5 for cleaner/faster, 9 for more detail.
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import imageio


# ════════════════════════════════════════════════════════════════════════
#  Style — matches animate_marching_cubes.py
# ════════════════════════════════════════════════════════════════════════

BG_COLOR = "#1c1d21"
GRID_BOUNDS_COLOR = "#5a5d66"
MESH_FACE_COLOR = (0.40, 0.70, 0.95)
MESH_EDGE_COLOR = (0.06, 0.06, 0.08)

# Diverging colormap for SDF values:
#   blue  = positive SDF = outside the surface
#   white = zero = the surface itself
#   red   = negative SDF = inside the object
SDF_CMAP = "RdBu_r"


# ════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ════════════════════════════════════════════════════════════════════════

def grid_bounds_segments(N):
    """12 edge segments of the N³ bounding cube."""
    c = np.array([
        (0, 0, 0), (N, 0, 0), (N, N, 0), (0, N, 0),
        (0, 0, N), (N, 0, N), (N, N, N), (0, N, N),
    ])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    return [(c[a], c[b]) for a, b in edges]


def add_bounds(ax, segments):
    """Draw bounding box as 12 separate artists so each edge depth-sorts
    correctly against other geometry (same fix as animate_marching_cubes.py)."""
    for p1, p2 in segments:
        ax.add_collection3d(Line3DCollection(
            [(p1, p2)], colors=GRID_BOUNDS_COLOR, linewidths=0.8, alpha=0.6))


def setup_ax(ax, N):
    """Standard dark-theme 3D axes reset — call at the top of every frame."""
    ax.clear()
    ax.set_facecolor(BG_COLOR)
    ax.xaxis.set_pane_color((0, 0, 0, 0))
    ax.yaxis.set_pane_color((0, 0, 0, 0))
    ax.zaxis.set_pane_color((0, 0, 0, 0))
    ax.set_xlim(0, N); ax.set_ylim(0, N); ax.set_zlim(0, N)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


def load_testbed(snapshot_path):
    try:
        import pyngp as ngp
    except ImportError:
        print("Error: pyngp not found. Run in the env with your NeuS2 build.")
        sys.exit(1)
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    if not os.path.exists(snapshot_path):
        print(f"Error: snapshot not found: {snapshot_path}")
        sys.exit(1)
    print(f"Loading snapshot from {snapshot_path}...")
    testbed.load_snapshot(snapshot_path)
    return testbed


def get_sdf_grid(testbed, resolution):
    """Pull SDF/density grid and return as (res, res, res) in [X, Y, Z] order.

    get_density_grid() requires the resolution to be a multiple of 128 (same
    constraint as the chunked extractor). We always fetch at the nearest valid
    multiple of 128, then downsample to the requested resolution with trilinear
    interpolation so you can still use low values like --res 32 for animations.
    """
    from scipy.ndimage import zoom

    aabb = testbed.render_aabb
    fetch_res = max(128, int(np.ceil(resolution / 128)) * 128)

    if fetch_res != resolution:
        print(f"  (get_density_grid requires multiples of 128 — "
              f"fetching at {fetch_res}, downsampling to {resolution})")

    raw = testbed.get_density_grid(fetch_res, fetch_res, fetch_res, aabb)
    grid = np.transpose(np.asarray(raw), (2, 1, 0))  # [Z,Y,X] -> [X,Y,Z]

    if fetch_res != resolution:
        grid = zoom(grid, resolution / fetch_res, order=1)  # trilinear

    return grid


def extract_mesh_at_level(grid, level):
    """Return (vertices, faces) of the isosurface at `level` in voxel coords.
    Vertices are in [X, Y, Z] order matching the grid.

    NOTE: grid must already be in [X, Y, Z] order (as returned by get_sdf_grid).
    We pass it to mcubes directly — mcubes iterates in C order (last axis
    fastest) and returns vertex indices along the same axes, so the output is
    also in [X, Y, Z] order, consistent with animate_marching_cubes.py's
    custom MC. Do NOT transpose before passing — that would swap X and Z,
    rotating the mesh by 90° around Y.
    """
    import mcubes
    v, f = mcubes.marching_cubes(grid, level)
    return v.astype(np.float32), f.astype(np.int32)


def sdf_colormap_range(grid, threshold):
    """Symmetric colormap range around `threshold`, clipping outlier voxels."""
    p_lo = np.percentile(grid, 2)
    p_hi = np.percentile(grid, 98)
    half = max(abs(p_lo - threshold), abs(p_hi - threshold))
    return threshold - half, threshold + half


def write_frame(fig, writer):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    writer.append_data(buf[:, :, :3])


def closing_spin(ax, fig, writer, N, bounds, v, f, azim_ref, fps, seconds):
    """360° turntable spin of the finished object, used by both modes."""
    spin_frames = max(1, int(fps * seconds))
    step = 360.0 / spin_frames
    azim = azim_ref[0]

    for _ in range(spin_frames):
        setup_ax(ax, N)
        azim += step
        ax.view_init(elev=25, azim=azim)
        add_bounds(ax, bounds)
        if len(f) > 0:
            tris = v[f]
            pc = Poly3DCollection(
                tris, facecolor=MESH_FACE_COLOR, edgecolor=MESH_EDGE_COLOR,
                linewidths=0.15, alpha=0.9)
            ax.add_collection3d(pc)
        write_frame(fig, writer)

    azim_ref[0] = azim


# ════════════════════════════════════════════════════════════════════════
#  Mode 1: SDF slice sweep
# ════════════════════════════════════════════════════════════════════════

def run_slices(grid, writer, fps, axis, rotate_per_frame, frames_per_slice,
               figsize, dpi, threshold=0.0):
    """
    Sweeps a plane through the SDF volume along `axis` (bottom→top, then
    top→bottom), holding each slice position for `frames_per_slice` frames.
    A colorbar on the right explains the heatmap: blue = positive SDF (outside
    the object), red = negative SDF (inside), white = zero = the surface.
    """
    N = grid.shape[0]
    bounds = grid_bounds_segments(N)
    vmin, vmax = sdf_colormap_range(grid, threshold)

    idx = np.arange(N)
    G0, G1 = np.meshgrid(idx, idx)   # G0[j,i]=i, G1[j,i]=j

    cmap_obj = plt.get_cmap(SDF_CMAP)
    sm = plt.cm.ScalarMappable(cmap=cmap_obj,
                                norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])

    # ── Figure layout: 3D axes on the left, thin colorbar axes on the right ─
    # We don't use subplots_adjust(right=1) here so there's room for the bar.
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(BG_COLOR)
    ax = fig.add_axes([0.0, 0.0, 0.82, 1.0], projection='3d')   # 3D view
    cax = fig.add_axes([0.84, 0.15, 0.03, 0.70])                 # colorbar strip

    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("SDF value", color='white', fontsize=10, labelpad=8)
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white', fontsize=8)
    cb.outline.set_edgecolor('#5a5d66')
    cax.set_facecolor(BG_COLOR)

    # Mark the zero level (surface) with a white tick + label
    cb.ax.axhline(y=(threshold - vmin) / (vmax - vmin),
                  color='white', linewidth=1.5, linestyle='--', alpha=0.8)
    # cb.ax.text(1.35, (threshold - vmin) / (vmax - vmin), 'surface',
    #            transform=cb.ax.transAxes,
    #            color='white', fontsize=7, va='center', ha='left')

    azim = -60.0
    t_start = time.time()
    n_frames = 0

    def draw_slice(k):
        nonlocal azim, n_frames
        setup_ax(ax, N)
        azim += rotate_per_frame
        ax.view_init(elev=25, azim=azim)
        add_bounds(ax, bounds)

        if axis == 0:
            sl = grid[k, :, :].T
            ax.contourf(G0, G1, sl, zdir='x', offset=k,
                        cmap=SDF_CMAP, vmin=vmin, vmax=vmax, levels=30, alpha=0.88)
            ax.contour(G0, G1, sl, levels=[threshold], zdir='x', offset=k,
                       colors='white', linewidths=2.5, alpha=0.95)
        elif axis == 1:
            sl = grid[:, k, :].T
            ax.contourf(G0, G1, sl, zdir='y', offset=k,
                        cmap=SDF_CMAP, vmin=vmin, vmax=vmax, levels=30, alpha=0.88)
            ax.contour(G0, G1, sl, levels=[threshold], zdir='y', offset=k,
                       colors='white', linewidths=2.5, alpha=0.95)
        else:
            sl = grid[:, :, k].T
            ax.contourf(G0, G1, sl, zdir='z', offset=k,
                        cmap=SDF_CMAP, vmin=vmin, vmax=vmax, levels=30, alpha=0.88)
            ax.contour(G0, G1, sl, levels=[threshold], zdir='z', offset=k,
                       colors='white', linewidths=2.5, alpha=0.95)

        write_frame(fig, writer)
        n_frames += 1

    # ── Forward pass: 0 → N-1 ────────────────────────────────────────────
    print(f"Forward sweep (0 → {N-1}) ...")
    for k in range(N):
        for _ in range(frames_per_slice):
            draw_slice(k)
        if k % 8 == 0:
            print(f"  slice {k:3d}/{N-1}", end="\r")
    print()

    # ── Backward pass: N-1 → 0 ───────────────────────────────────────────
    print(f"Backward sweep ({N-1} → 0) ...")
    for k in range(N - 1, -1, -1):
        for _ in range(frames_per_slice):
            draw_slice(k)
        if k % 8 == 0:
            print(f"  slice {k:3d}/{N-1}", end="\r")
    print()

    plt.close(fig)
    print(f"Slices done: {n_frames} frames, {time.time()-t_start:.1f}s")


# ════════════════════════════════════════════════════════════════════════
#  Mode 2: nested isosurface "onion shells"
# ════════════════════════════════════════════════════════════════════════

def run_shells(grid, writer, fps, n_shells, hold_frames, rotate_per_frame,
               final_spin_seconds, figsize, dpi, threshold=0.0):
    """
    Reveals nested isosurfaces at N different SDF levels, one by one from
    the outermost (most positive SDF = farthest outside the object) to the
    innermost (most negative SDF = deepest inside). Each shell is colored by
    its SDF value using the same diverging colormap as slice mode. Ends with
    a 360° turntable spin of the complete "onion."
    """
    N = grid.shape[0]
    bounds = grid_bounds_segments(N)
    vmin, vmax = sdf_colormap_range(grid, threshold)
    cmap = plt.get_cmap(SDF_CMAP)
    norm_fn = plt.Normalize(vmin=vmin, vmax=vmax)

    # Shell levels: evenly spaced across the SDF value range, outermost first
    levels = np.linspace(vmax * 0.85, vmin * 0.85, n_shells)

    # Alpha: outer shells (far from surface) are more transparent so you can
    # see through them to the ones inside.  Inner shells are more opaque.
    # We linearly ramp from alpha_outer to alpha_inner.
    alpha_outer = 0.18
    alpha_inner = 0.72
    alphas = np.linspace(alpha_outer, alpha_inner, n_shells)

    # Pre-extract all isosurfaces (fast at low res)
    print(f"Extracting {n_shells} isosurface shells...")
    shells = []
    for i, level in enumerate(levels):
        v, f = extract_mesh_at_level(grid, level)
        color = cmap(norm_fn(level))[:3]
        shells.append((level, v, f, color, alphas[i]))
        sign = "+" if level > threshold else ""
        print(f"  shell {i+1}/{n_shells}  level={sign}{level:.3f}  "
              f"{len(f)} faces  alpha={alphas[i]:.2f}")

    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(BG_COLOR)
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    azim = [-60.0]
    t_start = time.time()
    n_frames = 0
    visible_shells = []

    def render_current():
        setup_ax(ax, N)
        azim[0] += rotate_per_frame
        ax.view_init(elev=25, azim=azim[0])
        add_bounds(ax, bounds)
        # Draw accumulated shells, outermost (most transparent) first so
        # painter's algorithm gives the right visual layering.
        for _level, sv, sf, scolor, salpha in visible_shells:
            if len(sf) > 0:
                tris = sv[sf]
                pc = Poly3DCollection(
                    tris, facecolor=scolor, edgecolor=(*scolor, 0.25),
                    linewidths=0.1, alpha=salpha)
                ax.add_collection3d(pc)
        write_frame(fig, writer)

    # Brief opening: just the bounding box so the viewer sees the empty volume
    print("Rendering opening hold...")
    for _ in range(max(1, int(fps * 0.8))):
        render_current()
        n_frames += 1

    # Reveal shells one at a time
    print("Revealing shells...")
    for i, shell in enumerate(shells):
        level, sv, sf, scolor, salpha = shell
        visible_shells.append(shell)
        sign = "+" if level > threshold else ""
        print(f"  revealing shell {i+1}/{n_shells}  level={sign}{level:.3f}", end="\r")
        for _ in range(hold_frames):
            render_current()
            n_frames += 1

    print()

    # Closing 360° spin
    # For the spin, show all shells (already in visible_shells)
    print("Closing spin...")
    spin_frames = max(1, int(fps * final_spin_seconds))
    step = 360.0 / spin_frames
    _azim = azim[0]
    for _ in range(spin_frames):
        setup_ax(ax, N)
        _azim += step
        ax.view_init(elev=25, azim=_azim)
        add_bounds(ax, bounds)
        for _level, sv, sf, scolor, salpha in visible_shells:
            if len(sf) > 0:
                tris = sv[sf]
                pc = Poly3DCollection(
                    tris, facecolor=scolor, edgecolor=(*scolor, 0.25),
                    linewidths=0.1, alpha=salpha)
                ax.add_collection3d(pc)
        write_frame(fig, writer)
        n_frames += 1

    plt.close(fig)
    print(f"Shells done: {n_frames} frames, {time.time()-t_start:.1f}s")


# ════════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("snapshot_path")
    parser.add_argument("--mode", choices=["slices", "shells", "both"],
                        default="shells",
                        help="Visualization mode (default: shells)")
    parser.add_argument("--res", type=int, default=40,
                        help="SDF grid resolution. 32-48 recommended. "
                             "Slices look better at higher res; shells "
                             "look cleaner at lower res. Default 40.")
    parser.add_argument("--thresh", type=float, default=0.0,
                        help="SDF threshold = the surface level (default 0)")
    parser.add_argument("--axis", type=int, default=2, choices=[0, 1, 2],
                        help="Sweep axis for slice mode: 0=X, 1=Y, 2=Z "
                             "(default 2)")
    parser.add_argument("--n_shells", type=int, default=7,
                        help="Number of isosurface shells to show "
                             "(default 7). Odd numbers center a shell "
                             "exactly on the surface.")
    parser.add_argument("--hold_frames", type=int, default=18,
                        help="Frames to hold on each newly revealed shell "
                             "(default 18 ≈ 0.75s at 24fps)")
    parser.add_argument("--frames_per_slice", type=int, default=4,
                        help="[slices mode] Frames held on each slice position. "
                             "At 24fps, 4 frames ≈ 0.17s per slice. "
                             "Raise to 8-12 for a much slower, more readable "
                             "sweep. Default 4.")
    parser.add_argument("--mesh_hold_seconds", type=float, default=3.0,
                        help="[slices mode] How long the final mesh is shown "
                             "after the sweep (no rotation). Default 3s.")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--rotate_per_frame", type=float, default=0.25,
                        help="Camera rotation per frame during build phase "
                             "(degrees, default 0.25)")
    parser.add_argument("--final_spin_seconds", type=float, default=4.0,
                        help="Duration of the closing 360° spin (default 4s)")
    parser.add_argument("--fig_w", type=float, default=8.0)
    parser.add_argument("--fig_h", type=float, default=6.0)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--out_video", default="sdf_visualization.mp4")

    args = parser.parse_args()

    testbed = load_testbed(args.snapshot_path)
    print(f"Pulling {args.res}³ SDF grid...")
    grid = get_sdf_grid(testbed, args.res)
    print(f"  SDF range: [{grid.min():.3f}, {grid.max():.3f}]  "
          f"surface threshold: {args.thresh}")

    figsize = (args.fig_w, args.fig_h)

    writer = imageio.get_writer(
        args.out_video, fps=args.fps, codec="libx264", quality=8)

    if args.mode in ("slices", "both"):
        print("\n── Phase 1: slice sweep ──")
        run_slices(grid, writer, args.fps, args.axis, args.rotate_per_frame,
                   args.frames_per_slice, figsize, args.dpi, args.thresh)

    if args.mode in ("shells", "both"):
        print("\n── Phase 2: isosurface shells ──")
        run_shells(grid, writer, args.fps, args.n_shells, args.hold_frames,
                   args.rotate_per_frame, args.final_spin_seconds,
                   figsize, args.dpi, args.thresh)

    writer.close()
    print(f"\nSaved {args.out_video}")