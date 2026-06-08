# eval_pipeline

Evaluation pipeline for 3D surface reconstruction benchmarks. Computes Chamfer Distance, F-score, precision/recall curves, and generates metric visualizations (accuracy/completeness heatmaps, curvature, visibility).

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
# For visualization (requires OSMesa for headless rendering):
pip install -e ".[visualization]"
```

## Quick start

### 1. Prepare your data

```
data_root/
  ├── sfm.json              # Camera parameters (AliceVision SfMData format)
  ├── cameras.npz           # ... or NumPy format
  └── masks/                # (optional) Object segmentation masks
      ├── 000.png
      └── ...

eval_root/my_object/
  ├── Groundtruth/
  │   └── gt_mesh.ply       # Ground truth mesh (CT scan, etc.)
  └── my_method/
      └── results_raw/
          └── mesh.ply       # Reconstructed mesh to evaluate
```

### 2. Create a config

Copy `config/example.yaml` and set your paths. See `config/default.yaml` for all options.

### 3. Run the pipeline

```bash
# Step 1: Preprocess ground truth (sample point cloud, compute curvature & visibility)
eval-pipeline -c config/my_config.yaml preprocess-gt -o my_object

# Step 2: Clean up reconstructed mesh (remove non-visible parts)
eval-pipeline -c config/my_config.yaml cleanup -o my_object -m my_method

# Step 3: Evaluate (Chamfer distance, F-score)
eval-pipeline -c config/my_config.yaml evaluate -o my_object -m my_method

# Step 4: Visualize (accuracy/completeness heatmaps)
eval-pipeline -c config/my_config.yaml visualize -o my_object -m my_method
```

## Pipeline stages

### Preprocess GT (`preprocess-gt`)

Prepares the ground truth mesh for evaluation:
- Samples a uniform point cloud (`gt_pcd.npy`) from the mesh surface via upsampling + radius-based downsampling at the configured density (default: 0.05mm spacing)
- Computes per-vertex curvature via local quadratic surface fitting in a neighborhood radius (default: 2mm), transferred to the point cloud
- Computes per-point visibility count across all camera views (0-84)
- Optionally cleans the GT mesh by removing non-visible regions

### Cleanup (`cleanup`)

Removes non-visible parts from the reconstructed mesh:
- Projects each vertex into all camera views
- Checks occlusion via ray casting against the mesh
- Optionally filters by 2D segmentation masks (with dilation)
- Keeps only vertices visible from at least one view

### Evaluate (`evaluate`)

Computes distance metrics between reconstruction and ground truth:
- Samples both meshes to uniform point clouds
- Computes bidirectional nearest-neighbor distances (data→GT for accuracy, GT→data for completeness)
- Chamfer Distance: mean of both directions
- F-score at configurable thresholds
- Saves distances for later analysis (curves, per-region metrics)

### Visualize (`visualize`)

Renders metric heatmaps from configured camera views:
- **accuracy**: data→GT distance on the reconstructed mesh (jet colormap)
- **completeness**: GT→data distance on the GT mesh (jet colormap)
- **uniform**: plain gray render (both meshes)
- **visibility**: view count on the GT mesh (viridis colormap)
- **curvature**: principal curvature on the GT mesh (coolwarm colormap)

Supports three modes for excluded regions (`exclude_mode`):
- `none`: no masking, full heatmap everywhere
- `gray`: excluded zones rendered in gray
- `remove`: excluded faces removed from the mesh before rendering

Crop bounding boxes are computed from the GT uniform render and saved to `bbox.json` for consistent framing across methods. Output images are capped at 1000px (configurable).

### Additional commands

```bash
# Compute dense precision/recall/F-score curves
eval-pipeline -c config.yaml curves -o my_object -m my_method --n-thresholds 200 --max-threshold 4.0

# Recompute metrics from saved distances (no re-evaluation needed)
eval-pipeline -c config.yaml recompute -o my_object -m my_method

# Generate challenge masks from colored PLY files
eval-pipeline -c config.yaml preprocess-challenges -o my_object

# Generate LaTeX comparison table
eval-pipeline -c config.yaml latex

# Watch for new meshes and auto-submit SLURM jobs
eval-pipeline -c config.yaml watch
```

## Configuration reference

See `config/default.yaml` for all available options with documentation.

Key parameters:
- `evaluation.downsample_density`: point cloud sampling density in mm (default: 0.05)
- `evaluation.max_dist`: outlier distance threshold in mm (default: 4.0)
- `evaluation.curvature_radius`: neighborhood radius for curvature estimation in mm (default: 2.0)
- `visualization.scale`: render resolution multiplier (default: 1.0 = native camera resolution)
- `visualization.max_dist`: heatmap color range max in mm
- `visualization.exclude_mode`: handling of excluded regions (`none`/`gray`/`remove`)

## Camera formats

Supported camera input formats (searched in order):
1. `cameras.npz` — NumPy archive with projection matrices
2. `sfm.json` — AliceVision SfMData JSON
3. `cameras.json` — simple JSON format
4. `*.sfm` — AliceVision SfM files

## SLURM support

Set `execution.mode: slurm` in config to submit jobs to a SLURM cluster. The `watch` command auto-detects new meshes and submits chained cleanup→evaluate→visualize jobs with proper dependencies.
