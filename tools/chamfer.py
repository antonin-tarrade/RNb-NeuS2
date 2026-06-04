import numpy as np
import open3d as o3d
import argparse
import torch
from pathlib import Path
from pytorch3d.loss import chamfer_distance

def get_points(path, n_samples):
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        raise ValueError(f"Empty mesh: {path}")
    
    pcd = mesh.sample_points_uniformly(number_of_points=n_samples)
    pts = np.asarray(pcd.points)
    
    
    return torch.from_numpy(pts).float().cuda().unsqueeze(0)

def compute_metrics(gt_path, pred_path, n_samples):
    p_gt = get_points(gt_path, n_samples)
    p_pred = get_points(pred_path, n_samples)

    dist, _ = chamfer_distance(p_pred, p_gt)
    return np.sqrt(dist.item())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=str)
    parser.add_argument("--pred", type=str)
    parser.add_argument("--dir", type=str)
    parser.add_argument("--samples", type=int, default=500000)
    parser.add_argument("--out", type=str, default="results.txt", help="Output txt file")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    output_lines = []

    if args.gt and args.pred:
        score = compute_metrics(args.gt, args.pred, args.samples)
        line = f"Chamfer Distance: {score:.6f}"
        print(line)
        output_lines.append(line)

    elif args.dir:
        root = Path(args.dir)
        objs = list(root.glob("*.obj"))
        plys = list(root.glob("*.ply"))
        gt_path = objs[0] if objs else (plys[0] if plys else None)
        
        if not gt_path:
            print("No GT found")
            return

        results = {}
        for p in root.rglob("*"):
            if p.suffix not in [".obj", ".ply"] or p.resolve() == gt_path.resolve():
                continue
            
            try:
                parts = p.stem.split('_')
                if len(parts) < 3: continue
                
                res, method = int(parts[-2]), parts[-1]
                score = compute_metrics(gt_path, p, args.samples)
                
                if method not in results: results[method] = {}
                results[method][res] = score
            except Exception as e:
                print(f"Error {p.name}: {e}")

        header1 = f"\nResults for {root.name} (Samples: {args.samples})"
        header2 = f"{'Method':<15} | {'Res':<6} | {'CD':<15}"
        sep = "-" * 40
        
        output_lines.extend([header1, header2, sep])
        print(header1)
        print(header2)
        print(sep)

        for m in sorted(results.keys()):
            for r in sorted(results[m].keys()):
                line = f"{m:<15} | {r:<6} | {results[m][r]:.6f}"
                print(line)
                output_lines.append(line)
            output_lines.append(sep)
            print(sep)

    filename = args.out.split('.')[0] + '_' + root.name + ".txt"
    with open(filename, "w") as f:
        f.write("\n".join(output_lines))
    print(f"\nResults saved to {args.out}")

if __name__ == "__main__":
    main()