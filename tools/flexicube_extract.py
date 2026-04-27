import torch
import numpy as np
import argparse
import sys
from flexicubes import FlexiCubes

def main():
    parser = argparse.ArgumentParser(description='Extract mesh from density grid using flexicubes')
    parser.add_argument('--input', type=str, default='density_grid.npy',
                        help='Path to input density grid .npy file (default: density_grid.npy)')
    parser.add_argument('--output', type=str, default='mesh',
                        help='Output mesh filename without extension (default: mesh)')
    
    args = parser.parse_args()
    
    print(f"Loading density grid from {args.input}...")
    try:
        sdf = torch.from_numpy(np.load(args.input)).cuda()
        print(f"Loaded density grid with shape: {sdf.shape}")
    except FileNotFoundError:
        print(f"Error: File '{args.input}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    res = sdf.shape[0]  # Assuming cubic grid
    print(f"Extracting mesh with resolution {res}...")
    
    
    fc = FlexiCubes()
    x_nx3, cube_fx8 = fc.construct_voxel_grid(res)
    

    verts, faces, L_dev = fc(x_nx3, sdf, cube_fx8, res)
    
    print(f"Extraction complete!")
    print(f"Vertices: {verts.shape}, Faces: {faces.shape}")
    print(f"Mesh extracted with {len(verts)} vertices and {len(faces)} faces")

    # Save the mesh to an OBJ file
    output_path = f"{args.output}.obj"
    print(f"Saving mesh to {output_path}...")
    try:
        with open(output_path, 'w') as f:
            for v in verts:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in faces:
                # OBJ format uses 1-based indexing
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        print(f"Mesh saved successfully to {output_path}")
    except Exception as e:
        print(f"Error saving mesh: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()