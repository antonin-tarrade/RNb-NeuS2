import numpy as np
import mcubes
import sys
import os
import argparse

# 1. Import the NGP bindings
try:
    import pyngp as ngp
    ##TODO : look for build folder relatively to this script 
except ImportError:
    print("Error: pyngp not found.")
    sys.exit(1)

def save_obj_with_extras(vertices, triangles, colors=None, normals=None, path="mesh.obj"):
    """Custom OBJ exporter to handle vertex colors and normals."""
    print(f"Saving mesh to {path} ({len(vertices)} vertices, {len(triangles)} triangles)...")
    with open(path, "w") as f:
        # Write Vertices (with colors if available: v x y z r g b)
        for i in range(len(vertices)):
            v = vertices[i]
            line = f"v {v[0]} {v[1]} {v[2]}"
            if colors is not None:
                c = colors[i]
                line += f" {c[0]} {c[1]} {c[2]}"
            f.write(line + "\n")
        
        # Write Normals if available
        if normals is not None:
            for n in normals:
                f.write(f"vn {n[0]} {n[1]} {n[2]}\n")
        
        # Write Faces (1-based indexing)
        # If normals exist, we use the 'v//vn' or 'v/vt/vn' format, 
        # but for simple OBJ with vertex normals, 'f v1//v1' is common.
        for t in triangles:
            if normals is not None:
                f.write(f"f {t[0]+1}//{t[0]+1} {t[1]+1}//{t[1]+1} {t[2]+1}//{t[2]+1}\n")
            else:
                f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")

def extract_mesh(snapshot_path, resolution=256, threshold=0.0, output_path="extracted_mesh.obj", mode="cuda"):
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)

    print(f"Loading snapshot from {snapshot_path}...")
    if not os.path.exists(snapshot_path):
        print(f"Error: Snapshot file {snapshot_path} not found.")
        return
    
    testbed.load_snapshot(snapshot_path)
    
    aabb = testbed.render_aabb
    print(f"AABB: {aabb.min} to {aabb.max}")
    vertices = None
    triangles = None
    colors = None
    normals = None
    
    if mode == "cuda":
        print(f"Mode: Internal CUDA Marching Cubes...")
        # Access the dictionary keys returned by the C++ binding
        mesh_data = testbed.compute_marching_cubes_mesh(resolution, resolution, resolution, aabb, threshold)
        
        vertices = mesh_data['V']   # Vertex positions
        triangles = mesh_data['F']  # Face indices
        colors = mesh_data.get('C') # Vertex colors (RGB)
        normals = mesh_data.get('N') # Vertex normals
        save_obj_with_extras(vertices, triangles, colors, normals, output_path)

    elif mode == "pymcubes":
        print(f"Mode: PyMCubes...")
        density_grid = testbed.get_density_grid(resolution, resolution, resolution, aabb)
        vertices, triangles = mcubes.marching_cubes(density_grid, threshold)

        # The vertices returned are in (z,y,x) order due to the way the density grid is structured. The flat buffer is laid out as [Z][Y][X]
        v_old = vertices.copy()
        v_new = np.zeros_like(v_old)
        v_new[:, 0] = v_old[:, 2]
        v_new[:, 1] = v_old[:, 1]
        v_new[:, 2] = v_old[:, 0]

        #Print some vertices to see format
        print("Sample vertices (grid index space):")
        print(v_new[:5])  # Print first 5 vertices  
        print("Vertex position range (grid index space):")
        print("Min:", np.min(v_new, axis=0))
        print("Max:", np.max(v_new, axis=0))
        

        # Map from grid index space → world space
        res_scale = (aabb.max - aabb.min) / resolution 
        world_vertices = v_new * res_scale + aabb.min

        #Print some vertices to see format
        print("Sample vertices (world space):")
        print(world_vertices[:5])  # Print first 5 vertices
        #Print min and max
        print("Vertex position range (world space):")
        print("Min:", np.min(world_vertices, axis=0))
        print("Max:", np.max(world_vertices, axis=0))

        print("Querying vertex colors and normals...")
        data = testbed.query_vertex_colors_and_normals(world_vertices.astype(np.float32))

        colors  = data["colors"].astype(np.float64)
        normals = data["normals"].astype(np.float64)


        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices      = o3d.utility.Vector3dVector(world_vertices)
        mesh.triangles     = o3d.utility.Vector3iVector(triangles)
        mesh.vertex_colors  = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
        mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
        o3d.io.write_triangle_mesh(output_path, mesh)
        print(f"Saved {output_path}")

    elif mode == "flexicubes":
        from kaolin.ops.conversions import FlexiCubes
        import torch

        device = torch.device("cuda")
        fc = FlexiCubes(device=device)

        density_np = testbed.get_density_grid(resolution, resolution, resolution, aabb)

        # Match NeRF [Z, Y, X] to FlexiCubes [X, Y, Z]
        density_np = np.transpose(density_np, (2, 1, 0)).copy()
        
        sdf = torch.from_numpy(density_np).float().to(device).reshape(-1)

        # Flexicube resolution is number of voxels, which is one less than number of grid points
        fc_res = resolution - 1 
        x_nx3, cube_fx8 = fc.construct_voxel_grid(fc_res)
        x_nx3 = x_nx3.to(device)

        # def flexicubes_grad_wrapper(x_nx3):
        #     x_np = x_nx3.detach().cpu().numpy().astype(np.float32)
        #     x_np = x_np + 0.5  # Shift from [-0.5, 0.5] to [0,1] range expected by testbed
        #     grads_np = testbed.query_sdf_gradients(x_np)
        #     return torch.from_numpy(grads_np).to(x_nx3.device)

        verts, faces, _ = fc(x_nx3, sdf, cube_fx8, fc_res)

        # FlexiCubes outputs in [-0.5, 0.5] range, so we shift to [0,1] by adding 0,5
        world_vertices = (verts + 0.5).detach().cpu().numpy().astype(np.float32)

        print(f"Querying {len(world_vertices):,} vertices for colors...")
        data = testbed.query_vertex_colors_and_normals(world_vertices)

        colors  = data["colors"].astype(np.float64)
        normals = data["normals"].astype(np.float64)

        # 8. SAVE
        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices       = o3d.utility.Vector3dVector(world_vertices)
        mesh.triangles      = o3d.utility.Vector3iVector(faces.detach().cpu().numpy())
        mesh.orient_triangles()
        mesh.vertex_colors  = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
        mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
        o3d.io.write_triangle_mesh(output_path, mesh)


    elif mode == "chunkedCuda" :
        print(f"Mode: Chunked CUDA Marching Cubes...")
        mesh_data = testbed.compute_marching_cubes_mesh_chunked(resolution, resolution, resolution, aabb, threshold)
        
        vertices = mesh_data['V']   # Vertex positions
        triangles = mesh_data['F']  # Face indices
        colors = mesh_data.get('C') # Vertex colors (RGB)
        normals = mesh_data.get('N') # Vertex normals
        save_obj_with_extras(vertices, triangles, colors, normals, output_path)

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract mesh from NeRF")
    parser.add_argument("snapshot_path", type=str)
    parser.add_argument("--mode", type=str, choices=["cuda", "pymcubes","chunkedCuda","flexicubes"], default="cuda")
    parser.add_argument("--res", type=int, default=256)
    parser.add_argument("--thresh", type=float, default=0)
    parser.add_argument("--out", type=str, default="extracted_mesh.obj")
    
    args = parser.parse_args()
    extract_mesh(args.snapshot_path, args.res, args.thresh, args.out, args.mode)