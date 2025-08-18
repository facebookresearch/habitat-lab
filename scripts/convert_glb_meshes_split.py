#!/usr/bin/env python3
import os
import argparse
import tempfile
import trimesh
import numpy as np
import pymeshlab


def cleanup_mesh(mesh: trimesh.Trimesh, max_coord=1e6) -> trimesh.Trimesh:
    """Basic sanity cleanup before subdivision."""
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    verts = mesh.vertices
    if not np.all(np.isfinite(verts)):
        raise RuntimeError("Mesh has NaN or Inf vertices")

    max_found = np.max(np.abs(verts))
    if max_found > max_coord:
        raise RuntimeError(
            f"Mesh has vertex coordinate beyond ±{max_coord} "
            f"(max found {max_found})"
        )

    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return mesh


def split_until_target(in_path, out_path, target_verts=None, max_steps=8, steps=None):
    """Subdivide mesh until reaching target verts (minimum)."""
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(in_path)

    cur_v = ms.current_mesh().vertex_number()
    done = False
    nsteps = 0

    if target_verts is not None:
        while cur_v < target_verts and nsteps < max_steps:
            ms.apply_filter("meshing_surface_subdivision_midpoint")
            cur_v = ms.current_mesh().vertex_number()
            nsteps += 1
        print(f"  Subdivided {nsteps} times, final verts={cur_v} (target={target_verts})")
    else:
        for _ in range(steps or 1):
            ms.apply_filter("meshing_surface_subdivision_midpoint")
            nsteps += 1
        cur_v = ms.current_mesh().vertex_number()
        print(f"  Subdivided {nsteps} times, final verts={cur_v}")

    ms.apply_filter("compute_normal_per_vertex")
    ms.save_current_mesh(out_path)


def convert_split(input_dir, output_dir, out_ext, steps=1, target_verts=None):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(".glb"):
            continue
        in_path = os.path.join(input_dir, fname)
        out_name = os.path.splitext(fname)[0] + f".{out_ext}"
        out_path = os.path.join(output_dir, out_name)

        print(f"Processing {in_path} ...")

        mesh = trimesh.load(in_path)
        mesh = cleanup_mesh(mesh)

        # Export temp .obj for pymeshlab
        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tmp:
            mesh.export(tmp.name)
            tmp.flush()
            split_until_target(tmp.name, out_path,
                               target_verts=target_verts, steps=steps)

        print(f"→ Subdivided {in_path} -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch subdivide GLB meshes by splitting triangles."
    )
    parser.add_argument("input_dir", help="Folder containing .glb files")
    parser.add_argument("output_dir", help="Folder to save meshes")
    parser.add_argument("--ext", choices=["obj", "stl"], default="obj",
                        help="Output format (default: obj)")
    parser.add_argument("--steps", type=int, default=1,
                        help="Fixed number of subdivision steps (ignored if --target-verts is set)")
    parser.add_argument("--target-verts", type=int, default=None,
                        help="Minimum target vertices (stop subdividing once reached)")
    args = parser.parse_args()

    try:
        convert_split(args.input_dir, args.output_dir, args.ext,
                      steps=args.steps, target_verts=args.target_verts)
    except Exception as e:
        print(f"FATAL: {e}")
        raise


if __name__ == "__main__":
    main()
