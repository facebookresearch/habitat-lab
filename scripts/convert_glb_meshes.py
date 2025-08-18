#!/usr/bin/env python3
import os
import argparse
import tempfile
import trimesh
import numpy as np
import pymeshlab


def cleanup_mesh(mesh: trimesh.Trimesh, max_coord=1e6) -> trimesh.Trimesh:
    mesh = ensure_trimesh(mesh)   # flatten scene if needed

    verts = mesh.vertices
    if not np.all(np.isfinite(verts)):
        raise RuntimeError("Mesh has NaN or Inf vertices")

    max_found = np.max(np.abs(verts))
    if max_found > max_coord:
        raise RuntimeError(
            f"Mesh has vertex coordinate beyond ±{max_coord} "
            f"(max found {max_found})"
        )

    mesh.update_faces(mesh.nondegenerate_faces())  # replaces deprecated remove_degenerate_faces
    mesh.update_faces(mesh.unique_faces())         # replaces deprecated remove_duplicate_faces
    mesh.remove_unreferenced_vertices()
    mesh.fill_holes()
    mesh.fix_normals()
    return mesh


def simplify_with_pymeshlab(in_path, out_path, ratio=None):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(in_path)
    kwargs = {
        "preservenormal": True,
        "preservetopology": True,
        "preserveboundary": True,
    }
    if ratio is not None:
        kwargs["targetperc"] = ratio
    ms.apply_filter("meshing_decimation_quadric_edge_collapse", **kwargs)
    ms.save_current_mesh(out_path)

def ensure_trimesh(obj):
    if isinstance(obj, trimesh.Scene):
        return trimesh.util.concatenate(obj.dump())
    return obj

def enforce_watertight(in_path, out_path, maxholesize=100):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(in_path)

    # 1) Fix topology first
    ms.apply_filter("meshing_repair_non_manifold_edges", method='Split Vertices')
    ms.apply_filter("meshing_repair_non_manifold_vertices", vertdispratio=0)

    # 2) Try to bring open borders closer (optional but often helpful)
    ms.apply_filter("meshing_snap_mismatched_borders")

    # 3) Now close holes (edge-manifold is expected here)
    ms.apply_filter("meshing_close_holes", maxholesize=maxholesize)

    # 4) Save
    ms.save_current_mesh(out_path)


def convert_glbs(input_dir, output_dir, out_ext, ratio=None, watertight=False):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(".glb"):
            continue
        in_path = os.path.join(input_dir, fname)
        out_name = os.path.splitext(fname)[0] + f".{out_ext}"
        out_path = os.path.join(output_dir, out_name)

        print(f"Processing {in_path} ...")

        mesh = trimesh.load(in_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())

        # Pre-clean
        mesh = cleanup_mesh(mesh)

        if ratio is not None:
            # Simplify first
            with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tmp:
                mesh.export(tmp.name)
                tmp.flush()
                simplify_with_pymeshlab(tmp.name, out_path, ratio)

            mesh = trimesh.load(out_path)
            mesh = cleanup_mesh(mesh)
            mesh.export(out_path)
            print(f"Converted + simplified + cleaned {in_path} -> {out_path}")
        else:
            mesh.export(out_path)
            print(f"Converted + cleaned {in_path} -> {out_path}")

        if watertight:
            # enforce watertightness on the just-exported mesh
            enforce_watertight(out_path, out_path)
            mesh = trimesh.load(out_path)
            mesh = cleanup_mesh(mesh)
            mesh.export(out_path)
            print(f"→ enforced watertight {in_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert GLB files to OBJ/STL with optional decimation, cleanup, and watertight enforcement."
    )
    parser.add_argument("input_dir", help="Folder containing .glb files")
    parser.add_argument("output_dir", help="Folder to save converted files")
    parser.add_argument("--ext", choices=["obj", "stl"], default="obj",
                        help="Output format (default: obj)")
    parser.add_argument("--simplify-ratio", type=float, default=None,
                        help="Simplify with pymeshlab: keep this fraction of faces (e.g. 0.5 = 50%).")
    parser.add_argument("--max-coord", type=float, default=1e6,
                        help="Fatal error if any vertex coordinate exceeds this (default: 1e6).")
    parser.add_argument("--watertight", action="store_true",
                        help="Run hole closing + non-manifold repair after conversion/cleanup.")
    args = parser.parse_args()

    try:
        convert_glbs(args.input_dir, args.output_dir, args.ext,
                     ratio=args.simplify_ratio, watertight=args.watertight)
    except Exception as e:
        print(f"FATAL: {e}")
        raise


if __name__ == "__main__":
    main()
