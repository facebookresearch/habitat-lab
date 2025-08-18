#!/usr/bin/env python3
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

def fail(msg: str, code: int = 1):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)

def run_cmd(cmd):
    print(f"+ {' '.join(cmd)}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        if proc.stdout.strip():
            print(proc.stdout, file=sys.stderr)
        if proc.stderr.strip():
            print(proc.stderr, file=sys.stderr)
        fail(f"Command failed with return code {proc.returncode}", proc.returncode)

def move_overwrite(src: Path, dst: Path):
    """Move src to dst, deleting dst first if it exists."""
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    shutil.move(str(src), str(dst))

def natural_key(p: Path):
    s = p.name
    parts = re.split(r"(\d+)", s)
    return [int(t) if t.isdigit() else t for t in parts]

def main():
    ap = argparse.ArgumentParser(description="Assemble a Mochi prefab from a URDF")
    ap.add_argument("-f", "--urdf", required=True, type=Path, help="Path to input .urdf")
    ap.add_argument("-d", "--dest", required=True, type=Path, help="Destination directory")
    ap.add_argument("--static-root", action="store_true", help="Mark root joint as static instead of free")
    ap.add_argument("--skip-meshes", action="store_true", help="Skip mesh generation (reuse existing meshes)")
    args = ap.parse_args()

    urdf = args.urdf.resolve()
    dest = args.dest.resolve()
    dest_name = dest.name

    if not urdf.exists():
        fail(f"URDF not found: {urdf}")

    # ------------------------------
    # Run the pipeline
    # ------------------------------
    if args.skip_meshes:
        cmd = [sys.executable, "urdf2rootTransform.py", "-f", str(urdf), "--use_collision"]
        if args.static_root:
            cmd.append("--static-root")
        run_cmd(cmd)
    else:
        run_cmd([sys.executable, "urdf2individualmesh.py", "-f", str(urdf), "--use_collision"])
        # run_cmd([sys.executable, "urdf2individualmesh.py", "-f", str(urdf)])
        cmd = [sys.executable, "urdf2rootTransform.py", "-f", str(urdf), "--use_collision"]
        if args.static_root:
            cmd.append("--static-root")
        run_cmd(cmd)
        run_cmd([sys.executable, "convert_meshes.py", "-dir", "./individual_meshes"])

    # ------------------------------
    # Verify outputs and move
    # ------------------------------
    src_h5 = Path("mesh_transforms.mochi.h5")
    if not src_h5.is_file():
        fail(f"Missing transform file: {src_h5}")

    dest.mkdir(parents=True, exist_ok=True)
    move_overwrite(src_h5, dest / src_h5.name)

    prefab = {
        "actors": {
            "articulated": [
                {
                    "articulatedShape": f"hab_mochi_shared_data/{dest_name}/mesh_transforms.mochi.h5",
                    "linkParams": [{"colliderType": "Sdf", "layer": "EnvironmentLinks"}],
                    "linkShapes": [],
                    "name": dest_name,
                }
            ]
        }
    }

    # ------------------------------
    # Add meshes
    # ------------------------------
    moved_mesh_dir = dest / "individual_meshes"
    if not args.skip_meshes:
        src_mesh_dir = Path("individual_meshes")
        if not src_mesh_dir.is_dir():
            fail(f"Missing meshes dir: {src_mesh_dir}")
        move_overwrite(src_mesh_dir, moved_mesh_dir)

    if moved_mesh_dir.is_dir():
        mesh_files = sorted(moved_mesh_dir.glob("*.mochi.h5"), key=natural_key)
        prefab["actors"]["articulated"][0]["linkShapes"] = [
            f"hab_mochi_shared_data/{dest_name}/individual_meshes/{p.name}"
            for p in mesh_files
        ]
    else:
        print(f"WARNING: no individual_meshes found at {moved_mesh_dir}")    


    # ------------------------------
    # Write prefab
    # ------------------------------
    prefab_path = dest / f"{dest_name}.mochi_prefab"
    print(f"+ write {prefab_path}")
    with open(prefab_path, "w", encoding="utf-8") as f:
        json.dump(prefab, f, indent=2)
        f.write("\n")

    print("Done.")

if __name__ == "__main__":
    main()
