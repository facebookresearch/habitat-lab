#!/usr/bin/env python3
import os
import json
import argparse
import subprocess
import shutil

OLD_ROOT = "/home/eric/projects/hssd-hab2"

GLB_DECOMPRESS_SCRIPT = "/home/eric/projects/glb_tools/decompress_glb_with_toktx_v2.py"
KTX_TOOL_PATH = "/home/eric/projects/KTX-Software/build/Release/ktx"

def process_json(input_path, output_path, decompress_textures=False):
    with open(input_path, "r") as f:
        data = json.load(f)

    # Set new root correctly depending on mode
    NEW_ROOT = (
        "data/mochi_vr_data/uncompressed/hssd-hab"
        if decompress_textures
        else "data/mochi_vr_data/source/hssd-hab"
    )

    def replace_and_process(path):
        if not isinstance(path, str) or not path.endswith(".glb"):
            return path
        if not path.startswith(OLD_ROOT):
            return path

        # Map to new path
        rel_path = os.path.relpath(path, OLD_ROOT)
        new_path = os.path.join(NEW_ROOT, rel_path)

        # Ensure directory exists
        os.makedirs(os.path.dirname(new_path), exist_ok=True)

        if decompress_textures:
            subprocess.run(
                [
                    "python",
                    GLB_DECOMPRESS_SCRIPT,
                    path,
                    new_path,
                    "--tool-path",
                    KTX_TOOL_PATH,
                ],
                check=True,
            )
        else:
            shutil.copy2(path, new_path)

        return new_path

    def recurse(obj):
        if isinstance(obj, dict):
            return {k: recurse(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recurse(v) for v in obj]
        elif isinstance(obj, str):
            return replace_and_process(obj)
        else:
            return obj

    new_data = recurse(data)

    with open(output_path, "w") as f:
        json.dump(new_data, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rewrite GLB paths and copy/decompress files.")
    parser.add_argument("input_json", help="Input JSON file")
    parser.add_argument("output_json", help="Output JSON file")
    parser.add_argument(
        "--decompress-textures",
        action="store_true",
        help="Use GLB decompression script instead of copying",
    )
    args = parser.parse_args()

    process_json(args.input_json, args.output_json, args.decompress_textures)
