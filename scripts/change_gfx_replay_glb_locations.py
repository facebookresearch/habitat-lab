#!/usr/bin/env python3
import os
import json
import argparse
import shutil

OLD_ROOT = "/home/eric/projects/hssd-hab2"
NEW_ROOT = "data/mochi_vr_data/source/hssd-hab"

def process_json(input_path, output_path):
    with open(input_path, "r") as f:
        data = json.load(f)

    def replace_and_copy(path):
        if not isinstance(path, str) or not path.endswith(".glb"):
            return path
        if not path.startswith(OLD_ROOT):
            return path

        # Map to new path
        rel_path = os.path.relpath(path, OLD_ROOT)
        new_path = os.path.join(NEW_ROOT, rel_path)

        # Ensure directory exists and copy file
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        shutil.copy2(path, new_path)

        return new_path

    def recurse(obj):
        if isinstance(obj, dict):
            return {k: recurse(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recurse(v) for v in obj]
        elif isinstance(obj, str):
            return replace_and_copy(obj)
        else:
            return obj

    new_data = recurse(data)

    with open(output_path, "w") as f:
        json.dump(new_data, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rewrite GLB paths and copy files.")
    parser.add_argument("input_json", help="Input JSON file")
    parser.add_argument("output_json", help="Output JSON file")
    args = parser.parse_args()

    process_json(args.input_json, args.output_json)
