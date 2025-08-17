import json
import argparse
import os

def apply_translation_offset(input_file, output_file, offset):
    # Load JSON
    with open(input_file, "r") as f:
        data = json.load(f)

    # Walk through all keyframes
    for keyframe in data.get("keyframes", []):
        for update in keyframe.get("stateUpdates", []):
            state = update.get("state", {})
            abs_transform = state.get("absTransform", {})
            translation = abs_transform.get("translation")

            if translation and len(translation) == 3:
                abs_transform["translation"] = [
                    translation[i] + offset[i] for i in range(3)
                ]

    # Save JSON
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply translation offset to JSON keyframes")
    parser.add_argument("input_file", help="Path to input JSON file")
    parser.add_argument("output_file", nargs="?", help="Path to save output JSON file (default: input name with _shifted)")
    parser.add_argument("--offset", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"),
                        help="Translation offset (x y z)")

    args = parser.parse_args()

    if args.output_file is None:
        root, ext = os.path.splitext(args.input_file)
        args.output_file = f"{root}_shifted{ext}"

    apply_translation_offset(args.input_file, args.output_file, args.offset)
    print(f"Saved shifted file to {args.output_file}")
