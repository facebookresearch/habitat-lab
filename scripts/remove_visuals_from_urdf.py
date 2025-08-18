import xml.etree.ElementTree as ET
import sys

def strip_visuals(input_file, output_file):
    # Parse URDF
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Find and remove all <visual> tags
    for link in root.findall(".//link"):
        visuals = list(link.findall("visual"))
        for vis in visuals:
            link.remove(vis)

    # Write new URDF
    tree.write(output_file, encoding="utf-8", xml_declaration=True)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python strip_visuals.py input.urdf output.urdf")
        sys.exit(1)

    strip_visuals(sys.argv[1], sys.argv[2])
    print(f"Saved URDF without <visual> to {sys.argv[2]}")
