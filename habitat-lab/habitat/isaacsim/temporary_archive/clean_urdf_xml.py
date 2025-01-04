import argparse
from lxml import etree as ET  # Using lxml for better XML handling

def clean_urdf(input_file, output_file, remove_visual=False):
    """
    Cleans a URDF file:
    1. Optionally removes <visual> elements.
    2. Fixes invalid use of '.' in <link> and <joint> names while preserving references.

    :param input_file: Path to the input URDF file.
    :param output_file: Path to the output cleaned URDF file.
    :param remove_visual: If True, removes all <visual> elements.
    """
    tree = ET.parse(input_file)
    root = tree.getroot()

    name_map = {}  # Maps old names to new names for links and joints

    # Helper function to sanitize names
    def sanitize_name(name):
        if '.' in name:
            new_name = name.replace('.', '_')
            name_map[name] = new_name
            return new_name
        return name

    # Update <link> and <joint> names
    for element in root.xpath("//*[@name]"):
        original_name = element.get("name")
        sanitized_name = sanitize_name(original_name)
        element.set("name", sanitized_name)

    # Update references to <parent link> and <child link>
    for parent in root.xpath("//parent[@link]"):
        original_link = parent.get("link")
        parent.set("link", name_map.get(original_link, original_link))

    for child in root.xpath("//child[@link]"):
        original_link = child.get("link")
        child.set("link", name_map.get(original_link, original_link))

    # Optionally remove <visual> elements
    if remove_visual:
        for visual in root.xpath("//visual"):
            visual_parent = visual.getparent()
            visual_parent.remove(visual)

    # Write the cleaned URDF to the output file
    with open(output_file, "wb") as f:
        f.write(ET.tostring(root, pretty_print=True, xml_declaration=True, encoding="UTF-8"))
    print(f"Cleaned URDF written to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Clean and sanitize URDF files.")
    parser.add_argument("input_file", help="Path to the input URDF file")
    parser.add_argument("output_file", help="Path to the output cleaned URDF file")
    parser.add_argument(
        "--remove-visual", action="store_true", help="Remove all <visual> elements"
    )
    args = parser.parse_args()

    clean_urdf(args.input_file, args.output_file, args.remove_visual)


if __name__ == "__main__":
    main()
