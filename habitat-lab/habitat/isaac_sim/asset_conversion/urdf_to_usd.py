"""This module converts urdf files into usda files."""

import argparse
import os

from lxml import etree as ET  # Using lxml for better XML handling


def clean_urdf(input_file: str, output_file: str, remove_visual=False) -> None:
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
        if "." in name:
            new_name = name.replace(".", "_")
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
        f.write(
            ET.tostring(
                root, pretty_print=True, xml_declaration=True, encoding="UTF-8"
            )
        )
    print(f"Cleaned URDF written to: {output_file}")


def convert_urdf(urdf_filepath: str, out_usd_filepath: str) -> None:
    """Convert urdf file to usda file

    :param urdf_filepath: The filepath of urdf file
    :param out_usd_filepath: The desired output path of usda file
    """

    from omni.isaac.lab.sim.converters import UrdfConverter, UrdfConverterCfg

    out_dir, out_filename = os.path.split(out_usd_filepath)

    # Define the configuration for the URDF conversion
    config = UrdfConverterCfg(
        asset_path=urdf_filepath,  # Path to the input URDF file
        usd_dir=out_dir,  # Directory to save the converted USD file
        usd_file_name=out_filename,  # Name of the output USD file
        force_usd_conversion=True,  # Force conversion even if USD already exists
        make_instanceable=False,  # Make the USD asset instanceable
        link_density=1000.0,  # Default density for links
        import_inertia_tensor=True,  # Import inertia tensor from URDF
        convex_decompose_mesh=False,  # Decompose convex meshes
        fix_base=False,  # Fix the base link
        merge_fixed_joints=False,  # Merge links connected by fixed joints
        self_collision=False,  # Enable self-collision
        default_drive_type="position",  # Default drive type for joints
        override_joint_dynamics=False,  # Override joint dynamics
    )

    # Create the UrdfConverter with the specified configuration
    converter = UrdfConverter(config)

    # Get the path to the generated USD file
    usd_path = converter.usd_path
    print(f"USD file generated at: {usd_path}")


def add_habitat_visual_metadata_for_articulation(
    usd_filepath: str,
    reference_urdf_filepath: str,
    out_usd_filepath: str,
    project_root_folder: str,
) -> None:
    """Add habitat visual metadata into usda file.

    :param usd_filepath: Usd fileapath to add visual metadata
    :param reference_urdf_filepath: Filepath of reference urdf
    :param out_usd_filepath: Desired output usd path
    :param project_root_folder: Directory path of habitat-lab
    """
    from pxr import Gf, Sdf, Usd

    # Parse the URDF file
    urdf_tree = ET.parse(reference_urdf_filepath)
    urdf_root = urdf_tree.getroot()

    # Get the robot name from the URDF
    robot_name = urdf_root.get("name")

    # Extract visual metadata from the URDF
    visual_metadata = {}
    urdf_dir = os.path.dirname(os.path.abspath(reference_urdf_filepath))
    # usd_dir = os.path.dirname(os.path.abspath(usd_filepath)) # NOTE: Not used in this function
    for link in urdf_root.findall("link"):
        link_name = link.get("name")
        visual = link.find("visual")
        if visual is not None:
            geometry = visual.find("geometry")
            if geometry is not None:
                mesh = geometry.find("mesh")
                if mesh is not None:
                    filename = mesh.get("filename")
                    asset_path = os.path.relpath(
                        os.path.abspath(os.path.join(urdf_dir, filename)),
                        start=project_root_folder,
                    )
                    scale = (1.0, 1.0, 1.0)  # Default scale

                    # Check for scale in the <mesh> element
                    scale_element = mesh.find("scale")
                    if scale_element is not None:
                        scale = tuple(
                            map(float, scale_element.text.split())
                        )  # type: ignore # noqa: ALL

                    # Replace periods with underscores for USD-safe names
                    # todo: use a standard get_sanitized_usd_name function here
                    safe_link_name = link_name.replace(".", "_")
                    visual_metadata[safe_link_name] = {
                        "assetPath": asset_path,
                        "assetScale": scale,
                    }

    # Open the USD file
    stage = Usd.Stage.Open(usd_filepath)
    if not stage:
        raise ValueError(f"Could not open USD file: {usd_filepath}")

    # Add the habitatVisual metadata to each relevant prim
    for link_name, metadata in visual_metadata.items():
        prim_path = f"/{robot_name}/{link_name}"
        prim = stage.GetPrimAtPath(prim_path)
        if prim:
            # Add assetPath
            asset_path_attr = prim.CreateAttribute(
                "habitatVisual:assetPath", Sdf.ValueTypeNames.String
            )
            asset_path_attr.Set(metadata["assetPath"])

            # Add assetScale
            asset_scale_attr = prim.CreateAttribute(
                "habitatVisual:assetScale", Sdf.ValueTypeNames.Float3
            )
            asset_scale_attr.Set(Gf.Vec3f(*metadata["assetScale"]))
        else:
            print(f"Warning: Prim not found for link: {link_name}")

    # Save the updated USD to the output file
    stage.GetRootLayer().Export(out_usd_filepath)
    print(f"Updated USD file written to: {out_usd_filepath}")


def convert_urdf_to_usd(
    input_urdf_file: str,
    output_usda_file: str,
    project_root_folder: str,
    remove_visual: bool = False,
) -> None:
    filename, file_extension = os.path.splitext(input_urdf_file)
    clean_urdf_temp = f"{filename}_TEMP{file_extension}"
    clean_urdf(input_urdf_file, clean_urdf_temp, remove_visual)

    convert_urdf(clean_urdf_temp, output_usda_file)

    if os.path.exists(clean_urdf_temp):
        os.remove(clean_urdf_temp)

    add_habitat_visual_metadata_for_articulation(
        output_usda_file,
        input_urdf_file,
        output_usda_file,
        project_root_folder,
    )


if __name__ == "__main__":
    # Launch Issac Lab Applauncher
    from omni.isaac.lab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    # Build parser and subparser
    parser = argparse.ArgumentParser(description="Convert urdf file to usd.")

    parser.add_argument("input_urdf_file")
    parser.add_argument("output_usda_file")
    parser.add_argument("project_root_folder")
    parser.add_argument("--remove-visual")

    args = parser.parse_args()

    convert_urdf_to_usd(
        args.input_urdf_file, args.output_usda_file, args.project_root_folder
    )

    '''
    """This function converts urdf files to usda."""
    # Build parser and subparser
    parser = argparse.ArgumentParser(description="Convert urdf file to usd.")
    subparsers = parser.add_subparsers(dest="command")

    # Parser for clean_urdf
    parser_clean_urdf = subparsers.add_parser(
        "clean_urdf", help="Run clean_urdf function"
    )
    parser_clean_urdf.add_argument("input_file")
    parser_clean_urdf.add_argument("output_file")
    parser_clean_urdf.add_argument("--remove_visual")
    parser_clean_urdf.set_defaults(func=clean_urdf)

    # Parser for convert_urdf function
    parser_convert_urdf = subparsers.add_parser(
        "convert_urdf", help="Run convert_urdf function"
    )
    parser_convert_urdf.add_argument("urdf_filepath")
    parser_convert_urdf.add_argument("out_usd_filepath")
    parser_convert_urdf.set_defaults(func=convert_urdf)

    # Parser for add_habitat_visual_metadata_for_articulation function
    parser_add_hab_visual_metadata = subparsers.add_parser(
        "add_habitat_visual_metadata_for_articulation",
        help="add_habitat_visual_metadata_for_articulation",
    )
    parser_add_hab_visual_metadata.add_argument("usd_filepath")
    parser_add_hab_visual_metadata.add_argument("reference_urdf_filepath")
    parser_add_hab_visual_metadata.add_argument("out_usd_filepath")
    parser_add_hab_visual_metadata.add_argument("project_root_folder")
    parser_add_hab_visual_metadata.set_defaults(
        func=add_habitat_visual_metadata_for_articulation
    )

    args = parser.parse_args()

    # Argparse function selection
    if args.command:
        if args.command == "clean_urdf":
            args.func(args.input_file, args.output_file, args.remove_visual)
        elif args.command == "convert_urdf":
            args.func(args.urdf_filepath, args.out_usd_filepath)
        elif args.command == "add_habitat_visual_metadata_for_articulation":
            args.func(
                args.usd_filepath,
                args.reference_urdf_filepath,
                args.out_usd_filepath,
                args.project_root_folder,
            )
    else:
        parser.print_help()

    '''
