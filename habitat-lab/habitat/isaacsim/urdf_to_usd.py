import os
import asyncio
import re
import xml.etree.ElementTree as ET
import argparse
import json

import argparse
from omni.isaac.lab.app import AppLauncher


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
    for element in root.xpath("//*[@name]"):  # noqa
        original_name = element.get("name")
        sanitized_name = sanitize_name(original_name)
        element.set("name", sanitized_name)

    # Update references to <parent link> and <child link>
    for parent in root.xpath("//parent[@link]"):  # noqa
        original_link = parent.get("link")
        parent.set("link", name_map.get(original_link, original_link))

    for child in root.xpath("//child[@link]"):  # noqa
        original_link = child.get("link")
        child.set("link", name_map.get(original_link, original_link))

    # Optionally remove <visual> elements
    if remove_visual:
        for visual in root.xpath("//visual"):  # noqa
            visual_parent = visual.getparent()
            visual_parent.remove(visual)

    # Write the cleaned URDF to the output file
    with open(output_file, "wb") as f:
        f.write(
            ET.tostring(  # noqa
                root, pretty_print=True, xml_declaration=True, encoding="UTF-8"
            )
        )
    print(f"Cleaned URDF written to: {output_file}")


def convert_urdf(urdf_filepath: str, out_usd_filepath: str) -> None:
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
    # Parse the URDF file
    urdf_tree = ET.parse(reference_urdf_filepath)
    urdf_root = urdf_tree.getroot()

    # Get the robot name from the URDF
    robot_name = urdf_root.get("name")

    # Extract visual metadata from the URDF
    visual_metadata = {}
    urdf_dir = os.path.dirname(os.path.abspath(reference_urdf_filepath))
    usd_dir = os.path.dirname(os.path.abspath(usd_filepath))
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
                        )  # noqa

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


def convert_urdf_test():
    """
    This is an exmaple use case of going from an unformatted urdf file to usd file.
    Specifically, this is render the habitat spot arm urdf.
    """
    # /home/trandaniel/dev/habitat-lab/data/hab_spot_arm/urdf
    source_urdf_filepath = "data/hab_spot_arm/urdf/hab_spot_arm.urdf"
    clean_urdf_filepath = "data/hab_spot_arm/urdf/hab_spot_arm_clean.urdf"
    # Temp USD must be in same folder as final USD. It's okay to be the exact same file.
    temp_usd_filepath = "data/usd/robots/hab_spot_arm.usda"
    out_usd_filepath = "data/usd/robots/hab_spot_arm.usda"
    convert_urdf(clean_urdf_filepath, temp_usd_filepath)
    add_habitat_visual_metadata_for_articulation(
        temp_usd_filepath,
        source_urdf_filepath,
        out_usd_filepath,
        project_root_folder="./",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create an empty Issac Sim stage."
    )
    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    # parse the arguments
    ## args_cli = parser.parse_args()
    args_cli, _ = parser.parse_known_args()
    # launch omniverse app
    args_cli.headless = True  # Config to have Isaac Lab UI off
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # TODO: DELETE library if not used.
    # from omni.isaac.core.utils.extensions import enable_extension

    # TODO: DELETE line with extra modules if not needed
    # from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf
    from pxr import Usd, Gf, Sdf

    convert_urdf_test()
