

import os
import re
import xml.etree.ElementTree as ET
import json
from dataclasses import dataclass
from typing import List

from habitat.mujoco.convert_glb_to_obj import convert_glb_to_obj

@dataclass
class RenderAsset:
    """A render asset that can be provided to Habitat-sim ResourceManager::loadAndCreateRenderAssetInstance."""
    abs_filepath: str
    # todo: possible color override
    semantic_id: int
    scale: List[float]


def find_all_files(root_folder, file_suffix):
    files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith(file_suffix):
                files.append(os.path.abspath(os.path.join(dirpath, file)))
    return files

def find_file(folder, filename):
    result = None
    for root, dirs, files in os.walk(folder):
        if filename in files:
            assert not result
            result = os.path.join(root, filename)
    return os.path.abspath(result)

def convert_filepath_to_single_extension(path: str) -> str:
    # Split the path into directory and filename
    directory, filename = os.path.split(path)
    
    # Split the filename into the base and the extension
    base, extension = os.path.splitext(filename)
    
    # Replace all periods in the base with underscores
    base = base.replace('.', '_')
    
    # Reconstruct the filename
    new_filename = base + extension
    
    # Join with the original directory
    return os.path.join(directory, new_filename)


def sanitize_usd_name(name):
    """Sanitizes a string for use as a USD node name.

    Args:
        name: The input string to sanitize.

    Returns:
        The sanitized string, suitable for use as a USD node name.
    """

    assert len(name) > 0

    # Replace spaces, hyphens, colons, and other special characters with underscores
    sanitized_name = re.sub(r"[^\w]", "_", name)

    # Remove consecutive underscores
    sanitized_name = re.sub(r"_+", "_", sanitized_name)

    # Ensure the name doesn't start with a number
    if sanitized_name[0].isdigit():
        sanitized_name = "_" + sanitized_name

    return sanitized_name


def convert_object_to_obj_and_get_render_asset(object_config_filepath, out_obj_path):

    object_config_folder, _ = os.path.split(object_config_filepath)

    object_config_json_data = None
    with open(object_config_filepath, 'r') as file:
        object_config_json_data = json.load(file)

    if not os.path.exists(out_obj_path):
        # By convention, we fall back to render_asset if collision_asset is not set. See Habitat-sim Simulator::getJoinedMesh.
        collision_asset_rel_filepath = object_config_json_data["collision_asset"] if "collision_asset" in object_config_json_data else object_config_json_data["render_asset"]
        # Render assets should be merged; collision assets should not as they already represent a convex decomposition into nodes. Later, a render asset's mesh should be set to "convexDecomposition" and a collision asset's meshes should be set to "convexHull".
        merge_all_meshes = "collision_asset" not in object_config_json_data
        # collision_asset_filepath = find_file(objects_root_folder, collision_asset_filename)
        collision_asset_filepath = os.path.abspath(os.path.join(object_config_folder, collision_asset_rel_filepath))

        print(f"Converting {collision_asset_filepath}...")

        convert_glb_to_obj(collision_asset_filepath, out_obj_path)
        
        print(f"Wrote {out_obj_path}")   

    render_asset_rel_filepath = object_config_json_data["render_asset"]
    render_asset_abs_filepath = os.path.abspath(os.path.join(object_config_folder, render_asset_rel_filepath))
    render_asset_scale = object_config_json_data.get("scale", (1.0, 1.0, 1.0))

    return RenderAsset(abs_filepath=render_asset_abs_filepath, semantic_id=None, scale=render_asset_scale)


def convert_object_to_usd(object_config_filepath, out_usd_path, project_root_folder, enable_collision=True):

    object_config_folder, _ = os.path.split(object_config_filepath)

    object_config_json_data = None
    with open(object_config_filepath, 'r') as file:
        object_config_json_data = json.load(file)
    # By convention, we fall back to render_asset if collision_asset is not set. See Habitat-sim Simulator::getJoinedMesh.
    collision_asset_rel_filepath = object_config_json_data["collision_asset"] if "collision_asset" in object_config_json_data else object_config_json_data["render_asset"]
    # Render assets should be merged; collision assets should not as they already represent a convex decomposition into nodes. Later, a render asset's mesh should be set to "convexDecomposition" and a collision asset's meshes should be set to "convexHull".
    merge_all_meshes = "collision_asset" not in object_config_json_data
    # collision_asset_filepath = find_file(objects_root_folder, collision_asset_filename)
    collision_asset_filepath = os.path.abspath(os.path.join(object_config_folder, collision_asset_rel_filepath))

    print(f"Converting {collision_asset_filepath}...")

    if enable_collision:
        convert_mesh_to_usd(collision_asset_filepath, out_usd_path, load_materials=False, merge_all_meshes=merge_all_meshes)
    else:
        # create empty scene
        usd_stage = Usd.Stage.CreateNew(out_usd_path)
        xform_prim = UsdGeom.Xform.Define(usd_stage, "/World")
        usd_stage.SetDefaultPrim(xform_prim.GetPrim())
        usd_stage.GetRootLayer().Save()        

    render_asset_filepath_from_urdf = object_config_json_data["render_asset"]

    render_asset_filepath_for_usd = os.path.relpath(os.path.abspath(os.path.join(object_config_folder, render_asset_filepath_from_urdf)), start=project_root_folder)

    render_asset_scale = object_config_json_data.get("scale", (1.0, 1.0, 1.0))

    add_habitat_visual_to_usd_root(out_usd_path, render_asset_filepath_for_usd, render_asset_scale)

    print(f"Wrote {out_usd_path}")    

import math

def habitat_to_usd_position(position):
    """
    Convert a position from Habitat (Y-up) to USD (Z-up) coordinate system.

    Habitat (-x, z, y) -> Isaac (x, y, z)
    """
    x, y, z = position
    return [-x, z, y]

def habitat_to_usd_rotation(rotation):
    """
    Convert a quaternion rotation from Habitat to USD coordinate system.

    Parameters
    ----------
    rotation : list[float]
        Quaternion in Habitat coordinates [w, x, y, z] (wxyz).

    Returns
    -------
    list[float]
        Quaternion in USD coordinates [w, x, y, z] (wxyz).
    """
    HALF_SQRT2 = 0.70710678  # √0.5

    # Combined inverse quaternion transform: (inverse of q_trans)
    # q_x90_inv = [√0.5, √0.5, 0, 0] (wxyz format)
    # q_y180_inv = [0, 0, -1, 0] (wxyz format)
    q_y90_inv = [HALF_SQRT2, 0.0, HALF_SQRT2, 0.0]

    q_x90_inv = [HALF_SQRT2, HALF_SQRT2, 0.0, 0.0]
    q_z90_inv = [HALF_SQRT2, 0.0, 0.0, HALF_SQRT2]
    q_y180_inv = [0.0, 0.0, -1.0, 0.0]
    q_z180_inv = [0.0, 0.0, 0.0, 1.0]

    # todo: revise this to get the 180-degree rotation about y from the object_config.json

    # Multiply q_y180_inv * q_x90_inv to get the combined quaternion
    def quat_multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return [w, x, y, z]

    q_trans_inv = quat_multiply(q_x90_inv, q_y180_inv)

    # Multiply q_trans_inv with the input rotation quaternion
    w, x, y, z = rotation
    rotation_usd = quat_multiply(q_trans_inv, [w, x, y, z])

    return rotation_usd





def convert_hab_scene(scene_filepath, project_root_folder, enable_collision_for_stage=True):

    # not yet supported; we need to do convex hull decomposition
    assert not enable_collision_for_stage

    render_map = {}

    _, scene_filename = os.path.split(scene_filepath)
    scene_name = scene_filename.removesuffix(".scene_instance.json")
    scene_mjcf_folder = "data/mujoco/scenes"
    assert os.path.exists(scene_mjcf_folder) and os.path.isdir(scene_mjcf_folder)
    objects_mjcf_folder = "data/mujoco/objects"
    assert os.path.exists(objects_mjcf_folder) and os.path.isdir(objects_mjcf_folder)
    scene_mjcf_filepath = os.path.join(scene_mjcf_folder, f"{scene_name}.xml")
    render_map_filepath = os.path.join(scene_mjcf_folder, f"{scene_name}.render_map.json")

    
    # Create the root element
    mujoco_elem = ET.Element("mujoco")
    
    # Add <worldbody> element
    worldbody_elem = ET.SubElement(mujoco_elem, "worldbody")
    
    # Add <asset> element
    asset_elem = ET.SubElement(mujoco_elem, "asset")
    
    scenes_folder = os.path.dirname(scene_filepath)
    stages_folder = scenes_folder + "/../stages"
    assert os.path.exists(stages_folder) and os.path.isdir(stages_folder)
    objects_root_folder = scenes_folder + "/../objects"
    assert os.path.exists(objects_root_folder) and os.path.isdir(objects_root_folder)

    with open(scene_filepath, 'r') as file:
        scene_json_data = json.load(file)    

    object_counts = {}

    max_count = -1  # temp only convert the first N objects
    count = 0

    # todo: handle stage

    assert "object_instances" in scene_json_data
    for obj_instance_json in scene_json_data["object_instances"]:

        # todo: assert collision_asset_size is (1,1,1) or not present
        # todo: check is_collidable
        # todo: how to handle scale

        object_config_filename = f"{obj_instance_json['template_name']}.object_config.json"
        base_object_name = "OBJECT_" + object_config_filename.removesuffix(".object_config.json")
        # base_object_name = sanitize_usd_name(base_object_name)

        if base_object_name in object_counts:
            object_counts[base_object_name] += 1
            unique_object_name = f"{base_object_name}_dup{object_counts[base_object_name]}"
        else:
            unique_object_name = base_object_name
            object_counts[base_object_name] = 1

        out_obj_path = os.path.join(objects_mjcf_folder, f"{base_object_name}.obj")

        object_config_filepath = find_file(objects_root_folder, object_config_filename)
        assert object_config_filepath

        render_asset = convert_object_to_obj_and_get_render_asset(object_config_filepath, out_obj_path)

        mesh_name = base_object_name + "_mesh"
        if object_counts[base_object_name] == 1:
            # relative_usd_path = out_usd_path.removeprefix("./data/usd/")
            relative_obj_path = os.path.relpath(out_obj_path, start=scene_mjcf_folder)
            ET.SubElement(asset_elem, "mesh", name=mesh_name, file=relative_obj_path)

        rotation_wxyz = habitat_to_usd_rotation(obj_instance_json.get("rotation", [1.0, 0.0, 0.0, 0.0]))
        rotation_str = " ".join(map(str, rotation_wxyz))

        # Ensure translation op exists
        position = habitat_to_usd_position(obj_instance_json.get("translation", [0.0, 0.0, 0.0]))
        position_str = " ".join(map(str, position))

        # <geom name="thingy_geom0" type="mesh" mesh="thingy_mesh" pos="0 -0.2 0.5" />

        ET.SubElement(worldbody_elem, "geom", name=unique_object_name, type="mesh", mesh=mesh_name, pos=position_str, quat=rotation_str)

        render_asset_project_rel_filepath = os.path.relpath(render_asset.abs_filepath, start=project_root_folder)

        render_map[unique_object_name] = {
            "is_dynamic": False,  # all object instances are assumed to be static
            "render_asset_filepath": render_asset_project_rel_filepath,
            "semantic_id": render_asset.semantic_id,
            "scale": render_asset.semantic_id,
        }

        count += 1
        if count == max_count:
            break

    # usd_stage.GetRootLayer().Save()
    # print(f"wrote scene {scene_usd_filepath}")

    # Create an XML tree and write it to a file
    tree = ET.ElementTree(mujoco_elem)
    ET.indent(tree, space="    ")  # Pretty-print the XML
    with open(scene_mjcf_filepath, "w", encoding="utf-8") as f:
        f.write(ET.tostring(mujoco_elem, encoding="unicode"))

    with open(render_map_filepath, "w") as f:
        root_dict = {"render_map": render_map}
        json.dump(root_dict, f, indent=2)
    print(f"MuJoCo scene saved to {scene_mjcf_filepath}. Render map saved to {render_map_filepath}.")


def convert_urdf(urdf_filepath, out_usd_filepath):

    from omni.isaac.lab.sim.converters import UrdfConverter, UrdfConverterCfg

    out_dir, out_filename = os.path.split(out_usd_filepath)

    # Define the configuration for the URDF conversion
    config = UrdfConverterCfg(
        asset_path=urdf_filepath,  # Path to the input URDF file
        usd_dir=out_dir,         # Directory to save the converted USD file
        usd_file_name=out_filename,         # Name of the output USD file
        force_usd_conversion=True,                   # Force conversion even if USD already exists
        make_instanceable=False,                      # Make the USD asset instanceable
        link_density=1000.0,                         # Default density for links
        import_inertia_tensor=True,                  # Import inertia tensor from URDF
        convex_decompose_mesh=False,                 # Decompose convex meshes
        fix_base=False,                              # Fix the base link
        merge_fixed_joints=False,                    # Merge links connected by fixed joints
        self_collision=False,                        # Enable self-collision
        default_drive_type="position",               # Default drive type for joints
        override_joint_dynamics=False                # Override joint dynamics
    )

    # Create the UrdfConverter with the specified configuration
    converter = UrdfConverter(config)

    # Get the path to the generated USD file
    usd_path = converter.usd_path
    print(f"USD file generated at: {usd_path}")   


def add_habitat_visual_metadata_for_articulation(usd_filepath, reference_urdf_filepath, out_usd_filepath, project_root_folder):
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
            # todo: handle materials (color only, not texture)
            geometry = visual.find("geometry")
            if geometry is not None:
                mesh = geometry.find("mesh")
                if mesh is not None:
                    filename = mesh.get("filename")
                    asset_path = os.path.relpath(os.path.abspath(os.path.join(urdf_dir, filename)), start=project_root_folder)
                    scale = (1.0, 1.0, 1.0)  # Default scale

                    # Check for scale in the <mesh> element
                    scale_element = mesh.find("scale")
                    if scale_element is not None:
                        scale = tuple(map(float, scale_element.text.split()))

                    # Replace periods with underscores for USD-safe names
                    # todo: use a standard get_sanitized_usd_name function here
                    safe_link_name = link_name.replace(".", "_")
                    visual_metadata[safe_link_name] = {
                        "assetPath": asset_path,
                        "assetScale": scale
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
            asset_path_attr = prim.CreateAttribute("habitatVisual:assetPath", Sdf.ValueTypeNames.String)
            asset_path_attr.Set(metadata["assetPath"])

            # Add assetScale
            asset_scale_attr = prim.CreateAttribute("habitatVisual:assetScale", Sdf.ValueTypeNames.Float3)
            asset_scale_attr.Set(Gf.Vec3f(*metadata["assetScale"]))
        else:
            print(f"Warning: Prim not found for link: {link_name}")

    # Save the updated USD to the output file
    stage.GetRootLayer().Export(out_usd_filepath)
    print(f"Updated USD file written to: {out_usd_filepath}")

def convert_urdf_test():
    base_urdf_name = "hab_spot_arm"
    base_urdf_folder = "data/robots/hab_spot_arm/urdf"
    # base_urdf_name = "allegro_digit360_right_calib_free"
    # base_urdf_folder = "data/from_gum"

    source_urdf_filepath = f"{base_urdf_folder}/{base_urdf_name}.urdf"
    # create clean urdf with `python clean_urdf_xml.py --input_file [source_urdf_filepath] --output_file [clean_urdf_filepath] --remove_visual`
    # todo: combine cleaning and converting into single user-friendly function
    clean_urdf_filepath = f"{base_urdf_folder}/{base_urdf_name}_clean.urdf"

    # Temp USD must be in same folder as final USD. It's okay to be the exact same file.
    temp_usd_filepath = f"data/usd/robots/{base_urdf_name}.usda"
    out_usd_filepath = f"data/usd/robots/{base_urdf_name}.usda"
    convert_urdf(clean_urdf_filepath, temp_usd_filepath)
    add_habitat_visual_metadata_for_articulation(temp_usd_filepath, source_urdf_filepath, out_usd_filepath, project_root_folder="./")

def convert_objects_folder_to_usd(objects_root_folder, out_usd_folder, project_root_folder):

    filepaths = find_all_files(objects_root_folder, ".object_config.json")
    for object_config_filepath in filepaths:

        _, object_config_filename = os.path.split(object_config_filepath)

        # todo: avoid duplication with convert_hab_scene
        base_object_name = "OBJECT_" + object_config_filename.removesuffix(".object_config.json")
        base_object_name = sanitize_usd_name(base_object_name)
        # todo: consider preserving subfolder structure for objects, e.g. "usd/dataset_a/objects/b/my_object.usda" instead of "usd/objects/my_object.usda".
        out_usd_path = os.path.join(out_usd_folder, f"{base_object_name}.usda")

        # todo: gather these up and do them later with multiprocessing
        if not os.path.exists(out_usd_path):

            convert_object_to_usd(object_config_filepath, out_usd_path, project_root_folder)


if __name__ == "__main__":
    # example usage:

    # convert_urdf_test()
    convert_hab_scene("data/hssd-hab/scenes-uncluttered/102344193.scene_instance.json", project_root_folder="./", enable_collision_for_stage=False)
    # convert_objects_folder_to_usd("data/objects/ycb", "data/usd/objects/ycb/configs", "./")