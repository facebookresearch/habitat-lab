
import os
import asyncio
import re
import xml.etree.ElementTree as ET
import argparse
import json


import argparse
from omni.isaac.lab.app import AppLauncher

# parser = argparse.ArgumentParser(description="Create an empty Issac Sim stage.")
# # append AppLauncher cli args
# AppLauncher.add_app_launcher_args(parser)
# # parse the arguments
# ## args_cli = parser.parse_args()
# args_cli, _ = parser.parse_known_args()
# # launch omniverse app
# args_cli.headless = True # Config to have Isaac Lab UI off
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

# from omni.isaac.core.utils.extensions import enable_extension
# from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf

def add_habitat_visual_to_usd_root(usd_filepath, render_asset_filepath, render_asset_scale):
    # Open the USD file
    stage = Usd.Stage.Open(usd_filepath)
    if not stage:
        raise ValueError(f"Could not open USD file: {usd_filepath}")

    default_prim = stage.GetDefaultPrim()

    # Add habitatVisual:assetPath
    asset_path_attr = default_prim.CreateAttribute("habitatVisual:assetPath", Sdf.ValueTypeNames.String)
    asset_path_attr.Set(render_asset_filepath)

    # Add habitatVisual:assetScale
    asset_scale_attr = default_prim.CreateAttribute("habitatVisual:assetScale", Sdf.ValueTypeNames.Float3)
    asset_scale_attr.Set(Gf.Vec3f(*render_asset_scale))

    # Save the updated USD file
    stage.GetRootLayer().Save()
    print(f"Added habitat visual metadata to RigidBody prim in: {usd_filepath}")


def add_habitat_visual_to_usd_root_rigid_body(usd_filepath, render_asset_filepath, render_asset_scale):
    # Open the USD file
    stage = Usd.Stage.Open(usd_filepath)
    if not stage:
        raise ValueError(f"Could not open USD file: {usd_filepath}")

    # Find all prims with RigidBodyAPI
    rigid_body_prims = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]

    if not rigid_body_prims:
        raise ValueError("No prim with UsdPhysics.RigidBodyAPI found in USD file")

    if len(rigid_body_prims) > 1:
        raise ValueError("Multiple prims with UsdPhysics.RigidBodyAPI found in USD file")

    # Get the single rigid body prim
    rigid_body_prim = rigid_body_prims[0]

    # Add habitatVisual:assetPath
    asset_path_attr = rigid_body_prim.CreateAttribute("habitatVisual:assetPath", Sdf.ValueTypeNames.String)
    asset_path_attr.Set(render_asset_filepath)

    # Add habitatVisual:assetScale
    asset_scale_attr = rigid_body_prim.CreateAttribute("habitatVisual:assetScale", Sdf.ValueTypeNames.Float3)
    asset_scale_attr.Set(Gf.Vec3f(*render_asset_scale))

    # Save the updated USD file
    stage.GetRootLayer().Save()
    print(f"Added habitat visual metadata to RigidBody prim in: {usd_filepath}")


def convert_meshes_to_static_colliders(stage, root_path):
    """
    Iterates over all meshes in the USD subtree under `root_path` and adds convex hull collision shapes.

    :param stage: The USD stage.
    :param root_path: The root path of the subtree to process.
    """
    # Get the root prim
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        raise ValueError(f"The specified root path '{root_path}' is not valid.")
    
    # Iterate over all child prims in the subtree
    for prim in Usd.PrimRange(root_prim):
        # Check if the prim is a mesh
        if prim.IsA(UsdGeom.Mesh):
            # print(f"Processing mesh: {prim.GetPath()}")

            # Check if MeshCollisionAPI is already applied
            UsdPhysics.CollisionAPI.Apply(prim)
            collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            collision_api.GetApproximationAttr().Set("convexHull")  # "convexDecomposition")

            physicsAPI = UsdPhysics.RigidBodyAPI.Apply(prim)
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            physicsAPI.CreateRigidBodyEnabledAttr(True)
            physicsAPI.CreateKinematicEnabledAttr(True)

            # set purpose to "guide" so that this mesh doesn't render in Isaac?
            if True:
                mesh_geom = UsdGeom.Imageable(prim)
                mesh_geom.CreatePurposeAttr(UsdGeom.Tokens.guide)


def convert_mesh_to_usd(in_file: str, out_file: str, load_materials: bool = True) -> bool:

    asyncio.run(
        async_convert_mesh_to_usd(in_file=in_file, out_file=out_file, load_materials=load_materials)
    )

    from pxr import Usd
    stage = Usd.Stage.Open(out_file)

    convert_meshes_to_static_colliders(stage, "/World")

    stage.GetRootLayer().Export(out_file)


async def async_convert_mesh_to_usd(in_file: str, out_file: str, load_materials: bool = True) -> bool:
    """ported from IsaacLab mesh_converter.py _convert_mesh_to_usd"""

    enable_extension("omni.kit.asset_converter")

    import omni.kit.asset_converter
    import omni.usd

    # Create converter context
    converter_context = omni.kit.asset_converter.AssetConverterContext()
    # Set up converter settings
    # Don't import/export materials
    converter_context.ignore_materials = not load_materials
    converter_context.ignore_animations = True
    converter_context.ignore_camera = True
    converter_context.ignore_light = True
    # Merge all meshes into one
    converter_context.merge_all_meshes = False
    # Sets world units to meters, this will also scale asset if it's centimeters model.
    # This does not work right now :(, so we need to scale the mesh manually
    converter_context.use_meter_as_world_unit = True
    converter_context.baking_scales = True
    # Uses double precision for all transform ops.
    converter_context.use_double_precision_to_usd_transform_op = True

    # Create converter task
    instance = omni.kit.asset_converter.get_instance()
    task = instance.create_converter_task(in_file, out_file, None, converter_context)
    # Start conversion task and wait for it to finish
    success = await task.wait_until_finished()
    if not success:
        raise RuntimeError(f"Failed to convert {in_file} to USD. Error: {task.get_error_message()}")
    return success


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


def convert_object_to_usd(objects_folder, object_config_filename, out_usd_path, project_root_folder):

    object_config_filepath = find_file(objects_folder, object_config_filename)
    assert object_config_filepath

    object_config_json_data = None
    with open(object_config_filepath, 'r') as file:
        object_config_json_data = json.load(file)
    # By convention, we fall back to render_asset if collision_asset is not set. See Habitat-sim Simulator::getJoinedMesh.
    collision_asset_filename = object_config_json_data["collision_asset"] if "collision_asset" in object_config_json_data else object_config_json_data["render_asset"]
    collision_asset_filepath = find_file(objects_folder, collision_asset_filename)

    print(f"Converting {collision_asset_filepath}...")

    convert_mesh_to_usd(collision_asset_filepath, out_usd_path, load_materials=False)

    render_asset_filepath_from_urdf = object_config_json_data["render_asset"]

    object_config_dir, _ = os.path.split(object_config_filepath)
    usd_dir, _ = os.path.split(out_usd_path)
    render_asset_filepath_for_usd = os.path.relpath(os.path.abspath(os.path.join(object_config_dir, render_asset_filepath_from_urdf)), start=project_root_folder)

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

def usd_to_habitat_position(position):
    """
    Convert a position from USD (Z-up) to Habitat (Y-up) coordinate system.

    Issac (x, y, z) -> Habitat (-x, z, y)
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


def usd_to_habitat_rotation(rotation):
    """
    Convert a quaternion rotation from USD to Habitat coordinate system.

    Parameters
    ----------
    rotation : list[float]
        Quaternion in USD coordinates [w, x, y, z] (wxyz).

    Returns
    -------
    list[float]
        Quaternion in Habitat coordinates [w, x, y, z] (wxyz).
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

    def quat_multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return [w, x, y, z]
    
    def quat_inverse(q):
        w, x, y, z = q
        norm = w*w + x*x + y*y + z*z
        return [w/norm, -x/norm, -y/norm, -z/norm]

    # Calculate the inverse of q_x90_inv and q_y180_inv
    q_x90 = quat_inverse(q_x90_inv)
    q_y180 = quat_inverse(q_y180_inv)
    # Calculate q_trans by multiplying q_y180 and q_x90
    q_trans = quat_multiply(q_y180, q_x90)
    # Multiply q_trans with rotation_usd to get the original rotation
    w, x, y, z = rotation
    rotation_hab = quat_multiply(q_trans, rotation)

    return rotation_hab



def convert_hab_scene(scene_filepath: str, project_root_folder: str, objects_folder: str = '', scene_usd_filepath: str= "./test_scene_instance.usda"):
    
    # assert os.path.exists(scene_filepath)
    try: 
        assert os.path.exists(scene_filepath)
    except AssertionError:
        raise FileNotFoundError(f"Scene instance json {scene_filepath} does not exist")

    stage = Usd.Stage.CreateNew(scene_usd_filepath)
    xform_prim = UsdGeom.Xform.Define(stage, "/Scene")
    stage.SetDefaultPrim(xform_prim.GetPrim())

    scenes_folder = os.path.dirname(scene_filepath)
    
    if not objects_folder:
        objects_folder = scenes_folder + "/../objects"


    assert os.path.exists(objects_folder) and os.path.isdir(objects_folder)
    # try: 
    #     assert os.path.exists(objects_folder)
    # except AssertionError:
    #     raise FileNotFoundError(f"Object glb folder {objects_folder} does not exist")
    

    with open(scene_filepath, 'r') as file:
        scene_json_data = json.load(file)    

    object_counts = {}

    max_count = -1 # 50  # temp only convert the first N objects
    count = 0

    # assert "object_instances" in scene_json_data
    
    try: 
        assert "object_instances" in scene_json_data
    except AssertionError:
        print(f"'object_instances' key not found in {scene_filepath} ")
        return
        # raise KeyError(f"'object_instances' key not found in {scene_filepath} ")

    
    for obj_instance_json in scene_json_data["object_instances"]:

        # todo: assert collision_asset_size is (1,1,1) or not present
        # todo: check is_collidable
        # todo: how to handle scale

        object_config_filename = f"{obj_instance_json['template_name']}.object_config.json"
        # todo: maybe don't prepend OBJECT_. Just use sanitize_usd_name.
        # NOTE: You cannot start the unique object name with number in teh Xform class environment
        base_object_name = "OBJECT_" + object_config_filename.removesuffix(".object_config.json")
        base_object_name = sanitize_usd_name(base_object_name)

        if base_object_name in object_counts:
            object_counts[base_object_name] += 1
            unique_object_name = f"{base_object_name}_dup{object_counts[base_object_name]}"
        else:
            unique_object_name = base_object_name
            object_counts[base_object_name] = 1

        out_usd_path = f"./data/usd/objects/{base_object_name}.usda"

        # todo: gather these up and do them later with multiprocessing
        if not os.path.exists(out_usd_path):

            convert_object_to_usd(objects_folder, object_config_filename, out_usd_path, project_root_folder)

        relative_usd_path = out_usd_path.removeprefix("./data/usd/")

        object_xform = UsdGeom.Xform.Define(
            stage, f"/Scene/{unique_object_name}"
        )  
        object_xform.GetPrim().GetReferences().AddReference(relative_usd_path)

        # Collect or create transform ops in the desired order: scale, rotation, translation
        xform_op_order = []

        # Ensure scale op exists
        scale = [1.0, 1.0, 1.0] # obj_instance_json.get("non_uniform_scale", [1.0, 1.0, 1.0])
        
        scale_op = next(
            (op for op in object_xform.GetOrderedXformOps() if op.GetName() == "xformOp:scale"),
            None,
        )
        if scale_op is None:
            scale_op = object_xform.AddScaleOp()
        scale_op.Set(Gf.Vec3f(*scale))
        xform_op_order.append(scale_op)

        # Ensure rotation op exists
        # rotation = habitat_to_usd_rotation([0.0, 0.0, 0.0, 1.0])
        # rotation = [1.0, 0.0, 0.0, 0.0]
        # 90 degrees about x in wxyz format
        # rotation = [0.7071068,0.7071067,0.0,0.0]
        rotation = habitat_to_usd_rotation(obj_instance_json.get("rotation", [1.0, 0.0, 0.0, 0.0]))
        orient_op = next(
            (op for op in object_xform.GetOrderedXformOps() if op.GetName() == "xformOp:orient"),
            None,
        )
        if orient_op is None:
            orient_op = object_xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
        orient_op.Set(Gf.Quatd(*rotation))
        xform_op_order.append(orient_op)

        # Ensure translation op exists
        position = habitat_to_usd_position(obj_instance_json.get("translation", [0.0, 0.0, 0.0]))
        translate_op = next(
            (op for op in object_xform.GetOrderedXformOps() if op.GetName() == "xformOp:translate"),
            None,
        )
        if translate_op is None:
            translate_op = object_xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3f(*position))
        xform_op_order.append(translate_op)

        # Let's not change the xform order. I don't know what's going on with Xform order.
        # object_xform.SetXformOpOrder(xform_op_order)


        count += 1
        if count == max_count:
            break

    # Set the file format arguments to flatten the stage
    # file_format_args = pxr.Sdf.FileFormatArguments()
    # file_format_args.Set("flatten", True)
    # # Save the stage without references
    # # stage.Save("output_no_ref.usda", file_format_args)
    
    # stage.Flatten()
    stage.GetRootLayer().Save() # NOTE: OLD WAY
    print(f"wrote scene {scene_usd_filepath}")


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


def convert_urdf_test():
    source_urdf_filepath = "data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf"
    clean_urdf_filepath = "data/robots/hab_spot_arm/urdf/hab_spot_arm_clean.urdf"
    # Temp USD must be in same folder as final USD. It's okay to be the exact same file.
    temp_usd_filepath = "data/usd/robots/hab_spot_arm.usda"
    out_usd_filepath = "data/usd/robots/hab_spot_arm.usda"
    convert_urdf(clean_urdf_filepath, temp_usd_filepath)
    add_habitat_visual_metadata_for_articulation(temp_usd_filepath, source_urdf_filepath, out_usd_filepath, project_root_folder="./")

def combine_two_usd_files():

    #file1 = 'data/hab_spot_arm/urdf/hab_spot_arm.usda'
    #file2 = 'data/usd/test_scene.usda'
    # Open the input files
    stage1 = Usd.Stage.Open("/home/trandaniel/dev/habitat-lab/data/hab_spot_arm/urdf/hab_spot_arm.usda")
    stage2 = Usd.Stage.Open("/home/trandaniel/dev/habitat-lab/data/usd/test_scene.usda")
    
    
    # Create a new stage for the combined data
    combinedStage = Usd.Stage.CreateNew("/home/trandaniel/dev/habitat-lab/data/hab_spot_arm/urdf/robot_scene_combined_output.usda")
   
    # # Open first file and copy prims
    # for prim in stage1.Traverse():
    #     combinedStage.DefinePrim(prim.GetPath(), prim.GetTypeName())
        
    # # Open second file and copy prims
    # for prim in stage2.Traverse():
    #     combinedStage.DefinePrim(prim.GetPath(), prim.GetTypeName())
    
    ####
    
    # def copy_prims(source_file, target_stage):
    #     source_stage = Usd.Stage.Open(source_file)
        
    #     for prim in source_stage.GetPseudoRoot().GetAllChildren():
    #         target_stage.OverridePrim(prim.GetPath())
    #         source_layer = source_stage.GetEditTarget().GetLayer()
    #         target_layer = target_stage.GetRootLayer()
            
    #         for spec in source_layer.pseudoRoot.GetAllChildren():
    #             target_layer.TransferContent(spec)
                
    # copy_prims(file1, combinedStage)
    # copy_prims(file2, combinedStage)
    
    
    ####
    
    
    # for name, value in stage2.GetPrimAtPath('/').GetAllMetadata().items():
        
    #     print(name)
    #     print(value)
    # Check if the spec is an Xform spec
        # if spec.GetName().startswith('xform'):
        #     # Get the value of the Xform spec
        #     xform_value = spec.Get()
        #     # Create a new attribute on the target layer with the same name and value
        #     combinedStage.GetPrimAtPath('/').CreateAttribute(spec.GetName(), Sdf.ValueTypeNames.String)
    # source_layer = stage.GetRootLayer()
    # omni.usd.resolve_paths(source_layer.identifier, source_layer.identifier)
    # stage.Save()
    # combinedStage.GetRootLayer().Save() 
    # print()
    
    
    # # Add all objects from stage1 to the combined stage
    # for prim in stage1.GetPseudoRoot().GetChildren():
    #     combinedStage.OverridePrim(prim.GetName())

    # Add all objects from stage2 to the combined stage
    # count = 0
    # for prim in stage2.GetPseudoRoot().GetChildren():
    #     if count == 0:
    #         count += 1
    #         continue
        
    #     references = prim.GetReferences()
    #     # Iterate over the references
    #     for reference in references:
    #         # Check if the reference has the prepend attribute
    #         if reference.HasAttribute("prepend"):
    #             # Get the prepend reference
    #             prepend_reference = reference.GetAttribute("prepend")
    #             print(prepend_reference)
        
        #combinedStage.OverridePrim(prim.GetName())

    # # Save the combined stage
    # combinedStage.Save()
# if __name__ == "__main__":
#     combine_two_usd_files()

if __name__ == "__main__":
    scene_instance_filepath = '/home/trandaniel/dev/habitat-sim/data/hssd-hab/scenes/102343992.scene_instance.json'
    #output_folder = '/home/guest/dev/usd_converter/converted_usd_test1'
    # object_folder = "/home/trandaniel/dev/habitat-sim/data/hssd-hab/objects/"
    

    from omni.isaac.lab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Create an empty Issac Sim stage.")
    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    # parse the arguments
    args_cli = parser.parse_args()
    # launch omniverse app
    args_cli.headless = True # Config to have Isaac Lab UI off
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    from omni.isaac.core.utils.extensions import enable_extension
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf

#     # convert_urdf_test()
    convert_hab_scene(scene_instance_filepath, project_root_folder="/home/trandaniel/dev/habitat-lab/test_convert")

