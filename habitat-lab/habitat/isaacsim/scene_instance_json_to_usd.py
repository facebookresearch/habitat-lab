"""This module converts a scene instance json to a usda file.
"""

import argparse
import asyncio
import json
import math
import os
import re
import xml.etree.ElementTree as ET

from typing import List, Optional, Union


def sanitize_usd_name(name: str) -> Optional[str]:
    """Sanitizes a string for use as a USD node name.

    :param name: The input string to sanitize.

    :return: The sanitized string, suitable for use as a USD node name.
    """

    # assert len(name) > 0
    if len(name) == 0:
        print("Input string for USD node is empty")
        return None

    # Replace spaces, hyphens, colons, and other special characters with underscores
    sanitized_name = re.sub(r"[^\w]", "_", name)

    # Remove consecutive underscores
    sanitized_name = re.sub(r"_+", "_", sanitized_name)

    # Ensure the name doesn't start with a number
    if sanitized_name[0].isdigit():
        sanitized_name = "_" + sanitized_name

    return sanitized_name


def object_usd_info(obj_instance_json, object_counts):
    """From the values defined in the object json from the scence instance, generate various
    strings and object counts.
    :param obj_instance_json: This is object level information from the scence_instance json file, Dictionary object
    :param object counts: This dictionary contains information about multiples uses of the same mesh

    :return: object_config_filename is the name of the object config file.
    out_usd_path is the filepath where the converted object is located. object_counts might be updated if there is another
    instance of the same mesh. unique_object_name is the unique name of a mesh.
    """
    object_config_filename = (
        f"{obj_instance_json['template_name']}.object_config.json"
    )
    # TODO: maybe don't prepend OBJECT_. Just use sanitize_usd_name.
    # NOTE: You cannot start the unique object name with number in teh Xform class environment
    base_object_name = "OBJECT_" + object_config_filename.removesuffix(
        ".object_config.json"
    )
    base_object_name = sanitize_usd_name(base_object_name)

    if base_object_name in object_counts:
        object_counts[base_object_name] += 1
        unique_object_name = (
            f"{base_object_name}_dup{object_counts[base_object_name]}"
        )
    else:
        unique_object_name = base_object_name
        object_counts[base_object_name] = 1

    out_usd_path = f"./data/usd/objects/{base_object_name}.usda"

    return (
        object_config_filename,
        out_usd_path,
        object_counts,
        unique_object_name,
    )


############################################################################################
# Convert scence instance object to a usd file
############################################################################################


def convert_object_to_usd(
    objects_folder: str,
    object_config_filename: str,
    out_usd_path: str,
    project_root_folder: str,
) -> None:
    """This converts an hssd object to a usda file.

    :param objects_folder : Parent folder that contains all the objects to an hssd scene.
    :param object_config_filename: String name of the config file associated with an object.
    :param out_usd_path: Filepath where object usd will be located after conversion
    :project_root_folder: Path of habitat-lab repo
    """
    object_config_filepath = find_file(objects_folder, object_config_filename)
    assert object_config_filepath

    object_config_json_data = None
    with open(object_config_filepath, "r") as file:
        object_config_json_data = json.load(file)
    # By convention, we fall back to render_asset if collision_asset is not set. See Habitat-sim Simulator::getJoinedMesh.
    collision_asset_filename = (
        object_config_json_data["collision_asset"]
        if "collision_asset" in object_config_json_data
        else object_config_json_data["render_asset"]
    )
    collision_asset_filepath = find_file(
        objects_folder, collision_asset_filename
    )

    print(f"Converting {collision_asset_filepath}...")

    convert_mesh_to_usd(
        collision_asset_filepath, out_usd_path, load_materials=False
    )

    # TODO: DANIEL is this from urdf?
    render_asset_filepath_from_urdf = object_config_json_data["render_asset"]

    object_config_dir, _ = os.path.split(object_config_filepath)

    # TODO: DELETE LINE IF NOT USED
    # usd_dir, _ = os.path.split(out_usd_path)

    render_asset_filepath_for_usd = os.path.relpath(
        os.path.abspath(
            os.path.join(object_config_dir, render_asset_filepath_from_urdf)
        ),
        start=project_root_folder,
    )

    render_asset_scale = object_config_json_data.get("scale", (1.0, 1.0, 1.0))

    add_habitat_visual_to_usd_root(
        out_usd_path, render_asset_filepath_for_usd, render_asset_scale
    )

    print(f"Wrote {out_usd_path}")


def find_file(folder: str, filename: str) -> str:
    """Given a root folder, return the string of the absolute path of file contained within the root folder

    :param folder: Root folder to search in
    :param filename: Name of file to be searched for.

    :return: Absoluate path string of filename if it is found, None if none found.
    """
    result = None
    for root, _, files in os.walk(folder):
        if filename in files:
            assert not result
            result = os.path.join(root, filename)
    return os.path.abspath(result)


def convert_mesh_to_usd(
    in_file: str, out_file: str, load_materials: bool = True
) -> None:
    """Convert mesh to usd

    :param in_file: string filepath of input mesh
    :param out_file: string of output usda file
    :param load_materials: TODO: Add description

    :return: TODO: Add description

    """
    asyncio.run(
        async_convert_mesh_to_usd(
            in_file=in_file, out_file=out_file, load_materials=load_materials
        )
    )

    from pxr import Usd

    stage = Usd.Stage.Open(out_file)

    convert_meshes_to_static_colliders(stage, "/World")

    stage.GetRootLayer().Export(out_file)


async def async_convert_mesh_to_usd(
    in_file: str, out_file: str, load_materials: bool = True
) -> bool:
    """ported from IsaacLab mesh_converter.py _convert_mesh_to_usd
    TODO: Add param descriptions, and return description.
    """

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
    task = instance.create_converter_task(
        in_file, out_file, None, converter_context
    )
    # Start conversion task and wait for it to finish
    success = await task.wait_until_finished()
    if not success:
        raise RuntimeError(
            f"Failed to convert {in_file} to USD. Error: {task.get_error_message()}"
        )
    return success


def convert_meshes_to_static_colliders(stage, root_path) -> None:
    """
    Iterates over all meshes in the USD subtree under `root_path` and adds convex hull collision shapes.

    :param stage: The USD stage.
    :param root_path: The root path of the subtree to process.
    """
    # Get the root prim
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        raise ValueError(
            f"The specified root path '{root_path}' is not valid."
        )

    # Iterate over all child prims in the subtree
    for prim in Usd.PrimRange(root_prim):
        # Check if the prim is a mesh
        if prim.IsA(UsdGeom.Mesh):
            # print(f"Processing mesh: {prim.GetPath()}")

            # Check if MeshCollisionAPI is already applied
            UsdPhysics.CollisionAPI.Apply(prim)
            collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            collision_api.GetApproximationAttr().Set(
                "convexHull"
            )  # "convexDecomposition")

            physicsAPI = UsdPhysics.RigidBodyAPI.Apply(prim)
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            physicsAPI.CreateRigidBodyEnabledAttr(True)
            physicsAPI.CreateKinematicEnabledAttr(True)

            # set purpose to "guide" so that this mesh doesn't render in Isaac?
            if True:
                mesh_geom = UsdGeom.Imageable(prim)
                mesh_geom.CreatePurposeAttr(UsdGeom.Tokens.guide)


def add_habitat_visual_to_usd_root(
    usd_filepath, render_asset_filepath, render_asset_scale
):
    """TODO: Add function description, param description, and output description"""

    # Open the USD file
    stage = Usd.Stage.Open(usd_filepath)
    if not stage:
        raise ValueError(f"Could not open USD file: {usd_filepath}")

    default_prim = stage.GetDefaultPrim()

    # Add habitatVisual:assetPath
    asset_path_attr = default_prim.CreateAttribute(
        "habitatVisual:assetPath", Sdf.ValueTypeNames.String
    )
    asset_path_attr.Set(render_asset_filepath)

    # Add habitatVisual:assetScale
    asset_scale_attr = default_prim.CreateAttribute(
        "habitatVisual:assetScale", Sdf.ValueTypeNames.Float3
    )
    asset_scale_attr.Set(Gf.Vec3f(*render_asset_scale))

    # Save the updated USD file
    stage.GetRootLayer().Save()
    print(
        f"Added habitat visual metadata to RigidBody prim in: {usd_filepath}"
    )


############################################################################################
# Add xform spatial information
############################################################################################


def add_xform_scale(object_xform) -> None:
    """Add object scale value into xform class for scene instance json."""
    # Ensure scale op exists, default value
    scale = [
        1.0,
        1.0,
        1.0,
    ]  # obj_instance_json.get("non_uniform_scale", [1.0, 1.0, 1.0])

    scale_op = next(
        (
            op
            for op in object_xform.GetOrderedXformOps()
            if op.GetName() == "xformOp:scale"
        ),
        None,
    )
    if scale_op is None:
        scale_op = object_xform.AddScaleOp()
    scale_op.Set(Gf.Vec3f(*scale))


def add_xform_rotation(obj_instance_json, object_xform) -> None:
    """Add object rotation value into xform class for scene instance json."""
    # Ensure rotation op exists
    # rotation = habitat_to_usd_rotation([0.0, 0.0, 0.0, 1.0])
    # rotation = [1.0, 0.0, 0.0, 0.0]
    # 90 degrees about x in wxyz format
    # rotation = [0.7071068,0.7071067,0.0,0.0]
    rotation = habitat_to_usd_rotation(
        obj_instance_json.get("rotation", [1.0, 0.0, 0.0, 0.0])
    )
    orient_op = next(
        (
            op
            for op in object_xform.GetOrderedXformOps()
            if op.GetName() == "xformOp:orient"
        ),
        None,
    )
    if orient_op is None:
        orient_op = object_xform.AddOrientOp(
            precision=UsdGeom.XformOp.PrecisionDouble
        )
    orient_op.Set(Gf.Quatd(*rotation))


def habitat_to_usd_rotation(rotation) -> List[float]:
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
    # q_y90_inv = [HALF_SQRT2, 0.0, HALF_SQRT2, 0.0]
    # q_z90_inv = [HALF_SQRT2, 0.0, 0.0, HALF_SQRT2]
    # q_z180_inv = [0.0, 0.0, 0.0, 1.0]

    q_x90_inv = [HALF_SQRT2, HALF_SQRT2, 0.0, 0.0]
    q_y180_inv = [0.0, 0.0, -1.0, 0.0]

    # Multiply q_y180_inv * q_x90_inv to get the combined quaternion
    def quat_multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return [w, x, y, z]

    q_trans_inv = quat_multiply(q_x90_inv, q_y180_inv)

    # Multiply q_trans_inv with the input rotation quaternion
    w, x, y, z = rotation
    rotation_usd = quat_multiply(q_trans_inv, [w, x, y, z])

    return rotation_usd


def add_xform_translation(obj_instance_json, object_xform) -> None:
    """Add object translation value into xform class for scene instance json."""
    # Ensure translation op exists
    position = habitat_to_usd_position(
        obj_instance_json.get("translation", [0.0, 0.0, 0.0])
    )
    translate_op = next(
        (
            op
            for op in object_xform.GetOrderedXformOps()
            if op.GetName() == "xformOp:translate"
        ),
        None,
    )
    if translate_op is None:
        translate_op = object_xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3f(*position))


def habitat_to_usd_position(position) -> List[float]:
    """
    Convert a position from Habitat (Y-up) to USD (Z-up) coordinate system.

    Habitat (-x, z, y) -> Isaac (x, y, z)
    """
    x, y, z = position
    return [-x, z, y]


############################################################################################
# Main function
############################################################################################


def convert_hab_scene(
    scene_filepath: str,
    project_root_folder: str,
    objects_folder: str = "",
    scene_usd_filepath: str = "./data/test_scene_instance.usda",
) -> None:
    """
    This is the main function takes data from a scene_instance.json file to .usda.
    NOTE, folder directories for objects are speific to the hssd object dataset.
    """

    # TODO: DANIEL Not sure why check is here, when json library for scene_json_data variable does same check?
    try:
        assert os.path.exists(scene_filepath)
    except AssertionError:
        raise FileNotFoundError(
            f"Scene instance json {scene_filepath} does not exist"
        )

    with open(scene_filepath, "r") as file:
        scene_json_data = json.load(file)

    # Form Xform stage object for output .usda file.
    stage = Usd.Stage.CreateNew(scene_usd_filepath)
    xform_prim = UsdGeom.Xform.Define(stage, "/Scene")
    stage.SetDefaultPrim(xform_prim.GetPrim())

    # Get scene folder and objects folder. In the case of HSSD, the scene folder
    # contains the scene_instance.json files, and the object folder is the same hierarchy.
    # Adjust paths as needed.
    scenes_folder = os.path.dirname(scene_filepath)
    if not objects_folder:
        objects_folder = scenes_folder + "/../objects"

    try:
        assert os.path.exists(objects_folder)
    except AssertionError:
        raise FileNotFoundError(
            f"Object glb folder {objects_folder} does not exist"
        )

    # A scene may have multiple instances of the same object mesh. Xform usda needs
    # to have one unique name per mesh.
    object_counts: Dict[str, int] = {}
    max_count = -1  # 50  #TODO: temp only convert the first N objects
    count = 0

    try:
        assert "object_instances" in scene_json_data
    except AssertionError:
        print(f"'object_instances' key not found in {scene_filepath} ")
        return
        # raise KeyError(f"'object_instances' key not found in {scene_filepath} ")# TODO: may need return instead of raise, incase we need to for loop convert_hab_scene

    for obj_instance_json in scene_json_data["object_instances"]:
        # TODO: assert collision_asset_size is (1,1,1) or not present
        # TODO: check is_collidable
        # TODO: how to handle scale

        object_config_filename = (
            f"{obj_instance_json['template_name']}.object_config.json"
        )
        # TODO: maybe don't prepend OBJECT_. Just use sanitize_usd_name.
        # NOTE: You cannot start the unique object name with number in teh Xform class environment
        base_object_name = "OBJECT_" + object_config_filename.removesuffix(
            ".object_config.json"
        )
        base_object_name = sanitize_usd_name(base_object_name)

        (
            object_config_filename,
            out_usd_path,
            object_counts,
            unique_object_name,
        ) = object_usd_info(obj_instance_json, object_counts)

        # TODO: gather these up and do them later with multiprocessing
        # NOTE: This will be answered when trying to proccess multiple scene_instance_jsons.
        # if not os.path.exists(out_usd_path):
        convert_object_to_usd(
            objects_folder,
            object_config_filename,
            out_usd_path,
            project_root_folder,
        )

        relative_usd_path = out_usd_path.removeprefix("./data/usd/")

        # Form Scene Node
        object_xform = UsdGeom.Xform.Define(
            stage, f"/Scene/{unique_object_name}"
        )

        object_xform.GetPrim().GetReferences().AddReference(relative_usd_path)

        # Operation are performed in the order listed, change as desired
        # Current order is scale at global origin, local rotation at global origin, then translate from global origin
        add_xform_scale(object_xform)
        add_xform_rotation(obj_instance_json, object_xform)
        add_xform_translation(obj_instance_json, object_xform)

        # NOTE: Let's not change the xform order. I don't know what's going on with Xform order.
        # object_xform.SetXformOpOrder(xform_op_order)

        count += 1
        if count == max_count:
            break

    stage.GetRootLayer().Save()
    print(f"wrote scene {scene_usd_filepath}")


if __name__ == "__main__":
    scene_instance_filepath = "/home/trandaniel/dev/habitat-sim/data/hssd-hab/scenes/102343992.scene_instance.json"

    from omni.isaac.lab.app import AppLauncher

    parser = argparse.ArgumentParser(
        description="Create an empty Issac Sim stage."
    )
    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    # parse the arguments
    args_cli = parser.parse_args()
    # launch omniverse app
    args_cli.headless = True  # Config to have Isaac Lab UI off
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    from omni.isaac.core.utils.extensions import enable_extension
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

    convert_hab_scene(
        scene_instance_filepath,
        project_root_folder="/home/trandaniel/dev/habitat-lab/test_convert",
    )
