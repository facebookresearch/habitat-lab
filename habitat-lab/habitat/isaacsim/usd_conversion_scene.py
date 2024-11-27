"""This file converts a <EPISODE>.scene_instance.json to a .usd file, 
using various .glb scene objects to buld the .usd scene instance..
"""

from pxr import Usd, UsdGeom, Sdf, Gf
import json
from pathlib import Path
import os
from ssh_conversion import *
from multiprocessing import Pool, Barrier
from typing import List, Dict, Union, Tuple


# CONSTANTS
STAGE_INSTANCE = "stage_instance"
OBJECT_INSTANCES = "object_instances"
TEMPLATE_NAME = "template_name"
TRANSLATION = "translation"
ROTATION = "rotation"
SCALE = "non_uniform_scale"


def find_filepath(filename, parent_directory) -> str:
    file_found = False
    for root, dirs, files in os.walk(parent_directory):
        if filename in files:
            output_filepath = str(os.path.join(root, filename))
            file_found = True
            break

    if not file_found:
        print(f"{filename} not found in the parent directoy {parent_directory}")
        return None

    return output_filepath


def get_unique_object_names(scene_data) -> list[str]:
    """
    This looks at the scene_instance.json by looking at the "template_name" key.
    HSSD keeps all these objects as .glb. The scene json could use the same type of
    object multiple times, but we only need the unique list of objects
    """
    object_list = scene_data[OBJECT_INSTANCES]  

    object_name_list = []
    for object in object_list:
        object_name_list.append(object[TEMPLATE_NAME])

    return list(set(object_name_list))


def make_object_filepath_list(
    glb_data_folder: str,
    scene_instance_filepath: str,
    scene_data,   #NOTE: This should be typing for scene_instance.json Dict[str, Union[str, Dict[str, str] , List[ Dict[str, List[float]] ]]],
    file_extension: str = ".glb",
) -> List[str]:

    object_name_list = get_unique_object_names(scene_data)

    # Search through the object glb folder to make a list of object filepaths from scene_data
    object_filepath_list = []
    for object_name in object_name_list:
        object_glb = object_name + file_extension
        object_path = find_filepath(object_glb, glb_data_folder)
        if object_path:
            object_filepath_list.append(object_path)

    if not object_filepath_list:
        # Stop the entire script if no .glb objects were found
        print("No .glb objects found. Halting the program.")
        sys.exit(1)

    return object_filepath_list


def make_collider_glb_filepath(object_name: str, scene_instance_path: str) -> str:

    object_collider_glb = object_name + ".collider.glb"
    scene_folder_path = os.path.dirname(os.path.abspath(scene_instance_path))
    hssd_folder_path = os.path.dirname(scene_folder_path)
    object_folder_path = hssd_folder_path + "/objects"

    object_usd_path = object_folder_path + "/" + object_name + ".collider.usd"
    return object_usd_path


def initialize_stage(stage_usd_filepath: str, scene_instance_usd_filepath: str) -> Usd.Stage:
    """Initialize Usd.Stage"""


    stage = Usd.Stage.CreateNew(scene_instance_usd_filepath) # NOTE Might not work starting with nummber
    xform_prim = UsdGeom.Xform.Define(stage, "/Scene")

    #probably don't need this
    #prim = xform_prim.GetPrim()
    #prim.GetReferences().AddReference(stage_usd_filepath)

    return stage


def find_object_usd(object_data, object_usd_folder: str, file_extension=".collider.usd") -> Tuple[bool, str, str]:
    """
    Find the object .usd file from the object usd folder.
    """
    object_exists = True

    object_name = object_data[TEMPLATE_NAME]
    usd_filename = object_name + file_extension
    usd_filepath = object_usd_folder + usd_filename

    if not os.path.exists(usd_filepath):
        print(f"File {usd_filepath} not found. .usd file may not have been coverted or downloaded")
        object_exists = False

    return object_exists, usd_filepath, object_name


def add_object_to_stage(
    stage: Usd.Stage,
    object_data,
    object_filepath: str,
    count: str,
    position: List[float]=[0.0, 0.0, 0.0],
    rotation: List[float]=[0.0, 0.0, 0.0],
    scale: List[float]=[1.0, 1.0, 1.0],
) -> None:
    """
    This module adds an UsdGeom.Xform to an initialized Usd.Stage.
    """
    position = object_data[TRANSLATION]
    rotation = object_data[ROTATION]
    scale = object_data[SCALE]
    object_name = object_data[TEMPLATE_NAME]

    # PSEUDOCODE # Given template name from scene_instance.json from above
    # From root: /home/trandaniel/dev/habitat-sim/data/hssd-hab/scenes/<MY_SCENE>.scene_instance.json
    # Object folder: /home/trandaniel/dev/habitat-sim/data/hssd-hab/objects

    # Navigate from scene instance folder to object folder through script
    #

    # Recursively search through object folder for object json (Structure vary between object data sets. Write for HSSD for now)
    # Open JSON, and then use the "render_asset" key and "collision_asset" key to get .glb file

    # Convert .collision.glb to USD with isaacsim omniverse
    # Output .usd file in the object folder that contains collision.glb

    # CONVERSION

    # Turn object .glb to xform

    # Add count to string if object used more than once
    if count > 0:
        object_name = object_name + f"_{count}"

    object_stage_name = "/OBJECT_" + object_name # NOTE: Object state name must start with a char letter

    object_xform = UsdGeom.Xform.Define(
        stage, object_stage_name
    )  
    object_xform.GetPrim().GetReferences().AddReference(object_filepath)

    # Set Position
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
    translate_op.Set(value=Gf.Vec3d(*position))

    # Set Rotation
    orient_op = next(
        (
            op
            for op in object_xform.GetOrderedXformOps()
            if op.GetName() == "xformOp:orient"
        ),
        None,
    )
    if orient_op is None:
        orient_op = object_xform.AddOrientOp()
    orient_op.Set(value=Gf.Quatd(*rotation))

    # Set Scale
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
    scale_op.Set(value=Gf.Vec3d(*scale))


def convert_scene_instance(
    scene_instance_filepath: str, stage_glb_folder: str, local_download_folder: str, glb_data_folder: str
) -> None:
    """
    This function takes a folder of .glb objects, converts them remotely to .usd,
    downloads the converted files, and then makes a .usda scene. The idea is to
    convert a scene_instance json to a .usd file by aggregating all the defined 
    objects within the scene instance json.
    """
    
    # Open scnece_instance JSON
    with open(scene_instance_filepath, "r") as file:
        scene_data = json.load(file)
 
    
    stage_name: str = scene_data[STAGE_INSTANCE][TEMPLATE_NAME].split('/')[1
    ]
    
    # Form a list of existing object glb file paths 
    object_glb_filepath_list = make_object_filepath_list(
        glb_data_folder,
        scene_instance_filepath,
        scene_data,
        file_extension=".collider.glb",
    )

    # # Append stage glb to piggy back on conversion.py. #Note, stage glb not necessary probably
    #stage_glb_filepath = stage_glb_folder + stage_name + '.glb'
    #object_glb_filepath_list.append(stage_glb_filepath)

    # Upload glb objects to remote server
    num_processes = 10
    barrier = Barrier(num_processes)
    with Pool(processes=num_processes) as pool:
        pool.map(upload_object_glb_to_server, object_glb_filepath_list)

    # Process objects remotely
    run_remote_converter()

    # Download objects
    download_remote_usd_folder(local_download_folder)

    
    # TODO: Delete objects on remote server after download

    # Populate USDA file. USDA is the human readable version of a usd file.\

    stage_usd_filepath = local_download_folder + 'object_usd/' + stage_name + '.usd' # NOTE: Might not need this
    scene_instance_usd_filepath = './' + stage_name + '.scene_instance.usda'

    stage = initialize_stage(stage_usd_filepath, scene_instance_usd_filepath)

    object_usd_folder = local_download_folder + "/object_usd/" #The string is the name of the folder from remote server
    
    # This is to ensure the same object used multiple times have a unique ID number
    previous_object_name = "" 
    count = 0

    for object_data in scene_data[OBJECT_INSTANCES]:
        object_exists, usd_filepath, object_name = find_object_usd(
            object_data, object_usd_folder, file_extension=".collider.usd"
        )

        # This if/else is to ensure multiple copies of the same type of object have unique names.
        if previous_object_name == object_name:
            count += 1
        else:
            previous_object_name = object_name
            count = 0

        if object_exists:
            add_object_to_stage(stage, object_data, usd_filepath, count)

    stage.GetRootLayer().Save()

    print(f"Finished rendering convertin {scene_instance_filepath} to .usd")


if __name__ == "__main__":

    scene_instance_filepath = "/home/trandaniel/dev/habitat-sim/data/hssd-hab/scenes/102343992.scene_instance.json"
    stage_glb_folder = '/home/trandaniel/dev/IsaacSimConversion/'
    local_download_folder = "/home/trandaniel/dev/IsaacSimConversion/usd_scene/"
    glb_data_folder = "/home/trandaniel/dev/habitat-sim/data/hssd-hab/objects"

    convert_scene_instance(
        scene_instance_filepath, stage_glb_folder, local_download_folder, glb_data_folder
    )

    # TODO: Test with /home/trandaniel/dev/habitat-sim/data/hssd-hab/scenes/<MY_SCENE>.scene_instance.json
    # TODO; Open resulting scene output with blender, open it in viewer.py
    # Some things might be missing, only rigid objects render for now, articulated later
