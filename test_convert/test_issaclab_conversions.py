import pytest
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)
from isaaclab_convert_test import *
import json

# def test_init_issac_lab_env():
#     import argparse
#     from omni.isaac.lab.app import AppLauncher

#     parser = argparse.ArgumentParser(description="Create an empty Issac Sim stage.")
#     # append AppLauncher cli args
#     AppLauncher.add_app_launcher_args(parser)
#     # parse the arguments
#     ## args_cli = parser.parse_args()
#     args_cli, _ = parser.parse_known_args()
#     # launch omniverse app
#     args_cli.headless = True # Config to have Isaac Lab UI off
#     app_launcher = AppLauncher(args_cli)
#     simulation_app = app_launcher.app

#     from omni.isaac.core.utils.extensions import enable_extension
#     from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf
    
#     assert str(simulation_app.__class__) == "<class 'omni.isaac.kit.simulation_app.SimulationApp'>"
#     print("Simulation app instantiation test successful.")
#     # simulation_app.close()
    
def test_pyest_runs_in_isaaclab():
    assert 2 == 2
    

## GLB TO USD CONVERSION ##

def test_wrong_scene_instance_path():
    scene_instance_filepath = './NONEXISTENT.scene_instance.json'
    with pytest.raises(FileNotFoundError):
        convert_hab_scene(scene_instance_filepath, project_root_folder="./test_project_root")
        
def test_example2_scene_instance():
    scene_instance_filepath = './test_convert/data/EXAMPLE2.scene_instance.json'
    object_folder = './test_convert/data/objects_EXAMPLE2'
    convert_hab_scene(scene_instance_filepath, project_root_folder="./test_project_root", objects_folder=object_folder)
    
    output_usd_path = "./data/usd/test_scene.usda"
    
    stage = Usd.Stage.Open(output_usd_path)
    

    xform_path = "/Scene/OBJECT_1efdc3d37dfab1eb9f99117bb84c59003d684811"
    xform_prim = stage.GetPrimAtPath(xform_path)
    
    xformable = UsdGeom.Xformable(xform_prim)
    
    usda_orient_im = list(xform_prim.GetAttribute('xformOp:orient').Get().imaginary)
    usda_orient_real = xform_prim.GetAttribute('xformOp:orient').Get().real
    usda_scale = list(xform_prim.GetAttribute('xformOp:scale').Get())
    usda_translate = list(xform_prim.GetAttribute('xformOp:translate').Get())


    #change usd coords back to habitat coords
    
    usda_translate_hab_coord = usd_to_habitat_position(usda_translate)
    usda_rotation = [usda_orient_real] + usda_orient_im
    usda_rotation_hab_coord = usd_to_habitat_rotation(usda_rotation)

    with open(scene_instance_filepath , 'r') as file:
        scene_instance_json_data = json.load(file)
        
    
    scene_instance_translation = scene_instance_json_data['object_instances'][0]['translation']
    scene_instance_rotation = scene_instance_json_data['object_instances'][0]['rotation']
    scene_instance_uniform_scale = scene_instance_json_data['object_instances'][0]['non_uniform_scale']
    
    assert usda_translate_hab_coord == pytest.approx(scene_instance_translation)
    assert usda_rotation_hab_coord== pytest.approx(scene_instance_rotation)
    # TODO: Add Scale to show values are equal, looks like most objects are (1, 1, 1) for the (x, y, z) coordinates in hab space.  
    # Not sure, but pxr space is has values from 0 to infinity, and 1 is default, but hab space has
    # -1 values?  Don't know what hab space scaling is.

##########################################################################

## URDF TO USD CONVERSION ##

##########################################################################

def test_clean_urdf():
    input_file = "data/hab_spot_arm/urdf/hab_spot_arm.urdf"
    output_file = "data/hab_spot_arm/urdf/hab_spot_arm_clean.urdf"
    clean_urdf(input_file, output_file, remove_visual=False)

def test_convert_urdf():

    clean_urdf_filepath = "data/hab_spot_arm/urdf/hab_spot_arm_clean.urdf"
    # Temp USD must be in same folder as final USD. It's okay to be the exact same file.
    temp_usd_filepath = "data/hab_spot_arm/urdf/hab_spot_arm_no_hab_metadata.usda"
    convert_urdf(clean_urdf_filepath, temp_usd_filepath)
    
def test_add_habitat_visual_metadata_for_articulation():
    source_urdf_filepath = "data/hab_spot_arm/urdf/hab_spot_arm.urdf"
    temp_usd_filepath = "data/hab_spot_arm/urdf/hab_spot_arm_no_hab_metadata.usda"
    out_usd_filepath = "data/hab_spot_arm/urdf/hab_spot_arm.usda"
    add_habitat_visual_metadata_for_articulation(temp_usd_filepath, source_urdf_filepath, out_usd_filepath, project_root_folder="./")
    

##########################################################################

## LOADING OBJECTS ##

##########################################################################

def combine_two_usd_files():

    
    # Open the input files
    stage1 = Usd.Stage.Open('file1.usda')
    stage2 = Usd.Stage.Open('file2.usda')
    # Create a new stage for the combined data
    combinedStage = Usd.Stage.CreateNew('output.usda')
    
    
    # Add the contents of the first file to the combined stage
    for prim in stage1.GetPseudoRoot().GetChildren():
        combinedStage.OverridePrim(prim.GetName())
    # Add the contents of the second file to the combined stage
    for prim in stage2.GetPseudoRoot().GetChildren():
        combinedStage.OverridePrim(prim.GetName())
    # Save the combined stage
    combinedStage.Save()


##########################################################################

if __name__ == "__main__":

    test_add_habitat_visual_metadata_for_articulation()