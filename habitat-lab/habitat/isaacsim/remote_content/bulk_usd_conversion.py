import os
from typing import List, Dict
import asyncio
import argparse
from omni.isaac.lab.app import AppLauncher
import timer


def copy_directory_structure(source_dir: str, target_dir: str):

    for root, dirs, files in os.walk(source_dir):
        relative_path = os.path.relpath(root, source_dir)
        target_path = os.path.join(target_dir, relative_path)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
            

def list_child_folders(parent_folder: str) -> List[str]:
    try:
        return [folder for folder in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, folder))]
    except FileNotFoundError:
        print(f"Error: The folder '{parent_folder}' does not exist.")
        return []
    
def make_child_folder_paths(source_dir: str, child_folder_list: List[str]) -> List[str]:
    child_folder_paths = []
    for child_folder in child_folder_list:
        child_folder_paths.append(source_dir + child_folder + '/')
    return child_folder_paths


async def convert_glb_to_usd(glb_folder_path: str, map_source_to_output: Dict[str, str]) -> None:
    """
    This function converts glb files to usd. This function launches an empty Isaac Lab stage,
    Imports the respective Isaac Lab converter functions, then converts with another 
    MeshConverter._convert_mesh_to_usd async function. The simulation app then closes, and
    the converted functions are dumped onto the respective usd_folder_path.
    """
    
     # create argparser
    parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    # parse the arguments
    args_cli = parser.parse_args()
    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    from omni.isaac.lab.sim.converters import MeshConverter #NOTE: Import isaac lab converter extension after launcher runs
   
    loop = asyncio.get_event_loop()
    for object_folder_path in map_source_to_output:
        # print(f"{object_folder_path} started.")
        usd_folder_path = map_source_to_output[object_folder_path]
        
        glb_file_list = [file for file in os.listdir(object_folder_path) if '.collider.glb' in file and os.path.isfile(os.path.join(object_folder_path, file))]
        for glb_filename in glb_file_list:
            
    
            glb_filepath = object_folder_path + glb_filename
            if not os.path.exists(glb_filepath):
                continue
            usd_filename = f"{glb_filename.replace('.glb', '.usd')}"
            usd_filepath = usd_folder_path + usd_filename
            
            loop.run_until_complete(MeshConverter._convert_mesh_to_usd(in_file=glb_filepath, out_file=usd_filepath))
        print(f"{usd_folder_path} finished.")
    loop.close()
    print(f"Bulk conversion finished.")
    simulation_app.close()
    

async def bulk_convert_to_usd(source_dir: str, target_dir: str):
    object_folders = list_child_folders(source_dir)
    object_folder_paths = make_child_folder_paths(source_dir, object_folders)
    
    usd_folder_paths = make_child_folder_paths(target_dir, object_folders)
    
    # A dictionary of {Input filepath: output filepath}
    map_source_to_output = dict(zip(object_folder_paths, usd_folder_paths))
    

    for object_folder_path in object_folder_paths:
        usd_folder_path = map_source_to_output[object_folder_path]
        
    await convert_glb_to_usd(source_dir, map_source_to_output)


if __name__ == "__main__":
    source_dir = '/home/guest/repos/hssd-hab/objects/'
    target_dir = '/home/guest/repos/hssd-hab/usd_objects/'
    # copy_directory_structure(source_dir, target_dir)
    asyncio.run(bulk_convert_to_usd(source_dir, target_dir))