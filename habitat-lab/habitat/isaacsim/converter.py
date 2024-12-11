''' 
This module takes converts glb files to usd. 

The system that runs this folder needs have the Issac Lab repo installed.
The converter function from isaac lab can import only after the Isaac Lab stage
is launched. asyncio library is used because the source code for 
MeshConverter._convert_mesh_to_usd is also an async function. 

'''
import argparse
from omni.isaac.lab.app import AppLauncher
import os
import timer


def convert_glb_to_usd(glb_folder_path: str, usd_folder_path: str) -> None:
    """
    This function converts glb files to usd. This function launches an empty Isaac Lab stage,
    Imports the respective Isaac Lab converter functions, then converts with another 
    MeshConverter._convert_mesh_to_usd async function. The simulation app then closes, and
    the converted functions are dumped onto the respective usd_folder_path.
    """

    glb_folder_list = [glb for glb in os.listdir(glb_folder_path) if os.path.isfile(os.path.join(glb_folder_path, glb))]

    for glb_filename in glb_folder_list:
        glb_filepath = glb_folder_path + glb_filename
        usd_filename = f"{glb_filename.replace('.glb', '.usd')}"
        usd_filepath = usd_folder_path + usd_filename
        
        converter_config = MeshConverterCfg(
            asset_path=glb_filepath,
            usd_dir=usd_folder_path, 
            usd_file_name=usd_filename,
            force_usd_conversion=False,
            collision_approximation="convexDecomposition",
        )
        converter = MeshConverter(converter_config)
        print(f"Wrote {converter.usd_path}")
            

if __name__ == "__main__":
    glb_folder_path = '/path/object_glb/'
    usd_folder_path = '/path/object_usd/'
    
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

    convert_glb_to_usd(glb_folder_path, usd_folder_path)

    simulation_app.close()