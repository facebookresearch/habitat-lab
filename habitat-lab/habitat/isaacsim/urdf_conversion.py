"""This module converts urdf files to usd.""" 
# TODO: first setup mamba activate isaaclab2?

import argparse
from omni.isaac.lab.app import AppLauncher
import os

def convert_urdf_to_usd(source_urdf_filepath: str, usd_folder_path: str, usd_filename: str) -> None:
    """
    This function converts a .urdf file into a .usd file. NOTE: Please be aware of 
    pathing issues and references to other .glb files within the urdf file.
    
    :param source_urdf_filepath: The absolute string path of the urdf file
    :param usd_folder_path: The absolute directory of the desired directory location of .usd file. The output will generatate a Props-> instanceable_meshes.usd, .asset_hash, config.yaml, and the desired .usd.

    """
    
    # Define the configuration for the URDF conversion
    config = UrdfConverterCfg(
        asset_path=source_urdf_filepath,  # Path to the input URDF file
        usd_dir=usd_folder_path,         # Directory to save the converted USD file
        usd_file_name=usd_filename,         # Name of the output USD file
        force_usd_conversion=True,                   # Force conversion even if USD already exists
        make_instanceable=True,                      # Make the USD asset instanceable
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
    


if __name__ == "__main__":
    urdf_file_path = '/home/guest/dev/robot_arm/hab_spot_arm/urdf/hab_spot_arm.urdf'
    usd_folder_path = '/home/guest/dev/robot_arm/usd_output'
    usd_filename = 'converted_robot' # .usd and .usda both work
    
    
    # create argparser
    parser = argparse.ArgumentParser(description="Create an empty Issac Sim stage.")
    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    # parse the arguments
    args_cli = parser.parse_args()
    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    #NOTE: Import isaac lab converter extension after launcher runs
    from omni.isaac.lab.sim.converters import UrdfConverter, UrdfConverterCfg
    
    convert_urdf_to_usd(urdf_file_path, usd_folder_path, usd_filename)

    simulation_app.close()