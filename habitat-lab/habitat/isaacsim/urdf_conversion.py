"""This module converts urdf files to usd.""" 
# TODO: first setup mamba activate isaaclab2?

import argparse
from omni.isaac.lab.app import AppLauncher

def convert_urdf_to_usd(source_urdf_filepath: str, usd_folder_path: str) -> None:
    """
    This function converts a .urdf file into a .usd file. NOTE: Please be aware of 
    pathing issues and references to other .glb files within the urdf file.
    
    :param source_urdf_filepath: The absolute string path of the urdf file
    :param usd_folder_path: The absolute directory of the desired directory location of .usd file. The output will generatate a Props-> instanceable_meshes.usd, .asset_hash, config.yaml, and the desired .usd.

    """

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
    
    # urdf_filename = 'sink.urdf' 
    # usd_filepath = f"{source_urdf_filepath.replace('.urdf', '.usd')}"

    # Add config information
    config = UrdfConverterCfg()
    config.asset_path = source_urdf_filepath # Source file
    config.usd_dir = usd_folder_path # Output file
    config.fix_base = True

    # Convert asset
    converter = UrdfConverter(cfg=config)
    converter._convert_asset(cfg=config)

    simulation_app.close()

if __name__ == "__main__":
    urdf_folder_path = '/home/guest/dev/urdf_converter/object_urdf/sink.urdf'
    usd_folder_path = '/home/guest/dev/urdf_converter/object_usd/'
    convert_urdf_to_usd(urdf_folder_path, usd_folder_path)
