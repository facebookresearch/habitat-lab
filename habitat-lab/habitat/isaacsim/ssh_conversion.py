"""
This module helps a local computer interact with a remote server.
The remote server has the machine specs required for Issac Lab installation
to run MeshConverter._convert_mesh_to_usd(). The local machine does not.
See this url for documentation:  
https://isaac-sim.github.io/IsaacLab/main/_modules/omni/isaac/lab/sim/converters/mesh_converter.html#MeshConverter
"""

from scp import SCPClient
import os
import subprocess
from multiprocessing import Barrier

#NOTE: USER SHOULD INPUT REMOTE SERVER VALUES HERE
username = 'USERNAME'
hostname = 'IP_ADDRESS' #NOTE: THIS IS A SECURITY ISSUE
port = 23456
REMOTE_GLB_FOLDER = '/path/object_glb/'
REMOTE_USD_FOLDER = '/path/object_usd'


def upload_object_glb_to_server(object_glb_path: str) -> None:
    """
    Upload a local folder with .glb files and upload it to remote server. 
    """

    attempts = 3 #NOTE: The attempts and for loop ensure uploads complete if upload task is dumped from queue.
    for attempt in range(attempts):
        try:
            remote_file_path = REMOTE_GLB_FOLDER + os.path.basename(object_glb_path)
            scp_command = f'scp {object_glb_path} {hostname}:{remote_file_path}'
            subprocess.run(scp_command, shell=True)
            break
        except Exception as e:
            print(f"Error uploading {filepath} (attempt {i+1}/{attempts}): {e}")
            if attempt == attempts - 1:
                raise

        barrier.wait() 

def run_remote_converter() -> None:
    """
    This function runs the glb to usd conversion.py that sits on the given remote computer.
    This sends bash commands to active Isaac Lab library in conda and run converter.py with a local subprocess library.
    """

    #Check remote pathing.
    scp_command = f"ssh {username}@{hostname} 'source /home/guest/miniforge3/etc/profile.d/conda.sh; conda activate isaaclab2; python /home/guest/dev/isaac_lab_converter/converter.py' "
    subprocess.run(scp_command, shell=True)


def download_remote_usd_folder(local_download_file_path: str) -> None:
    """
    This function downloads folder contents from a remote computer.
    This function uses the bash script with the help of the subprocess library.
    """
    scp_command = f'scp -r -C {hostname}:{REMOTE_USD_FOLDER} {local_download_file_path}'
    subprocess.run(scp_command, shell=True)


if __name__ == "__main__":
    run_remote_converter()