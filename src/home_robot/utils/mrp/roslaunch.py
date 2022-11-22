from typing import Optional


def roslaunch_local_cmd(launch_file: str, pkg_name: Optional[str] = None):
    if pkg_name is None:
        pkg_name = ""

    cmd = ["roslaunch", pkg_name, launch_file]

    "conda deactivate"
    "source ~/catkin_ws/devel/setup.bash"
