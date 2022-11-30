import os

import mrp

import home_robot

PKG_ROOT_DIR = home_robot.__path__[0]
PROJECT_ROOT_DIR = os.path.join(PKG_ROOT_DIR, "../..")

pip_deps = [
    "numpy",
    "scipy",
    "sophuspy",
    f"-e {os.path.abspath(PROJECT_ROOT_DIR)}",
    "pybullet",
]

env = mrp.Conda.SharedEnv(
    name="stretch_control_env",
    channels=["conda-forge", "robostack"],
    dependencies=[
        "python=3.8",
        "cmake",
        "pybind11",
        "ros-noetic-desktop",
        "ros-noetic-ros-numpy",
        {"pip": pip_deps},
    ],
)

mrp.process(
    name="picknplace_script",
    runtime=mrp.Conda(
        shared_env=env,
        run_command=[
            "python3",
            "pick_cup_real_robot.py",
        ],
    ),
)

if __name__ == "__main__":
    mrp.main()
