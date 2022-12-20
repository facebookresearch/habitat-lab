# Home Robot

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/facebookresearch/home-robot/tree/main.svg?style=shield&circle-token=282f21120e0b390d466913ef0c0a92f0048d52a3)](https://dl.circleci.com/status-badge/redirect/gh/facebookresearch/home-robot/tree/main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Mostly Hello Stretch infrastructure

## Installation

1. Prepare a conda environment (ex: `conda create -n home_robot python=3.8 && conda activate home_robot`)
1. Install Mamba (optional but highly recommended): `conda install -c conda-forge mamba`
1. Install repo `pip install -e .`

### Additional instructions for setting up on hardware

1. Install firmware from Hello Robot
    ```sh
    # Copy robot factory data into your user workspace
    cp -r /etc/hello-robot/stretch-re* ~

    # Clone the official setup scripts
    cd ~
    git clone https://github.com/hello-robot/stretch_install
    cd stretch_install

    # Run setup script (DO NOT RUN BOTH)
    ./stretch_new_robot_install.sh  # if installing into a new robot
    ./stretch_new_user_install.sh  # if installing into a new user account on a already-set-up robot
    ```
1. Open `~/.bashrc`. You will see a block of commands that initializes Stretch, and another block that initializes Conda. If needed, move the stretch setup block BEFORE the conda initialization.
1. Launch a new bash shell. Activate an conda env with Python 3.8 installed.
1. Link `home_robot` and install ROS stack
    ```sh
    # Create symlink in catkin workspace
    ln -s /abs/path/to/home-robot/rospkg $HOME/catkin_ws/src/home_robot

    # Install dependencies for catkin
    pip install empy catkin_pkg rospkg

    # Build catkin workspace
    cd ~/catkin_ws
    catkin_make

    # Add newly built setup.bash to .bashrc
    echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
    ```
1. Calibrate robot following instructions [here](https://github.com/hello-robot/stretch_ros/tree/master/stretch_calibration#checking-the-current-calibration-with-new-observations).
1. Generate URDF from calibration data: `rosrun stretch_calibration update_urdf_after_xacro_change.sh`.
1. Run `stretch_robot_system_check.py` to make sure that things are normal.

#### Additional hardware stack dependencies
1. Hector SLAM: `sudo apt install ros-noetic-hector-*`
1. (For grasping only) Detectron 2: `python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'`

You also need to install a supported grasp prediction library. (TODO: clarify?)


## Usage

### Launching the hardware stack:
```sh
# Launch core components
roslaunch home_robot startup_stretch_hector_slam.launch

# Launch state estimator & goto controller
cd /path/to/home-robot/src/home_robot
mrp up agent_procs
```

### Launching a minimal kinematic simulation (no camera yet)
```sh
cd /path/to/home-robot/src/home_robot
mrp up sim_stack
```

This launches:
- Fake stretch node (A kinematic simulation that publishes 100% accurate odometry and slam information and accepts velocity control inputs)
- State estimation node
- Continuous controller node

### Stopping all processes
```sh
mrp down
```

### Launching a simple local command line interface (CLI) on the robot:

The CLI currently exposes a simple base control interface via the terminal.
The interface `home_robot.client.LocalHelloRobot` can also be imported and used within user scripts.

```sh
cd src/home_robot/client
mrp up local_cli --attach
```

Available commands:
```py
robot.get_base_state()  # returns base location in the form of [x, y, rz]
robot.set_nav_mode()  # enables continuous navigation
robot.set_pos_mode()  # enables position control
robot.set_yaw_tracking(value: bool)  # turns yaw tracking on/off (robot only tries to reach the xy position of goal if off)
robot.set_goal(xyt: list)  # sets the goal for the goto controller
robot.set_velocity(v: float, w: float)  # directly sets the linear and angular velocity of robot base (command gets overwritten immediately if goto controller is on)
```

#### Getting Started

```py
robot.set_nav_mode()  # Enables continuous control
robot.set_goal(]1.0, 0.0, 0.0])  # Sets XYZ target
robot.get_base_state()  # Shows the robot's XYZ coordinates
```

### Launching ROS Demo

You need to create a catkin workspace on your server in order to run this demo, as this is where we will run [Contact Graspnet](https://github.com/cpaxton/contact_graspnet/tree/cpaxton/devel).

Contact graspnet is downloaded under `third_party/`, but there is a `CATKIN_IGNORE` file in this directory. You want to symlink this file out into your workspace:
```
ROSWS=/path/to/ros_ws
ln -s `rospack find home_robot`/third_party/contact_graspnet $ROSWS/src/contact_graspnet
```
... but it actually shouldn't be necessary. What is necessary is to build the grasp service defined in `home_robot` by placing it into `$ROSWS`.


Put the robot in its initial position, e.g. so the arm is facing cups you can pick up. On the robot side:
```
roslaunch home_robot startup_stretch_hector_slam.launch
```

### Troubleshooting 

- `ImportError: cannot import name 'gcd' from 'fractions'`: Launch ros nodes from an env with Python 3.8 instead of 3.9
- Conflicting Processes Already Running: `mrp down`, restart robot if that doesn't work.


## Third Party Code

#### Contact GraspNet

Contact graspnet is supported as a way of generating candidate grasps for the Stretch to use on various objects. We have our own fork of [Contact Graspnet](https://github.com/cpaxton/contact_graspnet/tree/cpaxton/devel) which has been modified with a ROS interface.

Follow the installation instructions as normal and start it with:
```
conda activate contact_graspnet_env
~/src/contact_graspnet$ python contact_graspnet/graspnet_ros_server.py  --local_regions --filter_grasps
```

## Code Contribution

We enforce linters for our code. The `formatting` test will not pass if your code does not conform.

To make this easy for yourself, you can either
- Add the formattings to your IDE
- Install the git [pre-commit](https://pre-commit.com/) hooks by running
    ```bash
    pip install pre-commit
    pre-commit install
    ```

To enforce this in VSCode, install [black](https://github.com/psf/black), [set your Python formatter to black](https://code.visualstudio.com/docs/python/editing#_formatting) and [set Format On Save to true](https://code.visualstudio.com/updates/v1_6#_format-on-save).

To format manually, run: `black .`

## References (temp)

- [cpaxton/home_robot](https://github.com/cpaxton/home_robot)
  - Chris' repo for controlling stretch
- [facebookresearch/fairo](https://github.com/facebookresearch/fairo)
  - Robotics platform with a bunch of different stuff
  - [polymetis](https://github.com/facebookresearch/fairo/tree/main/polymetis): Contains Torchscript controllers useful for exposing low-level control logic to the user side.
  - [Meta Robotics Platform(MRP)](https://github.com/facebookresearch/fairo/tree/main/mrp): Useful for launching & managing multiple processes within their own sandboxes (to prevent dependency conflicts).
  - The [perception](https://github.com/facebookresearch/fairo/tree/main/perception) folder contains a bunch of perception related modules
    - Polygrasp: A grasping library that uses GraspNet to generate grasps and Polymetis to execute them.
    - iphone_reader: iPhone slam module.
    - realsense_driver: A thin realsense wrapper
  - [droidlet/lowlevel/hello_robot](https://github.com/facebookresearch/fairo/tree/main/droidlet/lowlevel/hello_robot)
    - Austin's branch with the continuous navigation stuff: austinw/hello_goto_odom
    - Chris & Theo's branch with the grasping stuff: cpaxton/grasping-with-semantic-slam
    - [Nearest common ancester of all actively developing branches](https://github.com/facebookresearch/fairo/tree/c39ec9b99115596a11cb1af93a31f1045f92775e): Should migrate this snapshot into home-robot then work from there.
- [hello-robot/stretch_body](https://github.com/hello-robot/stretch_body)
  - Base API for interacting with the Stretch robot
  - Some scripts for interacting with the Stretch
- [hello-robot/stretch_firmware](https://github.com/hello-robot/stretch_firmware)
  - Arduino firmware for the Stretch
- [hello-robot/stretch_ros](https://github.com/hello-robot/stretch_ros)
  - Builds on top of stretch_body
  - ROS-related code for Stretch
- [hello-robot/stretch_web_interface](https://github.com/hello-robot/stretch_ros2)
  - Development branch for ROS2
- [hello-robot/stretch_web_interface](https://github.com/hello-robot/stretch_web_interface)
  - Web interface for teleoping Stretch
- [RoboStack/ros-noetic](https://github.com/RoboStack/ros-noetic)
  - Conda stream with ROS binaries
- [codekansas/strech-robot](https://github.com/codekansas/stretch-robot)
  - Some misc code for interacting with RealSense camera, streaming

