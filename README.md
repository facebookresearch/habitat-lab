# Home Robot

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/facebookresearch/home-robot/tree/main.svg?style=shield&circle-token=282f21120e0b390d466913ef0c0a92f0048d52a3)](https://dl.circleci.com/status-badge/redirect/gh/facebookresearch/home-robot/tree/main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Mostly Hello Stretch infrastructure

## Installation

1. Prepare a conda environment (ex: `conda create -n home_robot python=3.8`)
1. Install Mamba (needed for MRP): `conda install -c conda-forge mamba`
1. Install repo `pip install -e .`

### Additional instructions for setting up on hardware

1. Setup the Stretch robot following official instructions [here](https://github.com/hello-robot/stretch_install)
1. Install stretch_ros following official instructions [here](https://github.com/hello-robot/stretch_ros/blob/dev/noetic/install_noetic.md)
1. Install Hector SLAM: `sudo apt install ros-noetic-hector-*`

#### Conflicting Processes Already Running

- Restart stretch
- See running processes using:

```sh
mrp info
```

## Usage

### Launching the hardware stack:
```sh
cd src/home_robot
mrp up hw_stack
```

This launches:
- Stretch ROS driver
- Hector SLAM
- State estimation node
- Continuous controller node

### Launching a minimal kinematic simulation (no camera yet)
```sh
cd src/home_robot
mrp up sim_stack
```

This launches:
- Fake stretch node (A kinematic simulation that publishes 100% accurate odometry and slam information and accepts velocity control inputs)
- State estimation node
- Continuous controller node

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

### Stopping all processes
```sh
mrp down
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

