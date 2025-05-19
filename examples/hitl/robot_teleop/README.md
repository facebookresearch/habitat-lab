# Robot Tele-op HITL application

TL;DR A HITL app that loads robots from URDF and simulates them in a Habitat Simulator instance with basic UI teleoperation and hot-reloading for quick morphology iteration.

# Build Steps

## Install Pymomentum
```
conda install -c conda-forge momentum-cpp
conda install -c conda-forge pymomentum # Windows is not supported yet
conda install -c conda-forge momentum
```

## Install habitat-sim

Install the habitat-sim from source using the following steps:

```
#start from your project root directory
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim

#we use python 3.10+ to support momentum IK
conda create -n habitat_teleop python=3.10
conda activate habitat_teleop
pip install cmake==3.31.6
pip install -r requirements.txt

#adjust the number of parallel threads for your system
python setup.py build_ext --parallel 6 install --bullet
```

If you encounter an error saying
```
error: Could not find suitable distribution for Requirement.parse('habitat-sim==0.3.3')
```

please re-run the command
```
python setup.py build_ext --parallel 6 install --bullet
```

you should see something like
```
Installing collected packages: magnum
Successfully installed magnum-0.0.0
```

## Install habitat-hitl and habitat-lab
NOTE: habitat-lab is currently only required for utilities. The intent is to keep dependencies as close to habitat-sim as possible.

Using the instructions below, clone and install this [asjad/sim_teleop](https://github.com/facebookresearch/habitat-lab/tree/asjad/sim_teleop) branch of habitat-lab.

```
#start from your project root directory
git clone --branch asjad/sim_teleop https://github.com/facebookresearch/habitat-lab.git

cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-hitl
pip install -e habitat-baselines

#optionally symblink your data directory to habitat-sim/data
ln -s ../habitat-sim/data data/
```

## Additional Installations To Support Pymomentum

```
conda install -c conda-forge torchvision
conda install -c conda-forge libstdcxx-ng --update-deps
conda install numpy=1.26.4
pip install spatialmath-python
```

## Get the data
This app requires both HSSD scene dataset and murp_hab robot models.

```
#run from habitat-lab root directory
python -m habitat_sim.utils.datasets_download --uids hssd-hab ycb
ln -s scene_datasets/hssd-hab/ data/hssd-hab
#run from your 'habitat-lab/data' directory
git clone https://huggingface.co/datasets/ai-habitat/hab_murp
```

# Quickstart

With data and code installed you can run the local app with default configuration:
```
python examples/hitl/robot_teleop/robot_teleop.py
```

- default robot is TMR base + Franka.
- default scene dataset is hssd-hab-articulated
- default scene is "103997403_171030405"
- above settings can be changed in `robot_teleop.yaml` and `robot_settings.yaml`
- default pose is defined in `robot_poses.json`

# UI Details and Features
This section is a living record of features and UI elements as they are developed.

## Camera Controls
The camera follows a cursor visualized as a yellow and white circle in the frame.
- `WASD` translate the cursor in-plane
- `ZX` translate the cursor elevation
- mouse wheel scroll zooms the camera
- `R` rotates the camera with mouse movement
- `SPACE` toggles cursor lock to the robot
  - In robot cursor lock mode, elevation and rotation can be controlled

## Robot UI

### Hot reloading the robot configuration
The robot's configuration settings are defined in a re-loadable config: `robot_settings.yaml`
- `T` removes the current robot and reloads the settings file including any changes you have made while the app was running

Use this feature to quickly iterate on URDF parameters, navmesh parameters, motor settings, etc...

### Teleop with mouse
The robot joints can be manipulated with the mouse.
- hovering the mouse over any object shows its details in the top-left corner text.
- hovering the mouse over a robot link shows the axis of joint rotation and local axis frame of the link
- `LEFT-click` on a robot link and drag to change the motor target. A number line bar is displayed over the link with a green dash indicating the joint state and end dashes indicating the joint limits.

### Caching and loading poses
The current robot pose can be saved to a json file and hot-reloaded from that file.
- `O` key "opens" the cache file and sets the requested pose
- `P` key serializes the current robot pose into the cache file
- defaults:
  - `"robot_poses.json"` is the default cache file
  - `"default"` is the default pose name
- The caching system keys on the robot's name `robot.ao.handle` and then a key string
  - key string `"initial"` is loaded and set when the application starts

### Controlling the robot base

The application has two options for placing the robot base.

- `RIGHT click` on any surface to place the robot on the navmesh as close to the point as possible. Use this to quickly test if a robot fits somewhere without fuss.

- `M` places the robot at a random navigable point in the scene. Use this to escape islands.
- `IK` move the robot forward and backward respecting the navmesh.
- `JL` rotate the robot
- `N` toggles a visualization of the navmesh
  - Note that a circle around the robot base shows its configured navmesh radius. This can be set in the robot_settings.yaml file.

# VR Integration

## Running Habitat
To perform teleoperation using Quest3, first run the following command on your system running habitat.

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python examples/hitl/robot_teleop/robot_teleop.py --config-name robot_teleop_vr.yaml
```

## Habitat Quest Viewer
Please run the latest build of Quest-Habitat Unity build on your Quest headset after launching habitat. The Headset and Laptop need to be connected to the same network without VPN. 

## User Interface

### Robot Teleoperation
This section describes teleoperating the robot. All commands are associated with the controllers.
- `'X'` on the quest to align the user's current head position and orientation to the robot's base frame. This is necessary before starting teleoperation as the end-effector poses are captured with respect to this frame.
- `'Y'` resets the robot to their `initial joint configuration`. This is particularly useful if the robot is accidentally driven into a singularity in trying to each beyond the robot's workspace OR the user moved in the physical world without using `'X'`
- `Left Controller Joy Stick` moves the robot's base.
- `Right Controller Joy Stick` moves the camera's orientation with respect to the robot's base. Enabling the user to not have to reorient themselves in the physical space.
- `Quest Grip Button` press and hold to teleoperate the robot arm for the hand on which the button is pressed. The axis marker represents the desired end-effector position for each arm.
- `Quest Trigger Button` press for closing the allegro hand gripper. Interpolation is implemented, i.e. the amount of the button pressed controls the amount the allegro gripper's closing.

### Changing Scenes and adding YCB Objects
- `0` on keyboard to change scenes.
- `Y` *( if `use_cursor` is set to `True` in `robot_teleop_vr.yaml` )* : Object is loaded at the position the cursor is pointing at. The user can select which YCB object to add using the terminal. List of possible options that may be added can be modified in the `robot_teleop_vr.yaml`
- `Y` *( if `use_cursor` is set to `False` in `robot_teleop_vr.yaml` )* : Objects are loaded in the scene at the defined positions inside yaml.
