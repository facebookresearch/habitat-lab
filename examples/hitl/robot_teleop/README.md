# Robot Tele-op HITL application

TL;DR A HITL app that loads robots from URDF and simulates them in a Habitat Simulator instance with basic UI teleoperation and hot-reloading for quick morphology iteration.

# Build Steps

## Install habitat-sim

Install the ([eundersander/isaac_vr](https://github.com/facebookresearch/habitat-sim/tree/eundersander/isaac_vr)) branch of habitat-sim from source:

```
#start from your project root directory
git clone --branch eundersander/isaac_vr https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim

#we use python 3.10+ to support momentum IK
conda create -n habitat_teleop python=3.10 cmake=3.14.0
conda activate habitat_teleop
pip install -r requirements.txt

#adjust the number of parallel threads for your system
python setup.py build_ext --parallel 6 install --bullet
```

## Install habitat-hitl and habitat-lab
NOTE: habitat-lab is currently only required for utilities. The intent is to keep dependencies as close to habitat-sim as possible.

Clone and install this [eundersander/isaacsim_viewer](https://github.com/facebookresearch/habitat-lab/tree/eundersander/isaacsim_viewer) branch of habitat-lab.

```
#start from your project root directory
git clone --branch eundersander/isaac_vr https://github.com/facebookresearch/habitat-lab.git

cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-hitl

#optionally symblink your data directory to habitat-sim/data
ln -s ../habitat-sim/data data/
```

## (optional) Install pymomentum for Inverse Kinematics
https://github.com/facebookincubator/momentum

*NOTE: waiting on IK feature to be added to the framework. ETA early March.*

> Momentum provides foundational algorithms for human kinematic motion and numerical optimization solvers to apply human motion in various applications.

Installing pymomentum from conda-forge :
```
mamba install -c conda-forge pymomentum=0.1.29
#need to downgrade numpy for Habitat compatibility.
mamba install numpy=1.26.4
```

## Get the data
This app requires both HSSD scene dataset and murp_hab robot models.

```
#run from your 'habitat-lab/data' directory
git clone https://huggingface.co/datasets/hssd/hssd-hab
git clone https://huggingface.co/datasets/ai-habitat/hab_murp
```

# Quickstart

With data and code installed you can run the local app with default configuration:
```
python examples/hitl/robot_teleop/robot_teleop.py
```

- default robot is TMR base + Franka with no end effectors
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

# TODOs
Items not yet implemented but intended to be added
- Additional murp robot platform configs
- IK via pymomentum
- meta hand integration and piecewise pose setting (for grasps)
- robot sensor config and visualization
- fixed base via constraint would be more dynamically stable than kinematic fixed base. Consider adding this.
