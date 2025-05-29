# Isaac Robot Tele-op HITL application

TL;DR A HITL app using the Isaac physics integration that loads a robot from URDF/USD and RearrangeEpisodes with basic mouse/keyboard and XR (e.g. Quest 3) UI teleoperation.

# Build Steps

### Create a Conda Env
```
# We recommend python=3.10 and cmake=3.31.6
conda create -n habitat-isaac python=3.10 cmake=3.31.6
conda activate habitat-isaac
```

## Install Isaac Sim

### Ubuntu 22.04
```
# install Isaac Sim 4.5
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

# verify install and accept EULA
python -c "import isaacsim; print(isaacsim)"

```

### (optional) Install Isaac Lab
Isaac Lab is only needed for doing asset conversion to USD format. If you don't need to rebuild USD you can skip this step for now.

```
git clone https://github.com/isaac-sim/IsaacLab
cd IsaacLab
git checkout b9a49caebc912b86f2ace0709c08d9884d167cda
./isaaclab.sh --install "none"
```

If you encounter issues, see [official instructions for installing Isaac Sim and Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html#installing-isaac-lab).

## Install habitat-sim

Install the habitat-sim from source using the following steps:

```
# start from your project root directory
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
git checkout --track origin/alex_05-13_render_instance_offsets

pip install -r requirements.txt
# adjust the number of parallel threads for your system
python setup.py build_ext --parallel 6 install --bullet

# verify install with
python -c "import habitat_sim; print(habitat_sim)"
```


## Install habitat-hitl and habitat-lab

Using the instructions below, clone and install this [alex_04-18_hab_isaac](https://github.com/facebookresearch/habitat-lab/tree/alex_04-18_hab_isaac) branch of habitat-lab.

```
#start from your project root directory
git clone --branch alex_04-18_hab_isaac https://github.com/facebookresearch/habitat-lab.git

cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-hitl
pip install -e habitat-baselines

#optionally symblink your data directory to habitat-sim/data
ln -s ../habitat-sim/data data/
```

Install additional dependencies for IK/quest integration
```
pip install spatialmath-python
pip install drake
pip install roboticstoolbox-python
```

## Get the data
This app requires HSSD scene dataset, YCB objects, and hab_murp robot models.

```
#run from habitat-lab root directory
python -m habitat_sim.utils.datasets_download --uids hssd-hab ycb
ln -s scene_datasets/hssd-hab/ data/hssd-hab
#NOTE: you will need to generate navmeshes from your scenes in advance and place them like: data/hssd-hab/navmeshes/<scene_name>.navmesh

cd data
#run from your 'habitat-lab/data' directory (reach out to request permissions if necessary)
git clone https://huggingface.co/datasets/ai-habitat/hab_murp
git checkout --track origin/new_murp_urdf
```

## Convert assets to USD

This step is necessary if you have not been provided USD files from another source or if you need to rebuild USD files after assets or scene changes. For example, any changes to the robot URDF or scene JSON contents requires rebuild.

NOTE: requires IsaacLab. See [Install Isaac Lab](#install-isaac-lab) above if you have not done so.

```
#run from habitat-lab root directory

#first check the data sources exist as expected
ls data/hab_murp/murp_tmr_franka/franka_with_hand_v2.1.urdf
ls data/hssd-hab/scenes-articulated/
ls data/objects/ycb/

#run the converter (settings can be modified in the main function of this python file)
python habitat-lab/habitat/isaac_sim/data_conversion_utils.py

#check that USD files were successfully written
ls data/usd
ls data/usd/objects/ #should contain OBJECT_<hssd_rigid_object_name>.usda and AOBJECT<hssd_articulated_object_name>.usda
ls data/usd/scenes #should contain 103997403_171030405.usda
ls data/usd/stages #should contain 103997403_171030405.usda
ls data/usd/robots #should contain franka_with_hand_v2.1.usda

ls data/usd/objects/ycb/configs #should contain OBJECT_<ycb_object_name>.usda
#NOTE: ycb conversions are currently incorrect. Instead, get YCB usd files from https://huggingface.co/datasets/ai-habitat/usd_assets/tree/main/objects/ycb/configs and copy them to data/usd/objects/ycb/configs
```

# Quickstart

With data and code installed you can run the local app with default configuration:
```
python examples/hitl/isaac_robot_teleop/isaac_robot_teleop.py
```

- app settings can be changed in `isaac_robot_teleop.yaml` and `robot_settings.yaml`
- cached robot poses are defined in `robot_poses.json`
- episodes can be generated from configuration in `hssd_ycb_hitl_rearrange_ep_gen.yaml`.
  ```
  #run the following to generate a dataset with one compatible episode
  python habitat-lab/habitat/datasets/rearrange/run_episode_generator.py --config examples/hitl/isaac_robot_teleop/hssd_ycb_hitl_rearrange_ep_gen.yaml --out data/datasets/hitl_teleop_episodes.json.gz --run
  ```


# UI Details and Features
This section details features and UI elements.

## (PC)(server) Camera Controls
The local mouse and keyboard application camera follows a cursor visualized as a yellow and white circle in the frame.
- `WASD` translate the cursor in-plane. (not active when cursor is locked to robot frame)
- `F` toggle cursor lock to robot position (initial setting from app config)
- `ZX` translate the cursor up and down
- `mouse wheel scroll` zooms the camera
- `R` or `mouse wheel press` rotates the camera with mouse movement
