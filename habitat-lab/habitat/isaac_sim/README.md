

# Habitat Isaac Sim Integration

## Isaac Sim Installation

### Ubuntu 22.04
```
# install Isaac Sim
pip install isaacsim==4.2.0.2 isaacsim-extscache-physics==4.2.0.2 isaacsim-extscache-kit==4.2.0.2 isaacsim-extscache-kit-sdk==4.2.0.2 --extra-index-url https://pypi.nvidia.com

# verify install and accept EULA
python -c "import isaacsim; print(isaacsim)"
```

### Ubuntu 20.04

1. Ensure your system is headed. If using ssh to a remote machine, use display forwarding.
2. Follow official [Workstation installation steps](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html).


### Isaac Lab
Isaac Lab is only needed for doing asset conversion to USD format. If you've already been provided Isaac Sim USD files, you can skip this.

```
git clone https://github.com/isaac-sim/IsaacLab
cd IsaacLab
git checkout b9a49caebc912b86f2ace0709c08d9884d167cda
./isaaclab.sh --install "none"
```

If you encounter issues, see [official instructions for installing Isaac Sim and Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html#installing-isaac-lab).

## Converting Habitat data to Isaac USD format

Beware this pipeline is not polished! Note `.usda` is simply the ascii, human-readable version of USD.

See [data_conversion_utils.py](./data_conversion_utils.py), [clean_urdf_xml.py](./clean_urdf_xml.py), and instructions below.

Robot:
1. Run `python habitat-lab/habitat/isaac_sim/clean_urdf_xml.py /path/to/my_robot.urdf ./data/usd/robots/my_robot.usda --remove-visual`.
2. Edit `data_conversion_utils.py convert_urdf_test` to use the correct filepaths. Edit `data_conversion_utils.py __main__` to run convert_urdf_test(). Run the script as `python habitat-lab/habitat/isaac_sim/data_conversion_utils.py`.
3. Verify output USDA file in Isaac Sim GUI or elsewhere.

Habitat scene instance:
1. Ensure your source scene is working in Habitat.
2. Call `data_conversion_utils.py convert_hab_scene`. See bottom of that file for example code.
3. **Beware that articulated objects are currently not handled by `convert_hab_scene`**. They will be missing from your converted scene!

YCB or other objects:
1. Ensure your source objects are working in Habitat.
2. Call `data_conversion_utils.py convert_objects_folder_to_usd`.
