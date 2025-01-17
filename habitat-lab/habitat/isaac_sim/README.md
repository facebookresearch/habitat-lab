# Habitat Isaac Sim Integration

This integration offers Isaac Sim as an additional physics backend for use in Habitat-lab, alongside Habitat-sim. Habitat-sim is used to render the Isaac simulation.

This branch is a work in progress without much functionality so far. See also branch [eundersander/isaac_vr](https://github.com/facebookresearch/habitat-lab/tree/eundersander/isaac_vr/examples/hitl/isaacsim_viewer) which is messy but includes a full integration including usage in a HITL VR app.

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
    * Beware you will likely need a web browser installed in order to log into your Nvidia account during the setup process.


### Isaac Lab
Isaac Lab is only needed for doing asset conversion to USD format. If you've already been provided Isaac Sim USD files, you can skip this.

```
# Clone and install Isaac Lab. We recommend cloning it outside your Habitat-lab folder.
git clone https://github.com/isaac-sim/IsaacLab
cd IsaacLab
./isaaclab.sh --install "none"
```

If you encounter issues, see [official instructions for installing Isaac Sim and Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html#installing-isaac-lab).

## Converting Physics Assets to USD
Todo

## Using Isaac Sim
Coming soon!
