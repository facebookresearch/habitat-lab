# IsaacSim Viewer

## Installation

### Isaac Sim

See [habitat-lab/habitat/isaac_sim/README.md](../../../habitat-lab/habitat/isaac_sim/README.md).

### Habitat-sim

1. Use the [eundersander/isaac_vr](https://github.com/facebookresearch/habitat-sim/tree/eundersander/isaac_vr) branch. See [BUILD_FROM_SOURCE.md](https://github.com/facebookresearch/habitat-sim/blob/eundersander/isaac_vr/BUILD_FROM_SOURCE.md).
2. Verify your install with `python -c "import habitat_sim; print(habitat_sim)"`. Verify you imported the habitat-sim you just cloned and built.

### Data

1. Download [hssd-hab](https://huggingface.co/datasets/hssd/hssd-hab) to `data/scene_datasets/hssd-hab`. I used revision `9ec06b0d25cba85b6ee422b02e52e14ec4acafc7`. (You alternately also use the Habitat-sim dataset downloader script.)
2. Download [ycb](https://huggingface.co/datasets/ai-habitat/ycb) to `data/objects/ycb`. I used revision `29be64fdd95b4881f244152ad653058e0a48c28f`.
3. Download [hab_spot_arm](https://huggingface.co/datasets/ai-habitat/hab_spot_arm) to `data/robots/hab_spot_arm`. I used revision `0267f9c3eaea0081a788716e87ee1088afa3cb50`.
4. Download [isaac_vr_extra_server_data.zip](https://drive.google.com/file/d/1AF5zpL6Uo_8nJ6H-Twn8vu34Ec_a06ip/view?usp=drive_link
 ) and extract at `data/`. This should create `data/from_gum` and `data/usd` folders.

### VR Client
Optional. Instructions here coming soon. 

## To run

```
# Run from habitat-lab repo root.

# Run desktop GUI app (no VR):
python examples/hitl/isaacsim_viewer/isaacsim_viewer.py 
```

## Re-exporting USD/USDA data from source data

Beware this pipeline is not polished! Note `.usda` is simply the ascii, human-readable version of USD.

The source data is from above installation steps:
* `data/scene_datasets/hssd-hab/scenes-uncluttered/102344193.scene_instance.json` 
* `data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf`
* `data/objects/ycb`

See [isaaclab_convert_test.py](../../../isaaclab_convert_test.py) and [clean_urdf_xml.py](../../../clean_urdf_xml.py) and instructions below.

Robot:
1. Run `python clean_urdf_xml.py /path/to/my_robot.urdf ./data/usd/robots/my_robot.usda --remove-visual`.
2. Edit `isaaclab_convert_test.py convert_urdf_test` to use the correct filepaths. Edit `isaaclab_convert_test.py __main__` to run convert_urdf_test(). Run the script.
3. Verify output USDA file in Isaac Sim GUI or elsewhere.

Habitat scene instance:
1. Ensure your source scene is working in Habitat.
2. Edit `isaaclab_convert_test.py __main__` to call `convert_hab_scene`.
    * Beware this will produce not only a new USDA for your scene but also new USDA files for the objects in the scene.
3. **Beware that articulated objects and the stage are currently not handled by `convert_hab_scene`**. For this app, it so happens that the only physics I needed was for rigid objects and YCB objects added later. Your mileage may vary. I also manually re-added the stage visuals to my scene USDA file as a hack (see top of `usd/scenes/102344193_with_stage.usda`).

YCB or other objects:
1. Ensure your source objects are working in Habitat.
2. Edit `isaaclab_convert_test.py __main__` to call `convert_objects_folder_to_usd`.


