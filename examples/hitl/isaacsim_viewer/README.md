# IsaacSim Viewer
![spot_and_hand_banner](https://github.com/user-attachments/assets/6aaf5ab1-2568-4b22-bd3c-54ab521b0ee7)

This is an updated version of the IsaacSim Viewer with an alternate Mochi backend.

The app is an interactive VR physics sandbox. It features an HSSD scene, YCB objects, and a policy-driven Spot. If you connect a VR client, you can also interact with the scene via a Metahand. The app is also a good starting point for testing/hacking/benchmarking.

## Requirements

* A headed Linux desktop. Ubuntu is recommended for compatibility with Isaac Sim.
* for VR (optional), you need a Quest 3, and the Linux server can be run headless.

## Installation

### Mochi

See [habitat-lab/habitat/mochi/README.md](../../../habitat-lab/habitat/mochi/README.md).

### Isaac Sim

See [habitat-lab/habitat/isaac_sim/README.md](../../../habitat-lab/habitat/isaac_sim/README.md).

### Habitat-sim

1. Use the [eundersander/mochi_vr](https://github.com/facebookresearch/habitat-sim/tree/eundersander/mochi_vr) branch. See [BUILD_FROM_SOURCE.md](https://github.com/facebookresearch/habitat-sim/blob/eundersander/mochi_vr/BUILD_FROM_SOURCE.md).
2. Verify your install with `python -c "import habitat_sim; print(habitat_sim)"`. Verify you imported the habitat-sim you just cloned and built.

### Data

1. Download [ycb](https://huggingface.co/datasets/ai-habitat/ycb) to `data/objects/ycb`. I used revision `29be64fdd95b4881f244152ad653058e0a48c28f`.
2. Download [mochi_vr_data](https://huggingface.co/datasets/undersan/mochi_vr_data) to `data/mochi_vr_data`.

### Setting IP address on the Quest client
```
adb shell "sed -i 's/[0-9]\{1,3\}\(\.[0-9]\{1,3\}\)\{3\}/192.168.1.17/' \
  /sdcard/Android/data/com.meta.siro_hitl_vr_client4/files/config.txt"
```

## To run the desktop app

```
# Run from habitat-lab repo root.

# Run desktop GUI app (no VR):
python examples/hitl/isaacsim_viewer/isaacsim_viewer.py

# Run desktop GUI app as a (headed) server for VR
python examples/hitl/isaacsim_viewer/isaacsim_viewer.py habitat_hitl.networking.enable=True

# Run app in headless mode as a server for VR. This will improve server SPS and app responsiveness.
python examples/hitl/isaacsim_viewer/isaacsim_viewer.py habitat_hitl.networking.enable=True habitat_hitl.experimental.headless.do_headless=true habitat_hitl.window=null
```

## Using the desktop GUI app (without VR)
1. On startup, beware the viewport initializes with a weird camera view outside the house. Start by zooming out using the mouse scroll wheel.
2. The help text includes `server SPS` (steps/second). This should be at least ~15, otherwise the app will be unusable.
3. Use WASD to move the camera lookat. Hold R + mousemove to rotate the camera. Find the Spot robot in the scene.
4. Press any number key from 1-8. After a brief pause, Spot will move to an object on the table and attempt to grasp it.


## VR Client
Tested on Quest 3

### VR Installation
1. Download [siro_hitl_unity_client4_quest.apk](https://drive.google.com/file/d/1T5ClMPu51fKrozOJsSudgLzInUNsHAFT/view?usp=drive_link) to your Mac or other desktop machine.
3. Install `adb` (Android Debug Bridge) on your desktop. Search the internet for suggestions for your OS.
2. Ensure Quest 3 developer mode is enabled. Connect your Quest to your desktop via USB cable.
4. Ensure `adb devices` lists the device. Install the app via `adb install ./siro_hitl_unity_client4_quest.apk`.
5. Put on the headset. From the Quest dashboard, find `Library > Unknown Sources > siro_hitl_unity_client4` and run it.
6. You'll see a black loading screen with a blue Habitat icon. Then you'll see an empty scene with a gray ground plane.
7. Put down your controllers and wave your hands in front of the headset. Your hands will be detected and visualized in gray.
8. Next, take off the headset. You need to edit  `/Android/data/com.meta.siro_hitl_vr_client4/config.txt` on your Quest to specify the IP address of your server (see below). This file is created the first time you run the VR client. One way to access this text file is by connecting the Quest to an Android phone. Click through appropriate USB permissions, then you can browse the Quest filesystem on the phone and open/edit `config.txt`.

### To run the VR app

1. Ensure the desktop app is running on your desktop or elsewhere as a server (see instructions above).
2. Ensure you've set the server IP address in `config.txt` as described in VR Installation above. We recommend putting the Quest and desktop machine on the same wired/wifi network, in which case you'll use your desktop machine's *internal* IP address, e.g. `ipconfig getifaddr en0` on Mac.
3. Put on the headset. From the Quest dashboard, find `Library > Unknown Sources > siro_hitl_unity_client4` and run it. It may also show up on your home bar as a recent app.
4. The VR client should automatically connect. You'll be spawned in the house scene, above the table with the objects. Your right hand will be tracked by Quest and you can teleop the Metahand.
5. If you're running the server as headed (with desktop GUI), you can take off your headset, and you'll see the Metahand in the desktop GUI simulation plus some extra debug visualization related to the VR client. You can use the 1-8 keys as before to make the Spot attempt to grasp objects.
6. With a bit of finesse, you can even place the headset near your desk, giving it a view of your hands, and then you can teleop the Metahand while watching the desktop GUI (instead of being in VR). Beware the Quest may go to sleep; sometimes you can prevent this by blocking the light sensor between the lenses with black tape or similar.


### Using the VR app
* Use the Metahand to pick up objects.
* If your desktop machine is nearby and you're running a *headed* server, use the 1-8 keys to make Spot attempt to pick objects. Alternately, see `isaacsim_viewer.py` L1182 for commented-out reference code for having Spot automatically pick a new object every 25 seconds. (To be clear, there is no interface within VR for controlling Spot.)


### Teleoperating a real Metahand
* Step 6 of "To run the VR app" above is a possible starting point for teleoperating a real Metahand. As a quick prototype, edit `isaacsim_viewer.py udpate_metahand_bones_from_art_hand` to issue commands to HW instead of calling into the sim API (`_metahand_wrapper._target_joint_positions`).
* As additional polish, you should create a new HITL app outside the `habitat-lab` repo, by simply copy-pasting the `examples/hitl/isaacsim_viewer` folder. Then, strip away the unneeded server functionality around loading and simulating the HSSD scene, Metahand, and Spot.
* As additional polish, you can update the Unity VR application to enable pass-through (instead of rendering a simulated scene). In this way, you can wear the headset while teleoperating the real Metahand, which should offer superior hand-tracking compared to setting the headset near your desk.


## Re-exporting USD/USDA data from source data

**Updated instructions [here](../../../habitat-lab/habitat/isaac_sim/README.md#converting-habitat-data-to-isaac-usd-format). Below instructions are outdated.**

Beware this pipeline is not polished! Note `.usda` is simply the ascii, human-readable version of USD. Beware, the VR client relies on on-device Unity-converted versions of all visual assets (GLB models used for rendering). If you're adding new visual assets to this app, you will also have to update the Unity build with these new assets (see "Rebuilding the VR client").

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


## Rebuilding the Unity VR client

See [siro_hitl_unity_client/tree/eundersander/isaac_vr2](https://github.com/eundersander/siro_hitl_unity_client/tree/eundersander/isaac_vr2).
