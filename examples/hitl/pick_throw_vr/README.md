# Pick_throw_vr HITL application

This is an example HITL application that allows a user to interact with a scene, controlling a human avatar with mouse/keyboard or a [VR headset](#vr). A policy-driven Spot robot also interacts with the scene. This is a proof of concept and not meant for rigorous evaluation or data-collection.

**Note:** The robot policy used in this example application will randomly move around and pick up objects. This is an example application from which you can base yourself to build your own VR application.

![pick_throw_vr_screenshot](https://github.com/facebookresearch/habitat-lab/assets/6557808/39972686-ec74-49f0-ac7d-fbd0128dd713)

---
- [Pick\_throw\_vr HITL application](#pick_throw_vr-hitl-application)
  - [Installing and using HITL apps](#installing-and-using-hitl-apps)
  - [Example launch command (mouse/keyboard)](#example-launch-command-mousekeyboard)
  - [Configuration](#configuration)
- [VR](#vr)
  - [Installation](#installation)
    - [Requirements](#requirements)
      - [Server](#server)
      - [Client](#client)
  - [Launch Command](#launch-command)
  - [Unity Data Folder](#unity-data-folder)
    - [Dataset Processing](#dataset-processing)
      - [Magnum Installation for Mac](#magnum-installation-for-mac)
      - [Magnum Installation for Linux](#magnum-installation-for-linux)
      - [Usage](#usage)
      - [Importing Data Into Unity](#importing-data-into-unity)
  - [Running Locally from Unity](#running-locally-from-unity)
  - [Running Remotely from Quest Headset](#running-remotely-from-quest-headset)
    - [Connection](#connection)
    - [Server Controls](#server-controls)
  - [Troubleshooting](#troubleshooting)
    - [Deployment to the VR Headset](#deployment-to-the-vr-headset)
    - [Connection Issues](#connection-issues)
    - [Slow Performance](#slow-performance)

## Installing and using HITL apps
See [habitat-hitl/README.md](../../../habitat-hitl/README.md).

## Example launch command (mouse/keyboard)

```bash
python examples/hitl/pick_throw_vr/pick_throw_vr.py
```

## Configuration
See `config/pick_throw_vr.yaml`. You can also use the configs in `experiment` as overrides, e.g. `python examples/hitl/pick_throw_vr/pick_throw_vr.py +experiment=headless_server`.

# VR

The human avatar can optionally be controlled from VR. In this mode, the Pick_throw_vr app must still be run on a headed desktop machine, and it still offers a 3D window and some limited keyboard/mouse controls. However, it also acts as a server that communicates with our Unity-based VR client (below), which immerses the VR user in the Habitat environment.

## Installation

The system is composed of the following components:

* The **Server**, which is the Pick_throw_vr app.
* The **Client** is a Unity app that can be run from within the Unity Editor or deployed to a VR headset.

### Requirements

#### Server

| Requirements | Notes |
|---|---|
| [habitat-sim](https://github.com/facebookresearch/habitat-sim) | Use a nightly conda build, or build from source. Bullet is required. |
| Datasets | After installing `habitat-sim`, run the following command from the root `habitat-lab` directory:<br>```python -m habitat_sim.utils.datasets_download --uids hab3-episodes habitat_humanoids hab_spot_arm ycb hssd-hab --data-path data/``` |
| [hssd-models](https://huggingface.co/datasets/hssd/hssd-models) | Required for processing datasets for Unity.<br>Clone it anywhere. It will be specified later as a command-line argument. |

#### Client

| Requirements | Notes |
|---|---|
| VR Headset | We recommend Quest 3 (best) or Quest Pro with ~300 MB free storage. Make sure that [developer mode](https://developer.oculus.com/documentation/native/android/mobile-device-setup/) is activated. On Quest 2, more complex HSSD scenes may run poorly or not at all. Other VR headsets supported by Unity should also work. |
| [siro_hitl_unity_client](https://github.com/eundersander/siro_hitl_unity_client) | **Beware that a Unity license may be required by your organization.**<br>Follow [these installation instructions](https://github.com/eundersander/siro_hitl_unity_client/blob/main/README.md). |

## Launch Command

The standard keyboard-mouse launch command-line arguments can be used with those differences:

* The `habitat_hitl.networking.enable=True` config override launches the Pick_throw_vr app as a server, allowing a remote client (e.g. VR headset) to connect and control the human avatar.

```bash
python examples/hitl/pick_throw_vr/pick_throw_vr.py habitat_hitl.networking.enable=True
```

We also have an experimental headless server:
```
python examples/hitl/pick_throw_vr/pick_throw_vr.py \
+experiment=headless_server
```


## Unity Data Folder

Because the Unity application is a remote client, it must have its own copy of the 3D models used for rendering scenes.

Habitat's 3D models are not directly compatible with Unity, and must be simplified to run at an acceptable performance on the VR devices.

Therefore, a script is provided so that you can process your datasets and add them to your Unity project.

### Dataset Processing

The dataset processing script requires latest Magnum binaries, which should be installed separately from Habitat as described below.

#### Magnum Installation for Mac

Magnum is easiest to install on Mac via Homebrew.

1. Follow [magnum-bindings installation instructions](https://doc.magnum.graphics/python/building/#homebrew-formulas-for-macos).
   * In addition to `corrade`, `magnum`, and `magnum-bindings`, you may need `magnum-plugins`.
2. Test your install: `python -c "from magnum import math, meshtools, scenetools, trade"`.
   * Beware homebrew installs python packages like magnum to its own Python location, not your current conda environment's Python.
   * Depending on how Homebrew has installed Python, you may need to use `python3` instead of `python`.

#### Magnum Installation for Linux

It is recommended that you create a new `conda` environment so that it can be reused in the future without interfering with Habitat.

1. Install magnum binaries for Linux.
   * Navigate to the [Magnum CI](https://github.com/mosra/magnum-ci/actions/workflows/magnum-tools.yml).
   * Select the latest green workflow run.
   * Scroll down to "Artifacts".
   * Download your the binaries that match your system (e.g. On Linux: `magnum-tools-v2020.06-...-linux-x64`)
   * Extract to a convenient location.
2. Create a new `conda` environment:
```
conda create --name magnum python=3.10
```
1. Navigate to the `site-packages` of your new environment, e.g. `~/anaconda/envs/magnum/lib/python3.10/site-packages/`.
2. Create a `magnum.pth` file in this directory.
3. Add the absolute path to `magnum-tools`'s `python` folder to this file, e.g:
```
/home/USER/Documents/magnum-tools/linux-x64/python/
```
1. The Magnum libraries will now be included upon activating your `magnum` environment. You may validate by assessing that the following commands don't return errors:
   * `conda activate magnum`
   * `python -c "from magnum import math, meshtools, scenetools, trade"`

#### Usage

To process the dataset, navigate to your `habitat-lab` root directory. Run the following command:

```
python ./scripts/unity_dataset_processing/unity_dataset_processing.py \
--hssd-hab-root-dir data/scene_datasets/hssd-hab \
--hssd-models-root-dir path_to/hssd-models/objects \
--scenes 105515448_173104512
```

The transformed assets will be output to `data/hitl_simplified_data`.

#### Importing Data Into Unity

In Unity, open the project and use `Tools/Update Data Folder...`. From the dialog window, copy the path to the generated `data/hitl_simplified/data` in the `External Data Path` field. The resources will be imported into Unity.

## Running Locally from Unity

At this point, you should be able to run HITL remotely from Unity Editor.
To validate that everything is in place, follow the following steps:

1. Start the HITL tool by running [this command](#launch-command) from the root `habitat-lab` directory.
2. In the Unity editor, load the `GfxReplayPlayerScene`.
3. Press play.
4. After a short while, your Unity client will be connected to your local server instance. You can navigate in the Unity viewport and the movements will reflect on the server.
  * Use WASD and the mouse to navigate.
  * With some familiarity, you can use the XR Device Simulator (see on-screen help).

## Running Remotely from Quest Headset

If the application works correctly from the Unity Editor, you may now deploy it to a Quest headset.

1. Quest is an Android device. In the Unity editor, go to `Build Settings`. From the platform list, select `Android`, then `Switch Platform`.
2. Plug your Quest to your machine via USB.
   * A popup will show up in your Quest headset to authorize the computer.
3. Still in `Build Settings`, refresh the device list, then look for your specific Quest device in the dropdown menu. Select it.
4. Click `Build and Run` and ensure that this completes without error. You'll be prompted for a build save location - any location will do.
5. Put on your headset. The app may already be running. You can find the application `siro_hitl_vr_client` in your applications list.
6. The application won't connect to the server. Follow the steps below to enable the connection.

### Connection

Upon launching the server, it will start listening for incoming connections. The client will attempt to connect to the addresses listed in `Android/data/com.meta.siro_hitl_vr_client/files/config.txt`. It rotates between the addresses periodically until a connection is established.

1. Make sure that your Quest is connected to your machine via USB.
   * A popup will show up in your Quest headset to authorize the computer.
2. Navigate to `Android/data/com.meta.siro_hitl_vr_client/files/config.txt`
3. Put your server IP addresses there (which can be found using `hostname -i`, for example).
4. Save and restart `siro_hitl_vr_client`.
   * You may now disconnect the USB cable.

See [troubleshooting notes](#connection-issues) if connection fails.

### Server Controls

See on-screen help for information about controls.

* Use `T` to toggle between server-controlled mouse-keyboard and client-controlled VR controls.

## Troubleshooting

### Deployment to the VR Headset

See [the troubleshooting steps on siro_hitl_unity_client](https://github.com/eundersander/siro_hitl_unity_client/blob/main/README.md#deployment-to-the-vr-headset) if you have issues deploying the client to your VR device.

### Connection Issues

* Make sure that your server firewall allows incoming connections to the port `8888`.
* Check that the Unity client `config.txt` file lists to the correct address. See [this section](#connection).
* Make sure that both devices are on the same network.
* Corporate networks may introduce additional hurdles. To circumvent these, you can use the wifi hotspot on your phone or a separate router.
* Check that only 1 server is running on your PC.

### Slow Performance

* If your server runs on Mac, consider disabling Retina. You can use [displayplacer](https://github.com/jakehilborn/displayplacer) to achieve this.
  * If you need to mirror your screen, do it before using the tool.
  * Use `displayplacer list` to see all supported display modes.
  * Find a mode that does not use Retina and runs at 60FPS.
  * Apply the mode using `displayplacer "id:X mode:Y"`.
* If using a laptop, make sure that power is connected.
* If running on a busy network, consider using the wifi hotspot on your phone or a separate router.
* Performance is poor on Quest Pro. Consider using smaller scenes, or increasing mesh simplification aggressiveness in the dataset processing tool. See [decimate.py](../../scripts/unity_dataset_processing/decimate.py).
* The VR device transforms are currently sent to from the client to the server at frequency that is lower than the simulation. This causes grabbed objects to jitter. This is the current expected behavior.
