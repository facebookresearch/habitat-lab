# VR Human-in-the-loop (HITL) Evaluation

HITL evaluation can be driven from VR. In this mode, the HITL app acts as a server that can be remotely accessed by a client.

As it stands, the VR integration can only be used with the `fetch` app state. In this mode, the user controls a human avatar that can play fetch with a policy-driven Spot robot.

## Table of Contents
- [VR Human-in-the-loop (HITL) Evaluation](#vr-human-in-the-loop-hitl-evaluation)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Requirements](#requirements)
      - [Server](#server)
      - [Client](#client)
  - [Launch Command](#launch-command)
  - [Unity Data Folder](#unity-data-folder)
    - [Dataset Processing](#dataset-processing)
      - [Installation](#installation-1)
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


## Installation

The system is composed of the following components:

* The **Server**, which is the HITL app.
* The **Client**, which runs on Unity and can be deployed to a VR headset.

### Requirements

#### Server

| Requirements | Notes |
|---|---|
| [habitat-sim](https://github.com/facebookresearch/habitat-sim) | Use the `main` branch. Bullet is required. |
| Datasets | After installing `habitat-sim`, run the following command from the root `habitat-lab` directory:<br>```python -m habitat_sim.utils.datasets_download --uids hab3-episodes habitat_humanoids hab_spot_arm ycb hssd-hab --data-path data/``` |
| [hssd-models](https://huggingface.co/datasets/hssd/hssd-models) | Required for processing datasets for Unity.<br>Clone it anywhere. It will be specified later as a command-line argument. |

#### Client

| Requirements | Notes |
|---|---|
| Quest Headset | Tested on Quest Pro and Quest 3. Make sure that [developer mode](https://developer.oculus.com/documentation/native/android/mobile-device-setup/) is activated. |
| [siro_hitl_unity_client](https://github.com/eundersander/siro_hitl_unity_client) | Clone it anywhere. |
| Unity | **Beware that a license may be required by your organization.**<br>- Install the Unity Hub<br>- Open `siro_hitl_unity_client` in Unity Hub. This will trigger installation of the correct version of Unity Editor.<br>- During Unity installation, select `Android Build Tools`, including `OpenJDK` and `Android SDK and NDK tools`. |

## Launch Command

The standard keyboard-mouse launch commands can be used with those differences:

* The client-server application can be activated by including the `--remote-gui-mode` command-line argument.
* Only `--app-state fetch` supports remote VR evaluation.

```bash
HABITAT_SIM_LOG=warning MAGNUM_LOG=warning \
python examples/siro_sandbox/sandbox_app.py \
--remote-gui-mode \
--app-state fetch \
--disable-inverse-kinematics \
--never-end \
--gui-controlled-agent-index 1 \
--ang-speed "15" \
--cfg social_rearrange/pop_play.yaml \
--cfg-opts habitat_baselines.num_environments=1 \
habitat.task.measurements.rearrange_cooperate_reward.end_on_collide=False
```

## Unity Data Folder

Unity is intended to run on a remote client (VR headset). Therefore, it must have its own copy of the datasets.

Furthermore, HSSD is not directly compatible with Unity. Meshes must also be simplified to run at an acceptable performance on the VR devices.

Therefore, a script is provided so that you can process your datasets and add them to your Unity project.

### Dataset Processing

#### Installation

This step requires [Magnum](https://github.com/mosra/magnum) data processing tools. They are not built along with Habitat.

It is recommended that you create a new `conda` environment so that it can be reused in the future without interfering with Habitat.

1. Get the latest `magnum-tools`.
   * Navigate to the [Magnum CI](https://github.com/mosra/magnum-ci/actions/workflows/magnum-tools.yml).
   * Select the latest green workflow run.
   * Scroll down to "Artifacts".
   * Download your the binaries that match your system (e.g. On MacOS: `magnum-tools-v2020.06-1579-g68eed-2737-gc9e13-1374-g70dca-macos11-x64`)
   * Extract to a convenient location.
2. Create a new `conda` environment:
```
conda create --name magnum python=3.10
```
3. Navigate to the `site-packages` of your new environment, e.g. `~/anaconda/envs/magnum/lib/python3.10/site-packages/`.
4. Create a `magnum.pth` file in this directory.
5. Add the absolute path to `magnum-tools`'s `python` folder to this file, e.g:
```
/home/USER/Documents/magnum-tools/linux-x64/python/
```
6. The Magnum libraries will now be included upon activating your `magnum` environment. Validate the installation as such:
   * `conda activate magnum`
   * `python -c "from magnum import math, meshtools, scenetools, trade"`

#### Usage

To process the dataset, navigate to your `habitat-lab` root directory and run the following command:

```
python unity_dataset_processing.py \
--hssd-hab-root-dir data/scene_datasets/hssd-hab \
--hssd-models-root-dir path_to/hssd-models \
--scenes 105515448_173104512
```

The transformed assets will be output to `data/hitl_simplified_data`.

#### Importing Data Into Unity

In Unity, open the project and use `Tools/Update Data Folder...`. From the dialog window, copy the path to the generated `data/hitl_simplified/data` in the `External Data Path` field. The resources will be imported into Unity.

## Running Locally from Unity

To validate that everything is in place, follow the following steps:

1. Start the HITL tool by running [this command](#launch-command) from the root `habitat-lab` directory.
2. In the Unity editor, load the `GfxReplayPlayerScene`.
3. Press play.
4. After a short while, your Unity client will be connected to your local server instance. You can navigate in the Unity viewport and the movements will reflect on the server.
  * Use WASD and the mouse to navigate.
  * With some familiarity, you can use the XR Device Simulator (see on-screen help).

## Running Remotely from Quest Headset

1. Quest is an Android device. In the Unity editor, go to `Build Settings`. From the platform list, select `Android`, then `Switch Platform`.
2. Plug your Quest to your machine via USB.
   * A popup will show up in your Quest headset to authorize the computer.
3. Still in `Build Settings`, refresh the device list, then look for your specific Quest device in the dropdown menu. Select it.
4. Click `Build and Run` and ensure that this completes without error. You'll be prompted for a build save location - any location will do.
5. Put on your headset. The app may already be running. You can find the application `siro_hitl_vr_client` in your applications list.

### Connection

Upon launching the server, it will start listening for incoming connections. The client will attempt to connect to the addresses listed in `Android/data/com.meta.siro_hitl_vr_client/files/config.txt`. It rotates between the addresses periodically until a connection is established.

Put your server IP addresses there (which can be found using `hostname -i`, for example).

See [troubleshooting notes](#connection-issues) if connection fails.

### Server Controls

* Use `T` to toggle between server-controlled mouse-keyboard and client-controlled VR controls.

## Troubleshooting

### Deployment to the VR Headset

* Make sure that [developer mode](https://developer.oculus.com/documentation/native/android/mobile-device-setup/) is activated.
* When connecting your headset to your development computer via USB, a pop-up will ask you to confirm the connection within the VR headset.
  * If the pop-up doesn't show up, reconnect your USB cable.
  * You may also have to re-do this after the headset goes into sleep mode.
* Deployment occasionally fails when the application is already installed. You can delete the old build from the Quest storage menu. The following error will often show in Unity when that occurs:
```
CommandInvokationFailure: Unable to install APK to device. Please make sure the Android SDK is installed and is properly configured in the Editor. See the Console for more details.
/home/user/Unity/Hub/Editor/2022.3.7f1/Editor/Data/PlaybackEngines/AndroidPlayer/SDK/platform-tools/adb -s "4G3YA1ZF571D4D" install -r -d "/home/user/git/siro_hitl_unity_client/Build/build.apk"
```

### Connection Issues

* Make sure that your server firewall allows incoming connections.
* Check that the Unity client `config.txt` file lists to the correct address. See [this section](#connection).
* Make sure that both devices are on the same network.
* Corporate networks may introduce additional obstacles. To circumvent these, you can use the wifi hotspot on your phone or a separate router.

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
