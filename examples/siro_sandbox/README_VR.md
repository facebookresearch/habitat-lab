# VR Human-in-the-loop (HITL) Evaluation

**Note: VR HITL evaluation currently only works from this branch.**

HITL evaluation can be driven from VR. In this mode, the sandbox app acts as a server that can be remotely accessed by a client.

As it stands, the VR integration can only be used with the `fetch` app state. In this mode, the user controls a human avatar that can play fetch with a policy-driven Spot robot.

## Table of Contents
- [VR Human-in-the-loop (HITL) Evaluation](#vr-human-in-the-loop-hitl-evaluation)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Requirements](#requirements)
      - [Server](#server)
      - [Client](#client)
  - [Launch Commands](#launch-commands)
      - [Example](#example)
  - [Unity Data Folder](#unity-data-folder)
    - [Dataset Processing](#dataset-processing)
      - [Installation](#installation-1)
      - [Usage](#usage)
      - [Importing Data Into Unity](#importing-data-into-unity)
  - [Connection](#connection)
  - [Server Controls](#server-controls)
  - [Troubleshooting](#troubleshooting)
    - [Deployment to the VR Headset](#deployment-to-the-vr-headset)
    - [Connection Issues](#connection-issues)
    - [Slow Performance](#slow-performance)
  - [Other information](#other-information)
    - [Serving HTTPS content from your local machine](#serving-https-content-from-your-local-machine)


## Installation

The system is composed of the following components:

* The **Server**, which is the sandbox app.
* The **Client**, which runs on Unity and is deployed on a VR headset.

### Requirements

#### Server

| Requirements | Notes |
|---|---|
| `habitat-sim` | Use the `main` branch. Bullet is required. |
| `habitat-lab` | This specific `habitat-lab` version must be installed. See [instructions](../../README.md#installation). The `main` branch is currently incompatible. |
| Datasets | `python -m habitat_sim.utils.datasets_download --uids habitat_humanoids hab_spot_arm ycb hssd-hab --data-path data/` |
| [hssd-models](https://huggingface.co/datasets/hssd/hssd-models) | Required for dataset processing. |
| [habitat_humanoids](https://huggingface.co/datasets/ai-habitat/habitat_humanoids) | Use the `main` branch. |
| [Server files](https://huggingface.co/datasets/ai-habitat/siro_fetch_extra_data) | Copy the files as-is to `data/`, following the directory structure. |
| `websockets` | `pip install websockets` |

#### Client

| Requirements | Notes |
|---|---|
| Unity 2022.3.7f1 | Beware that a license may be required by your organization. |
| Quest Headset | Tested on Quest Pro and Quest 3. Make sure that [developer mode](https://developer.oculus.com/documentation/native/android/mobile-device-setup/) is activated. |
| [siro_hitl_unity_client](https://github.com/eundersander/siro_hitl_unity_client) | Use the `main` branch. See [instructions](https://github.com/eundersander/siro_hitl_unity_client/blob/main/README.md). |
| Unity data folder | See [this section](#dataset-processing) for instructions for generating the data folder. |

## Launch Commands

The standard keyboard-mouse launch commands can be used with those differences:

* The client-server application can be activated by including the `--remote-gui-mode` command-line argument.
* Only `--app-state fetch` is supported.
* Use the `examples/siro_sandbox/configs/fetch.yaml` configuration.

#### Example

```bash
HABITAT_SIM_LOG=warning MAGNUM_LOG=warning \
python examples/siro_sandbox/sandbox_app.py \
--app-state fetch \
--disable-inverse-kinematics \
--never-end \
--gui-controlled-agent-index 1 \
--ang-speed "15" \
--cfg examples/siro_sandbox/configs/fetch.yaml \
--cfg-opts ~habitat.task.measurements.agent_blame_measure
```

## Unity Data Folder

HSSD is not directly compatible with Unity. Furthermore, meshes must be simplified to run at an acceptable performance on the VR devices.

Therefore, a script is provided so that you can process your HSSD scenes. Usage details are provided below.

Note that the provided episodes support the following HSSD scenes:
* 102344193
* 102344280
* 102817200
* 103997424_171030444
* 103997541_171030615

### Dataset Processing

#### Installation

This step requires [Magnum](https://github.com/mosra/magnum) data processing tools. They are not built along with Habitat.

It is recommended that you create a new `conda` environment so that it can be reused in the future without interfering with Habitat.

1. Get the latest `magnum-tools` from [Magnum CI](https://github.com/mosra/magnum-ci/actions/workflows/magnum-tools.yml). Extract to a convenient location.
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

Example command:

```
python get_scene_object_glbs.py \
--hssd-hab-root-dir path_to/hssd-hab/ \
--hssd-models-root-dir path_to/hssd-models/ \
--scenes 102344193 102344280 102817200 103997424_171030444 103997541_171030615
```

#### Importing Data Into Unity

In Unity, use `Tools/Update Data Folder...`. From the dialog window, copy the path to the generated `data/hitl_simplified/data` in the `External Data Path` field. The resources will be imported into Unity.

After this step, you may deploy the application to your headset. See [instructions](https://github.com/eundersander/siro_hitl_unity_client/blob/main/README.md).

## Connection

Upon launching the server, it will start listening for incoming connections. The client will attempt to connect to the addresses listed in `Android/data/com.meta.siro_hitl_vr_client/files/config.txt`. It rotates between the addresses periodically until a connection is established.

See [troubleshooting notes](#connection-issues) is connection fails.

## Server Controls

* Use the `1-5` keys to change scenes. Scenes are currently hardcoded here.
* Use `T` to toggle between server-controlled mouse-keyboard and client-controlled VR controls.
* If the robot gets stuck, hold `O` to temporarily use an oracle policy.

## Troubleshooting

### Deployment to the VR Headset

* Make sure that [developer mode](https://developer.oculus.com/documentation/native/android/mobile-device-setup/) is activated.
* When connecting your headset to your development computer via USB, a pop-up will ask you to confirm the connection within the VR headset.
  * If the pop-up doesn't show up, reconnect your USB cable.
  * You may also have to re-do this after the headset goes into sleep mode.
* Deployment occasionally fails when the application is already installed.

### Connection Issues

* Make sure that your server firewall allows incoming connections.
* Check that the Unity client `config.txt` file lists to the correct address. See [this section](#connection).
* Make sure that both devices are on the same network.
* Corporate networks may introduce additional obstacles. To circumvent these, you can use the wifi hotspot on your phone or a separate router.

### Slow Performance

**Note: As it stands, pose tracking is updated at a low frequency.**

* If your server runs on Mac, consider disabling Retina. You can use [displayplacer](https://github.com/jakehilborn/displayplacer) to achieve this.
  * If you need to mirror your screen, do it before using the tool.
  * Use `displayplacer list` to see all supported display modes.
  * Find a mode that does not use Retina and runs at 60FPS.
  * Apply the mode using `displayplacer "id:X mode:Y"`.
* If using a laptop, make sure that power is connected.
* If running on a busy network, consider using the wifi hotspot on your phone or a separate router.
* Performance is poor on Quest Pro. Consider using smaller scenes, or increasing mesh simplification aggressiveness in the dataset processing tool. See [decimate.py](../../scripts/unity_dataset_processing/decimate.py).

## Other information

### Serving HTTPS content from your local machine

Note: This is not required for using the native Unity app.

Note: Quest's browser (and probably all browsers) will only load WebXR experiences if served as a secure (HTTPS) web page. Luckily for us, this includes self-signed HTTPS.

1. install openssl on your OS if necessary.

2. generate private.key
```
openssl genpkey -algorithm RSA -out private.key -pkeyopt rsa_keygen_bits:2048
```

3. generate temp.csr
```
openssl req -new -key private.key -out temp.csr
```

4. generate self_signed.pem
* There are several prompts for info like country and organization. You can press return to use defaults for all of these.
```
openssl x509 -req -days 365 -in temp.csr -signkey private.key -out self_signed.pem -outform PEM
```

5. Launch your HTTPS server in a folder that you want to serve
```
openssl s_server -accept 8443 -cert self_signed.pem -key private.key -WWW
```

6. Test on the same machine
* Browse to https://0.0.0.0:8443/example.html
* In Chrome and other browsers, you will have to navigate past a "Your connection is not private" warning but it will otherwise work.

7. Test on another machine on the same local network (same router)
* First, make sure that your firewall is configured to allow the the connection to port 8443.
* Find your *private* IP address, e.g. `hostname -I`
* From the other machine, browse to https://IP:8443/example.html
