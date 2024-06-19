# Human-in-the-loop (HITL) Framework

The HITL framework brings real human users into Habitat virtual environments. Use it to build interactive desktop and VR applications that enable interaction between users and simulated robots and other virtual agents. Deploy these apps to users to collect interaction data for agent evaluation and training.

The HITL framework consists of the `habitat-hitl` Python library, example [desktop applications](../examples/hitl/), and our Unity-based [VR client](../examples/hitl/pick_throw_vr/README.md#vr).

The framework builds on `habitat-lab` and `habitat-baselines`. It provides wrappers to load, simulate, and render Habitat environments in real time, including virtual agents and policy inference. To enable users to interact with these agents, the framework provides graphical user interfaces (GUIs), including a window with a 3D viewport, camera-control and avatar-control helpers, and VR integration.

<p align="center">
  <img src="../res/img/hitl_tool.gif" height=400>
</p>

- [Human-in-the-loop (HITL) Framework](#human-in-the-loop-hitl-framework)
  - [System Requirements](#system-requirements)
  - [Installation](#installation)
  - [Example HITL applications](#example-hitl-applications)
  - [VR HITL applications](#vr-hitl-applications)
  - [Configuration](#configuration)
  - [Minimal HITL application](#minimal-hitl-application)

## System Requirements
* **Operating System:** macOS or Linux. We've tested Fedora but other Linux flavors should also work.
* **CPU/GPU:** Apple M1 Pro/Max, Intel Core i7 + dedicated GPU, or equivalent.
* **Display:** laptop or desktop with an attached monitor. We haven't tested remote desktop or other headless options.
* **Storage:** ~20 GB including Habitat dependencies and runtime data like HSSD. This excludes common third-party and OS libraries. An Ubuntu machine needs about 60 GB of storage for the OS and everything else required to run HITL apps.
* **VR:** HITL VR uses a client/server model which requires both a desktop system (above) and a VR headset. See [`pick_throw_vr/README.md`](../examples/hitl/pick_throw_vr/README.md) for details.

Example HITL apps are configured to run at 30 steps per second (SPS). If your system doesn't meet the above specs, you'll have a lower SPS and a degraded user experience.

## Installation
1. Clone Habitat-lab [main branch](https://github.com/facebookresearch/habitat-lab).
2. Install Habitat-lab using [instructions](https://github.com/facebookresearch/habitat-lab#installation).
    * The HITL framework depends on the `habitat-lab` and `habitat-baselines` packages. While these packages are in the same repository as the HITL framework, it's not strictly necessary for the HITL framework to import the packages that live in th- [Human-in-the-loop (HITL) Framework](#human-in-the-loop-hitl-framework)
3. Install Habitat-sim [main branch](https://github.com/facebookresearch/habitat-sim).
    * [Build from source](https://github.com/facebookresearch/habitat-sim/blob/main/BUILD_FROM_SOURCE.md), or install the [conda packages](https://github.com/facebookresearch/habitat-sim#recommended-conda-packages).
        * Be sure to include Bullet physics, e.g. `python setup.py install --bullet`.
4. Install the `habitat-hitl` package.
    * From the `habitat-lab` root directory, run `pip install -e habitat-hitl`.
5. Download required assets for our example HITL applications (Note that the dataset downloader should be run from habitat-lab/.):
    ```bash
    python -m habitat_sim.utils.datasets_download \
    --uids hab3-episodes habitat_humanoids hab_spot_arm ycb hssd-hab \
    --data-path data/
    ```

## Data directory and running location

HITL apps (and Habitat libraries in general) expect a `data/` directory in the running location (aka current working directory). Notice the `--data-path` argument in our installation steps above. Here are two options to consider:

1. Download data to `habitat-lab/data` as shown in our installation steps above. This is the default location for many Habitat tutorials and utilities. Run your HITL app from this location, e.g. `habitat-lab/$ python /path/to/my_hitl_app/my_hitl_app.py`.
2. Download (or symlink) data to your HITL app's root directory, e.g. `/path/to/my_hitl_app/data`. Run your HITL app from this location, e.g. `/path/to/my_hitl_app/$ python my_hitl_app.py`."

## Example HITL applications

Check out our example HITL apps [here](../examples/hitl/).

Use these as reference to create your own HITL application. we recommend starting by copy-pasting one of the example application folders like `hitl/pick_throw_vr/` into your own git repository, for example `my_repo/my_pick_throw_vr/`.

## VR HITL applications

The HITL framework can be used to build desktop applications (controlled with keyboard/mouse) as well as **VR** applications. See our [Pick_throw_vr](../examples/hitl/pick_throw_vr/README.md) example application.

## Configuration

HITL apps use Hydra for configuration, for example, to control the desktop window width and height. See [`hitl_defaults.yaml`](./config/hitl_defaults.yaml) as well as each example app's individual config, e.g. [`pick_throw_vr.yaml`](../examples/hitl/pick_throw_vr/config/pick_throw_vr.yaml). See also [`habitat-lab/habitat/config/README.md`](../habitat-lab/habitat/config/README.md).

## Minimal HITL application

Build a minimal desktop HITL app by implementing a derived AppState, loading a suitable Hydra config, and calling `hitl_main`. Let's take a closer look:
```
# minimal_cfg.yaml

# Read more about Hydra at habitat-lab/habitat/config/README.md.
# @package _global_

defaults:
  # We load the `pop_play` Habitat baseline featuring a Spot robot
  # and a humanoid in HSSD scenes. See habitat-baselines/README.md.
  - social_rearrange: pop_play
  # Load default parameters for the HITL framework. See
  # habitat-hitl/habitat_hitl/config/hitl_defaults.yaml.
  - hitl_defaults
  - _self_

```
```
# minimal.py

import hydra
import magnum

from habitat_hitl.app_states.app_state_abc import AppState
from habitat_hitl.core.gui_input import GuiInput
from habitat_hitl.core.hitl_main import hitl_main
from habitat_hitl.core.hydra_utils import register_hydra_plugins


class AppStateMinimal(AppState):
    """
    A minimal HITL app that loads and steps a Habitat environment, with
    a fixed overhead camera.
    """

    def __init__(self, app_service):
        self._app_service = app_service

    def sim_update(self, dt, post_sim_update_dict):
        """
        The HITL framework calls sim_update continuously (for each
        "frame"), before rendering the app's GUI window.
        """
        # run the episode until it ends
        if not self._app_service.env.episode_over:
            self._app_service.compute_action_and_step_env()

        # set the camera for the main 3D viewport
        post_sim_update_dict["cam_transform"] = magnum.Matrix4.look_at(
            eye=magnum.Vector3(-20, 20, -20),
            target=magnum.Vector3(0, 0, 0),
            up=magnum.Vector3(0, 1, 0),
        )

        # exit when the ESC key is pressed
        if self._app_service.gui_input.get_key_down(KeyCode.ESC):
            post_sim_update_dict["application_exit"] = True


@hydra.main(version_base=None, config_path="./", config_name="minimal_cfg")
def main(config):
    hitl_main(config, lambda app_service: AppStateMinimal(app_service))


if __name__ == "__main__":
    register_hydra_plugins()
    main()
```
See the latest version of the minimal app [here](../examples/hitl/minimal/).
