# Human-in-the-loop (HITL) Framework

The HITL framework brings real human users into Habitat virtual environments. Use it to build interactive desktop and VR applications that enable interaction between users and simulated robots and other virtual agents. Deploy these apps to users to collect interaction data for agent evaluation and training.

The HITL framework consists of the `habitat-hitl` Python library, HITL example [desktop applications](../examples/hitl/), and our Unity-based [VR client](../examples/hitl/pick_throw_vr/README.md#vr).

The framework builds on `habitat-lab` and `habitat-baselines`. It provides wrappers to load, simulate, and render Habitat environments in real time, including virtual agents and policy inference. To enable users to interact with these agents, the framework provides graphical user interfaces (GUIs), including a 3D viewport window, camera-control and avatar-control helpers, and VR integration.

<p align="center">
  <img src="../../res/img/hitl_tool.gif" height=400>
</p>

- [Human-in-the-loop (HITL) Framework](#human-in-the-loop-hitl-framework)
  - [System Requirements](#system-requirements)
  - [Installation](#installation)
  - [Minimal HITL application](#minimal-hitl-application)
  - [Example HITL applications](#example-hitl-applications)
  - [VR HITL applications](#vr-hitl-applications)
  - [AppService and helpers](#appservice-and-helpers)
  - [Configuration](#configuration)

## System Requirements
* **Operating System:** macOS or Linux. We've tested Fedora but other flavors should also work.
* **CPU/GPU:** Apple M1 Max, Intel Core i7 + dedicated GPU, or equivalent.
* **Display:** laptop or desktop with an attached monitor. We haven't tested remote desktop or other headless options.
* **Storage:** ~20 GB. Example HITL apps use the Habitat Synthetic Scenes Dataset (HSSD), which is about 20 GB.
* **VR:** HITL VR uses a client/server model which requires both a desktop system (above) and a VR headset. See [`pick_throw_vr/README.md`](../hitl/pick_throw_vr/README.md) for details.

Example HITL apps are configured to run at 30 steps per second (SPS). If your system doesn't meet the above specs, you'll have a lower SPS and a degraded user experience.

## Installation
1. Clone Habitat-lab [main branch](https://github.com/facebookresearch/habitat-lab).
2. Install Habitat-lab using [instructions](https://github.com/facebookresearch/habitat-lab#installation).
    * The HITL framework depends on the `habitat-lab` and `habitat-baselines` packages. While these packages are in the same repository as the HITL framework, it's not strictly necessary for the HITL framework to import the packages that live in th- [Human-in-the-loop (HITL) Framework](#human-in-the-loop-hitl-framework)
3. Install Habitat-sim [main branch](https://github.com/facebookresearch/habitat-sim).
    * [Build from source](https://github.com/facebookresearch/habitat-sim/blob/main/BUILD_FROM_SOURCE.md), or install the [conda packages](https://github.com/facebookresearch/habitat-sim#recommended-conda-packages).
        * Be sure to include Bullet physics, e.g. `python setup.py install --bullet`.
4. Download required assets for our example HITL applications:
    ```bash
    python -m habitat_sim.utils.datasets_download \
    --uids hab3-episodes habitat_humanoids hab_spot_arm ycb hssd-hab \
    --data-path data/
    ```

## Minimal HITL application

Build a minimal desktop HITL app by implementing a derived AppState (`MyAppState`), loading a suitable Hydra config, and calling `hitl_main.hitl_main(config, lambda: MyAppState())`. Let's take a closer look:
```
todo: full python source for a minimal HITL app, plus config yaml source
```

## Example HITL applications

Check out our example HITL apps [here](../hitl/).

Use these as additional reference to create your own HITL application. we recommend starting by copy-pasting one of the example application folders like `hitl/pick_throw_vr/` into your own git repository, for example `my_repo/my_pick_throw_vr/`.

## VR HITL applications

The HITL framework can be used to build desktop applications (controlled with keyboard/mouse) as well as **VR** applications. See our [Pick_throw_vr](../hitl/pick_throw_vr/README.md) example application.

## AppService and helpers

todo

## Configuration

HITL apps use Hydra for configuration, for example, to control the desktop window width and height. See [`hitl_defaults.yaml`](./config/hitl_defaults.yaml) as well as each example app's individual config, e.g. [`pick_throw_vr.yaml`](../hitl/pick_throw_vr/config/pick_throw_vr.yaml). See also [`habitat-lab/habitat/config/README.md`](../../habitat-lab/habitat/config/README.md).
