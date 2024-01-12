# Human-in-the-loop (HITL) Framework

The HITL framework brings real human users into Habitat virtual environments. Use it to build interactive desktop and VR applications that enable interaction between users and simulated robots and other virtual agents. Deploy these apps to users to collect interaction data for agent evaluation and training.

The framework builds on `habitat-lab` and `habitat-baselines`. It provides wrappers to load, simulate, and render Habitat environments in real time, including virtual agents and policy inference. To enable users to interact with these agents, the framework provides graphical user interfaces (GUIs), including a 3D viewport window, camera-control and avatar-control helpers, and VR integration.

<p align="center">
  <img src="../../res/img/hitl_tool.gif" height=400>
</p>

- [Human-in-the-loop (HITL) Framework](#human-in-the-loop-hitl-framework)
  - [System Requirements](#system-requirements)
  - [Installation](#installation)
  - [Example HITL applications](#example-hitl-applications)
  - [VR HITL applications](#vr-hitl-applications)
  - [Configuration](#configuration)
  - [HITL Framework Architecture](#hitl-framework-architecture)

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

## Example HITL applications

Check out our example HITL applications [here](../hitl/).

Use these as a reference to create your own HITL application. we recommend starting by copy-pasting one of the example application folders like `hitl/pick_throw_vr/` into your own git repository, for example `my_repo/my_pick_throw_vr/`.

## VR HITL applications

The HITL framework can be used to build desktop applications (controlled with keyboard/mouse) as well as **VR** applications. See our [Pick_throw_vr](../hitl/pick_throw_vr/README.md) example application.

## Configuration

HITL apps use Hydra for configuration, for example, to control the desktop window width and height. See [`hitl_defaults.yaml`](./config/hitl_defaults.yaml) as well as each example app's individual config, e.g. [`pick_throw_vr.yaml`](../hitl/pick_throw_vr/config/pick_throw_vr.yaml). See also [`habitat-lab/habitat/config/README.md`](../../habitat-lab/habitat/config/README.md).

## HITL Framework Architecture
* The HITL framework is logically divided into a Habitat-lab environment wrapper (`SandboxDriver`) and a GUI component (`GuiApplication` and `ReplayGuiAppRenderer`).
* `SandboxDriver`
    * It creates a `habitat.Env` instance.
    * Camera sensors are rendered by the `habitat.Env` instance in the usual way; see `self.obs = self.env.step(action)` in `SandboxDriver.sim_update`.
    * This class is provided a `gui_input` object that encapsulates OS input (keyboard and mouse input). We should avoid making direct calls to PyGame, GLFW, and other OS-specific APIs.
    * `sim_update` returns a `post_sim_update_dict` that contains info needed by the app renderer (below). E.g. a gfx-replay keyframe and a camera transform for rendering, plus optional "debug images" to be shown to the user.
    * This class also has access to a `debug_line_render` instance for visualizing lines in the GUI (the lines aren't rendered into camera sensors). This access is somewhat hacky; future versions of HITL apps will likely convey lines via `post_sim_update_dict` instead of getting direct access to this object.
    * This class is provided an AppState which provides application-specific logic, for example, specific keyboard/mouse controls and specific on-screen help text.
* `GuiApplication`
    * manages the OS window (via GLFW for now), including OS-input-handling (keyboard and mouse) and updating the display (invoking the renderer).
* `ReplayGuiAppRenderer`
    * `ReplayGuiAppRenderer` is a "render client". It receives the `post_sim_update_dict` from `SandboxDriver` and updates the OS window by rendering the scene from the requested camera pose.
    * In theory, render clients should be app-agnostic, i.e. `ReplayGuiAppRenderer` could be re-used for other GUI apps, but in practice we may find situations where we have to inject some app-specific code into this class (we should avoid if possible).
