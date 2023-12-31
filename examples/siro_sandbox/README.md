# Sandbox Tool

This is a 3D interactive GUI app that enables human-in-the-loop (HITL) evaluation of agents trained on Habitat.

<p align="center">
  <img src="../../res/img/hitl_tool.gif" height=400>
</p>

---

- [Installation](#installation)
- [Applications](#applications)
  - [AppState: rearrange](#appstate-rearrange)
    - [Example launch command](#example-launch-command)
  - [AppState: pick\_throw\_vr](#appstate-pick_throw_vr)
  - [AppState: free\_camera](#appstate-free_camera)
    - [Example launch command](#example-launch-command-1)
  - [AppState: socialnav](#appstate-socialnav)
  - [AppState: tutorial](#appstate-tutorial)
- [Controls](#controls)
- [Workaround for poor runtime perf on slower machines](#workaround-for-poor-runtime-perf-on-slower-machines)
- [Command-line Options](#command-line-options)
  - [Hack to hide the skinned humanoid in the GUI viewport](#hack-to-hide-the-skinned-humanoid-in-the-gui-viewport)
  - [Saving episode data](#saving-episode-data)
  - [Debugging visual sensors](#debugging-visual-sensors)
  - [Debugging simulator-rendering](#debugging-simulator-rendering)
  - [GUI-controlled agents and free camera mode](#gui-controlled-agents-and-free-camera-mode)
  - [First-person and third-person mode for GUI-controlled humanoid](#first-person-and-third-person-mode-for-gui-controlled-humanoid)
  - [Can grasp/place area](#can-graspplace-area)
  - [Disable episode end on collision](#disable-episode-end-on-collision)
  - [Play episodes filter](#play-episodes-filter)
  - [Saving episode data](#saving-episode-data-1)
  - [Capturing Gfx-Replay Files](#capturing-gfx-replay-files)
  - [Human-in-the-loop tutorial sequence](#human-in-the-loop-tutorial-sequence)
  - [Testing BatchReplayRenderer](#testing-batchreplayrenderer)
- [Known Issues](#known-issues)
- [Sandbox Tool Architecture](#sandbox-tool-architecture)
- [VR HITL Evaluation](#vr-hitl-evaluation)


## Installation
1. Clone Habitat-lab [main branch](https://github.com/facebookresearch/habitat-lab).
2. Install Habitat-lab using [instructions](https://github.com/facebookresearch/habitat-lab#installation).
    * The HITL tool depends on the `habitat-lab` and `habitat-baselines` packages. While these packages are in the same repository as the HITL tool, it's not strictly necessary for the HITL tool to import the packages that live in the parent directory. You can use any version installed to your current environment.
3. Install Habitat-sim [main branch](https://github.com/facebookresearch/habitat-sim).
    * [Build from source](https://github.com/facebookresearch/habitat-sim/blob/main/BUILD_FROM_SOURCE.md), or install the [conda packages](https://github.com/facebookresearch/habitat-sim#recommended-conda-packages).
        * Be sure to include Bullet physics, e.g. `python setup.py install --bullet`.
4. Download required assets:
    ```bash
    python -m habitat_sim.utils.datasets_download \
    --uids hab3-episodes habitat_humanoids hab_spot_arm ycb hssd-hab \
    --data-path data/
    ```

## Applications

The HITL app has a variety of built-in applications, called `AppState`. These are independent from each other and use the HITL app as a framework.

The app state can be selected with the `--app-state` command-line argument.

To create a new application, we recommend forking the files in `examples/siro_sandbox`. We recommend adding a new `AppState` for your use case, starting by copy-pasting an existing `AppState` class like `app_state_pick_throw_vr.py`.

### AppState: rearrange

In this application, the human and robot must accomplish a collaborative rearrangement task.
See on-screen help for controls.

#### Example launch command
GUI-controlled humanoid and random-policy-controlled (initialized with random weights) Spot in HSSD:
```bash
HABITAT_SIM_LOG=warning MAGNUM_LOG=warning \
python examples/siro_sandbox/sandbox_app.py \
--disable-inverse-kinematics \
--never-end \
--gui-controlled-agent-index 1 \
--app-state rearrange \
--cfg social_rearrange/pop_play.yaml \
--cfg-opts \
habitat.environment.iterator_options.cycle=False \
habitat_baselines.evaluate=True \
habitat_baselines.num_environments=1 \
habitat_baselines.eval.should_load_ckpt=False \
habitat_baselines.rl.agent.num_pool_agents_per_type=[1,1]
```

### AppState: pick_throw_vr

Example application that allows a user to interact with a scene, controlling a human avatar with mouse/keyboard or a VR headset. A policy-driven Spot robot also interacts with the scene. This is a proof of concept and not meant for rigorous evaluation or data-collection.

See [VR_HITL.md](./VR_HITL.md) for instructions.


### AppState: free_camera

This application allows to observe two policy-driven agents from an independent controllable camera.
See on-screen help for controls.

#### Example launch command

Policy-controlled humanoid (initialized with random weights) in HSSD:
```bash
HABITAT_SIM_LOG=warning MAGNUM_LOG=warning \
python examples/siro_sandbox/sandbox_app.py \
--disable-inverse-kinematics \
--never-end \
--app-state free_camera \
--cfg social_rearrange/pop_play.yaml \
--cfg-opts \
habitat.environment.iterator_options.cycle=False \
habitat_baselines.evaluate=True \
habitat_baselines.num_environments=1 \
habitat_baselines.eval.should_load_ckpt=False \
habitat_baselines.rl.agent.num_pool_agents_per_type=[1,1]
```

### AppState: socialnav

This app state is a work in progress and doesn't currently run.


### AppState: tutorial

Tutorial is a special app state that shows a tutorial sequence at the start of the application to introduce the user to the scene and goals in a human-in-the-loop context.

Unlike other app states, it is enabled with the `--show-tutorial` command-line argument.

It is currently only supported with the `rearrange` app state.

## Controls

* See on-screen help text for common keyboard and mouse controls
* `N` to toggle navmesh visualization in the debug third-person view (`--debug-third-person-width`)
* For `--first-person-mode`, you can toggle mouse-look by left-clicking anywhere

## Workaround for poor runtime perf on slower machines

If your FPS is very low, consider this workaround. This habitat-sim commit replaces render meshes for high-vertex-density objects with white bounding-box outlines. Beware, many parts of the scene will appear to be missing!
* Follow [SIRo install instructions](../../SIRO_README.md#installation) for building habitat-sim from source.
* Apply this habitat-sim commit: `git cherry-pick f031c975`
* Rebuild habitat-sim.

## Command-line Options

### Hack to hide the skinned humanoid in the GUI viewport
Use `--hide-humanoid-in-gui` to hide the humanoid in the GUI viewport. Note it will still be rendered into observations fed to policies. This option is a workaround for broken skinned humanoid rendering in the GUI viewport.

### Saving episode data
Use `--save-filepath-base my_session`. When the user presses `M` to reset the env, the first episode will be saved as `my_session.0.json.gz` and `my_session.0.pkl.gz`. These files contain mostly-identical data; we save both so that developers have two choices for how to consume the data later. After pressing `M` again, the second episode will be saved as `my_session.1.json.gz`, etc. For an example of consuming this data, see `test_episode_save_files.py` .

### Debugging visual sensors

Add `--debug-images` argument followed by the camera sensors ids to enable debug observations visualization in the app GUI. For example, to visualize agent1's head depth sensor observations add: `--debug-images agent_1_head_depth`.

### Debugging simulator-rendering

Add `--debug-third-person-width 600` to enable the debug third-person camera. Like all visual sensors, this is simulator-rendered, unlike the main sandbox app viewport, which is replay-rendered.

### GUI-controlled agents and free camera mode
Add `--gui-controlled-agent-index` followed by the agent's index you want to control via GUI (for example, `--gui-controlled-agent-index 0` to control the first agent).

If not set, it is assumed that scene is empty or all agents are policy-controlled. App switches to free camera mode in this case. User-controlled free camera lets the user observe the scene (instead of controlling one of the agents). For instance, one use case is to (eventually) observe policy-controlled agents. Update Aug 11: free camera is temporarily unsupported!

Use `--lin-speed` and `--ang-speed` arguments to control GUI-controlled agent's linear and angular speed respectively. For example, `--lin-speed 10 --ang-speed 10` to set both linear and angular speed to 10.

**Note:** Currently, only Spot and Humanoid agents can be policy-controlled (PDDL planner + oracle skills). If you want to test the free camera mode, omit `--gui-controlled-agent-index` argument.

### First-person and third-person mode for GUI-controlled humanoid
Include `--first-person-mode`, or omit it to use third-person mode. With first-person mode, use  `--max-look-up-angle` and `--min-look-down-angle` arguments to limit humanoid's look up/down angle. For example, `--max-look-up-angle 0 --min-look-down-angle -45` to let the humanoid look down -45 degrees. You should also generally use `--hide-humanoid-in-gui` with `--first-person-mode`, because it doesn't make sense to visualize the humanoid with this camera.

### Can grasp/place area
Use `--can-grasp-place-threshold` argument to set/change grasp/place area radius.

### Disable episode end on collision
In the multi agent tidy house task, episode is considered over when humanoid and robot agents collide. Sandbox app will crash in this case as the actions can't be executed if env episode is over. In this case, you may want too disable episode end on collision. It can be done by appending the following line to your `--cfg-opts`:
```
habitat.task.measurements.rearrange_cooperate_reward.end_on_collide=False
```

### Play episodes filter
Specify a subset of play episodes on the command line by adding `--episodes-filter`  argument followed by the filter string. Episodes filter string should be in the form `"0:10 12 14:20:2"`, where single integer number ('12' in this case) represents an episode id and colon separated integers ('0:10' and '14:20:2') represent start:stop:step episodes ids range.

### Saving episode data
Add `--save-episode-record` flag to enable saving recorded episode data to file and `--save-filepath-base my_session` argument to specify a custom save location (filepath base). When the user presses `M` to reset the env, the first episode will be saved as `my_session.0.json.gz` and `my_session.0.pkl.gz`. These files contain mostly-identical data; we save both so that developers have two choices for how to consume the data later. After pressing `M` again, the second episode will be saved as `my_session.1.json.gz`, etc. For an example of consuming this data, see `test_episode_save_files.py` .

### Capturing Gfx-Replay Files
Gfx-Replay files are graphics captures that can be replayed by other applications, such as Blender. Recording (and saving to disk) can be enabled by adding `--enable-gfx-replay-save` flag and `--save-filepath-base my_session` argument specifying a custom save location (filepath base). Capturing ends (is saved) when the session is over (pressed ESC). The file will be saved as `my_session.gfx_replay.json.gz`.

### Human-in-the-loop tutorial sequence
The sandbox tool can show a tutorial sequence at the start of every episode to introduce the user to the scene and goals in a human-in-the-loop context. To enable this, use the `--show-tutorial` command-line argument.

### Testing BatchReplayRenderer
This is an experimental feature aimed at those of us building the batch renderer. Run the above command but also include `--use-batch-renderer` as one of the first arguments.

## Known Issues
* The batch renderer doesn't work on Mac due to Mac's poor OpenGL support. We may resolve this later this year.
* The humanoid isn't visualized because 3D primitives aren't yet supported in the batch renderer.
* Ruslan reported an issue with the mouse-controlled humanoid navigation not working correctly.

## Sandbox Tool Architecture
* The sandbox app is logically divided into a Sim/Task/RL component (`SandboxDriver`) and a GUI component (`GuiApplication` and `ReplayGuiAppRenderer`).
* `SandboxDriver`
    * It creates a `habitat.Env` instance.
    * Camera sensors are rendered by the `habitat.Env` instance in the usual way; see `self.obs = self.env.step(action)` in `SandboxDriver.sim_update`.
    * This class is provided a `gui_input` object that encapsulates OS input (keyboard and mouse input). We should avoid making direct calls to PyGame, GLFW, and other OS-specific APIs.
    * `sim_update` returns a `post_sim_update_dict` that contains info needed by the app renderer (below). E.g. a gfx-replay keyframe and a camera transform for rendering, plus optional "debug images" to be shown to the user.
    * This class also has access to a `debug_line_render` instance for visualizing lines in the GUI (the lines aren't rendered into camera sensors). This access is somewhat hacky; future versions of HITL apps will likely convey lines via `post_sim_update_dict` instead of getting direct access to this object.
    * See `app_states/app_state_rearrange.py` and similar classes for per-step logic that is specific to various use cases (rearrange, pick_throw_vr, etc.).
* `GuiApplication`
    * manages the OS window (via GLFW for now), including OS-input-handling (keyboard and mouse) and updating the display (invoking the renderer).
* `ReplayGuiAppRenderer`
    * `ReplayGuiAppRenderer` is a "render client". It receives the `post_sim_update_dict` from `SandboxDriver` and updates the OS window by rendering the scene from the requested camera pose.
    * In theory, render clients should be app-agnostic, i.e. `ReplayGuiAppRenderer` could be re-used for other GUI apps, but in practice we may find situations where we have to inject some app-specific code into this class (we should avoid if possible).

## VR HITL Evaluation

VR HITL evaluation is supported in the `pick_throw_vr` app state. See [VR_HITL.md](./VR_HITL.md) for instructions for running the sandbox app in VR.
