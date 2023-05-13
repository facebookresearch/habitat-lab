# Sandbox Tool

![siro_sandbox_screenshot](https://user-images.githubusercontent.com/6557808/230213487-f4812c2f-ec7f-4d68-9bbe-0b65687f769b.png)

## Overview
This is a 3D interactive GUI app for testing various pieces of SIRo, e.g. rearrangement episode datasets, Fetch and Spot robots, humanoids (controllers, animation, skinning), trained agent policies, batch rendering and other visualization.

## Known Issues
* The policy-driven agent isn't working in terms of producing interesting actions. As a placeholder, we've injected random-base-movement behavior in `BaselinesController.act`; see comment "temp do random base actions".
* One-time visual flicker shortly after app startup
* When using Floorplanner scenes (see below), the app has very bad runtime perf on older Macbooks (2021 is fine; 2019 is bad).
* Spot robot stops and doesn't move once it collides with any object (try pressing `M` to reset to a next episode).

## Running HITL eval with a user-controlled humanoid and policy-driven Fetch or Spot

* Make sure you've followed the [SIRo install instructions](../../SIRO_README.md#installation), including grabbing latest habitat-sim `main`.
* To use Fetch, run:
```
HABITAT_SIM_LOG=warning MAGNUM_LOG=warning \
python examples/siro_sandbox/sandbox_app.py \
--disable-inverse-kinematics \
--never-end \
--gui-controlled-agent-index 0 \
--cfg benchmark/rearrange/rearrange_easy_human_and_fetch.yaml \
--cfg-opts habitat.dataset.split=minival \
--sample-random-baseline-base-vel
```
* To use Spot, run:
```
HABITAT_SIM_LOG=warning MAGNUM_LOG=warning \
python examples/siro_sandbox/sandbox_app.py \
--disable-inverse-kinematics \
--never-end \
--gui-controlled-agent-index 0 \
--cfg benchmark/rearrange/rearrange_easy_human_and_spot.yaml \
--cfg-opts habitat.dataset.split=minival \
--sample-random-baseline-base-vel
```
* Solo user-controlled humanoid mode, with sliding enabled:
```
HABITAT_SIM_LOG=warning MAGNUM_LOG=warning \
python examples/siro_sandbox/sandbox_app.py \
--disable-inverse-kinematics \
--gui-controlled-agent-index 0 \
--never-end \
--cfg benchmark/rearrange/rearrange_easy_human.yaml \
--cfg-opts habitat.dataset.split=minival \
habitat.simulator.habitat_sim_v0.allow_sliding=True
```


## Controls
* See on-screen help text for common keyboard and mouse controls
* `N` to toggle navmesh visualization in the debug third-person view (`--debug-third-person-width`)

## Debugging visual sensors

Add `--debug-images` argument followed by the camera sensors ids to enable debug observations visualization in the app GUI. For example, to visualize agent1's head depth sensor observations add: `--debug-images agent_1_head_depth`.

## Debugging simulator-rendering

Add `--debug-third-person-width 600` to enable the debug third-person camera. Like all visual sensors, this is simulator-rendered, unlike the main sandbox app viewport, which is replay-rendered.

## GUI-controlled agents and free camera mode
Add `--gui-controlled-agent-index` followed by the agent's index you want to control via GUI (for example, `--gui-controlled-agent-index 0` to control the first agent).

If not set, it is assumed that scene is empty or all agents are policy-controlled. App switches to free camera mode in this case. User-controlled free camera lets the user observe the scene (instead of controlling one of the agents). For instance, one use case is to (eventually) observe policy-controlled agents.

**Note:** Currently, only robot controllers can be policy-controlled (as for now they perform random actions). Policy-controlled humanoid controller is not implemented yet. So, if you want to test the free camera mode, make sure you are using robot-robot config as a `--cfg` argument value (for example, `--cfg benchmark/rearrange/rearrange_easy_fetch_and_fetch.yaml`).

## Solo humanoid mode
Set `--cfg benchmark/rearrange/rearrange_easy_human.yaml` to run app with only a user-controlled humanoid (no robot).

## First-person view humanoid control
Add `--first-person-mode` to switch to first-person view humanoid control mode. Use  `--max-look-up-angle` and `--min-look-down-angle` arguments to limit humanoid's look up/down angle. For example, `--max-look-up-angle 0 --min-look-down-angle -45` to let the humanoid look down -45 degrees.

## Using FP dataset
To use FP dataset follow the FP installation instructions in [SIRO_README.md](../../SIRO_README.md) and run any of the above Sandbox launch command with the following config overrides:
```
...
--cfg-opts \
habitat.task.task_spec=rearrange_easy_fp \
habitat.task.pddl_domain_def=fp \
+habitat.simulator.additional_object_paths="[data/objects/ycb/configs/, data/objects/amazon_berkeley/configs/, data/objects/google_object_dataset/configs/]" \
habitat.dataset.data_path=data/datasets/floorplanner/rearrange/scratch/train/s108294897_176710602.json.gz
```

## Capturing Gfx-Replay Files
Gfx-Replay files are graphics captures that can be replayed by other applications, such as Blender. Recording can be enabled with the `--enable-gfx-replay-save` argument. Capturing starts at the first frame and ends (is saved) when pressing the period (`.`) key. The `--gfx-replay-save-path` argument can be set to specify a custom save location.

## Testing BatchReplayRenderer

This is an experimental feature aimed at those of us building the batch renderer. Run the above command but also include `--use-batch-renderer` as one of the first arguments.

### Known Issues
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
* `GuiApplication`
    * manages the OS window (via GLFW for now), including OS-input-handling (keyboard and mouse) and updating the display (invoking the renderer).
* `ReplayGuiAppRenderer`
    * `ReplayGuiAppRenderer` is a "render client". It receives the `post_sim_update_dict` from `SandboxDriver` and updates the OS window by rendering the scene from the requested camera pose.
    * In theory, render clients should be app-agnostic, i.e. `ReplayGuiAppRenderer` could be re-used for other GUI apps, but in practice we may find situations where we have to inject some app-specific code into this class (we should avoid if possible).
