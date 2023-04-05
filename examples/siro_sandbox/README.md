# Sandbox Tool

## Overview
This is a 3D interactive GUI app for testing various pieces of SIRo, e.g. rearrangement episode datasets, Fetch and Spot robots, humanoids (controllers, animation, skinning), trained agent policies, batch rendering and other visualization.

## Known Issues
* The policy-driven agent doesn't seem to be working in terms of producing interesting actions. As a placeholder, I've injected random-base-movement behavior in `BaselinesController.act`; see comment "temp do random base actions".
* One-time visual flicker shortly after app startup on Mac

## Running HITL eval with a user-controlled humanoid and policy-driven Fetch

1. Make sure you've followed the [SIRo install instructions](../../SIRO_README.md#installation).
2. Run this command:
```
HABITAT_SIM_LOG=quiet python examples/interactive_play.py --disable-inverse-kinematics --use-humanoid-controller --cfg habitat-lab/habitat/config/benchmark/rearrange/rearrange_easy_human_and_fetch.yaml --never-end habitat.dataset.split=minival
```
Controls:
* Mouse to move humanoid
* Mouse scroll wheel to zoom camera
* `M` to reset to new episode
* Note there's no way to grasp objects yet

## Testing BatchReplayRenderer

Run the above command but also include `--use-batch-renderer`. Note the humanoid isn't visualized because 3D primitives aren't yet supported in the batch renderer.

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
