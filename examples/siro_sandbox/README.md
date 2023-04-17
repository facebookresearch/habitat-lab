# Sandbox Tool

![siro_sandbox_screenshot](https://user-images.githubusercontent.com/6557808/230213487-f4812c2f-ec7f-4d68-9bbe-0b65687f769b.png)

## Overview
This is a 3D interactive GUI app for testing various pieces of SIRo, e.g. rearrangement episode datasets, Fetch and Spot robots, humanoids (controllers, animation, skinning), trained agent policies, batch rendering and other visualization.

## Known Issues
* The policy-driven agent doesn't seem to be working in terms of producing interesting actions. As a placeholder, I've injected random-base-movement behavior in `BaselinesController.act`; see comment "temp do random base actions".
* One-time visual flicker shortly after app startup on Mac
* Spot robot stops and doesn't move once it collides with any object (try pressing `M` to reset to a next episode)

## Running HITL eval with a user-controlled humanoid and policy-driven Fetch or Spot

1. Make sure you've followed the [SIRo install instructions](../../SIRO_README.md#installation).
2. To use Fetch run:
```
HABITAT_SIM_LOG=warning MAGNUM_LOG=warning \
python examples/siro_sandbox/sandbox_app.py \
--disable-inverse-kinematics \
--humanoid-user-agent \
--never-end \
--cfg benchmark/rearrange/rearrange_easy_human_and_fetch.yaml \
--cfg-opts habitat.dataset.split=minival
```
3. To use Spot run:
```
HABITAT_SIM_LOG=warning MAGNUM_LOG=warning
python examples/siro_sandbox/sandbox_app.py \
--disable-inverse-kinematics \
--humanoid-user-agent \
--never-end \
--cfg benchmark/rearrange/rearrange_easy_human_and_spot.yaml \
--cfg-opts habitat.dataset.split=minival
```

Add `--debug-images` argument followed by the camera sensors ids to enable debug observations visualization in the app GUI. For example, to visualize agent1's head depth sensor observations add: `--debug-images agent_1_head_depth`.

### Controls
* Mouse scroll wheel to zoom the camera in/out.
* Right-click on the floor and hold to move the humanoid.
* Mouse-over an object. When you see a yellow highlight, left-click to grasp.
    * Note grasping isn't restricted to proximity to the humanoid.
    * Left-click again to drop the object anywhere in the scene.
* Camera yaw and pitch control (TODO: decide which one has better UX):
    1. WASD keys
    2. hold Q and move mouse
* `M` to reset to a new episode.

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
