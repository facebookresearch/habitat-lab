# Sandbox Tool

![siro_sandbox_screenshot](https://user-images.githubusercontent.com/6557808/230213487-f4812c2f-ec7f-4d68-9bbe-0b65687f769b.png)

# Overview
This is a 3D interactive GUI app for testing various pieces of Habitat 3.0, e.g. rearrangement episode datasets, Fetch and Spot robots, humanoids (controllers, animation, skinning), trained agent policies, batch rendering and other visualization.

# Known Issues
* When using HSSD scenes (see below), the app has bad runtime perf on older Macbooks (2021 is fine; 2019 is bad). See "Workaround for poor runtime perf on slower machines".

# Example usage
> [!NOTE]
> Installation steps and example terminal commands coming soon!

# Controls

* See on-screen help text for common keyboard and mouse controls
* `N` to toggle navmesh visualization in the debug third-person view (`--debug-third-person-width`)
* For `--first-person-mode`, you can toggle mouse-look by left-clicking anywhere

# Workaround for poor runtime perf on slower machines

If your FPS is very low, consider this workaround. This habitat-sim commit replaces render meshes for high-vertex-density objects with white bounding-box outlines. Beware, many parts of the scene will appear to be missing!
* Follow [SIRo install instructions](../../SIRO_README.md#installation) for building habitat-sim from source.
* Apply this habitat-sim commit: `git cherry-pick f031c975`
* Rebuild habitat-sim.

# Command-line Options

## App State and use cases
Use `--app-state rearrange` to run rearrange task in the sandbox app. Supported app states: `rearrange` (default), `fetch`, `socialnav`, `free_camera`. These correspond to the different use cases for the HITL tool (see corresponding implementations in `app_states/app_state_<state name>.py` files). Note, `free_camera` is special: it's not a "task" state, it's a state that lets the user control the camera instead of controlling an agent (useful for debugging and policies' behaviour visualisation). Also, `tutorial` is a special state that shows a tutorial sequence at the start of every episode to introduce the user to the scene and goals in a human-in-the-loop context. Tutorial should be followed by the task app state (`rearrange`, `fetch` or `socialnav`) and is currently supported only for the rearrange task.

## Hack to hide the skinned humanoid in the GUI viewport
Use `--hide-humanoid-in-gui` to hide the humanoid in the GUI viewport. Note it will still be rendered into observations fed to policies. This option is a workaround for broken skinned humanoid rendering in the GUI viewport.

## Saving episode data
Use `--save-filepath-base my_session`. When the user presses `M` to reset the env, the first episode will be saved as `my_session.0.json.gz` and `my_session.0.pkl.gz`. These files contain mostly-identical data; we save both so that developers have two choices for how to consume the data later. After pressing `M` again, the second episode will be saved as `my_session.1.json.gz`, etc. For an example of consuming this data, see `test_episode_save_files.py` .

## Debugging visual sensors

Add `--debug-images` argument followed by the camera sensors ids to enable debug observations visualization in the app GUI. For example, to visualize agent1's head depth sensor observations add: `--debug-images agent_1_head_depth`.

## Debugging simulator-rendering

Add `--debug-third-person-width 600` to enable the debug third-person camera. Like all visual sensors, this is simulator-rendered, unlike the main sandbox app viewport, which is replay-rendered.

## GUI-controlled agents and free camera mode
Add `--gui-controlled-agent-index` followed by the agent's index you want to control via GUI (for example, `--gui-controlled-agent-index 0` to control the first agent).

If not set, it is assumed that scene is empty or all agents are policy-controlled. App switches to free camera mode in this case. User-controlled free camera lets the user observe the scene (instead of controlling one of the agents). For instance, one use case is to (eventually) observe policy-controlled agents. Update Aug 11: free camera is temporarily unsupported!

Use `--lin-speed` and `--ang-speed` arguments to control GUI-controlled agent's linear and angular speed respectively. For example, `--lin-speed 10 --ang-speed 10` to set both linear and angular speed to 10.

**Note:** Currently, only Spot and Humanoid agents can be policy-controlled (PDDL planner + oracle skills). If you want to test the free camera mode, omit `--gui-controlled-agent-index` argument.

## First-person and third-person mode for GUI-controlled humanoid
Include `--first-person-mode`, or omit it to use third-person mode. With first-person mode, use  `--max-look-up-angle` and `--min-look-down-angle` arguments to limit humanoid's look up/down angle. For example, `--max-look-up-angle 0 --min-look-down-angle -45` to let the humanoid look down -45 degrees. You should also generally use `--hide-humanoid-in-gui` with `--first-person-mode`, because it doesn't make sense to visualize the humanoid with this camera.

## Can grasp/place area
Use `--can-grasp-place-threshold` argument to set/change grasp/place area radius.

## Disable episode end on collision
In the multi agent tidy house task, episode is considered over when humanoid and robot agents collide. Sandbox app will crash in this case as the actions can't be executed if env episode is over. In this case, you may want too disable episode end on collision. It can be done by appending the following line to your `--cfg-opts`:
```
habitat.task.measurements.rearrange_cooperate_reward.end_on_collide=False
```

## Play episodes filter
Specify a subset of play episodes on the command line by adding `--episodes-filter`  argument followed by the filter string. Episodes filter string should be in the form `"0:10 12 14:20:2"`, where single integer number ('12' in this case) represents an episode id and colon separated integers ('0:10' and '14:20:2') represent start:stop:step episodes ids range.

## Saving episode data
Add `--save-episode-record` flag to enable saving recorded episode data to file and `--save-filepath-base my_session` argument to specify a custom save location (filepath base). When the user presses `M` to reset the env, the first episode will be saved as `my_session.0.json.gz` and `my_session.0.pkl.gz`. These files contain mostly-identical data; we save both so that developers have two choices for how to consume the data later. After pressing `M` again, the second episode will be saved as `my_session.1.json.gz`, etc. For an example of consuming this data, see `test_episode_save_files.py` .

## Capturing Gfx-Replay Files
Gfx-Replay files are graphics captures that can be replayed by other applications, such as Blender. Recording (and saving to disk) can be enabled by adding `--enable-gfx-replay-save` flag and `--save-filepath-base my_session` argument specifying a custom save location (filepath base). Capturing ends (is saved) when the session is over (pressed ESC). The file will be saved as `my_session.gfx_replay.json.gz`.

## Human-in-the-loop tutorial sequence
The sandbox tool can show a tutorial sequence at the start of every episode to introduce the user to the scene and goals in a human-in-the-loop context. To enable this, use the `--show-tutorial` command-line argument. Update Aug 11: temporarily unsupported!

## Testing BatchReplayRenderer
This is an experimental feature aimed at those of us building the batch renderer. Run the above command but also include `--use-batch-renderer` as one of the first arguments.

### Known Issues
* The batch renderer doesn't work on Mac due to Mac's poor OpenGL support. We may resolve this later this year.
* The humanoid isn't visualized because 3D primitives aren't yet supported in the batch renderer.
* Ruslan reported an issue with the mouse-controlled humanoid navigation not working correctly.

# Sandbox Tool Architecture
* The sandbox app is logically divided into a Sim/Task/RL component (`SandboxDriver`) and a GUI component (`GuiApplication` and `ReplayGuiAppRenderer`).
* `SandboxDriver`
    * It creates a `habitat.Env` instance.
    * Camera sensors are rendered by the `habitat.Env` instance in the usual way; see `self.obs = self.env.step(action)` in `SandboxDriver.sim_update`.
    * This class is provided a `gui_input` object that encapsulates OS input (keyboard and mouse input). We should avoid making direct calls to PyGame, GLFW, and other OS-specific APIs.
    * `sim_update` returns a `post_sim_update_dict` that contains info needed by the app renderer (below). E.g. a gfx-replay keyframe and a camera transform for rendering, plus optional "debug images" to be shown to the user.
    * This class also has access to a `debug_line_render` instance for visualizing lines in the GUI (the lines aren't rendered into camera sensors). This access is somewhat hacky; future versions of HITL apps will likely convey lines via `post_sim_update_dict` instead of getting direct access to this object.
    * See `app_states/app_state_rearrange.py` and similar classes for per-step logic that is specific to various use cases (rearrange, fetch, etc.).
* `GuiApplication`
    * manages the OS window (via GLFW for now), including OS-input-handling (keyboard and mouse) and updating the display (invoking the renderer).
* `ReplayGuiAppRenderer`
    * `ReplayGuiAppRenderer` is a "render client". It receives the `post_sim_update_dict` from `SandboxDriver` and updates the OS window by rendering the scene from the requested camera pose.
    * In theory, render clients should be app-agnostic, i.e. `ReplayGuiAppRenderer` could be re-used for other GUI apps, but in practice we may find situations where we have to inject some app-specific code into this class (we should avoid if possible).
