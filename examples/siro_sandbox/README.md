# Sandbox Tool 

![siro_sandbox_screenshot](https://user-images.githubusercontent.com/6557808/230213487-f4812c2f-ec7f-4d68-9bbe-0b65687f769b.png)

## Overview
This is a 3D interactive GUI app for testing various pieces of SIRo, e.g. rearrangement episode datasets, Fetch and Spot robots, humanoids (controllers, animation, skinning), trained agent policies, batch rendering and other visualization.

## Known Issues
* The skinned humanoid doesn't render correctly; see workaround below.
* When using Floorplanner scenes (see below), the app has very bad runtime perf on older Macbooks (2021 is fine; 2019 is bad).

## Running HITL eval
Make sure you've followed the [SIRo install instructions](../../SIRO_README.md#installation), including grabbing latest habitat-sim `main`.

### GUI-controlled humanoid and PDDL planner + oracle skills policy-controlled Spot
<!-- * To use Fetch, run:
```
HABITAT_SIM_LOG=warning MAGNUM_LOG=warning \
python examples/siro_sandbox/sandbox_app.py \
--disable-inverse-kinematics \
--never-end \
--gui-controlled-agent-index 0 \
--cfg benchmark/rearrange/rearrange_easy_human_and_fetch.yaml \
--cfg-opts habitat.dataset.split=minival \
--sample-random-baseline-base-vel
``` -->
* To launch GUI-controlled humanoid and planner-controlled Spot, run:
```
HABITAT_SIM_LOG=warning MAGNUM_LOG=warning \
python examples/siro_sandbox/sandbox_app.py \
--disable-inverse-kinematics \
--never-end \
--gui-controlled-agent-index 1 \
--cfg experiments_hab3/pop_play_kinematic_oracle_humanoid_spot_fp.yaml \
--cfg-opts \
habitat_baselines.evaluate=True \
habitat_baselines.num_environments=1
```
* To launch [solo GUI-controlled humanoid](#solo-humanoid-mode), with sliding enabled, run:
```
HABITAT_SIM_LOG=warning MAGNUM_LOG=warning \
python examples/siro_sandbox/sandbox_app.py \
--disable-inverse-kinematics \
--gui-controlled-agent-index 0 \
--never-end \
--cfg experiments_hab3/single_agent_pddl_planner_kinematic_oracle_humanoid.yaml \
--cfg-opts \
habitat_baselines.evaluate=True \
habitat_baselines.num_environments=1 \
habitat.simulator.habitat_sim_v0.allow_sliding=True
```

### GUI-controlled humanoid and learned-policy-controlled Spot

* To launch GUI-controlled humanoid and random-policy-controlled (initialized with random weights) Spot, run:
```
HABITAT_SIM_LOG=warning MAGNUM_LOG=warning \
python examples/siro_sandbox/sandbox_app.py \
--disable-inverse-kinematics \
--never-end \
--gui-controlled-agent-index 1 \
--cfg experiments_hab3/pop_play_kinematic_oracle_humanoid_spot.yaml \
--cfg-opts \
habitat_baselines.evaluate=True \
habitat_baselines.num_environments=1 \
habitat_baselines.eval.should_load_ckpt=False
```

* To launch random-policy-controlled humanoid and Spot in [free camera mode](#gui-controlled-agents-and-free-camera-mode), run:
```
HABITAT_SIM_LOG=warning MAGNUM_LOG=warning \
python examples/siro_sandbox/sandbox_app.py \
--disable-inverse-kinematics \
--never-end \
--cfg experiments_hab3/pop_play_kinematic_oracle_humanoid_spot.yaml \
--cfg-opts \
habitat_baselines.evaluate=True \
habitat_baselines.num_environments=1 \
habitat_baselines.eval.should_load_ckpt=False
```

To use **trained**-policy-controlled agent(s) instead of random-policy-controlled:
1. Download the pre-trained [checkpoint](https://drive.google.com/file/d/1swH5ZUgxe3xQn_k0s5OD7Ow6-mwCN_ic/view?usp=share_link) (150 updates).
2.  Run two above commands with the following `--cfg-opts`:
```
--cfg-opts \
habitat_baselines.evaluate=True \
habitat_baselines.num_environments=1 \
habitat_baselines.eval.should_load_ckpt=True \
habitat_baselines.eval_ckpt_path_dir=path/to/latest.pth
```


## Controls
* See on-screen help text for common keyboard and mouse controls
* `N` to toggle navmesh visualization in the debug third-person view (`--debug-third-person-width`)
* For `--first-person-mode`, you can toggle mouse-look by left-clicking anywhere

## Saving episode data
Use `--save-filepath-base my_session`. When the user presses `M` to reset the env, the first episode will be saved as `my_session.0.json.gz` and `my_session.0.pkl.gz`. These files contain mostly-identical data; we save both so that developers have two choices for how to consume the data later. After pressing `M` again, the second episode will be saved as `my_session.1.json.gz`, etc. For an example of consuming this data, see `test_episode_save_files.py` .

## Workaround to avoid broken skinned humanoid

Following the instructions above, a broken skinned humanoid is rendered which blocks the first-person camera view at times. This is a known issue: the sandbox app uses replay-rendering, which doesn't yet support skinning.

Steps to work around this by reverting to a rigid-skeleton humanoid:
1. Make a copy (or symlink) of `female2_0.urdf`.
    * `cp data/humanoids/humanoid_data/female2_0.urdf data/humanoids/humanoid_data/female2_0_rigid.urdf`
2. Update or override your config. Your humanoid is probably either `main_agent` or `agent_1`.
    * `habitat.simulator.agents.main_agent.articulated_agent_urdf='data/humanoids/humanoid_data/female2_0_rigid.urdf`
    * or `habitat.simulator.agents.agent_1.articulated_agent_urdf='data/humanoids/humanoid_data/female2_0_rigid.urdf'`
3. Run the sandbox app and you should now see a rigid-skeleton humanoid that animates properly.

## Debugging visual sensors

Add `--debug-images` argument followed by the camera sensors ids to enable debug observations visualization in the app GUI. For example, to visualize agent1's head depth sensor observations add: `--debug-images agent_1_head_depth`.

## Debugging simulator-rendering

Add `--debug-third-person-width 600` to enable the debug third-person camera. Like all visual sensors, this is simulator-rendered, unlike the main sandbox app viewport, which is replay-rendered.

## GUI-controlled agents and free camera mode
Add `--gui-controlled-agent-index` followed by the agent's index you want to control via GUI (for example, `--gui-controlled-agent-index 0` to control the first agent).

If not set, it is assumed that scene is empty or all agents are policy-controlled. App switches to free camera mode in this case. User-controlled free camera lets the user observe the scene (instead of controlling one of the agents). For instance, one use case is to (eventually) observe policy-controlled agents.

**Note:** Currently, only Spot and Humanoid agents can be policy-controlled (PDDL planner + oracle skills). If you want to test the free camera mode, omit `--gui-controlled-agent-index` argument.

## Solo humanoid mode
Set `--cfg experiments_hab3/single_agent_pddl_planner_kinematic_oracle_humanoid.yaml` to run app with only a user-controlled humanoid (no robot).

## First-person view humanoid control
Add `--first-person-mode` to switch to first-person view humanoid control mode. Use  `--max-look-up-angle` and `--min-look-down-angle` arguments to limit humanoid's look up/down angle. For example, `--max-look-up-angle 0 --min-look-down-angle -45` to let the humanoid look down -45 degrees.

## Can grasp/place area
Use `--can-grasp-place-threshold` argument to set/change grasp/place area radius.

## Disable episode end on collision
In the multi agent tidy house task, episode is considered over when humanoid and robot agents collide. Sandbox app will crash in this case as the actions can't be executed if env episode is over. In this case, you may want too disable episode end on collision. It can be done by appending the following line to your `--cfg-opts`:
```
habitat.task.measurements.cooperate_subgoal_reward.end_on_collide=False
```

## Play episodes filter
Specify a subset of play episodes on the command line by adding `--episodes-filter`  argument followed by the filter string. Episodes filter string should be in the form `"0:10 12 14:20:2"`, where single integer number ('12' in this case) represents an episode id and colon separated integers ('0:10' and '14:20:2') represent start:stop:step episodes ids range.


## Using FP dataset
To use FP dataset follow the FP installation instructions in [SIRO_README.md](../../SIRO_README.md) and run any of the above Sandbox launch command with the following config overrides appended:
```
...
--cfg-opts \
... \
habitat.task.task_spec=rearrange_easy_fp \
habitat.task.pddl_domain_def=fp \
+habitat.simulator.additional_object_paths="[data/objects/ycb/configs/, data/objects/amazon_berkeley/configs/, data/objects/google_object_dataset/configs/]" \
habitat.dataset.data_path=data/datasets/floorplanner/rearrange/scratch/train/microtrain_v0.1.json.gz
```

## Capturing Gfx-Replay Files
Gfx-Replay files are graphics captures that can be replayed by other applications, such as Blender. Recording can be enabled with the `--enable-gfx-replay-save` argument. Capturing starts at the first frame and ends (is saved) when pressing the period (`.`) key. The `--gfx-replay-save-path` argument can be set to specify a custom save location.

## Human-in-the-loop tutorial sequence
The sandbox tool can show a tutorial sequence at the start of every episode to introduce the user to the scene and goals in a human-in-the-loop context. To enable this, use the `--show-tutorial` command-line argument.

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
