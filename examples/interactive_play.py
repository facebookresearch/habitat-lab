#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Manually control the robot to interact with the environment. Run as
```
python examples/interative_play.py
```

To Run you need PyGame installed (to install run `pip install pygame==2.0.1`).

By default this controls with velocity control (which makes controlling the
robot hard). To use IK control instead add the `--add-ik` command line argument.

Controls:
- For velocity control
    - 1-7 to increase the motor target for the robot arm joints
    - Q-U to decrease the motor target for the robot arm joints
- For IK control
    - W,S,A,D to move side to side
    - E,Q to move up and down
- I,J,K,L to move the robot base around
- PERIOD to print the current world coordinates of the robot base.
- Z to toggle the camera to free movement mode. When in free camera mode:
    - W,S,A,D,Q,E to translate the camera
    - I,J,K,L,U,O to rotate the camera
    - B to reset the camera position
- X to change the robot that is being controlled (if there are multiple robots).

Change the task with `--cfg benchmark/rearrange/close_cab.yaml` (choose any task under the `habitat-lab/habitat/config/task/rearrange/` folder).

Change the grip type:
- Suction gripper `task.actions.arm_action.grip_controller "SuctionGraspAction"`

To record a video: `--save-obs` This will save the video to file under `data/vids/` specified by `--save-obs-fname` (by default `vid.mp4`).

Record and play back trajectories:
- To record a trajectory add `--save-actions --save-actions-count 200` to
  record a truncated episode length of 200.
- By default the trajectories are saved to data/interactive_play_replays/play_actions.txt
- Play the trajectories back with `--load-actions data/interactive_play_replays/play_actions.txt`
"""
# exit()
# breakpoint()
import argparse
import os
import os.path as osp
import time
from collections import defaultdict

import habitat_sim
import magnum as mn
import numpy as np

import habitat
import habitat.tasks.rearrange.rearrange_task
from habitat.config.default_structured_configs import (
    GfxReplayMeasureMeasurementConfig,
    ThirdRGBSensorConfig,
)
from habitat.core.logging import logger
# from habitat.tasks.rearrange.actions.actions import WalkAction
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import euler_to_quat, write_gfx_replay
from habitat.utils.render_wrapper import overlay_frame
from habitat.utils.visualizations.utils import observations_to_image
from habitat_sim.utils import viz_utils as vut
from habitat_baselines import AmassHumanController


try:
    import pygame
except ImportError:
    pygame = None

DEFAULT_CFG = "benchmark/rearrange/play.yaml"
DEFAULT_RENDER_STEPS_LIMIT = 60
SAVE_VIDEO_DIR = "./data/vids"
SAVE_ACTIONS_DIR = "./data/interactive_play_replays"


def step_env(env, action_name, action_args):
    # breakpoint()
    return env.step({"action": action_name, "action_args": action_args})

def reached_dest(agent_controller, path):
    final_point = path.points[-1]
    
    distance = np.linalg.norm((agent_controller.translation_offset - final_point) * (mn.Vector3.x_axis() + mn.Vector3.z_axis()))
    if distance < 0.1:
        return True
    return False

def compute_displ(next_point, agent_controller):
    diff_dist = next_point - agent_controller.translation_offset
    
    return [diff_dist[0], diff_dist[2]]

def get_input_vel_ctlr(
    human_controller, skip_pygame, arm_action, env, not_block_input, agent_to_control, agent_path, path_ind, repeat_walk
):
    
    if skip_pygame:
        return step_env(env, "empty", {}), None, False
    multi_agent = len(env._sim.robots_mgr) > 1
    print(env.task.actions)
    arm_action_name = "arm_action"
    base_action_name = "empty"
    arm_key = "arm_action"
    # grip_key = "grip_action"
    base_key = "human_joints_trans"
    # if multi_agent:
    #     agent_k = f"agent_{agent_to_control}"
    #     arm_action_name = f"{agent_k}_{arm_action_name}"
    #     base_action_name = f"{agent_k}_{base_action_name}"
    #     arm_key = f"{agent_k}_{arm_key}"
    #     grip_key = f"{agent_k}_{grip_key}"
    #     base_key = f"{agent_k}_{base_key}"
    # self.path_ind = 0
    # breakpoint()
    base_action = [0,0]
    # breakpoint()
    # if reached_dest(human_controller, agent_path):
    if reached_dest(human_controller, agent_path):
        base_action_name = 'humanjoint_action'
        base_key = 'human_joints_trans'
        new_pose, new_trans = human_controller.stop()
        base_action = AmassHumanController.transformAction(new_pose, new_trans)

    else:
        if repeat_walk:
            base_action_name = 'humanjoint_action'
            base_key = "human_joints_trans"


        else:
            base_action_name = 'empty'
            base_key = "human_joints_trans"

        # if path_ind != 1:
        #     breakpoint()

        displ = compute_displ(agent_path.points[path_ind], human_controller)
        # displ = compute_displ(agent_path.points[path_ind], env._sim.robot)
        
        
        displ2 = mn.Vector3([displ[0], 0, displ[1]])
        new_pose, new_trans = human_controller.walk(displ2)
        # breakpoint()
        base_action = AmassHumanController.transformAction(new_pose, new_trans)
        base_action_name = "humanjoint_action"
        base_key = "human_joints_trans"
        # base_action = displ
    # breakpoint()
    if arm_action_name in env.action_space.spaces:
        arm_action_space = env.action_space.spaces[arm_action_name].spaces[
            arm_key
        ]
        arm_ctrlr = env.task.actions[arm_action_name].arm_ctrlr
        # base_action = None
    else:
        arm_action_space = np.zeros(7)
        arm_ctrlr = None
        # base_action = [0, 0]

    if arm_action is None:
        arm_action = np.zeros(arm_action_space.shape[0])
        given_arm_action = False
    # else:
    #     given_arm_action = True

    end_ep = False
    magic_grasp = None

    keys = pygame.key.get_pressed()

    if keys[pygame.K_ESCAPE]:
        return None, None, False
    elif keys[pygame.K_m]:
        end_ep = True
    elif keys[pygame.K_n]:
        env._sim.navmesh_visualization = not env._sim.navmesh_visualization
    
    
    if not_block_input:
        # Base control
        
        # if keys[pygame.K_j]:
        #     # Left
        #     base_action_name = 'walk_action'
        #     # base_action = [0, 1]
        #     # breakpoint()
        # elif keys[pygame.K_k]:
        #     # Left
        #     base_action_name = 'walk_action'
        #     base_action = [1, 0]
        # elif keys[pygame.K_l]:
        #     # Left
        #     base_action_name = 'walk_action'
        #     base_action = [0, -1]
        # elif keys[pygame.K_i]:
        #     # Left
        #     base_action_name = 'walk_action'
        #     base_action = [-1, 0]
        if keys[pygame.K_i]:
            base_action_name = 'grab_left_action'
            base_key = 'base_pos'
        elif keys[pygame.K_j]:
            base_action_name = 'grab_right_action'
            base_key = 'base_pos'
        
        # elif keys[pygame.K_y]:
        #     # Left
        #     # ee_pos = env._sim.robot.ee_transform.translation
        #     # print(ee_pos)
        #     repeat_walk = False
        #     base_action_name = 'humanjoint_action'
        #     base_action = mn.Vector3([0, 0.05, 0])
        #     env._sim.robot.curr_trans = base_action
        # elif keys[pygame.K_t]:
        #     # Left
        #     repeat_walk = False
        #     base_action_name = 'humanjoint_action'
        #     base_action = mn.Vector3([0, -0.05, 0])
        #     env._sim.robot.curr_trans = base_action

        elif keys[pygame.K_e]:
            # Left
            # ee_pos = env._sim.robot.ee_transform.translation
            # print(ee_pos)
            repeat_walk = False
            base_action_name = 'release_left_action'
            base_action = mn.Vector3([0, 0, 0.05])
            env._sim.robot.curr_trans = base_action
        elif keys[pygame.K_r]:
            # Right
            repeat_walk = False
            base_action_name = 'release_right_action'
            base_action = mn.Vector3([0, 0, -0.05])
            env._sim.robot.curr_trans = base_action
            # print(base_action)
            # breakpoint()



    if keys[pygame.K_PERIOD]:
        # Print the current position of the robot, useful for debugging.
        pos = [float("%.3f" % x) for x in env._sim.robot.sim_obj.translation]
        rot = env._sim.robot.sim_obj.rotation
        ee_pos = env._sim.robot.ee_transform.translation
        logger.info(
            f"Robot state: pos = {pos}, rotation = {rot}, ee_pos = {ee_pos}"
        )
    elif keys[pygame.K_COMMA]:
        # Print the current arm state of the robot, useful for debugging.
        joint_state = [float("%.3f" % x) for x in env._sim.robot.arm_joint_pos]
        logger.info(f"Robot arm joint state: {joint_state}")

    args = {}
    if base_action is not None and base_action_name in env.action_space.spaces:
        name = base_action_name
        args = {base_key: base_action}
    else:
        breakpoint()
        name = arm_action_name
        if given_arm_action:
            # The grip is also contained in the provided action
            args = {
                arm_key: arm_action[:-1],
                grip_key: arm_action[-1],
            }
        else:
            args = {arm_key: arm_action, grip_key: magic_grasp}

    if magic_grasp is None:
        arm_action = [*arm_action, 0.0]
    else:
        arm_action = [*arm_action, magic_grasp]
    # breakpoint()
    
    return step_env(env, name, args), arm_action, end_ep, repeat_walk


def get_wrapped_prop(venv, prop):
    if hasattr(venv, prop):
        return getattr(venv, prop)
    elif hasattr(venv, "venv"):
        return get_wrapped_prop(venv.venv, prop)
    elif hasattr(venv, "env"):
        return get_wrapped_prop(venv.env, prop)

    return None


class FreeCamHelper:
    def __init__(self):
        self._is_free_cam_mode = False
        self._last_pressed = 0
        self._free_rpy = np.zeros(3)
        self._free_xyz = np.zeros(3)

    @property
    def is_free_cam_mode(self):
        return self._is_free_cam_mode

    def update(self, env, step_result, update_idx):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_z] and (update_idx - self._last_pressed) > 60:
            self._is_free_cam_mode = not self._is_free_cam_mode
            logger.info(f"Switching camera mode to {self._is_free_cam_mode}")
            self._last_pressed = update_idx

        if self._is_free_cam_mode:
            offset_rpy = np.zeros(3)
            if keys[pygame.K_u]:
                offset_rpy[1] += 1
            elif keys[pygame.K_o]:
                offset_rpy[1] -= 1
            elif keys[pygame.K_i]:
                offset_rpy[2] += 1
            elif keys[pygame.K_k]:
                offset_rpy[2] -= 1
            elif keys[pygame.K_j]:
                offset_rpy[0] += 1
            elif keys[pygame.K_l]:
                offset_rpy[0] -= 1

            offset_xyz = np.zeros(3)
            if keys[pygame.K_q]:
                offset_xyz[1] += 1
            elif keys[pygame.K_e]:
                offset_xyz[1] -= 1
            elif keys[pygame.K_w]:
                offset_xyz[2] += 1
            elif keys[pygame.K_s]:
                offset_xyz[2] -= 1
            elif keys[pygame.K_a]:
                offset_xyz[0] += 1
            elif keys[pygame.K_d]:
                offset_xyz[0] -= 1
            offset_rpy *= 0.1
            offset_xyz *= 0.1
            self._free_rpy += offset_rpy
            self._free_xyz += offset_xyz
            if keys[pygame.K_b]:
                self._free_rpy = np.zeros(3)
                self._free_xyz = np.zeros(3)

            quat = euler_to_quat(self._free_rpy)
            trans = mn.Matrix4.from_(
                quat.to_matrix(), mn.Vector3(*self._free_xyz)
            )
            env._sim._sensors[
                "robot_third_rgb"
            ]._sensor_object.node.transformation = trans
            step_result = env._sim.get_sensor_observations()
            return step_result
        return step_result


def update_location_walk(curr_location, env, curr_ind_map, new_loc=None):
    sim = env.sim
    curr_location = mn.Vector3([curr_location[0], 0, curr_location[2]]) 
    if new_loc is None:
        snapped_pos = env.sim.pathfinder.get_random_navigable_point()  
    else:
        snapped_pos = env.sim.pathfinder.snap_point(new_loc)
    sim.viz_ids['target_loc'] = sim.visualize_position(
        snapped_pos, sim.viz_ids['target_loc']
    )
    path = habitat_sim.ShortestPath()
    path.requested_start = curr_location
    path.requested_end = snapped_pos
    found_path = sim.pathfinder.find_path(path)
    colors = [mn.Color3.red(), mn.Color3.yellow(), mn.Color3.green()]
    # breakpoint()
    if 'trajectory' in curr_ind_map:
        sim.get_rigid_object_manager().remove_object_by_id(curr_ind_map['trajectory'])
            # curr_ind_map['trajectory'] = -1
    # else:
    if 'cont' not in curr_ind_map:
        curr_ind_map['cont'] = 0
    curr_ind_map['trajectory'] = sim.add_gradient_trajectory_object("current_path_{}".format(curr_ind_map['cont']), path.points, colors=colors, radius=0.03)
    curr_ind_map['cont'] += 1
    # breakpoint()
    return found_path, snapped_pos, path

def play_env(env, args, config):
    render_steps_limit = None
    if args.no_render:
        render_steps_limit = DEFAULT_RENDER_STEPS_LIMIT

    use_arm_actions = None
    if args.load_actions is not None:
        with open(args.load_actions, "rb") as f:
            use_arm_actions = np.load(f)
            logger.info("Loaded arm actions")

    obs = env.reset()

    if not args.no_render:
        draw_obs = observations_to_image(obs, {})
        
        pygame.init()
        screen = pygame.display.set_mode(
            [draw_obs.shape[1], draw_obs.shape[0]]
        )
    curr_ind_map = {}
    update_idx = 0
    target_fps = 60.0
    prev_time = time.time()
    all_obs = []
    total_reward = 0
    all_arm_actions = []
    agent_to_control = 0

    free_cam = FreeCamHelper()
    gfx_measure = env.task.measurements.measures.get(
        GfxReplayMeasure.cls_uuid, None
    )
    is_multi_agent = len(env._sim.robots_mgr) > 1
    env._sim.robot.translation_offset = env._sim.robot.sim_obj.translation + mn.Vector3([0,0.9, 0])
    agent_location = env._sim.robot.translation_offset
    print(agent_location)

    

    urdf_path = config.habitat.simulator.agent_0.robot_urdf
    amass_path = config.habitat.simulator.agent_0.amass_path
    body_model_path = config.habitat.simulator.agent_0.body_model_path
    obj_translation = env._sim.robot.sim_obj.translation
    
    link_ids = env._sim.robot.sim_obj.get_link_ids()
    human_controller = AmassHumanController(urdf_path, amass_path, body_model_path, obj_translation, link_ids)
    
    # TODO: remove
    human_controller.sim_obj = env._sim.robot.sim_obj
    
    # breakpoint()
    found_path, goal_location, path = update_location_walk(agent_location, env, curr_ind_map)
    path_ind = 1
    # breakpoint()
    repeat_walk = True
    for path_i in range(path_ind, len(path.points)):
        env.sim.viz_ids[f'next_loc_{path_i}'] = env.sim.visualize_position(
            path.points[path_i], env.sim.viz_ids[f'next_loc_{path_i}']
        )
    while True:
        # breakpoint()

        if (
            args.save_actions
            and len(all_arm_actions) > args.save_actions_count
        ):
            # quit the application when the action recording queue is full
            break
        if render_steps_limit is not None and update_idx > render_steps_limit:
            break

        if args.no_render:
            keys = defaultdict(lambda: False)
        else:
            keys = pygame.key.get_pressed()

        do_update = True
        if keys[pygame.K_w]:
            goal_location += mn.Vector3([0.05, 0, 0])
        elif keys[pygame.K_s]:
            goal_location += mn.Vector3([-0.05, 0, 0])
        elif keys[pygame.K_a]:
            goal_location += mn.Vector3([0, 0, 0.05])
        elif keys[pygame.K_d]:
            goal_location += mn.Vector3([0, 0, -0.05])
        else:
            do_update = False

        if do_update:
            repeat_walk = True
            agent_location = human_controller.translation_offset
            # agent_location = env._sim.robot.translation_offset
            found_path, goal_location, path = update_location_walk(agent_location, env, curr_ind_map, goal_location)
            path_ind = 1
        
        if not args.no_render and is_multi_agent and keys[pygame.K_x]:
            agent_to_control += 1
            agent_to_control = agent_to_control % len(env._sim.robots_mgr)
            logger.info(
                f"Controlled agent changed. Controlling agent {agent_to_control}."
            )

        delta_dist = 0.1
        # dist = (path.points[path_ind] - env._sim.robot.translation_offset) * (mn.Vector3.x_axis() + mn.Vector3.z_axis())
        
        # Get the thing below back
        dist = (path.points[path_ind] - human_controller.translation_offset) * (mn.Vector3.x_axis() + mn.Vector3.z_axis())
        if np.linalg.norm(dist) < delta_dist:
            
            path_ind = min(path_ind+1, len(path.points) - 1)
                
        step_result, arm_action, end_ep, repeat_walk = get_input_vel_ctlr(
            human_controller,
            args.no_render,
            use_arm_actions[update_idx]
            if use_arm_actions is not None
            else None,
            env,
            not free_cam.is_free_cam_mode,
            agent_to_control,
            path,
            path_ind,
            repeat_walk
        )

        if not args.no_render and keys[pygame.K_c]:
            pddl_action = env.task.actions["PDDL_APPLY_ACTION"]
            logger.info("Actions:")
            actions = pddl_action._action_ordering
            for i, action in enumerate(actions):
                logger.info(f"{i}: {action}")
            entities = pddl_action._entities_list
            logger.info("Entities")
            for i, entity in enumerate(entities):
                logger.info(f"{i}: {entity}")
            action_sel = input("Enter Action Selection: ")
            entity_sel = input("Enter Entity Selection: ")
            action_sel = int(action_sel)
            entity_sel = [int(x) + 1 for x in entity_sel.split(",")]
            ac = np.zeros(pddl_action.action_space["pddl_action"].shape[0])
            ac_start = pddl_action.get_pddl_action_start(action_sel)
            ac[ac_start : ac_start + len(entity_sel)] = entity_sel

            step_env(env, "PDDL_APPLY_ACTION", {"pddl_action": ac})

        if not args.no_render and keys[pygame.K_g]:
            pred_list = env.task.sensor_suite.sensors[
                "all_predicates"
            ]._predicates_list
            pred_values = step_result["all_predicates"]
            logger.info("\nPredicate Truth Values:")
            for i, (pred, pred_value) in enumerate(
                zip(pred_list, pred_values)
            ):
                logger.info(f"{i}: {pred.compact_str} = {pred_value}")

        if step_result is None:
            break

        if end_ep:
            total_reward = 0
            # Clear the saved keyframes.
            if gfx_measure is not None:
                gfx_measure.get_metric(force_get=True)
            env.reset()

        if not args.no_render:
            step_result = free_cam.update(env, step_result, update_idx)

        all_arm_actions.append(arm_action)
        update_idx += 1
        if use_arm_actions is not None and update_idx >= len(use_arm_actions):
            break

        obs = step_result
        info = env.get_metrics()
        reward_key = [k for k in info if "reward" in k]
        if len(reward_key) > 0:
            reward = info[reward_key[0]]
        else:
            reward = 0.0

        total_reward += reward
        info["Total Reward"] = total_reward

        if free_cam.is_free_cam_mode:
            cam = obs["robot_third_rgb"]
            use_ob = np.zeros(draw_obs.shape)
            use_ob[:, : cam.shape[1]] = cam[:, :, :3]

        else:
            use_ob = observations_to_image(obs, info)
            if not args.skip_render_text:
                use_ob = overlay_frame(use_ob, info)

        draw_ob = use_ob[:]

        if not args.no_render:
            draw_ob = np.transpose(draw_ob, (1, 0, 2))
            draw_obuse_ob = pygame.surfarray.make_surface(draw_ob)
            screen.blit(draw_obuse_ob, (0, 0))
            pygame.display.update()
        if args.save_obs:
            all_obs.append(draw_ob)

        if not args.no_render:
            pygame.event.pump()
        if env.episode_over:
            total_reward = 0
            env.reset()

        curr_time = time.time()
        diff = curr_time - prev_time
        delay = max(1.0 / target_fps - diff, 0)
        time.sleep(delay)
        prev_time = curr_time

    if args.save_actions:
        if len(all_arm_actions) < args.save_actions_count:
            raise ValueError(
                f"Only did {len(all_arm_actions)} actions but {args.save_actions_count} are required"
            )
        all_arm_actions = np.array(all_arm_actions)[: args.save_actions_count]
        os.makedirs(SAVE_ACTIONS_DIR, exist_ok=True)
        save_path = osp.join(SAVE_ACTIONS_DIR, args.save_actions_fname)
        with open(save_path, "wb") as f:
            np.save(f, all_arm_actions)
        logger.info(f"Saved actions to {save_path}")
        pygame.quit()
        return

    if args.save_obs:
        all_obs = np.array(all_obs)
        all_obs = np.transpose(all_obs, (0, 2, 1, 3))
        os.makedirs(SAVE_VIDEO_DIR, exist_ok=True)
        vut.make_video(
            np.expand_dims(all_obs, 1),
            0,
            "color",
            osp.join(SAVE_VIDEO_DIR, args.save_obs_fname),
        )
    if gfx_measure is not None:
        gfx_str = gfx_measure.get_metric(force_get=True)
        write_gfx_replay(
            gfx_str, config.habitat.task, env.current_episode.episode_id
        )

    if not args.no_render:
        pygame.quit()


def has_pygame():
    return pygame is not None


if __name__ == "__main__":
    # breakpoint()
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-render", action="store_true", default=False)
    parser.add_argument("--save-obs", action="store_true", default=False)
    parser.add_argument("--save-obs-fname", type=str, default="play.mp4")
    parser.add_argument("--save-actions", action="store_true", default=False)
    parser.add_argument(
        "--save-actions-fname", type=str, default="play_actions.txt"
    )
    parser.add_argument(
        "--save-actions-count",
        type=int,
        default=200,
        help="""
            The number of steps the saved action trajectory is clipped to. NOTE
            the episode must be at least this long or it will terminate with
            error.
            """,
    )
    parser.add_argument("--play-cam-res", type=int, default=512)
    parser.add_argument(
        "--skip-render-text", action="store_true", default=False
    )
    parser.add_argument(
        "--same-task",
        action="store_true",
        default=False,
        help="If true, then do not add the render camera for better visualization",
    )
    parser.add_argument(
        "--skip-task",
        action="store_true",
        default=False,
        help="If true, then do not add the render camera for better visualization",
    )
    parser.add_argument(
        "--never-end",
        action="store_true",
        default=False,
        help="If true, make the task never end due to reaching max number of steps",
    )
    parser.add_argument(
        "--add-ik",
        action="store_true",
        default=False,
        help="If true, changes arm control to IK",
    )
    parser.add_argument(
        "--gfx",
        action="store_true",
        default=False,
        help="Save a GFX replay file.",
    )
    parser.add_argument("--load-actions", type=str, default=None)
    parser.add_argument("--cfg", type=str, default=DEFAULT_CFG)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()
    if not has_pygame() and not args.no_render:
        raise ImportError(
            "Need to install PyGame (run `pip install pygame==2.0.1`)"
        )
    config = habitat.get_config(args.cfg, args.opts)
    with habitat.config.read_write(config):
        env_config = config.habitat.environment
        sim_config = config.habitat.simulator
        task_config = config.habitat.task

        if not args.same_task:
            sim_config.debug_render = True
            sim_config.agent_0.sim_sensors.update(
                {
                    "third_rgb_sensor": ThirdRGBSensorConfig(
                        height=args.play_cam_res, width=args.play_cam_res
                    )
                }
            )
            if "composite_success" in task_config.measurements:
                task_config.measurements.composite_success.must_call_stop = (
                    False
                )
            if "rearrange_nav_to_obj_success" in task_config.measurements:
                task_config.measurements.rearrange_nav_to_obj_success.must_call_stop = (
                    False
                )
            if "force_terminate" in task_config.measurements:
                task_config.measurements.force_terminate.max_accum_force = -1.0
                task_config.measurements.force_terminate.max_instant_force = (
                    -1.0
                )
        if args.gfx:
            sim_config.habitat_sim_v0.enable_gfx_replay_save = True
            task_config.measurements.update(
                {"gfx_replay_measure": GfxReplayMeasureMeasurementConfig()}
            )

        if args.never_end:
            env_config.max_episode_steps = 0

        if args.add_ik:
            if "arm_action" not in task_config.actions:
                raise ValueError(
                    "Action space does not have any arm control so incompatible with `--add-ik` option"
                )
            sim_config.agent_0.ik_arm_urdf = (
                "./data/robots/hab_fetch/robots/fetch_onlyarm.urdf"
            )
            task_config.actions.arm_action.arm_controller = "ArmEEAction"
    # breakpoint()
    with habitat.Env(config=config) as env:
        # breakpoint()
        play_env(env, args, config)
