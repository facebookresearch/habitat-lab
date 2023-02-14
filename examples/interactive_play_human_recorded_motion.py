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
from typing import Any, Dict, List

import magnum as mn
import numpy as np
from human_controllers import AmassHumanController, SeqPoseHumanController

import habitat
import habitat.tasks.rearrange.rearrange_task
import habitat_sim
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    GfxReplayMeasureMeasurementConfig,
    ThirdDepthSensorConfig,
    ThirdRGBSensorConfig,
)
from habitat.core.logging import logger

# from habitat.tasks.rearrange.actions.actions import WalkAction
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import euler_to_quat, write_gfx_replay
from habitat.utils.render_wrapper import overlay_frame
from habitat.utils.visualizations.utils import observations_to_image
from habitat_sim.utils import viz_utils as vut

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

    distance = np.linalg.norm(
        (agent_controller.translation_offset - final_point)
        * (mn.Vector3.x_axis() + mn.Vector3.z_axis())
    )
    if distance < 0.1:
        return True
    return False


def compute_displ(next_point, agent_controller):
    diff_dist = next_point - agent_controller.translation_offset

    return [diff_dist[0], diff_dist[2]]


def get_input_vel_ctlr(
    human_controller,
    skip_pygame,
    arm_action,
    env,
    not_block_input,
    agent_to_control,
    agent_path,
    path_ind,
):
    not_block_input = True
    if skip_pygame:
        return step_env(env, "empty", {}), None, False
    multi_agent = len(env._sim.agents_mgr) > 1
    # print(env.task.actions)
    new_pose, new_trans = human_controller.pose_per_frame()

    base_action = AmassHumanController.transformAction(new_pose, new_trans)
    base_action_name = "humanjoint_action"
    base_key = "human_joints_trans"

    end_ep = False
    magic_grasp = None

    keys = pygame.key.get_pressed()

    if keys[pygame.K_ESCAPE]:
        return None, None, False
    # elif keys[pygame.K_m]:
    #     end_ep = True
    # elif keys[pygame.K_n]:
    #     env._sim.navmesh_visualization = not env._sim.navmesh_visualization

    if not_block_input:
        if keys[pygame.K_c]:
            # Left
            # ee_pos = env._sim.robot.ee_transform.translation
            # print(ee_pos)
            # base_action_name = 'next_frame'
            base_action = mn.Vector3([0, 0, 0.05])
            human_controller.next_frame()

            base_key = "human_joints_trans"
            new_pose, new_trans = human_controller.pose_per_frame()
            base_action = AmassHumanController.transformAction(
                new_pose, new_trans
            )

        elif keys[pygame.K_v]:
            # Right
            # base_action_name = 'prev_frame'
            base_action = mn.Vector3([0, 0, -0.05])
            env._sim.agent.curr_trans = base_action
            human_controller.prev_frame()

            base_key = "human_joints_trans"
            new_pose, new_trans = human_controller.pose_per_frame()
            base_action = AmassHumanController.transformAction(
                new_pose, new_trans
            )

        if keys[pygame.K_n]:
            # Left
            # ee_pos = env._sim.robot.ee_transform.translation
            # print(ee_pos)
            # base_action_name = 'next_frame'
            base_action = mn.Vector3([0, 0, 0.05])
            human_controller.prev_video()

            base_key = "human_joints_trans"
            new_pose, new_trans = human_controller.pose_per_frame()
            base_action = AmassHumanController.transformAction(
                new_pose, new_trans
            )

        elif keys[pygame.K_m]:
            # Right
            # base_action_name = 'prev_frame'
            base_action = mn.Vector3([0, 0, -0.05])
            env._sim.agent.curr_trans = base_action
            human_controller.next_video()

            base_key = "human_joints_trans"
            new_pose, new_trans = human_controller.pose_per_frame()
            base_action = AmassHumanController.transformAction(
                new_pose, new_trans
            )
            # print(base_action)
            # breakpoint()

    if keys[pygame.K_PERIOD]:
        # Print the current position of the robot, useful for debugging.
        pos = [float("%.3f" % x) for x in env._sim.agent.sim_obj.translation]
        rot = env._sim.agent.sim_obj.rotation
        ee_pos = env._sim.agent.ee_transform.translation
        logger.info(
            f"Robot state: pos = {pos}, rotation = {rot}, ee_pos = {ee_pos}"
        )
    elif keys[pygame.K_COMMA]:
        # Print the current arm state of the robot, useful for debugging.
        joint_state = [float("%.3f" % x) for x in env._sim.agent.arm_joint_pos]
        logger.info(f"Robot arm joint state: {joint_state}")

    args: Dict[str, Any] = {}
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

    # if magic_grasp is None:
    #     arm_action = [*arm_action, 0.0]
    # else:
    #     arm_action = [*arm_action, magic_grasp]
    # # breakpoint()
    arm_action = []
    return step_env(env, name, args), arm_action, end_ep


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
        self._is_free_cam_mode = True
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
    all_arm_actions: List[float] = []
    agent_to_control = 0

    env._sim.agent.sim_obj.translation += mn.Vector3([-5.0, -1.0, 0])

    free_cam = FreeCamHelper()
    gfx_measure = env.task.measurements.measures.get(
        GfxReplayMeasure.cls_uuid, None
    )
    is_multi_agent = len(env._sim.agents_mgr) > 1
    env._sim.agent.translation_offset = (
        env._sim.agent.sim_obj.translation + mn.Vector3([0, 0.9, 0])
    )
    agent_location = env._sim.agent.translation_offset
    # print(agent_location)

    urdf_path = config.habitat.simulator.agents.main_agent.agent_urdf
    amass_path = config.habitat.simulator.agents.main_agent.amass_path
    body_model_path = (
        config.habitat.simulator.agents.main_agent.body_model_path
    )
    draw_fps = config.habitat.simulator.agents.main_agent.draw_fps_human
    obj_translation = env._sim.agent.sim_obj.translation
    grab_path = config.habitat.simulator.agents.main_agent.grab_path

    link_ids = env._sim.agent.sim_obj.get_link_ids()
    # pose_path = '/Users/xavierpuig/sample00_rep00_smpl_params.npy'
    pose_paths = [
        "data/text_motion/sample01_rep02.npz",
        "data/text_motion/sample03_rep01.npz",
        "data/text_motion/sample04_rep01.npz",
    ]
    human_controller = SeqPoseHumanController(
        urdf_path,
        body_model_path,
        pose_paths,
        obj_translation,
        draw_fps=draw_fps,
    )

    # TODO: remove

    human_controller.reset(env._sim.agent.sim_obj.translation)
    # breakpoint()
    path_ind = 1
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
        dist = 0.05

        if not args.no_render and is_multi_agent and keys[pygame.K_x]:
            agent_to_control += 1
            agent_to_control = agent_to_control % len(env._sim.agents_mgr)
            logger.info(
                f"Controlled agent changed. Controlling agent {agent_to_control}."
            )

        delta_dist = 0.1
        # dist = (path.points[path_ind] - env._sim.robot.translation_offset) * (mn.Vector3.x_axis() + mn.Vector3.z_axis())

        step_result, arm_action, end_ep = get_input_vel_ctlr(
            human_controller,
            args.no_render,
            use_arm_actions[update_idx]
            if use_arm_actions is not None
            else None,
            env,
            not free_cam.is_free_cam_mode,
            agent_to_control,
            None,
            0,
        )

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
            all_obs.append(draw_ob)  # type: ignore[assignment]

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
        all_obs = np.array(all_obs)  # type: ignore[assignment]
        all_obs = np.transpose(all_obs, (0, 2, 1, 3))  # type: ignore[assignment]
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
        "--disable-inverse-kinematics",
        action="store_true",
        help="If specified, does not add the inverse kinematics end-effector control.",
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
    # print(args.opts)
    # breakpoint()
    # args.opts=["habitat.task.actions"]
    config = habitat.get_config(args.cfg, args.opts)
    actions_valid = [
        "humanjoint_action",
    ]

    with habitat.config.read_write(config):
        env_config = config.habitat.environment
        sim_config = config.habitat.simulator
        task_config = config.habitat.task

        if not args.same_task:
            sim_config.debug_render = True
            agent_config = get_agent_config(sim_config=sim_config)
            agent_config.sim_sensors.update(
                {
                    "third_rgb_sensor": ThirdRGBSensorConfig(
                        height=args.play_cam_res, width=args.play_cam_res
                    ),
                    "third_depth_sensor": ThirdDepthSensorConfig(
                        height=args.play_cam_res, width=args.play_cam_res
                    ),
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

        if not args.disable_inverse_kinematics:
            if "arm_action" not in task_config.actions:
                raise ValueError(
                    "Action space does not have any arm control so cannot add inverse kinematics. Specify the `--disable-inverse-kinematics` option"
                )
            sim_config.agents.main_agent.ik_arm_urdf = (
                "./data/robots/hab_fetch/robots/fetch_onlyarm.urdf"
            )
            task_config.actions.arm_action.arm_controller = "ArmEEAction"
    with habitat.Env(config=config) as env:
        env.task.actions = {
            key: value
            for key, value in env.task.actions.items()
            if key in actions_valid
        }
        play_env(env, args, config)
