#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
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

Change the task with `--cfg configs/tasks/rearrange/close_cab.yaml` (choose any task under the `configs/tasks/rearrange/` folder).

Change the grip type:
- Suction gripper `TASK.ACTIONS.ARM_ACTION.GRIP_CONTROLLER "SuctionGraspAction"`

To record a video: `--save-obs` This will save the video to file under `data/vids/` specified by `--save-obs-fname` (by default `vid.mp4`).

Record and play back trajectories:
- To record a trajectory add `--save-actions --save-actions-count 200` to
  record a truncated episode length of 200.
- By default the trajectories are saved to data/interactive_play_replays/play_actions.txt
- Play the trajectories back with `--load-actions data/interactive_play_replays/play_actions.txt`
"""

import argparse
import os
import os.path as osp
import time
from collections import defaultdict

import magnum as mn
import matplotlib.pyplot as plt
import numpy as np

import habitat
import habitat.tasks.rearrange.rearrange_task
import habitat_sim
from habitat.core.logging import logger
from habitat.tasks.rearrange.actions.actions import ArmEEAction
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import euler_to_quat, write_gfx_replay
from habitat.utils.render_wrapper import overlay_frame
from habitat.utils.visualizations.utils import observations_to_image
from habitat_sim.utils import viz_utils as vut

try:
    import pygame
except ImportError:
    pygame = None

# DEFAULT_CFG = "configs/tasks/rearrange/play_spot.yaml"
# DEFAULT_CFG = (
#     "configs/tasks/rearrange/play_stretch_gripper_roll_pitch_yaw.yaml"
# )
# DEFAULT_CFG = "configs/tasks/rearrange/play.yaml"
DEFAULT_RENDER_STEPS_LIMIT = 60
AGENT_RADIUS = 0.1
MULTIPLIER_OF_POINTS = 3
SAVE_VIDEO_DIR = "./data/vids"
SAVE_ACTIONS_DIR = "./data/interactive_play_replays"

import os

# Quiet the Habitat simulator logging
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"


def check_nav(env):

    # Get the navmesh setting.

    for r in [
        0.001,
        0.01,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
    ]:
        navmesh_settings = env._sim.pathfinder.nav_mesh_settings
        navmesh_settings.agent_radius = r
        env._sim.recompute_navmesh(env._sim.pathfinder, navmesh_settings, True)
        pts = env._sim.pathfinder.build_navmesh_vertices()
        area = env._sim.pathfinder.navigable_area
        num_islands = env._sim.pathfinder.num_islands
        print(
            "r:",
            r,
            "true r:",
            navmesh_settings.agent_radius,
            "area:",
            area,
            "islands:",
            num_islands,
            "pts:",
            len(pts),
        )

    # origin_pts = env._sim.pathfinder.build_navmesh_vertices()

    # print("pts:", len(pts))
    # navmesh_settings = env._sim.pathfinder.nav_mesh_settings
    # print(env._sim.pathfinder.nav_mesh_settings.agent_radius)
    # navmesh_settings.agent_radius *= 1.1

    # print(env._sim.pathfinder.nav_mesh_settings.agent_radius)

    # pts = env._sim.pathfinder.build_navmesh_vertices()

    # num_is_nav_pt = 0
    # for pt in pts:
    #     num_is_nav_pt += env._sim.pathfinder.is_navigable(pt)
    # print("pertange:", num_is_nav_pt, float(num_is_nav_pt)/float(len(pts)))

    navmesh_settings = env._sim.pathfinder.nav_mesh_settings
    navmesh_settings.agent_radius = AGENT_RADIUS
    env._sim.recompute_navmesh(env._sim.pathfinder, navmesh_settings, True)
    return env


def step_env(env, action_name, action_args):
    return env.step({"action": action_name, "action_args": action_args})


def get_input_vel_ctlr(
    skip_pygame, arm_action, env, not_block_input, agent_to_control
):
    if skip_pygame:
        return step_env(env, "EMPTY", {}), None, False
    multi_agent = len(env._sim.robots_mgr) > 1

    arm_action_name = "ARM_ACTION"
    base_action_name = "BASE_VELOCITY"
    arm_key = "arm_action"
    grip_key = "grip_action"
    base_key = "base_vel"
    if multi_agent:
        agent_k = f"AGENT_{agent_to_control}"
        arm_action_name = f"{agent_k}_{arm_action_name}"
        base_action_name = f"{agent_k}_{base_action_name}"
        arm_key = f"{agent_k}_{arm_key}"
        grip_key = f"{agent_k}_{grip_key}"
        base_key = f"{agent_k}_{base_key}"

    if arm_action_name in env.action_space.spaces:
        arm_action_space = env.action_space.spaces[arm_action_name].spaces[
            arm_key
        ]
        arm_ctrlr = env.task.actions[arm_action_name].arm_ctrlr
        base_action = None
    else:
        arm_action_space = np.zeros(7)
        arm_ctrlr = None
        base_action = [0, 0]

    if arm_action is None:
        arm_action = np.zeros(arm_action_space.shape[0])
        given_arm_action = False
    else:
        given_arm_action = True

    end_ep = False
    magic_grasp = None

    keys = pygame.key.get_pressed()

    if keys[pygame.K_ESCAPE]:
        return None, None, False
    elif keys[pygame.K_m]:
        end_ep = True
    elif keys[pygame.K_n]:
        env._sim.navmesh_visualization = not env._sim.navmesh_visualization
    env._sim.navmesh_visualization = True

    if not_block_input:
        # Base control
        if keys[pygame.K_j]:
            # Left
            base_action = [0, 1]
        elif keys[pygame.K_l]:
            # Right
            base_action = [0, -1]
        elif keys[pygame.K_k]:
            # Back
            base_action = [-1, 0]
        elif keys[pygame.K_i]:
            # Forward
            base_action = [1, 0]
        if arm_action_space.shape[0] == 7:
            # Velocity control. A different key for each joint
            if keys[pygame.K_q]:
                arm_action[0] = 1.0
            elif keys[pygame.K_1]:
                arm_action[0] = -1.0

            elif keys[pygame.K_w]:
                arm_action[1] = 1.0
            elif keys[pygame.K_2]:
                arm_action[1] = -1.0

            elif keys[pygame.K_e]:
                arm_action[2] = 1.0
            elif keys[pygame.K_3]:
                arm_action[2] = -1.0

            elif keys[pygame.K_r]:
                arm_action[3] = 1.0
            elif keys[pygame.K_4]:
                arm_action[3] = -1.0

            elif keys[pygame.K_t]:
                arm_action[4] = 1.0
            elif keys[pygame.K_5]:
                arm_action[4] = -1.0

            elif keys[pygame.K_y]:
                arm_action[5] = 1.0
            elif keys[pygame.K_6]:
                arm_action[5] = -1.0

            elif keys[pygame.K_u]:
                arm_action[6] = 1.0
            elif keys[pygame.K_7]:
                arm_action[6] = -1.0
        elif arm_action_space.shape[0] == 8:
            # Velocity control. A different key for each joint

            if keys[pygame.K_q]:
                arm_action[0] = 1.0
            elif keys[pygame.K_1]:
                arm_action[0] = -1.0

            elif keys[pygame.K_w]:
                arm_action[1] = 1.0
            elif keys[pygame.K_2]:
                arm_action[1] = -1.0

            elif keys[pygame.K_e]:
                arm_action[2] = 1.0
            elif keys[pygame.K_3]:
                arm_action[2] = -1.0

            elif keys[pygame.K_r]:
                arm_action[3] = 1.0
            elif keys[pygame.K_4]:
                arm_action[3] = -1.0

            elif keys[pygame.K_t]:
                arm_action[4] = 1.0
            elif keys[pygame.K_5]:
                arm_action[4] = -1.0

            elif keys[pygame.K_y]:
                arm_action[5] = 1.0
            elif keys[pygame.K_6]:
                arm_action[5] = -1.0

            elif keys[pygame.K_u]:
                arm_action[6] = 1.0
            elif keys[pygame.K_7]:
                arm_action[6] = -1.0

            elif keys[pygame.K_8]:
                arm_action[7] = 1.0
            elif keys[pygame.K_9]:
                arm_action[7] = -1.0
        elif arm_action_space.shape[0] == 10:
            # Velocity control. A different key for each joint
            # arm_control, arm_joints index, name
            # 0 28: joint_arm_l0
            # 1 27: joint_arm_l1
            # 2 26: joint_arm_l2
            # 3 25: joint_arm_l3
            # 4 23: joint_lift
            # 5 31: joint_wrist_yaw
            # 6 39: joint_wrist_pitch
            # 7 40: joint_wrist_roll
            # 8 7: joint_head_pan
            # 9 8: joint_head_tilt

            if keys[
                pygame.K_q
            ]:  # joint_arm_l0, joint_arm_l1, joint_arm_l2, joint_arm_l3
                arm_action[0] = 1.0
            elif keys[pygame.K_1]:
                arm_action[0] = -1.0

            elif keys[pygame.K_w]:  # joint_lift
                arm_action[4] = 1.0
            elif keys[pygame.K_2]:
                arm_action[4] = -1.0

            elif keys[pygame.K_e]:  # joint_wrist_yaw
                arm_action[5] = 1.0
            elif keys[pygame.K_3]:
                arm_action[5] = -1.0

            elif keys[pygame.K_r]:  # joint_wrist_pitch
                arm_action[6] = 1.0
            elif keys[pygame.K_4]:
                arm_action[6] = -1.0

            elif keys[pygame.K_t]:  # joint_wrist_roll
                arm_action[7] = 1.0
            elif keys[pygame.K_5]:
                arm_action[7] = -1.0

            elif keys[pygame.K_y]:  # joint_head_pan
                arm_action[8] = 1.0
            elif keys[pygame.K_6]:
                arm_action[8] = -1.0

            elif keys[pygame.K_u]:  # joint_head_tilt
                arm_action[9] = 1.0
            elif keys[pygame.K_7]:
                arm_action[9] = -1.0

        elif isinstance(arm_ctrlr, ArmEEAction):
            EE_FACTOR = 0.5
            # End effector control
            if keys[pygame.K_d]:
                arm_action[1] -= EE_FACTOR
            elif keys[pygame.K_a]:
                arm_action[1] += EE_FACTOR
            elif keys[pygame.K_w]:
                arm_action[0] += EE_FACTOR
            elif keys[pygame.K_s]:
                arm_action[0] -= EE_FACTOR
            elif keys[pygame.K_q]:
                arm_action[2] += EE_FACTOR
            elif keys[pygame.K_e]:
                arm_action[2] -= EE_FACTOR
        else:
            raise ValueError("Unrecognized arm action space")

        if keys[pygame.K_p]:
            logger.info("[play.py]: Unsnapping")
            # Unsnap
            magic_grasp = -1
        elif keys[pygame.K_o]:
            # Snap
            logger.info("[play.py]: Snapping")
            magic_grasp = 1

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
    # import pdb; pdb.set_trace()
    return step_env(env, name, args), arm_action, end_ep


def get_wrapped_prop(venv, prop):
    if hasattr(venv, prop):
        return getattr(venv, prop)
    elif hasattr(venv, "venv"):
        return get_wrapped_prop(venv.venv, prop)
    elif hasattr(venv, "env"):
        return get_wrapped_prop(venv.env, prop)

    return None


def setup_path_visualization(points):

    obj_attr_mgr = env.sim.get_object_template_manager()
    rigid_obj_mgr = env.sim.get_rigid_object_manager()
    vis_objs = []
    sphere_handle = obj_attr_mgr.get_template_handles("uvSphereSolid")[0]
    sphere_template_cpy = obj_attr_mgr.get_template_by_handle(sphere_handle)
    sphere_template_cpy.scale *= 0.2
    template_id = obj_attr_mgr.register_template(
        sphere_template_cpy, "mini-sphere"
    )

    if template_id < 0:
        return None
    vis_objs.append(rigid_obj_mgr.add_object_by_template_handle(sphere_handle))

    # Here is the place you should add your waypoints
    # The path_follower._points give you
    # [array([ 3.1160288 ,  0.22979212, -1.0067971 ], dtype=float32),
    #  array([2.1882367 , 0.15225519, 0.6902213 ], dtype=float32),
    #  array([1.7598894 , 0.15225519, 7.133915  ], dtype=float32)]
    for point in points:
        cp_obj = rigid_obj_mgr.add_object_by_template_handle(sphere_handle)
        # cp_obj = rigid_obj_mgr.add_object_by_template_handle("mini-sphere")
        if cp_obj.object_id < 0:
            print(cp_obj.object_id)
            return None
        cp_obj.translation = point
        vis_objs.append(cp_obj)

    for obj in vis_objs:
        if obj.object_id < 0:
            print(obj.object_id)
            return None

    for obj in vis_objs:
        obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC

    return vis_objs


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

    # Motified the nav mesh
    check_nav(env)
    pts = env._sim.pathfinder.build_navmesh_vertices()
    cur_num_pts = len(pts)
    for i in range(int(cur_num_pts * MULTIPLIER_OF_POINTS)):
        pts.append(env._sim.pathfinder.get_random_navigable_point())
    # random.shuffle(pts)
    pt_i = 0

    init_height = env.sim.robot.base_pos[1]

    num_non_contact_pts = 0

    # Get the x y locations of the visited points.
    plot_x = []
    plot_y = []
    plot_c = []
    plot_c_2 = []

    vis_pts = []
    while pt_i < len(pts):
        # Record if that point can be placed without any contact on all orientations
        no_contact_all_orientation = True
        no_contact_at_least_one_orientation = 0
        for rotation_y_rad in [
            0,
            np.pi * 0.25,
            np.pi * 0.5,
            np.pi * 0.75,
            np.pi * 1.0,
            -np.pi * 0.25,
            -np.pi * 0.5,
            -np.pi * 0.75,
        ]:
            env.reset()
            # Get the point
            pt = pts[pt_i]
            # Motify the base pos
            robot_base_pos = env.sim.robot.base_pos
            robot_base_pos[0] = pt[0]
            robot_base_pos[1] = init_height
            robot_base_pos[2] = pt[2]
            env.sim.robot.base_pos = robot_base_pos
            # Motify the base rot
            env.sim.robot.base_rot = rotation_y_rad

            is_contact = env.sim.contact_test(env.sim.robot.get_robot_sim_id())
            num_non_contact_pts += not is_contact

            if is_contact:
                no_contact_all_orientation = False
            if not is_contact:
                no_contact_at_least_one_orientation += 1

            if (
                args.save_actions
                and len(all_arm_actions) > args.save_actions_count
            ):
                # quit the application when the action recording queue is full
                break
            if (
                render_steps_limit is not None
                and update_idx > render_steps_limit
            ):
                break

            if args.no_render:
                keys = defaultdict(lambda: False)
            else:
                keys = pygame.key.get_pressed()

            if not args.no_render and is_multi_agent and keys[pygame.K_x]:
                agent_to_control += 1
                agent_to_control = agent_to_control % len(env._sim.robots_mgr)
                logger.info(
                    f"Controlled agent changed. Controlling agent {agent_to_control}."
                )

            step_result, arm_action, end_ep = get_input_vel_ctlr(
                args.no_render,
                use_arm_actions[update_idx]
                if use_arm_actions is not None
                else None,
                env,
                not free_cam.is_free_cam_mode,
                agent_to_control,
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
            if use_arm_actions is not None and update_idx >= len(
                use_arm_actions
            ):
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

        pt_i += 1

        # Store the location
        plot_x.append(pt[0])
        plot_y.append(pt[2])
        plot_c.append(int(no_contact_all_orientation))
        if no_contact_at_least_one_orientation > 0:
            plot_c_2.append(1)
        else:
            plot_c_2.append(0)
        if not no_contact_all_orientation:
            robot_base_array = np.array(
                [
                    env.sim.robot.base_pos[0],
                    env.sim.robot.base_pos[1],
                    env.sim.robot.base_pos[2],
                ]
            )
            vis_pts.append(robot_base_array)

    print("====Result===")
    print("num_non_contact_pts:", num_non_contact_pts)
    print("total_num_nav_mesh_pts:", len(pts) * 8)
    print(
        "pertange of nav pts:",
        float(num_non_contact_pts) / float(len(pts) * 8),
    )

    # Plot the figure for visualization.

    plot_x_safe = []
    plot_y_safe = []
    plot_x_contact = []
    plot_y_contact = []
    for c_i in range(len(plot_c)):
        if plot_c[c_i]:
            plot_x_safe.append(plot_x[c_i])
            plot_y_safe.append(plot_y[c_i])
        else:
            plot_x_contact.append(plot_x[c_i])
            plot_y_contact.append(plot_y[c_i])

    plot_x_safe_2 = []
    plot_y_safe_2 = []
    plot_x_contact_2 = []
    plot_y_contact_2 = []
    for c_i in range(len(plot_c_2)):
        if plot_c_2[c_i]:
            plot_x_safe_2.append(plot_x[c_i])
            plot_y_safe_2.append(plot_y[c_i])
        else:
            plot_x_contact_2.append(plot_x[c_i])
            plot_y_contact_2.append(plot_y[c_i])

    fig, ax = plt.subplots()
    plt.figure(figsize=(7, 7), dpi=300)
    percent = (
        len(plot_x_safe) / (len(plot_x_safe) + len(plot_x_contact)) * 100.0
    )
    plt.scatter(
        plot_x_safe,
        plot_y_safe,
        c="green",
        label="no contact: "
        + str(len(plot_x_safe))
        + ", "
        + str(percent)
        + "%",
    )
    percent = (
        len(plot_x_contact) / (len(plot_x_safe) + len(plot_x_contact)) * 100.0
    )
    plt.scatter(
        plot_x_contact,
        plot_y_contact,
        c="red",
        label="contact: "
        + str(len(plot_x_contact))
        + ", "
        + str(percent)
        + "%",
    )
    # plt.legend(loc='upper left', numpoints=1, ncol=1, fontsize=8, bbox_to_anchor=(0, 0))
    plt.legend(ncol=1, fontsize=15)
    if "spot" in DEFAULT_CFG:
        SAVE_NAME = "spot"
    elif "stretch" in DEFAULT_CFG:
        SAVE_NAME = "stretch"
    else:
        SAVE_NAME = "fetch"
    plt.title(SAVE_NAME, fontsize=20)
    SAVE_NAME = SAVE_NAME + "_radius" + str(AGENT_RADIUS)
    plt.xlabel(r"x", fontsize=20)
    plt.ylabel(r"y", fontsize=20)
    plt.savefig(
        "/Users/jimmytyyang/Documents/"
        + SAVE_NAME
        + "_area_1214_no_contact_in_all_orientation.png"
    )

    fig, ax = plt.subplots()
    plt.figure(figsize=(7, 7), dpi=300)
    percent = (
        len(plot_x_safe_2)
        / (len(plot_x_safe_2) + len(plot_x_contact_2))
        * 100.0
    )
    plt.scatter(
        plot_x_safe_2,
        plot_y_safe_2,
        c="green",
        label="no contact: "
        + str(len(plot_x_safe_2))
        + ", "
        + str(percent)
        + "%",
    )
    percent = (
        len(plot_x_contact_2)
        / (len(plot_x_safe_2) + len(plot_x_contact_2))
        * 100.0
    )
    plt.scatter(
        plot_x_contact_2,
        plot_y_contact_2,
        c="red",
        label="contact: "
        + str(len(plot_x_contact_2))
        + ", "
        + str(percent)
        + "%",
    )
    # plt.legend(loc='upper left', numpoints=1, ncol=1, fontsize=8, bbox_to_anchor=(0, 0))
    plt.legend(ncol=1, fontsize=15)
    if "spot" in DEFAULT_CFG:
        SAVE_NAME = "spot"
    elif "stretch" in DEFAULT_CFG:
        SAVE_NAME = "stretch"
    else:
        SAVE_NAME = "fetch"
    plt.title(SAVE_NAME, fontsize=20)
    SAVE_NAME = SAVE_NAME + "_radius" + str(AGENT_RADIUS)
    plt.xlabel(r"x", fontsize=20)
    plt.ylabel(r"y", fontsize=20)
    plt.savefig(
        "/Users/jimmytyyang/Documents/"
        + SAVE_NAME
        + "_area_1214_no_contact_at_least_in_one_orientation.png"
    )

    # After the running the collision examination. Explore the enviornment again.
    # Setting up the visual points
    vis_objs = setup_path_visualization(vis_pts)
    while True:
        # print(env.sim.robot.base_pos, env.sim.robot.base_rot)
        # print("ee_transform:", env.sim.robot.ee_transform.translation)
        trans = env.sim.robot.base_transformation
        ee_pos = env.sim.robot.ee_transform.translation
        local_ee_pos = trans.inverted().transform_point(ee_pos)
        # print(
        #     "@interactive_play.py: env.sim.robot.arm_joint_pos:",
        #     env.sim.robot.arm_joint_pos,
        # )
        # print(
        #     "@interactive_play.py: env.sim.robot.arm_motor_pos:",
        #     env.sim.robot.arm_motor_pos,
        # )
        # print("@interactive_play.py: arm_joint_angle",env.sim.robot)
        print(
            "@interactive_play.py, location of robot:",
            env.sim.robot.base_transformation.translation,
        )
        # print("@interactive_play.py, rotation of robot:", env.sim.robot.sim_obj.rotation)
        # print("rel target pos:", rel_targ_pos)
        # import pdb; pdb.set_trace()
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

        if not args.no_render and is_multi_agent and keys[pygame.K_x]:
            agent_to_control += 1
            agent_to_control = agent_to_control % len(env._sim.robots_mgr)
            logger.info(
                f"Controlled agent changed. Controlling agent {agent_to_control}."
            )

        step_result, arm_action, end_ep = get_input_vel_ctlr(
            args.no_render,
            use_arm_actions[update_idx]
            if use_arm_actions is not None
            else None,
            env,
            not free_cam.is_free_cam_mode,
            agent_to_control,
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

            for vis_obj in vis_objs:
                env.sim.get_rigid_object_manager().remove_object_by_id(
                    vis_obj.object_id
                )
            vis_objs = setup_path_visualization(vis_pts)

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
        write_gfx_replay(gfx_str, config.TASK, env.current_episode.episode_id)

    if not args.no_render:
        pygame.quit()


def has_pygame():
    return pygame is not None


if __name__ == "__main__":
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
    config.defrost()
    if not args.same_task:
        config.SIMULATOR.THIRD_RGB_SENSOR.WIDTH = args.play_cam_res
        config.SIMULATOR.THIRD_RGB_SENSOR.HEIGHT = args.play_cam_res
        config.SIMULATOR.AGENT_0.SENSORS.append("THIRD_RGB_SENSOR")
        config.SIMULATOR.DEBUG_RENDER = True
        config.TASK.COMPOSITE_SUCCESS.MUST_CALL_STOP = False
        config.TASK.REARRANGE_NAV_TO_OBJ_SUCCESS.MUST_CALL_STOP = False
        config.TASK.FORCE_TERMINATE.MAX_ACCUM_FORCE = -1.0
        config.TASK.FORCE_TERMINATE.MAX_INSTANT_FORCE = -1.0
    if args.gfx:
        config.SIMULATOR.HABITAT_SIM_V0.ENABLE_GFX_REPLAY_SAVE = True
        config.TASK.MEASUREMENTS.append("GFX_REPLAY_MEASURE")
    if args.never_end:
        config.ENVIRONMENT.MAX_EPISODE_STEPS = 0
    if args.add_ik:
        if "ARM_ACTION" not in config.TASK.ACTIONS:
            raise ValueError(
                "Action space does not have any arm control so incompatible with `--add-ik` option"
            )
        config.TASK.ACTIONS.ARM_ACTION.ARM_CONTROLLER = "ArmEEAction"
        config.SIMULATOR.IK_ARM_URDF = (
            "data/robots/hab_spot_arm/urdf/hab_spot_onlyarm.urdf"
        )
    config.freeze()

    with habitat.Env(config=config) as env:
        play_env(env, args, config)
