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
import random
import time
from collections import defaultdict

import magnum as mn
import numpy as np

import habitat
import habitat.tasks.rearrange.rearrange_task
import habitat_sim
from habitat.core.logging import logger
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.tasks.rearrange.actions.actions import ArmEEAction
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import euler_to_quat, write_gfx_replay
from habitat.utils.render_wrapper import overlay_frame
from habitat.utils.visualizations.utils import observations_to_image
from habitat_sim.utils import viz_utils as vut
from scipy.spatial.transform import Rotation

try:
    import pygame
except ImportError:
    pygame = None

# DEFAULT_CFG = "configs/tasks/rearrange/play_stretch.yaml"
DEFAULT_CFG = "configs/tasks/rearrange/play_stretch_gripper_roll_pitch_yaw.yaml"
DEFAULT_RENDER_STEPS_LIMIT = 60
SAVE_VIDEO_DIR = "./data/vids"
SAVE_ACTIONS_DIR = "./data/interactive_play_replays"

import os

# Quiet the Habitat simulator logging
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

# How many random way points you want to generate
LEN_WAY_POINT = 1500

# The output limit
V_MAX_DEFAULT = 1.0 #0.45  # base.params["motion"]["default"]["vel_m"]
W_MAX_DEFAULT = 1.0 #0.45  # (vel_m_max - vel_m_default) / wheel_separation_m
# The coeffcienet
ACC_LIN = 2.4 #1.2  # 0.5 * base.params["motion"]["max"]["accel_m"]
ACC_ANG = 2.4 #1.2  # 0.5 * (accel_m_max - accel_m_default) / wheel_separation_m
MAX_HEADING_ANG = np.pi / 10 #/ 2.0

LIN_ERROR_TOL = V_MAX_DEFAULT / 120 # 120
ANG_ERROR_TOL = W_MAX_DEFAULT / 120 # 120

def transform_global_to_base(XYT, current_pose, env=None):
    """
    Transforms the point cloud into geocentric frame to account for
    camera position
    Input:
        XYT                     : target goal ...x3
        current_pose            : base position (x, y, theta (radians))
    Output:
        XYT : ...x3
    """

    trans = env.sim.robot.base_transformation
    local_pos = trans.inverted().transform_point(env.sim.robot.base_pos)
    goal_pos = trans.inverted().transform_point(np.array([XYT[0],env.sim.robot.base_pos[1],XYT[1]]))

    error_t = XYT[2] - current_pose[2]
    error_x = goal_pos[0] - local_pos[0]
    error_y = goal_pos[1] - local_pos[1]

    return [error_x, error_y, error_t]

class Controller:
    def __init__(self, track_yaw=True):
        self.track_yaw = track_yaw

        # Params
        self.v_max = V_MAX_DEFAULT
        self.w_max = W_MAX_DEFAULT


        # Init
        self.xyt_goal = np.zeros(3)
        self.dxyt_goal = np.zeros(3)

    def set_goal(self, goal, vel_goal=None):
        self.xyt_goal = goal
        if vel_goal is not None:
            self.dxyt_goal = vel_goal

    def _compute_error_pose(self, xyt_base, env=None):
        """
        Updates error based on robot localization
        """
        xyt_err = transform_global_to_base(self.xyt_goal, xyt_base, env)
        if not self.track_yaw:
            xyt_err[2] = 0.0


        return xyt_err

    @staticmethod
    def _velocity_feedback_control(x_err, a, v_max):
        """
        Computes velocity based on distance from target.
        Used for both linear and angular motion.

        Current implementation: Trapezoidal velocity profile
        """
        t = np.sqrt(2.0 * abs(x_err) / a)  # x_err = (1/2) * a * t^2
        v = min(a * t, v_max)
        return v * np.sign(x_err)

    @staticmethod
    def _turn_rate_limit(lin_err, heading_diff, w_max, tol=0.0):
        """
        Compute velocity limit that prevents path from overshooting goal

        heading error decrease rate > linear error decrease rate
        (w - v * np.sin(phi) / D) / phi > v * np.cos(phi) / D
        v < (w / phi) / (np.sin(phi) / D / phi + np.cos(phi) / D)
        v < w * D / (np.sin(phi) + phi * np.cos(phi))

        (D = linear error, phi = angular error)
        """
        assert lin_err >= 0.0
        assert heading_diff >= 0.0
        #import pdb; pdb.set_trace()
        if heading_diff > MAX_HEADING_ANG:
            return 0.0
        else:
            return w_max * lin_err / (np.sin(heading_diff) + heading_diff * np.cos(heading_diff) + 1e-5)

    def _feedback_traj_track(self, xyt_err):
        xyt_err = self._compute_error_pose(xyt)
        v_raw = V_MAX_DEFAULT * (K1 * xyt_err[0] + xyt_err[1] * np.tan(xyt_err[2])) / np.cos(xyt_err[2])
        w_raw = V_MAX_DEFAULT * (K2 * xyt_err[1] + K3 * np.tan(xyt_err[2])) / np.cos(xyt_err[2])**2
        v_out = min(v_raw, V_MAX_DEFAULT)
        w_out = min(w_raw, W_MAX_DEFAULT)
        return np.array([v_out, w_out])

    def _feedback_simple(self, xyt_err):
        v_cmd = w_cmd = 0

        lin_err_abs = np.linalg.norm(xyt_err[0:2])
        ang_err = xyt_err[2]

        print("A:", lin_err_abs, LIN_ERROR_TOL, lin_err_abs > LIN_ERROR_TOL)
        print("B:", abs(ang_err), ANG_ERROR_TOL, abs(ang_err) > ANG_ERROR_TOL and self.track_yaw)
        # Go to goal XY position if not there yet
        if lin_err_abs > LIN_ERROR_TOL:
            heading_err = np.arctan2(xyt_err[1], xyt_err[0])
            heading_err_abs = abs(heading_err)

            # Compute linear velocity
            v_raw = self._velocity_feedback_control(
                lin_err_abs, ACC_LIN, self.v_max
            )
            v_limit = self._turn_rate_limit(
                lin_err_abs,
                heading_err_abs,
                self.w_max / 2.0,
                tol=LIN_ERROR_TOL,
            )
            #import pdb; pdb.set_trace()
            v_cmd = np.clip(v_raw, 0.0, v_limit)

            # Compute angular velocity
            w_cmd = self._velocity_feedback_control(
                heading_err, ACC_ANG, self.w_max
            )

        # Rotate to correct yaw if yaw tracking is on and XY position is at goal
        elif abs(ang_err) > ANG_ERROR_TOL and self.track_yaw:
            # Compute angular velocity
            w_cmd = self._velocity_feedback_control(
                ang_err, ACC_ANG, self.w_max
            )

        # v_cmd and w_cmd will be clip for [-1, 1]
        return v_cmd, w_cmd

    def forward(self, xyt, env):
        xyt_err = self._compute_error_pose(xyt, env)
        return self._feedback_simple(xyt_err)

def step_env(env, action_name, action_args):
    return env.step({"action": action_name, "action_args": action_args})


def waypoint_generator(env, args, config):

    """Generate the waypoints that the robot should navigate"""
    # Get the velocity of control
    base_vel_ctrl = habitat_sim.physics.VelocityControl()
    base_vel_ctrl.controlling_lin_vel = True
    base_vel_ctrl.lin_vel_is_local = True
    base_vel_ctrl.controlling_ang_vel = True
    base_vel_ctrl.ang_vel_is_local = True

    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()

    env.sim.recompute_navmesh(
        env.sim.pathfinder,
        navmesh_settings,
        include_static_objects=True,
    )

    before_base_pos = env.sim.robot.base_pos
    before_base_rot = env.sim.robot.base_rot

    success_flag = False
    collision_count = 1
    while not success_flag:
        visited_points = []
        used_actions = []

        env.sim.robot.base_pos = before_base_pos
        env.sim.robot.base_rot = before_base_rot

        agent = env.sim.agents[0]
        # navmesh_settings.agent_height = agent.height
        # navmesh_settings.agent_radius = agent.radius

        # Get the path finder
        pf = env.sim.pathfinder
        state = habitat_sim.AgentState()
        while True:
            state.position = before_base_pos
            rotation = [env.sim.robot.sim_obj.rotation.vector[0], env.sim.robot.sim_obj.rotation.vector[1], env.sim.robot.sim_obj.rotation.vector[2], env.sim.robot.sim_obj.rotation.scalar]
            state.rotation = rotation
            goal_pos = pf.get_random_navigable_point()
            path = habitat_sim.ShortestPath()
            path.requested_start = state.position
            path.requested_end = goal_pos

            if pf.is_navigable(goal_pos) and pf.find_path(path) and path.geodesic_distance > 5.0:
                break

        agent = env.sim.agents[0]
        follower = habitat_sim.GreedyGeodesicFollower(
            pf,
            agent,
            forward_key="move_forward",
            left_key="turn_left",
            right_key="turn_right",
        )

        agent.state = state
        next_action = follower.next_action_along(goal_pos)
        action_list = follower.find_path(goal_pos)

        success_flag = True

        last_pos = agent.state.position
        for a in action_list:
            if a == "move_forward":
                agent.act(1)
            elif a == "turn_left":
                agent.act(2)
            elif a == "turn_right":
                agent.act(3)

            # apply action
            env.sim.robot.base_pos = agent.state.position
            env.sim.robot.base_rot = agent.state.rotation.angle()

            is_navigable = env.sim.pathfinder.is_navigable(agent.state.position)
            is_contact = env.sim.contact_test(env.sim.robot.get_robot_sim_id())

            if not pf.is_navigable(agent.state.position) or not is_navigable or is_contact:
                print("collision detected:", collision_count)
                collision_count += 1
                success_flag = False

            # This is to filter out the waypoints that are not relevent
            if ((last_pos[0] - agent.state.position[0])**2 + (last_pos[2] - agent.state.position[2])**2)**0.5 > 1.0:
                visited_points.append([agent.state.position, agent.state.rotation.angle()])
                used_actions.append(a)
                last_pos = agent.state.position

    env.sim.robot.base_pos = before_base_pos
    env.sim.robot.base_rot = before_base_rot

    # visited_points = []
    # visited_points.append([[1.33366, 0.652255, 0.403866], 0.6732290])
    # visited_points.append([[-0.934055, 0.652255, 0.192359], 0.0718358])
    # visited_points.append([[-1.43059, 0.652255, -3.80539], 0.673224])

    print("origin length:", len(action_list))
    print("filter-out length:", len(visited_points))


    return visited_points, used_actions


def distance_angle(alpha, beta):
    alpha = float(alpha)
    beta = float(beta)
    phi = abs(beta - alpha) % (2 * np.pi)
    # This is either the distance or 360 - distance
    if phi > np.pi:
        return 2 * np.pi - phi
    else:
        return phi


def point2vel(action_list):
    """The point2vel in pratice should take the input of waypoints, but here
    I just use the ground-truth action control.
    """
    return action_list[0]


def get_input_vel_ctlr(
    skip_pygame,
    arm_action,
    env,
    not_block_input,
    agent_to_control,
    base_action,
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
        # base_action = None
    else:
        arm_action_space = np.zeros(7)
        arm_ctrlr = None
        # base_action = [0, 0]

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
        elif arm_action_space.shape[0] == 10:
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
                arm_action[6] = 1.0
            elif keys[pygame.K_9]:
                arm_action[6] = -1.0

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

    return step_env(env, name, args), arm_action, end_ep


def get_wrapped_prop(venv, prop):
    if hasattr(venv, prop):
        return getattr(venv, prop)
    elif hasattr(venv, "venv"):
        return get_wrapped_prop(venv.venv, prop)
    elif hasattr(venv, "env"):
        return get_wrapped_prop(venv.env, prop)

    return None


def reached(cur_pos, target_pos, cur_rot, target_rot):
    p = 0
    for i in [0, 2]:
        p += (cur_pos[i] - target_pos[i])**2

    a = abs(float(cur_rot)-float(target_rot))
    print("error position:", p**0.5)
    print("error angle:", a)
    if p**0.5 <= LIN_ERROR_TOL and a <= ANG_ERROR_TOL:
        return True
    else:
        return False

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



    before_base_pos = env.sim.robot.base_pos
    before_base_rot = env.sim.robot.base_rot

    # Define the fixed starting points
    # before_base_pos[0] = 1.972760
    # before_base_pos[1] = 0.652255
    # before_base_pos[2] = 7.207250
    # before_base_rot = 0.673229

    env.sim.robot.base_pos = before_base_pos
    env.sim.robot.base_rot = before_base_rot

    waypoint_list, used_action_list = waypoint_generator(env, args, config)

    env.sim.robot.base_pos = before_base_pos
    env.sim.robot.base_rot = before_base_rot

    first_flag = True

    index = 0
    while True:
        print("==============")
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

        # Here is the part to decide the velocity control.
        #import pdb; pdb.set_trace()
        if len(used_action_list) == 0:
            base_action = [0, 0]
        else:
            flag = False
            if len(waypoint_list) > 0:
                flag = reached(env.sim.robot.base_pos, waypoint_list[0][0], env.sim.robot.base_rot,  waypoint_list[0][1])
            if flag and len(waypoint_list) > 0:
                waypoint_list = waypoint_list[1:]
                index += 1
            if (flag or first_flag) and len(waypoint_list) > 0:
                agent = Controller()
                xyt_goal = [ waypoint_list[0][0][0],  waypoint_list[0][0][2],  float(waypoint_list[0][1])]
                agent.set_goal(xyt_goal)
                first_flag = False



            xyt = [env.sim.robot.base_pos[0], env.sim.robot.base_pos[2], float(env.sim.robot.base_rot)]
            base_action = agent.forward(xyt, env)

            if len(waypoint_list) == 0:
                print("Done")
                base_action = [0.0, 0.0]
                index = 0
            else:
                cam = env._sim.agents[0].scene_node.node_sensor_suite.get("robot_third_rgb")
                camera_position = cam.render_camera.node.absolute_translation
                env._sim.get_debug_line_render().draw_circle(
                    translation= waypoint_list[0][0],
                    radius=1.0,
                    color=mn.Color4.red(),
                    normal=camera_position-waypoint_list[0][0]
                )
            print("base_action:", base_action)
            print("current and goal:", [env.sim.robot.base_pos[0], env.sim.robot.base_pos[2], env.sim.robot.base_rot], xyt_goal)
            print("index:", index)
            # if flag:
            #     import pdb; pdb.set_trace()

            # Mine
            #base_action = point2vel(used_action_list)
            #used_action_list = used_action_list[1:]

        step_result, arm_action, end_ep = get_input_vel_ctlr(
            args.no_render,
            use_arm_actions[update_idx]
            if use_arm_actions is not None
            else None,
            env,
            not free_cam.is_free_cam_mode,
            agent_to_control,
            base_action,
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
            before_base_pos = env.sim.robot.base_pos
            before_base_rot = env.sim.robot.base_rot

            waypoint_list, used_action_list = waypoint_generator(env, args, config)

            env.sim.robot.base_pos = before_base_pos
            env.sim.robot.base_rot = before_base_rot

            first_flag = True

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
