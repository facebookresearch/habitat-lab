#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import magnum as mn
import numpy as np
from gym import spaces

import habitat_sim
from habitat.core.embodied_task import SimulatorTaskAction
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions

# flake8: noqa
# These actions need to be imported since there is a Python evaluation
# statement which dynamically creates the desired grip controller.
from habitat.tasks.rearrange.actions.grip_actions import (
    GripSimulatorTaskAction,
    MagicGraspAction,
    SuctionGraspAction,
    GazeGraspAction
)
from habitat.tasks.rearrange.actions.robot_action import RobotAction
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.utils import rearrange_collision, rearrange_logger


@registry.register_task_action
class EmptyAction(RobotAction):
    """A No-op action useful for testing and in some controllers where we want
    to wait before the next operation.
    """

    @property
    def action_space(self):
        return spaces.Dict(
            {
                "empty_action": spaces.Box(
                    shape=(1,),
                    low=-1,
                    high=1,
                    dtype=np.float32,
                )
            }
        )

    def step(self, *args, **kwargs):
        return self._sim.step(HabitatSimActions.empty)


@registry.register_task_action
class RearrangeStopAction(SimulatorTaskAction):
    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.does_want_terminate = False

    def step(self, task, *args, is_last_action, **kwargs):
        should_stop = kwargs.get("rearrange_stop", [1.0])
        if should_stop[0] > 0.0:
            rearrange_logger.debug(
                "Rearrange stop action requesting episode stop."
            )
            self.does_want_terminate = True

        if is_last_action:
            return self._sim.step(HabitatSimActions.rearrange_stop)
        else:
            return {}


@registry.register_task_action
class ArmAction(RobotAction):
    """An arm control and grip control into one action space."""

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        arm_controller_cls = eval(self._config.arm_controller)
        self._sim: RearrangeSim = sim
        self.arm_ctrlr = arm_controller_cls(
            *args, config=config, sim=sim, **kwargs
        )

        if self._config.grip_controller is not None:
            grip_controller_cls = eval(self._config.grip_controller)
            self.grip_ctrlr: Optional[
                GripSimulatorTaskAction
            ] = grip_controller_cls(*args, config=config, sim=sim, **kwargs)
        else:
            self.grip_ctrlr = None

        self.disable_grip = False
        if "disable_grip" in config:
            self.disable_grip = config["disable_grip"]

    def reset(self, *args, **kwargs):
        self.arm_ctrlr.reset(*args, **kwargs)
        if self.grip_ctrlr is not None:
            self.grip_ctrlr.reset(*args, **kwargs)

    @property
    def action_space(self):
        action_spaces = {
            self._action_arg_prefix
            + "arm_action": self.arm_ctrlr.action_space,
        }
        if self.grip_ctrlr is not None and self.grip_ctrlr.requires_action:
            action_spaces[
                self._action_arg_prefix + "grip_action"
            ] = self.grip_ctrlr.action_space
        return spaces.Dict(action_spaces)

    def step(self, is_last_action, *args, **kwargs):
        arm_action = kwargs[self._action_arg_prefix + "arm_action"]
        self.arm_ctrlr.step(arm_action)
        if self.grip_ctrlr is not None and not self.disable_grip:
            grip_action = kwargs[self._action_arg_prefix + "grip_action"]
            self.grip_ctrlr.step(grip_action)
        if is_last_action:
            return self._sim.step(HabitatSimActions.arm_action)
        else:
            return {}


@registry.register_task_action
class ExtendArmAction(RobotAction):

    def step(self, *args, is_last_action, **kwargs):
        extend = kwargs.get("extend_arm", [-1.0])
        if extend[0] > 0:
            self._sim.robot.arm_motor_pos = [0.13] * 4 + [1.0] + self._sim.robot.arm_motor_pos[5:].tolist()
            self._sim.robot.arm_joint_pos = [0.13] * 4 + [1.0] + self._sim.robot.arm_motor_pos[5:].tolist()
        if is_last_action:
            return self._sim.step(HabitatSimActions.arm_action)
        else:
            return {}

@registry.register_task_action
class FaceArmAction(RobotAction):

    def step(self, *args, is_last_action, **kwargs):
        face = kwargs.get("face_arm", [-1.0])
        if face[0] > 0:
            motor_pos = self._sim.robot.arm_motor_pos.tolist()
            joint_pos = self._sim.robot.arm_motor_pos.tolist()
            motor_pos[8] = -1.7375
            joint_pos[8] = -1.7375
            self._sim.robot.arm_motor_pos = motor_pos
            self._sim.robot.arm_joint_pos = joint_pos
        if is_last_action:
            return self._sim.step(HabitatSimActions.arm_action)
        else:
            return {}

@registry.register_task_action
class ResetJointsAction(RobotAction):
    def step(self, *args, is_last_action, **kwargs):
        reset = kwargs.get("reset_joints", [-1.0])
        if reset[0] > 0:
            self._sim.robot.arm_motor_pos = [0, 0, 0, 0, 0.775, 0, -1.57000005, 0, 0.0, -0.7125]
            self._sim.robot.arm_joint_pos = [0, 0, 0, 0, 0.775, 0, -1.57000005, 0, 0.0, -0.7125]
        if is_last_action:
            return self._sim.step(HabitatSimActions.arm_action)
        else:
            return {}


@registry.register_task_action
class ArmRelPosAction(RobotAction):
    """
    The arm motor targets are offset by the delta joint values specified by the
    action
    """

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._delta_pos_limit = self._config.delta_pos_limit

    @property
    def action_space(self):
        return spaces.Box(
            shape=(self._config.arm_joint_dimensionality,),
            low=-1,
            high=1,
            dtype=np.float32,
        )

    def step(self, delta_pos, should_step=True, *args, **kwargs):
        # clip from -1 to 1
        delta_pos = np.clip(delta_pos, -1, 1)
        delta_pos *= self._delta_pos_limit
        # The actual joint positions
        self._sim: RearrangeSim
        self.cur_robot.arm_motor_pos = delta_pos + self.cur_robot.arm_motor_pos


@registry.register_task_action
class ArmRelPosKinematicAction(RobotAction):
    """
    The arm motor targets are offset by the delta joint values specified by the
    action
    """

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._delta_pos_limit = self._config.delta_pos_limit
        self._should_clip = self._config.get("should_clip", True)

    @property
    def action_space(self):
        return spaces.Box(
            shape=(self._config.arm_joint_dimensionality,),
            low=0,
            high=1,
            dtype=np.float32,
        )

    def step(self, delta_pos, *args, **kwargs):
        if self._should_clip:
            # clip from -1 to 1
            delta_pos = np.clip(delta_pos, -1, 1)
        delta_pos *= self._delta_pos_limit
        self._sim: RearrangeSim

        set_arm_pos = delta_pos + self.cur_robot.arm_joint_pos
        self.cur_robot.arm_joint_pos = set_arm_pos
        self.cur_robot.fix_joint_values = set_arm_pos


@registry.register_task_action
class ArmAbsPosAction(RobotAction):
    """
    The arm motor targets are directly set to the joint configuration specified
    by the action.
    """

    @property
    def action_space(self):
        return spaces.Box(
            shape=(self._config.arm_joint_dimensionality,),
            low=0,
            high=1,
            dtype=np.float32,
        )

    def step(self, set_pos, *args, **kwargs):
        # No clipping because the arm is being set to exactly where it needs to
        # go.
        self._sim: RearrangeSim
        self.cur_robot.arm_motor_pos = set_pos


@registry.register_task_action
class ArmAbsPosKinematicAction(RobotAction):
    """
    The arm is kinematically directly set to the joint configuration specified
    by the action.
    """

    @property
    def action_space(self):
        return spaces.Box(
            shape=(self._config.arm_joint_dimensionality,),
            low=0,
            high=1,
            dtype=np.float32,
        )

    def step(self, set_pos, *args, **kwargs):
        # No clipping because the arm is being set to exactly where it needs to
        # go.
        self._sim: RearrangeSim
        self.cur_robot.arm_joint_pos = set_pos


@registry.register_task_action
class ArmRelPosReducedActionStretch(RobotAction):
    """
    The arm motor targets are offset by the delta joint values specified by the
    action and the mask. This function is used for Stretch.
    """

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self.last_arm_action = None
        self._delta_pos_limit = self._config.delta_pos_limit
        self._should_clip = self._config.get("should_clip", True)
        self._arm_joint_mask = self._config.arm_joint_mask

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.last_arm_action = None

    @property
    def action_space(self):
        self.step_c = 0
        return spaces.Box(
            shape=(self._config.arm_joint_dimensionality,),
            low=-1,
            high=1,
            dtype=np.float32,
        )

    def step(self, delta_pos, *args, **kwargs):
        if self._should_clip:
            # clip from -1 to 1
            delta_pos = np.clip(delta_pos, -1, 1)
        delta_pos *= self._delta_pos_limit
        self._sim: RearrangeSim

        # Expand delta_pos based on mask
        expanded_delta_pos = np.zeros(len(self._arm_joint_mask))
        src_idx = 0
        tgt_idx = 0
        for mask in self._arm_joint_mask:
            if mask == 0:
                tgt_idx += 1
                continue
            expanded_delta_pos[tgt_idx] = delta_pos[src_idx]
            tgt_idx += 1
            src_idx += 1

        min_limit, max_limit = self.cur_robot.arm_joint_limits
        set_arm_pos = expanded_delta_pos + self.cur_robot.arm_motor_pos
        # Perform roll over to the joints so that the user cannot control
        # the motor 2, 3, 4 for the arm.
        if expanded_delta_pos[0] >= 0:
            for i in range(3):
                if set_arm_pos[i] > max_limit[i]:
                    set_arm_pos[i + 1] += set_arm_pos[i] - max_limit[i]
                    set_arm_pos[i] = max_limit[i]
        else:
            for i in range(3):
                if set_arm_pos[i] < min_limit[i]:
                    set_arm_pos[i + 1] -= min_limit[i] - set_arm_pos[i]
                    set_arm_pos[i] = min_limit[i]
        set_arm_pos = np.clip(set_arm_pos, min_limit, max_limit)

        self.cur_robot.arm_motor_pos = set_arm_pos


@registry.register_task_action
class BaseVelAction(RobotAction):
    """
    The robot base motion is constrained to the NavMesh and controlled with velocity commands integrated with the VelocityControl interface.

    Optionally cull states with active collisions if config parameter `allow_dyn_slide` is True
    """

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._sim: RearrangeSim = sim
        self.base_vel_ctrl = habitat_sim.physics.VelocityControl()
        self.base_vel_ctrl.controlling_lin_vel = True
        self.base_vel_ctrl.lin_vel_is_local = True
        self.base_vel_ctrl.controlling_ang_vel = True
        self.base_vel_ctrl.ang_vel_is_local = True
        self._allow_dyn_slide = self._config.get("allow_dyn_slide", True)
        self._lin_speed = self._config.lin_speed
        self._ang_speed = self._config.ang_speed
        self._allow_back = self._config.allow_back

    @property
    def action_space(self):
        lim = 20
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "base_vel": spaces.Box(
                    shape=(2,), low=-lim, high=lim, dtype=np.float32
                )
            }
        )

    def _capture_robot_state(self):
        return {
            "forces": self.cur_robot.sim_obj.joint_forces,
            "vel": self.cur_robot.sim_obj.joint_velocities,
            "pos": self.cur_robot.sim_obj.joint_positions,
        }

    def _set_robot_state(self, set_dat):
        self.cur_robot.sim_obj.joint_positions = set_dat["forces"]
        self.cur_robot.sim_obj.joint_velocities = set_dat["vel"]
        self.cur_robot.sim_obj.joint_forces = set_dat["pos"]

    def update_base(self):

        ctrl_freq = self._sim.ctrl_freq

        before_trans_state = self._capture_robot_state()

        trans = self.cur_robot.sim_obj.transformation
        rigid_state = habitat_sim.RigidState(
            mn.Quaternion.from_matrix(trans.rotation()), trans.translation
        )

        target_rigid_state = self.base_vel_ctrl.integrate_transform(
            1 / ctrl_freq, rigid_state
        )
        end_pos = self._sim.step_filter(
            rigid_state.translation, target_rigid_state.translation
        )

        target_trans = mn.Matrix4.from_(
            target_rigid_state.rotation.to_matrix(), end_pos
        )
        self.cur_robot.sim_obj.transformation = target_trans

        if not self._allow_dyn_slide:
            # Check if in the new robot state the arm collides with anything.
            # If so we have to revert back to the previous transform
            self._sim.internal_step(-1)
            colls = self._sim.get_collisions()
            did_coll, _ = rearrange_collision(
                colls, self._sim.snapped_obj_id, False
            )
            if did_coll:
                # Don't allow the step, revert back.
                self._set_robot_state(before_trans_state)
                self.cur_robot.sim_obj.transformation = trans
        if self.cur_grasp_mgr.snap_idx is not None:
            # Holding onto an object, also kinematically update the object.
            # object.
            self.cur_grasp_mgr.update_object_to_grasp()

    def step(self, *args, is_last_action, **kwargs):
        lin_vel, ang_vel = kwargs[self._action_arg_prefix + "base_vel"]
        lin_vel = np.clip(lin_vel, -1, 1) * self._lin_speed
        ang_vel = np.clip(ang_vel, -1, 1) * self._ang_speed
        if not self._allow_back:
            lin_vel = np.maximum(lin_vel, 0)

        self.base_vel_ctrl.linear_velocity = mn.Vector3(lin_vel, 0, 0)
        self.base_vel_ctrl.angular_velocity = mn.Vector3(0, ang_vel, 0)

        if lin_vel != 0.0 or ang_vel != 0.0:
            self.update_base()

        if is_last_action:
            return self._sim.step(HabitatSimActions.base_velocity)
        else:
            return {}


@registry.register_task_action
class ArmEEAction(RobotAction):
    """Uses inverse kinematics (requires pybullet) to apply end-effector position control for the robot's arm."""

    def __init__(self, *args, sim: RearrangeSim, **kwargs):
        self.ee_target: Optional[np.ndarray] = None
        super().__init__(*args, sim=sim, **kwargs)
        self._sim: RearrangeSim = sim
        self._render_ee_target = self._config.get("render_ee_target", False)
        self._ee_ctrl_lim = self._config.ee_ctrl_lim

    def reset(self, *args, **kwargs):
        super().reset()
        cur_ee = self._ik_helper.calc_fk(
            np.array(self._sim.robot.arm_joint_pos)
        )

        self.ee_target = cur_ee

    @property
    def action_space(self):
        return spaces.Box(shape=(3,), low=-1, high=1, dtype=np.float32)

    def apply_ee_constraints(self):
        self.ee_target = np.clip(
            self.ee_target,
            self._sim.robot.params.ee_constraint[:, 0],
            self._sim.robot.params.ee_constraint[:, 1],
        )

    def set_desired_ee_pos(self, ee_pos: np.ndarray) -> None:
        self.ee_target += np.array(ee_pos)

        self.apply_ee_constraints()

        joint_pos = np.array(self._sim.robot.arm_joint_pos)
        joint_vel = np.zeros(joint_pos.shape)

        self._ik_helper.set_arm_state(joint_pos, joint_vel)

        des_joint_pos = self._ik_helper.calc_ik(self.ee_target)
        des_joint_pos = list(des_joint_pos)
        self._sim.robot.arm_motor_pos = des_joint_pos

    def step(self, ee_pos, **kwargs):
        ee_pos = np.clip(ee_pos, -1, 1)
        ee_pos *= self._ee_ctrl_lim
        self.set_desired_ee_pos(ee_pos)

        if self._render_ee_target:
            global_pos = self._sim.robot.base_transformation.transform_point(
                self.ee_target
            )
            self._sim.viz_ids["ee_target"] = self._sim.visualize_position(
                global_pos, self._sim.viz_ids["ee_target"]
            )


class Controller:
    """
    A controller that takes the input of the waypoints and
    produces the velocity command.
    """

    def __init__(
        self,
        v_max,
        w_max,
        lin_gain=5,
        ang_gain=5,
        lin_error_tol=0.05,
        ang_error_tol=0.05,
        max_heading_ang=np.pi / 10,
        track_yaw=True,
    ):
        # If we want track yaw or not
        self.track_yaw = track_yaw

        # Parameter of the controller
        self.v_max = v_max
        self.w_max = w_max

        self.lin_error_tol = lin_error_tol
        self.ang_error_tol = ang_error_tol

        self.acc_lin = lin_gain
        self.acc_ang = ang_gain

        self.max_heading_ang = max_heading_ang

        # Initialize the parameters
        self.base_pose_goal = np.zeros(3)
        self.base_velocity_goal = np.zeros(3)

    def set_goal(self, goal, vel_goal=None):
        """
        Set the goal of the agent
        """
        self.base_pose_goal = goal
        if vel_goal is not None:
            self.base_velocity_goal = vel_goal

    def _compute_error_pose(self, base_pose, sim=None):
        """
        Updates error based on robot localization
        """
        # We compute the base pose error and the function return list of integer
        base_pose_err = self._transform_global_to_base(
            self.base_pose_goal, base_pose, sim
        )

        if not self.track_yaw:
            base_pose_err[2] = 0.0

        return base_pose_err

    @staticmethod
    def _velocity_feedback_control(base_pose_err, a, v_max):
        """
        Computes velocity based on distance from target.
        Used for both linear and angular motion.
        Current implementation: Trapezoidal velocity profile
        """
        t = np.sqrt(2.0 * abs(base_pose_err) / a)
        v = min(a * t, v_max)
        return v * np.sign(base_pose_err)

    @staticmethod
    def _turn_rate_limit(
        lin_err, heading_diff, w_max, max_heading_ang, tol=0.0
    ):
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

        if heading_diff > max_heading_ang:
            return 0.0
        else:
            return (
                w_max
                * lin_err
                / (
                    np.sin(heading_diff)
                    + heading_diff * np.cos(heading_diff)
                    + 1e-5
                )
            )

    def _feedback_controller(self, base_pose_err):
        """
        Feedback controllers
        """
        v_cmd = w_cmd = 0

        lin_err_abs = np.linalg.norm(base_pose_err[0:2])
        ang_err = base_pose_err[2]

        # Go to goal XY position if not there yet
        if lin_err_abs > self.lin_error_tol:
            heading_err = np.arctan2(base_pose_err[1], base_pose_err[0])
            heading_err_abs = abs(heading_err)

            if abs(heading_err_abs - np.pi) <= 0.01:
                heading_err_abs = 0
                heading_err = 0

            # Compute linear velocity and allow the agent to move backward
            v_raw = self._velocity_feedback_control(
                np.sign(base_pose_err[0]) * lin_err_abs,
                self.acc_lin,
                self.v_max,
            )

            v_limit = self._turn_rate_limit(
                lin_err_abs,
                heading_err_abs,
                self.w_max / 2.0,
                max_heading_ang=self.max_heading_ang,
                tol=self.lin_error_tol,
            )

            # Clip the speed
            v_cmd = np.clip(v_raw, -v_limit, v_limit)

            # Compute angular velocity
            w_cmd = self._velocity_feedback_control(
                heading_err, self.acc_ang, self.w_max
            )

        # Rotate to correct yaw if yaw tracking is on and XY position is at goal
        elif abs(ang_err) > self.ang_error_tol and self.track_yaw:
            # Compute angular velocity
            w_cmd = self._velocity_feedback_control(
                ang_err, self.acc_ang, self.w_max
            )

        return v_cmd, w_cmd

    def _transform_global_to_base(self, base_pose, current_pose, sim=None):
        """
        Transforms the point cloud into geocentric frame to account for
        camera position
        Input:
            base_pose                     : target goal ...x3
            current_pose            : base position (x, y, theta (radians))
        Output:
            base_pose : ...x3
        """

        trans = sim.robot.base_transformation
        goal_pos = trans.inverted().transform_point(
            np.array([base_pose[0], sim.robot.base_pos[1], base_pose[1]])
        )

        error_t = base_pose[2] - current_pose[2]
        error_t = (error_t + np.pi) % (2.0 * np.pi) - np.pi
        error_x = goal_pos[0]
        error_y = goal_pos[1]

        return [error_x, error_y, error_t]

    def forward(self, base_pose, sim):
        """
        Generate the velocity command
        """
        base_pose_err = self._compute_error_pose(base_pose, sim)
        return self._feedback_controller(base_pose_err)


# TODO: Remove this once Teleport Action is merged.
@registry.register_task_action
class BaseWaypointVelAction(RobotAction):
    """
    The robot base motion is constrained to the NavMesh and controlled with velocity commands integrated with the VelocityControl interface.
    Optionally cull states with active collisions if config parameter `allow_dyn_slide` is True
    The robot is given a target waypoint and output a velocity command
    """

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._sim: RearrangeSim = sim
        self.base_vel_ctrl = habitat_sim.physics.VelocityControl()
        self.base_vel_ctrl.controlling_lin_vel = True
        self.base_vel_ctrl.lin_vel_is_local = True
        self.base_vel_ctrl.controlling_ang_vel = True
        self.base_vel_ctrl.ang_vel_is_local = True
        # Initialize the controller
        max_lin_speed = 30  # real Stretch max linear velocity
        max_ang_speed = 20.94  # real Stretch max angular velocity
        lin_speed_gain = 10000.0  # the linear gain
        ang_speed_gain = 10000.0  # the angular gain
        lin_tol = 0.001  # 5cm linear error tol
        ang_tol = 0.001  # 5cm angular error tol
        max_heading_ang = np.pi / 10.0
        self.controller = Controller(
            v_max=max_lin_speed,
            w_max=max_ang_speed,
            lin_gain=lin_speed_gain,
            ang_gain=ang_speed_gain,
            lin_error_tol=lin_tol,
            ang_error_tol=ang_tol,
            max_heading_ang=max_heading_ang,
        )

    @property
    def action_space(self):
        lim = 20
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "base_vel": spaces.Box(
                    shape=(2,), low=-lim, high=lim, dtype=np.float32
                )
            }
        )

    def _capture_robot_state(self):
        return {
            "forces": self.cur_robot.sim_obj.joint_forces,
            "vel": self.cur_robot.sim_obj.joint_velocities,
            "pos": self.cur_robot.sim_obj.joint_positions,
        }

    def _set_robot_state(self, set_dat):
        """ "
        Keep track of robot's basic info
        """
        self.cur_robot.sim_obj.joint_positions = set_dat["forces"]
        self.cur_robot.sim_obj.joint_velocities = set_dat["vel"]
        self.cur_robot.sim_obj.joint_forces = set_dat["pos"]

    def update_base(self):
        """
        Update the robot base
        """

        ctrl_freq = self._sim.ctrl_freq

        before_trans_state = self._capture_robot_state()

        trans = self.cur_robot.sim_obj.transformation
        rigid_state = habitat_sim.RigidState(
            mn.Quaternion.from_matrix(trans.rotation()), trans.translation
        )

        target_rigid_state = self.base_vel_ctrl.integrate_transform(
            1 / ctrl_freq, rigid_state
        )
        end_pos = self._sim.step_filter(
            rigid_state.translation, target_rigid_state.translation
        )

        target_trans = mn.Matrix4.from_(
            target_rigid_state.rotation.to_matrix(), end_pos
        )
        self.cur_robot.sim_obj.transformation = target_trans

        if not self._config.get("allow_dyn_slide", True):
            # Check if in the new robot state the arm collides with anything.
            # If so we have to revert back to the previous transform
            self._sim.internal_step(-1)
            # colls = get_collisions()
            did_coll, _ = rearrange_collision(self._sim, count_obj_colls=False)
            if did_coll:
                # Don't allow the step, revert back.
                self._set_robot_state(before_trans_state)
                self.cur_robot.sim_obj.transformation = trans
        if self.cur_grasp_mgr.snap_idx is not None:
            # Holding onto an object, also kinematically update the object.
            # object.
            self.cur_grasp_mgr.update_object_to_grasp()

    def step(self, *args, is_last_action, **kwargs):
        waypoint, sel = kwargs[self._action_arg_prefix + "base_vel"]
        # Scale the target waypoints and the rotation of the robot
        if sel > 0:
            lin_pos = np.clip(waypoint, -1, 1) * self._config.lin_speed
            ang_pos = 0.0
        else:
            lin_pos = 0.0
            ang_pos = np.clip(waypoint, -1, 1) * self._config.ang_speed
        if not self._config.allow_back:
            lin_pos = np.maximum(lin_pos, 0)

        # Get the transformation of the robot
        trans = self._sim.robot.base_transformation
        # Get the global pos from the local target waypoints
        global_pos = trans.transform_point(np.array([lin_pos, 0, 0]))
        # Get the target rotation
        target_theta = float(self._sim.robot.base_rot) + ang_pos

        # Set the goal
        base_pose_goal = [
            global_pos[0],
            global_pos[2],
            target_theta,
        ]
        self.controller.set_goal(base_pose_goal)

        # Get the current position
        base_pose = [
            self._sim.robot.base_pos[0],
            self._sim.robot.base_pos[2],
            float(self._sim.robot.base_rot),
        ]
        # Get the velocity cmd
        base_action = self.controller.forward(base_pose, self._sim)

        # Feed them into habitat's velocity controller
        self.base_vel_ctrl.linear_velocity = mn.Vector3(base_action[0], 0, 0)
        self.base_vel_ctrl.angular_velocity = mn.Vector3(0, base_action[1], 0)

        if lin_pos != 0.0 or ang_pos != 0.0:
            self.update_base()

        if is_last_action:
            return self._sim.step(HabitatSimActions.base_velocity)
        else:
            return {}
