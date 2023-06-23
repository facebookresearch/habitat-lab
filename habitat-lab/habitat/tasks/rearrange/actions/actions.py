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
from habitat.robots.stretch_robot import StretchJointStates, StretchRobot
from habitat.sims.habitat_simulator.actions import HabitatSimActions

# flake8: noqa
# These actions need to be imported since there is a Python evaluation
# statement which dynamically creates the desired grip controller.
from habitat.tasks.rearrange.actions.grip_actions import (
    GazeGraspAction,
    GripSimulatorTaskAction,
    MagicGraspAction,
    SuctionGraspAction,
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
class ArmRelPosAction(RobotAction):
    """
    The arm motor targets are offset by the delta joint values specified by the
    action
    """

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._max_delta_pos = self._config.max_delta_pos
        self._min_delta_pos = self._config.min_delta_pos

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
        delta_pos *= self._max_delta_pos
        delta_pos[np.abs(delta_pos) < self._min_delta_pos] = 0
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
        self._max_delta_pos = self._config.max_delta_pos
        self._min_delta_pos = self._config.min_delta_pos
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
        delta_pos *= self._max_delta_pos
        delta_pos[np.abs(delta_pos) < self._min_delta_pos] = 0
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
        self._max_delta_pos = self._config.max_delta_pos
        self._min_delta_pos = self._config.min_delta_pos
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
        delta_pos *= self._max_delta_pos
        delta_pos[np.abs(delta_pos) < self._min_delta_pos] = 0
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
        self.cur_robot.arm_joint_pos = set_arm_pos
        if self.cur_grasp_mgr.snap_idx is not None:
            # Holding onto an object, also kinematically update the object.
            self.cur_grasp_mgr.update_object_to_grasp()


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


@registry.register_task_action
class ManipulationModeAction(RobotAction):
    """
    The robot joints and base is changed for performing manipulation. In the case of Stretch, the head is turned to face the arm and the base is rotated left by 90 degrees
    """

    def __init__(self, *args, config, **kwargs):
        self._threshold = config.threshold
        super().__init__(self, *args, config=config, **kwargs)

    def step(self, task, *args, is_last_action, **kwargs):
        manip_mode = kwargs.get("manipulation_mode", [-1.0])
        if manip_mode[0] > self._threshold and not task._in_manip_mode:
            if isinstance(self._sim.robot, StretchRobot):
                # Turn the head to face the arm
                task._in_manip_mode = True
                self._sim.robot.arm_motor_pos = StretchJointStates.PRE_GRASP
                self._sim.robot.arm_joint_pos = StretchJointStates.PRE_GRASP
                # now turn the robot's base left by 90 degrees
                obj_trans = self.cur_robot.sim_obj.transformation
                turn_angle = np.pi / 2  # Turn left by 90 degrees
                rot_quat = mn.Quaternion(
                    mn.Vector3(0, np.sin(turn_angle / 2), 0),
                    np.cos(turn_angle / 2),
                )
                # Get the target rotation
                target_rot = rot_quat.to_matrix() @ obj_trans.rotation()
                target_trans = mn.Matrix4.from_(
                    target_rot,
                    obj_trans.translation,
                )
                self.cur_robot.sim_obj.transformation = target_trans
                if self.cur_grasp_mgr.snap_idx is not None:
                    # Holding onto an object, also kinematically update the object.
                    self.cur_grasp_mgr.update_object_to_grasp()

        if is_last_action:
            return self._sim.step(HabitatSimActions.manipulation_mode)
        else:
            return {}


@registry.register_task_action
class BaseWaypointTeleportAction(RobotAction):
    """
    The robot is teleported to the target waypoints while being constrained to the navmesh. In one step, The robot can only move forward or turn.
    """

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._sim: RearrangeSim = sim
        self._allow_back = config.allow_back
        self._collision_threshold = config.collision_threshold
        self._navmesh_offset = config.navmesh_offset
        self._min_displacement = (
            config.min_displacement
        )  # minimum displacement
        self._max_displacement_along_axis = config.max_displacement_along_axis
        self._max_turn_radians = (
            config.max_turn_degrees * np.pi / 180
        )  # maximum turn waypoint
        self._min_turn_radians = (
            config.min_turn_degrees * np.pi / 180
        )  # minimum turn waypoint
        self._allow_lateral_movement = config.allow_lateral_movement
        self._allow_simultaneous_turn = config.allow_simultaneous_turn
        self._discrete_movement = config.discrete_movement
        self._constraint_base_in_manip_mode = (
            config.constraint_base_in_manip_mode
        )

    def collision_check(self, trans, target_trans):
        """
        trans: the transformation of the current location of the robot
        target_trans: the transformation of the target location of the robot given the center original Navmesh
        """
        # Get the offset positions
        num_check_cylinder = len(self._navmesh_offset)
        nav_pos_3d = [
            np.array([xz[0], xz[1], 0.0]) for xz in self._navmesh_offset
        ]
        cur_pos = [trans.transform_point(xyz) for xyz in nav_pos_3d]
        goal_pos = [target_trans.transform_point(xyz) for xyz in nav_pos_3d]

        # For step filter of offset positions
        end_pos = []
        # Planar move distance clamped by NavMesh
        move = []
        for i in range(num_check_cylinder):
            pos = self._sim.step_filter(cur_pos[i], goal_pos[i])
            # Ignore any height differences that may pop up
            pos[1] = 0.0
            cur_pos[i][1] = 0.0
            goal_pos[i][1] = 0.0
            end_pos.append(pos)
            move.append((end_pos[i] - goal_pos[i]).length())

        # There is a collision if the distance between the clamped navmesh position and target position is greater than the self._collision_threshold.
        diff = len([v for v in move if v > self._collision_threshold])

        if diff > 0:
            return True, trans
        else:
            return False, target_trans

    @property
    def action_space(self):
        lim = 1
        action_space_shape = 2  # for turning and moving forward
        if self._allow_lateral_movement:
            action_space_shape += 1  # for lateral movement
        if not self._allow_simultaneous_turn:
            action_space_shape += (
                1  # for determining whether to turn or move forward
            )
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "base_vel": spaces.Box(
                    shape=(action_space_shape,),
                    low=-lim,
                    high=lim,
                    dtype=np.float32,
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
        """
        Keep track of robot's basic info
        """
        self.cur_robot.sim_obj.joint_positions = set_dat["forces"]
        self.cur_robot.sim_obj.joint_velocities = set_dat["vel"]
        self.cur_robot.sim_obj.joint_forces = set_dat["pos"]

    def update_base(self, target_rigid_state):
        """
        Update the robot base
        """

        trans = self.cur_robot.sim_obj.transformation

        target_trans = mn.Matrix4.from_(
            target_rigid_state.rotation.to_matrix(),
            target_rigid_state.translation,
        )
        self.cur_robot.sim_obj.transformation = target_trans
        # Check if there is a collision
        navmesh_violation, new_target_trans = self.collision_check(
            trans, target_trans
        )
        # Update the base
        self.cur_robot.sim_obj.transformation = new_target_trans
        if self.cur_grasp_mgr.snap_idx is not None:
            # Holding onto an object, also kinematically update the object.
            self.cur_grasp_mgr.update_object_to_grasp()
        return navmesh_violation

    def step(self, *args, task, is_last_action, **kwargs):
        base_action = kwargs[self._action_arg_prefix + "base_vel"]
        lin_pos_x = base_action[0]
        turn_offset = 1
        lin_pos_z = 0.0
        if self._allow_lateral_movement:
            lin_pos_z = base_action[1]
            turn_offset += 1
        turn = base_action[turn_offset]
        if not self._allow_simultaneous_turn:
            # Select between translation and turn
            sel = base_action[-1]
            if sel > 0:
                turn = 0
            else:
                lin_pos_x = 0
                lin_pos_z = 0

        if self._discrete_movement:
            # Either move/rotate in one direction by fixed amount or do not move at all
            lin_pos_x = np.sign(lin_pos_x) if lin_pos_x != 0 else 0
            lin_pos_z = np.sign(lin_pos_z) if lin_pos_z != 0 else 0
            turn = np.sign(turn) if turn != 0 else 0

        lin_pos_x = (
            np.clip(lin_pos_x, -1, 1) * self._max_displacement_along_axis
        )
        lin_pos_z = (
            np.clip(lin_pos_z, -1, 1) * self._max_displacement_along_axis
        )
        ang_pos = np.clip(turn, -1, 1) * self._max_turn_radians

        # Do not allow small movements
        if np.abs(ang_pos) < self._min_turn_radians:
            ang_pos = 0
        if np.linalg.norm([lin_pos_x, lin_pos_z]) < self._min_displacement:
            lin_pos_x = 0
            lin_pos_z = 0

        if not self._allow_back:
            lin_pos_x = np.maximum(lin_pos_x, 0)

        # Get the transformation of the robot
        base_trans = self._sim.robot.base_transformation
        obj_trans = self.cur_robot.sim_obj.transformation
        # Get the global pos from the local target waypoints
        target_pos = base_trans.transform_point(
            mn.Vector3([lin_pos_x, lin_pos_z, 0])
        )
        target_rot = obj_trans.rotation()
        rot_quat = mn.Quaternion(
            mn.Vector3(0, np.sin(ang_pos / 2), 0), np.cos(ang_pos / 2)
        )
        # Get the target rotation
        target_rot = rot_quat.to_matrix() @ obj_trans.rotation()

        # combine target translation and rotation to get target rigid state
        target_rigid_state = habitat_sim.RigidState(
            mn.Quaternion.from_matrix(target_rot), target_pos
        )

        if self._constraint_base_in_manip_mode and task._in_manip_mode:
            lin_pos_x = 0.0
            lin_pos_z = 0.0
            ang_pos = 0.0

        if lin_pos_x != 0.0 or lin_pos_z != 0.0 or ang_pos != 0.0:
            task._is_navmesh_violated = self.update_base(target_rigid_state)
        else:
            # no violation if no movement was required in the first place
            task._is_navmesh_violated = False
        if is_last_action:
            return self._sim.step(HabitatSimActions.base_velocity)
        else:
            return {}
