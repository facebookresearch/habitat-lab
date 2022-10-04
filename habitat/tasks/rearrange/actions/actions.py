#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
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
        return self._sim.step(HabitatSimActions.EMPTY)


@registry.register_task_action
class RearrangeStopAction(SimulatorTaskAction):
    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.does_want_terminate = False

    def step(self, task, *args, is_last_action, **kwargs):
        should_stop = kwargs.get("REARRANGE_STOP", [1.0])
        if should_stop[0] > 0.0:
            rearrange_logger.debug(
                "Rearrange stop action requesting episode stop."
            )
            self.does_want_terminate = True

        if is_last_action:
            return self._sim.step(HabitatSimActions.REARRANGE_STOP)
        else:
            return {}


@registry.register_task_action
class ArmAction(RobotAction):
    """An arm control and grip control into one action space."""

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        arm_controller_cls = eval(self._config.ARM_CONTROLLER)
        self._sim: RearrangeSim = sim
        self.arm_ctrlr = arm_controller_cls(
            *args, config=config, sim=sim, **kwargs
        )

        if self._config.GRIP_CONTROLLER is not None:
            grip_controller_cls = eval(self._config.GRIP_CONTROLLER)
            self.grip_ctrlr: Optional[
                GripSimulatorTaskAction
            ] = grip_controller_cls(*args, config=config, sim=sim, **kwargs)
        else:
            self.grip_ctrlr = None

        self.disable_grip = False
        if "DISABLE_GRIP" in config:
            self.disable_grip = config["DISABLE_GRIP"]

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
            return self._sim.step(HabitatSimActions.ARM_ACTION)
        else:
            return {}


@registry.register_task_action
class ArmRelPosAction(RobotAction):
    """
    The arm motor targets are offset by the delta joint values specified by the
    action
    """

    @property
    def action_space(self):
        return spaces.Box(
            shape=(self._config.ARM_JOINT_DIMENSIONALITY,),
            low=-1,
            high=1,
            dtype=np.float32,
        )

    def step(self, delta_pos, should_step=True, *args, **kwargs):
        # clip from -1 to 1
        delta_pos = np.clip(delta_pos, -1, 1)
        delta_pos *= self._config.DELTA_POS_LIMIT
        # The actual joint positions
        self._sim: RearrangeSim
        self.cur_robot.arm_motor_pos = delta_pos + self.cur_robot.arm_motor_pos


@registry.register_task_action
class ArmRelPosKinematicAction(RobotAction):
    """
    The arm motor targets are offset by the delta joint values specified by the
    action
    """

    @property
    def action_space(self):
        return spaces.Box(
            shape=(self._config.ARM_JOINT_DIMENSIONALITY,),
            low=0,
            high=1,
            dtype=np.float32,
        )

    def step(self, delta_pos, *args, **kwargs):
        if self._config.get("SHOULD_CLIP", True):
            # clip from -1 to 1
            delta_pos = np.clip(delta_pos, -1, 1)
        delta_pos *= self._config.DELTA_POS_LIMIT
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
            shape=(self._config.ARM_JOINT_DIMENSIONALITY,),
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
            shape=(self._config.ARM_JOINT_DIMENSIONALITY,),
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
class BaseVelAction(RobotAction):
    """
    The robot base motion is constrained to the NavMesh and controlled with velocity commands integrated with the VelocityControl interface.

    Optionally cull states with active collisions if config parameter `ALLOW_DYN_SLIDE` is True
    """

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._sim: RearrangeSim = sim
        self.base_vel_ctrl = habitat_sim.physics.VelocityControl()
        self.base_vel_ctrl.controlling_lin_vel = True
        self.base_vel_ctrl.lin_vel_is_local = True
        self.base_vel_ctrl.controlling_ang_vel = True
        self.base_vel_ctrl.ang_vel_is_local = True
        self.prev_collision_free_point = None

    @property
    def end_on_stop(self):
        return self._config.END_ON_STOP

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

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.does_want_terminate = False

    def update_base(self):

        ctrl_freq = self._sim.ctrl_freq

        before_trans_state = self._capture_robot_state()
        cur_state = self._sim.robot.sim_obj.transformation
        trans = self.cur_robot.sim_obj.transformation
        rigid_state = habitat_sim.RigidState(
            mn.Quaternion.from_matrix(trans.rotation()), trans.translation
        )
        target_rigid_state = self.base_vel_ctrl.integrate_transform(
            1 / ctrl_freq, rigid_state
        )

        # Conduct the collision detection
        has_attr = hasattr(
            self._sim.habitat_config, "COLLISION_DETECTION_METHOD"
        )
        if (
            has_attr
            and self._sim.habitat_config.COLLISION_DETECTION_METHOD
            == "ContactTestRevert"
        ):
            end_pos = target_rigid_state.translation
            proposed_target_trans = mn.Matrix4.from_(
                target_rigid_state.rotation.to_matrix(), end_pos
            )
            self._sim.robot.sim_obj.transformation = proposed_target_trans
            if self._sim.contact_test(self.cur_robot.sim_obj.object_id):
                self._sim.robot.sim_obj.transformation = cur_state
        elif (
            has_attr
            and self._sim.habitat_config.COLLISION_DETECTION_METHOD
            == "ContactTestProj"
        ):
            did_collide = self._sim.contact_test(
                self.cur_robot.sim_obj.object_id
            )
            robot_id = self.cur_robot.sim_obj.object_id
            end_pos = target_rigid_state.translation

            if did_collide:
                cur_pos = rigid_state.translation
                target_pos = target_rigid_state.translation
                move_dir = target_pos - cur_pos

                # Find the point that is closed to the move_dir
                contact_points = self._sim.get_physics_contact_points()
                num_contact_points = len(
                    self._sim.get_physics_contact_points()
                )
                robot_contact_list = []
                for i in range(num_contact_points):
                    if robot_id == contact_points[i].object_id_a:
                        robot_contact_list.append(contact_points[i])
                # Perform sliding
                contact_points = robot_contact_list
                if len(contact_points) != 0:
                    # Get the average normal vector
                    n_vec = contact_points[0].contact_normal_on_b_in_ws
                    for i in range(1, len(contact_points)):
                        n_vec += contact_points[i].contact_normal_on_b_in_ws
                    n_vec = n_vec / len(contact_points)
                    # Get the average contact point
                    c_pot = contact_points[0].position_on_a_in_ws
                    for i in range(1, len(contact_points)):
                        c_pot += contact_points[i].position_on_a_in_ws
                    c_pot = c_pot / len(contact_points)
                    # Get the next target point
                    p_pot = target_rigid_state.translation
                    # Do the projection
                    s0 = n_vec[0] * c_pot[0] - n_vec[0] * p_pot[0]
                    s2 = n_vec[2] * c_pot[2] - n_vec[2] * p_pot[2]
                    s = s0 + s2
                    proj_pot = p_pot + s * n_vec
                    # Get the final movement vector
                    move_vec = target_pos - proj_pot
                    move_vec[1] = 0
                    end_pos = rigid_state.translation + move_vec
            target_trans = mn.Matrix4.from_(
                target_rigid_state.rotation.to_matrix(), end_pos
            )
            self.cur_robot.sim_obj.transformation = target_trans
        elif (
            has_attr
            and self._sim.habitat_config.COLLISION_DETECTION_METHOD
            == "ContactTestProjRevert"
        ):
            did_collide = self._sim.contact_test(
                self.cur_robot.sim_obj.object_id
            )
            robot_id = self.cur_robot.sim_obj.object_id
            end_pos = target_rigid_state.translation

            if did_collide:
                # import pdb;pdb.set_trace()
                cur_pos = rigid_state.translation
                target_pos = target_rigid_state.translation
                move_dir = target_pos - cur_pos

                # Find the point that is closed to the move_dir
                contact_points = self._sim.get_physics_contact_points()
                num_contact_points = len(
                    self._sim.get_physics_contact_points()
                )
                robot_contact_list = []
                for i in range(num_contact_points):
                    if robot_id == contact_points[i].object_id_a:
                        robot_contact_list.append(contact_points[i])
                # Perform sliding
                contact_points = robot_contact_list
                if len(contact_points) != 0:
                    # Get the average normal vector
                    n_vec = contact_points[0].contact_normal_on_b_in_ws
                    for i in range(1, len(contact_points)):
                        n_vec += contact_points[i].contact_normal_on_b_in_ws
                    n_vec = n_vec / len(contact_points)
                    # Get the average contact point
                    c_pot = contact_points[0].position_on_a_in_ws
                    for i in range(1, len(contact_points)):
                        c_pot += contact_points[i].position_on_a_in_ws
                    c_pot = c_pot / len(contact_points)
                    # Get the next target point
                    p_pot = target_rigid_state.translation
                    # Do the projection
                    s0 = n_vec[0] * c_pot[0] - n_vec[0] * p_pot[0]
                    s2 = n_vec[2] * c_pot[2] - n_vec[2] * p_pot[2]
                    s = s0 + s2
                    proj_pot = p_pot + s * n_vec
                    # Get the final movement vector
                    move_vec = target_pos - proj_pot
                    move_vec[1] = 0
                    # Project the move vector the plane
                    v0 = (
                        move_dir[0] * n_vec[0]
                        + move_dir[1] * n_vec[1]
                        + move_dir[2] * n_vec[2]
                    )
                    v1 = n_vec[0] ** 2 + n_vec[1] ** 2 + n_vec[2] ** 2
                    proj_move_vec_plane = move_dir - v0 / v1 * n_vec
                    proj_move_vec_plane[1] = 0
                    # Add together
                    end_pos = (
                        rigid_state.translation
                        + 0.1 * move_vec
                        + 0.1 * proj_move_vec_plane
                    )
                proposed_target_trans = mn.Matrix4.from_(
                    target_rigid_state.rotation.to_matrix(), end_pos
                )
                self._sim.robot.sim_obj.transformation = proposed_target_trans
                if self._sim.contact_test(self.cur_robot.sim_obj.object_id):
                    self.end_pos = self.prev_collision_free_point

            self.prev_collision_free_point = rigid_state.translation
            target_trans = mn.Matrix4.from_(
                target_rigid_state.rotation.to_matrix(), end_pos
            )
            self.cur_robot.sim_obj.transformation = target_trans

        elif (
            has_attr
            and self._sim.habitat_config.COLLISION_DETECTION_METHOD
            == "ContactTestProjPreCheckRevert"
        ):
            end_pos = target_rigid_state.translation
            proposed_target_trans = mn.Matrix4.from_(
                target_rigid_state.rotation.to_matrix(), end_pos
            )
            self._sim.robot.sim_obj.transformation = proposed_target_trans
            if self._sim.contact_test(self.cur_robot.sim_obj.object_id):
                cur_pos = rigid_state.translation
                target_pos = target_rigid_state.translation
                move_dir = target_pos - cur_pos

                # Find the point that is closed to the move_dir
                contact_points = self._sim.get_physics_contact_points()
                num_contact_points = len(
                    self._sim.get_physics_contact_points()
                )
                robot_id = self.cur_robot.sim_obj.object_id
                robot_contact_list = []
                for i in range(num_contact_points):
                    # if robot_id == contact_points[i].object_id_a:
                    robot_contact_list.append(contact_points[i])
                # Perform sliding
                contact_points = robot_contact_list
                if len(contact_points) != 0:
                    # Get the average normal vector
                    n_vec = contact_points[0].contact_normal_on_b_in_ws
                    for i in range(1, len(contact_points)):
                        n_vec += contact_points[i].contact_normal_on_b_in_ws
                    n_vec = n_vec / len(contact_points)
                    # Get the average contact point
                    c_pot = contact_points[0].position_on_a_in_ws
                    for i in range(1, len(contact_points)):
                        c_pot += contact_points[i].position_on_a_in_ws
                    c_pot = c_pot / len(contact_points)
                    # Get the next target point
                    p_pot = target_rigid_state.translation
                    # Do the projection
                    s0 = n_vec[0] * c_pot[0] - n_vec[0] * p_pot[0]
                    s2 = n_vec[2] * c_pot[2] - n_vec[2] * p_pot[2]
                    s = s0 + s2
                    proj_pot = p_pot + s * n_vec
                    # Get the final movement vector
                    move_vec = target_pos - proj_pot
                    move_vec[1] = 0
                    end_pos = rigid_state.translation + move_vec
                # Check the proposed projection point
                proposed_target_trans = mn.Matrix4.from_(
                    target_rigid_state.rotation.to_matrix(), end_pos
                )
                self._sim.robot.sim_obj.transformation = proposed_target_trans
                # If it collides, we should revert to its previous state
                if self._sim.contact_test(self.cur_robot.sim_obj.object_id):
                    self._sim.robot.sim_obj.transformation = cur_state
        elif (
            ~has_attr
            or self._sim.habitat_config.COLLISION_DETECTION_METHOD == "NevMesh"
        ):
            end_pos = self._sim.step_filter(
                rigid_state.translation, target_rigid_state.translation
            )
            # Offset the end position
            end_pos -= self.cur_robot.params.base_offset
            target_trans = mn.Matrix4.from_(
                target_rigid_state.rotation.to_matrix(), end_pos
            )
            self.cur_robot.sim_obj.transformation = target_trans

        # Fix the leg joints
        self.cur_robot.leg_joint_pos = [0.0, 0.7, -1.5] * 4

        if not self._config.get("ALLOW_DYN_SLIDE", True):
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
        lin_vel = np.clip(lin_vel, -1, 1) * self._config.LIN_SPEED
        ang_vel = np.clip(ang_vel, -1, 1) * self._config.ANG_SPEED
        if not self._config.ALLOW_BACK:
            lin_vel = np.maximum(lin_vel, 0)

        if (
            abs(lin_vel) < self._config.MIN_ABS_LIN_SPEED
            and abs(ang_vel) < self._config.MIN_ABS_ANG_SPEED
        ):
            self.does_want_terminate = True
        else:
            self.does_want_terminate = False

        self.base_vel_ctrl.linear_velocity = mn.Vector3(lin_vel, 0, 0)
        self.base_vel_ctrl.angular_velocity = mn.Vector3(0, ang_vel, 0)

        if lin_vel != 0.0 or ang_vel != 0.0:
            self.update_base()

        if is_last_action:
            return self._sim.step(HabitatSimActions.BASE_VELOCITY)
        else:
            return {}


@registry.register_task_action
class ArmEEAction(RobotAction):
    """Uses inverse kinematics (requires pybullet) to apply end-effector position control for the robot's arm."""

    def __init__(self, *args, sim: RearrangeSim, **kwargs):
        self.ee_target: Optional[np.ndarray] = None
        super().__init__(*args, sim=sim, **kwargs)
        self._sim: RearrangeSim = sim

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
        ee_pos *= self._config.EE_CTRL_LIM
        self.set_desired_ee_pos(ee_pos)

        if self._config.get("RENDER_EE_TARGET", False):
            global_pos = self._sim.robot.base_transformation.transform_point(
                self.ee_target
            )
            self._sim.viz_ids["ee_target"] = self._sim.visualize_position(
                global_pos, self._sim.viz_ids["ee_target"]
            )
