#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
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
class ArmRelPosKinematicReducedAction(RobotAction):
    """
    The arm motor targets are offset by the delta joint values specified by the
    action
    """

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self.last_arm_action = None
        self.last_no_contact_angle = []
        self.kinematic = True

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.last_arm_action = None
        self.last_no_contact_angle = []

    @property
    def action_space(self):
        self.step_c = 0
        return spaces.Box(
            shape=(self._config.ARM_JOINT_DIMENSIONALITY,),
            low=-1,
            high=1,
            dtype=np.float32,
        )

    def step(self, delta_pos, *args, **kwargs):
        if self._config.get("SHOULD_CLIP", True):
            # clip from -1 to 1
            delta_pos = np.clip(delta_pos, -1, 1)
        delta_pos *= self._config.DELTA_POS_LIMIT
        self._sim: RearrangeSim

        # # clip from -1 to 1
        # delta_pos = np.clip(delta_pos, -1, 1)
        # delta_pos *= self._config.DELTA_POS_LIMIT
        # # The actual joint positions
        # self._sim: RearrangeSim
        # self.cur_robot.arm_motor_pos = delta_pos + self.cur_robot.arm_motor_pos

        # if self.step_c >= 100:
        #     try:
        #         print(self._sim.contact_test(self.cur_robot.sim_obj.object_id))
        #     except:
        #         print("ee")
        #     try:
        #         link_ids = self.cur_robot.sim_obj.get_link_ids()
        #     except:
        #         print("ee")
        #     for ids in link_ids:
        #         try:
        #             if ids == 19:
        #                 break
        #             print(ids,'-',self._sim.contact_test(ids))
        #         except:
        #             print("ee")
        # self.step_c += 1

        # Expand delta_pos based on mask
        expanded_delta_pos = np.zeros(len(self._config.ARM_JOINT_MASK))
        src_idx = 0
        tgt_idx = 0
        for mask in self._config.ARM_JOINT_MASK:
            if mask == 0:
                tgt_idx += 1
                continue
            expanded_delta_pos[tgt_idx] = delta_pos[src_idx]
            tgt_idx += 1
            src_idx += 1

        min_limit, max_limit = self.cur_robot.arm_joint_limits
        if True:
            set_arm_pos = expanded_delta_pos + self.cur_robot.arm_motor_pos
        else:
            # self.cur_robot.arm_joint_pos drifts becuase that it is not
            # getting the information of the motor target position
            set_arm_pos = expanded_delta_pos + self.cur_robot.arm_joint_pos
        set_arm_pos = np.clip(set_arm_pos, min_limit, max_limit)
        # Reset to zero based on the mask
        for i, v in enumerate(self._config.ARM_JOINT_MASK):
            if v == 0:
                set_arm_pos[i] = 0

        contact = self.cur_robot._sim.contact_test(
            self.cur_robot.sim_obj.object_id
        )
        # If there is a contact, this means that the motor angle should revert to the
        # previous uncontact state.
        if False:  # contact:
            # This is the dynamics simulation
            if len(self.last_no_contact_angle) >= 5:
                if not self.kinematic:
                    self.cur_robot.arm_motor_pos = self.last_no_contact_angle[
                        -5
                    ]
                else:
                    self.cur_robot.arm_joint_pos = self.last_no_contact_angle[
                        -5
                    ]
            else:
                if not self.kinematic:
                    self.cur_robot.arm_motor_pos = set_arm_pos
                else:
                    self.cur_robot.arm_joint_pos = set_arm_pos
        else:
            # This means that there is no contact after applying the last motor angles. So we should
            # store this motor angles.
            if not self.kinematic:
                self.last_no_contact_angle.append(
                    self.cur_robot.arm_motor_pos.copy()
                )
                self.cur_robot.arm_motor_pos = set_arm_pos
            else:
                self.last_no_contact_angle.append(
                    self.cur_robot.arm_joint_pos.copy()
                )
                self.cur_robot.arm_joint_pos = set_arm_pos

        if len(self.last_no_contact_angle) >= 10:
            self.last_no_contact_angle[-10:]

        # #if self._config.PREVENT_PENETRATION:
        # if False:
        #     # Get the ee position
        #     ee_pos = self._sim.robot.ee_transform.translation
        #     # Get the arm bottom base position
        #     arm_base_pos = self._sim.robot.sim_obj.get_link_scene_node(0).transformation
        #     arm_base_pos.translation = arm_base_pos.transform_point(
        #         self._sim.robot.ee_local_offset
        #     )
        #     arm_base_pos = arm_base_pos.translation
        #     # Compute the directional vector for the arm and the base
        #     ee_dir_vec = ee_pos - arm_base_pos # For ee vector
        #     bs_dir_vec = self._sim.robot.base_pos - arm_base_pos # For base vector
        #     bs_dir_vec[1] = 0.0 # Remove the z direction offset
        #     # Find theta angle between two directional vectors
        #     angle = self.arccos(ee_dir_vec, bs_dir_vec)
        #     if angle <= np.pi/2.0:
        #         self.cur_robot.arm_motor_pos = self.last_arm_action
        #     elif angle > np.pi/2.0+0.05:
        #         self.last_arm_action = self.cur_robot.arm_motor_pos

    def arccos(self, v1, v2):
        inner_product = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
        norm_v1 = (v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2) ** 0.5
        norm_v2 = (v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2) ** 0.5
        return np.arccos(inner_product / (norm_v1 * norm_v2))


@registry.register_task_action
class ArmRelPosReducedActionStretch(RobotAction):
    """
    The arm motor targets are offset by the delta joint values specified by the
    action
    """

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self.last_arm_action = None
        self.control_method = self._config.CONTROL_METHOD
        self.contact_detection = self._config.CONTACT_DETECTION
        self.contact = False
        self.previous_no_contact_arm_joint_pos_list = []

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.last_arm_action = None
        self.contact = False
        self.previous_no_contact_arm_joint_pos_list = []
        for i in range(5):
            self.previous_no_contact_arm_joint_pos_list.append(
                list(self.cur_robot.arm_joint_pos)
            )

    @property
    def action_space(self):
        return spaces.Box(
            shape=(self._config.ARM_JOINT_DIMENSIONALITY,),
            low=-1,
            high=1,
            dtype=np.float32,
        )

    def step(self, delta_pos, *args, **kwargs):
        if self._config.get("SHOULD_CLIP", True):
            # clip from -1 to 1
            delta_pos = np.clip(delta_pos, -1, 1)
        delta_pos *= self._config.DELTA_POS_LIMIT
        self._sim: RearrangeSim

        # Expand delta_pos based on mask
        expanded_delta_pos = np.zeros(len(self._config.ARM_JOINT_MASK))
        src_idx = 0
        tgt_idx = 0
        for mask in self._config.ARM_JOINT_MASK:
            if mask == 0:
                tgt_idx += 1
                src_idx += 1
                continue
            expanded_delta_pos[tgt_idx] = delta_pos[src_idx]
            tgt_idx += 1
            src_idx += 1

        # Get the max and min of each arm joint
        min_limit, max_limit = self.cur_robot.arm_joint_limits
        # Determine the arm position by adding the control value into
        # the previous arm motor position. Note that self.cur_robot.arm_motor_pos
        # will output the target motor angle, which is fixed unless you
        # motify it. In contrast, arm_joint_pos drifts.
        set_arm_pos = expanded_delta_pos + self.cur_robot.arm_motor_pos

        # Perform roll over to the joints.
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

        # Clip the arm.
        set_arm_pos = np.clip(set_arm_pos, min_limit, max_limit)
        # Get the current arm position.
        if self.control_method == "dynamic":
            # For the dynamics simulation, since the true joint angle, arm_joint_pos, needs
            # some time to reach the arm_motor_pos angles, we have to wait until arm does
            # not have collision.
            contact = self.cur_robot._sim.contact_test(
                self.cur_robot.sim_obj.object_id
            )
            # We want to make sure that there are at least two consecutive states the robot
            # does not collide with the object.
            # Case #1: contact True and self.contact False: the arm contorl input produces the collision
            # Case #2: contact False and self.contact True: the robot is going to revert the collision angles
            if not contact and not self.contact:
                cur_arm_pos = self.cur_robot.arm_motor_pos.copy()
                if (
                    list(cur_arm_pos)
                    not in self.previous_no_contact_arm_joint_pos_list
                ):
                    self.previous_no_contact_arm_joint_pos_list.append(
                        list(cur_arm_pos)
                    )

        elif self.control_method == "kinematic":
            cur_arm_pos = self.cur_robot.arm_motor_pos.copy()
        else:
            print("There is no such a simulation method")

        # Set the motor position.
        if self.control_method == "dynamic":
            self.cur_robot.arm_motor_pos = set_arm_pos
        elif self.control_method == "kinematic":
            self.cur_robot.arm_joint_pos = set_arm_pos
        else:
            print("There is no such a simulation method")

        # If there is a contact after controlling the arm
        contact = self.cur_robot._sim.contact_test(
            self.cur_robot.sim_obj.object_id
        )
        self.contact = contact
        if contact and self.contact_detection:
            # Revert the motor position.
            if self.control_method == "dynamic":
                cur_arm_pos = np.array(
                    self.previous_no_contact_arm_joint_pos_list[-5]
                )
                self.cur_robot.arm_motor_pos = cur_arm_pos
                if len(self.previous_no_contact_arm_joint_pos_list) > 10:
                    self.previous_no_contact_arm_joint_pos_list = (
                        self.previous_no_contact_arm_joint_pos_list[-10:]
                    )
            elif self.control_method == "kinematic":
                self.cur_robot.arm_joint_pos = cur_arm_pos
            else:
                print("There is no such a simulation method")


@registry.register_task_action
class ArmRelPosReducedActionSpot(RobotAction):
    """
    The arm motor targets are offset by the delta joint values specified by the
    action
    """

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self.last_arm_action = None
        self.control_method = self._config.CONTROL_METHOD
        self.contact_detection = self._config.CONTACT_DETECTION
        self.contact = False
        self.previous_no_contact_arm_joint_pos_list = []

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.last_arm_action = None
        self.contact = False
        self.previous_no_contact_arm_joint_pos_list = []
        for i in range(5):
            self.previous_no_contact_arm_joint_pos_list.append(
                list(self.cur_robot.arm_joint_pos)
            )

    @property
    def action_space(self):
        return spaces.Box(
            shape=(self._config.ARM_JOINT_DIMENSIONALITY,),
            low=-1,
            high=1,
            dtype=np.float32,
        )

    def step(self, delta_pos, *args, **kwargs):
        if self._config.get("SHOULD_CLIP", True):
            # clip from -1 to 1
            delta_pos = np.clip(delta_pos, -1, 1)
        delta_pos *= self._config.DELTA_POS_LIMIT
        self._sim: RearrangeSim

        # Expand delta_pos based on mask
        expanded_delta_pos = np.zeros(len(self._config.ARM_JOINT_MASK))
        src_idx = 0
        tgt_idx = 0
        for mask in self._config.ARM_JOINT_MASK:
            if mask == 0:
                tgt_idx += 1
                src_idx += 1
                continue
            expanded_delta_pos[tgt_idx] = delta_pos[src_idx]
            tgt_idx += 1
            src_idx += 1

        # Get the max and min of each arm joint
        min_limit, max_limit = self.cur_robot.arm_joint_limits
        # Determine the arm position by adding the control value into
        # the previous arm motor position. Note that self.cur_robot.arm_motor_pos
        # will output the target motor angle, which is fixed unless you
        # motify it. In contrast, arm_joint_pos drifts.
        set_arm_pos = expanded_delta_pos + self.cur_robot.arm_motor_pos
        # Clip the arm.
        set_arm_pos = np.clip(set_arm_pos, min_limit, max_limit)
        # Get the current arm position.
        if self.control_method == "dynamic":
            # For the dynamics simulation, since the true joint angle, arm_joint_pos, needs
            # some time to reach the arm_motor_pos angles, we have to wait until arm does
            # not have collision.
            contact = self.cur_robot._sim.contact_test(
                self.cur_robot.sim_obj.object_id
            )
            # We want to make sure that there are at least two consecutive states the robot
            # does not collide with the object.
            # Case #1: contact True and self.contact False: the arm contorl input produces the collision
            # Case #2: contact False and self.contact True: the robot is going to revert the collision angles
            if not contact and not self.contact:
                cur_arm_pos = self.cur_robot.arm_motor_pos.copy()
                if (
                    list(cur_arm_pos)
                    not in self.previous_no_contact_arm_joint_pos_list
                ):
                    self.previous_no_contact_arm_joint_pos_list.append(
                        list(cur_arm_pos)
                    )

        elif self.control_method == "kinematic":
            cur_arm_pos = self.cur_robot.arm_motor_pos.copy()
        else:
            print("There is no such a simulation method")

        # Set the motor position.
        if self.control_method == "dynamic":
            self.cur_robot.arm_motor_pos = set_arm_pos
        elif self.control_method == "kinematic":
            self.cur_robot.arm_joint_pos = set_arm_pos
        else:
            print("There is no such a simulation method")

        # If there is a contact after controlling the arm
        contact = self.cur_robot._sim.contact_test(
            self.cur_robot.sim_obj.object_id
        )
        self.contact = contact
        if contact and self.contact_detection:
            # Revert the motor position.
            if self.control_method == "dynamic":
                cur_arm_pos = np.array(
                    self.previous_no_contact_arm_joint_pos_list[-5]
                )
                self.cur_robot.arm_motor_pos = cur_arm_pos
                if len(self.previous_no_contact_arm_joint_pos_list) > 10:
                    self.previous_no_contact_arm_joint_pos_list = (
                        self.previous_no_contact_arm_joint_pos_list[-10:]
                    )
            elif self.control_method == "kinematic":
                self.cur_robot.arm_joint_pos = cur_arm_pos
            else:
                print("There is no such a simulation method")


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

        self.data_time_1 = []
        self.data_time_2 = []
        self.counter = 0

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

    def get_collisions(self):
        def extract_coll_info(coll, n_point):
            parts = coll.split(",")
            coll_type, name, link = parts[:3]
            return {
                "type": coll_type.strip(),
                "name": name.strip(),
                "link": link.strip(),
                "n_points": n_point,
            }

        sum_str = self._sim.get_physics_step_collision_summary()
        colls = sum_str.split("\n")
        if "no active" in colls[0]:
            return []
        n_points = [
            int(c.split(",")[-1].strip().split(" ")[0])
            for c in colls
            if c != ""
        ]
        colls = [
            [extract_coll_info(x, n) for x in re.findall("\[(.*?)\]", s)]
            for n, s in zip(n_points, colls)
        ]
        colls = [x for x in colls if len(x) != 0]
        return colls

    def check_step(self):
        before_trans_state = self._capture_robot_state()
        trans = self.cur_robot.sim_obj.transformation

        if not self._config.get("ALLOW_DYN_SLIDE", True):
            # Check if in the new robot state the arm collides with anything.
            # If so we have to revert back to the previous transform
            self._sim.internal_step(-1)
            colls = self.get_collisions()
            did_coll, _ = rearrange_collision(
                colls, self.cur_grasp_mgr._snapped_obj_id, False
            )
            did_coll = False
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

        if self._config.VEL_BASE_MASK is not None:
            if self._config.VEL_BASE_MASK[0] == 0:
                lin_vel = 0.0
            if self._config.VEL_BASE_MASK[1] == 0:
                ang_vel = 0.0

        self.base_vel_ctrl.linear_velocity = mn.Vector3(lin_vel, 0, 0)
        self.base_vel_ctrl.angular_velocity = mn.Vector3(0, ang_vel, 0)

        height = self.cur_robot.base_pos[1]

        if lin_vel != 0.0 or ang_vel != 0.0:
            trans = self.cur_robot.sim_obj.transformation
            rigid_state = habitat_sim.RigidState(
                mn.Quaternion.from_matrix(trans.rotation()), trans.translation
            )
            target_rigid_state = self.base_vel_ctrl.integrate_transform(
                1 / self._sim.ctrl_freq, rigid_state
            )
            # import time
            # start_time = time.time()
            # self._sim.contact_test(self.cur_robot.sim_obj.object_id)
            # elapsed_time_1 = time.time() - start_time
            # start_time = time.time()
            # self._sim.step_filter(rigid_state.translation, target_rigid_state.translation)
            # elapsed_time_2 = time.time() - start_time

            self.cur_robot.update_base(rigid_state, target_rigid_state)

            # self.data_time_1.append(elapsed_time_1)
            # self.data_time_2.append(elapsed_time_2)
            self.check_step()

        cur_pos = self.cur_robot.base_pos
        cur_pos[1] = height
        self.cur_robot.base_pos = cur_pos

        self.counter += 1

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
