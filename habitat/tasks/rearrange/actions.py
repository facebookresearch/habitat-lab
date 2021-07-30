#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
import numpy as np
from gym import spaces

import habitat_sim
from habitat.core.embodied_task import SimulatorTaskAction
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.utils import rearrange_collision


@registry.register_task_action
class EmptyAction(SimulatorTaskAction):
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
class ArmAction(SimulatorTaskAction):
    """An arm control and grip control into one action space."""

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        arm_controller_cls = eval(self._config.ARM_CONTROLLER)
        self.arm_ctrlr = arm_controller_cls(
            *args, config=config, sim=sim, **kwargs
        )

        if self._config.GRIP_CONTROLLER is not None:
            grip_controller_cls = eval(self._config.GRIP_CONTROLLER)
            self.grip_ctrlr = grip_controller_cls(
                *args, config=config, sim=sim, **kwargs
            )
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
            "arm_action": self.arm_ctrlr.action_space,
        }
        if self.grip_ctrlr is not None:
            action_spaces["grip_action"] = self.grip_ctrlr.action_space
        return spaces.Dict(action_spaces)

    def step(self, arm_action, grip_action=None, *args, **kwargs):
        self.arm_ctrlr.step(arm_action, should_step=False)
        if self.grip_ctrlr is not None and not self.disable_grip:
            self.grip_ctrlr.step(grip_action, should_step=False)
        return self._sim.step(HabitatSimActions.ARM_ACTION)


@registry.register_task_action
class MagicGraspAction(SimulatorTaskAction):
    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._sim: RearrangeSim = sim

    @property
    def action_space(self):
        return spaces.Box(shape=(1,), high=1.0, low=-1.0)

    def _grasp(self):
        scene_obj_pos = self._sim.get_scene_pos()
        ee_pos = self._sim.robot.ee_transform.translation
        if len(scene_obj_pos) != 0:
            # Get the target the EE is closest to.
            closest_obj_idx = np.argmin(
                np.linalg.norm(scene_obj_pos - ee_pos, ord=2, axis=-1)
            )

            closest_obj_pos = scene_obj_pos[closest_obj_idx]
            to_target = np.linalg.norm(ee_pos - closest_obj_pos, ord=2)
            sim_idx = self._sim.scene_obj_ids[closest_obj_idx]

            if to_target < self._config.GRASP_THRESH_DIST:
                self._sim.grasp_mgr.snap_to_obj(sim_idx)

    def _ungrasp(self):
        self._sim.grasp_mgr.desnap()

    def step(self, grip_action, should_step=True, *args, **kwargs):
        if grip_action is None:
            return
        if grip_action >= 0 and not self._sim.grasp_mgr.is_grasped:
            self._grasp()
        elif grip_action < 0 and self._sim.grasp_mgr.is_grasped:
            self._ungrasp()


@registry.register_task_action
class ArmRelPosAction(SimulatorTaskAction):
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

    def step(self, delta_pos, should_step=True, *args, **kwargs):
        # clip from -1 to 1
        delta_pos = np.clip(delta_pos, -1, 1)
        delta_pos *= self._config.DELTA_POS_LIMIT
        # The actual joint positions
        self._sim: RearrangeSim
        self._sim.robot.arm_motor_pos = (
            delta_pos + self._sim.robot.arm_motor_pos
        )
        if should_step:
            return self._sim.step(HabitatSimActions.ARM_VEL)
        return None


@registry.register_task_action
class ArmRelPosKinematicAction(SimulatorTaskAction):
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

    def step(self, delta_pos, should_step=True, *args, **kwargs):
        if self._config.get("SHOULD_CLIP", True):
            print("CLIPPING!")
            # clip from -1 to 1
            delta_pos = np.clip(delta_pos, -1, 1)
        delta_pos *= self._config.DELTA_POS_LIMIT
        # The actual joint positions
        self._sim: RearrangeSim
        self._sim.robot.arm_joint_pos = (
            delta_pos + self._sim.robot.arm_joint_pos
        )
        if should_step:
            return self._sim.step(HabitatSimActions.ARM_VEL)
        return None


@registry.register_task_action
class ArmAbsPosAction(SimulatorTaskAction):
    """
    The arm motor targets are directly set to the joint configuration specified by the
    action.
    """

    @property
    def action_space(self):
        return spaces.Box(
            shape=(self._config.ARM_JOINT_DIMENSIONALITY,),
            low=0,
            high=1,
            dtype=np.float32,
        )

    def step(self, set_pos, should_step=True, *args, **kwargs):
        # No clipping because the arm is being set to exactly where it needs to
        # go.
        self._sim: RearrangeSim
        self._sim.robot.arm_motor_pos = set_pos
        if should_step:
            return self._sim.step(HabitatSimActions.ARM_ABS_POS)
        else:
            return None


@registry.register_task_action
class ArmAbsPosKinematicAction(SimulatorTaskAction):
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

    def step(self, set_pos, should_step=True, *args, **kwargs):
        # No clipping because the arm is being set to exactly where it needs to
        # go.
        self._sim: RearrangeSim
        self._sim.robot.arm_joint_pos = set_pos
        if should_step:
            return self._sim.step(HabitatSimActions.ARM_ABS_POS_KINEMATIC)
        else:
            return None


@registry.register_task_action
class BaseVelAction(SimulatorTaskAction):
    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._sim: RearrangeSim = sim
        self.base_vel_ctrl = habitat_sim.physics.VelocityControl()
        self.base_vel_ctrl.controlling_lin_vel = True
        self.base_vel_ctrl.lin_vel_is_local = True
        self.base_vel_ctrl.controlling_ang_vel = True
        self.base_vel_ctrl.ang_vel_is_local = True

        self.end_on_stop = self._config.get("END_ON_STOP", False)

    @property
    def action_space(self):
        lim = 20
        return spaces.Box(shape=(2,), low=-lim, high=lim, dtype=np.float32)

    def _capture_robot_state(self, sim):
        return {
            "forces": sim.robot.sim_obj.joint_forces,
            "vel": sim.robot.sim_obj.joint_velocities,
            "pos": sim.robot.sim_obj.joint_positions,
        }

    def _set_robot_state(self, sim: RearrangeSim, set_dat):
        sim.robot.sim_obj.joint_positions = set_dat["forces"]
        sim.robot.sim_obj.joint_velocities = set_dat["vel"]
        sim.robot.sim_obj.joint_forces = set_dat["pos"]

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.does_want_terminate = False

    def update_base(self):
        ctrl_freq = self._sim.ctrl_freq

        before_trans_state = self._capture_robot_state(self._sim)

        trans = self._sim.robot.sim_obj.transformation
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
        self._sim.robot.sim_obj.transformation = target_trans

        if not self._config.get("ALLOW_DYN_SLIDE", True):
            # Check if in the new robot state the arm collides with anything. If so
            # we have to revert back to the previous transform
            self._sim.internal_step(-1)
            colls = self._sim.get_collisions()
            did_coll, _ = rearrange_collision(
                colls, self._sim.snapped_obj_id, False
            )
            if did_coll:
                # Don't allow the step, revert back.
                self._set_robot_state(self._sim, before_trans_state)
                self._sim.robot.sim_obj.transformation = trans

    def step(self, base_vel, should_step=True, *args, **kwargs):
        lin_vel, ang_vel = base_vel
        lin_vel = np.clip(lin_vel, -1, 1)
        lin_vel *= self._config.LIN_SPEED
        ang_vel = np.clip(ang_vel, -1, 1) * self._config.ANG_SPEED

        if (
            self.end_on_stop
            and abs(lin_vel) < self._config.MIN_ABS_LIN_SPEED
            and abs(ang_vel) < self._config.MIN_ABS_ANG_SPEED
        ):
            self.does_want_terminate = True

        self.base_vel_ctrl.linear_velocity = mn.Vector3(lin_vel, 0, 0)
        self.base_vel_ctrl.angular_velocity = mn.Vector3(0, ang_vel, 0)

        if lin_vel != 0.0 or ang_vel != 0.0:
            self.update_base()

        if should_step:
            return self._sim.step(HabitatSimActions.BASE_VELOCITY)
        else:
            return None


@registry.register_task_action
class ArmEEAction(SimulatorTaskAction):
    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        self.ee_target = None
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._sim: RearrangeSim = sim
        self.robot_ee_constraints = np.array(
            [
                [0.4, 1.2],
                [-0.7, 0.7],
                [0.25, 1.5],
            ]
        )

    def reset(self, *args, **kwargs):
        super().reset()
        self.ee_target = np.zeros((3,))

        arm_pos = self.set_desired_ee_pos(np.array([0.5, 0.0, 1.0]))

        self._sim.robot.arm_joint_pos = arm_pos
        self._sim.settle_sim(0.1)

    @property
    def action_space(self):
        return spaces.Box(shape=(3,), low=-1, high=1, dtype=np.float32)

    def apply_ee_constraints(self):
        self.ee_target = np.clip(
            self.ee_target,
            self.robot_ee_constraints[:, 0],
            self.robot_ee_constraints[:, 1],
        )

    def set_desired_ee_pos(self, ee_pos: np.ndarray) -> np.ndarray:
        self.ee_target += np.array(ee_pos)

        self.apply_ee_constraints()

        ik = self._sim.ik_helper

        joint_pos = np.array(self._sim.robot.arm_joint_pos)
        joint_vel = np.zeros(joint_pos.shape)

        ik.set_arm_state(joint_pos, joint_vel)

        des_joint_pos = ik.calc_ik(self.ee_target)
        des_joint_pos = list(des_joint_pos)
        self._sim.robot.arm_motor_pos = des_joint_pos

        return des_joint_pos

    def step(self, ee_pos, should_step=True, **kwargs):
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

        if should_step:
            return self._sim.step(HabitatSimActions.ARM_EE)
        else:
            return None
