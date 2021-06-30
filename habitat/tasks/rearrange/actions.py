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
from habitat.tasks.rearrange.utils import rearrang_collision


@registry.register_task_action
class ArmAction(SimulatorTaskAction):
    """An arm control and grip control into one action space."""

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        arm_controller_cls = eval(self._config.ARM_CONTROLLER)
        grip_controller_cls = eval(self._config.GRIP_CONTROLLER)
        self.arm_ctrlr = arm_controller_cls(
            *args, config=config, sim=sim, **kwargs
        )
        self.grip_ctrlr = grip_controller_cls(
            *args, config=config, sim=sim, **kwargs
        )
        self.disable_grip = False
        if "DISABLE_GRIP" in config:
            self.disable_grip = config["DISABLE_GRIP"]

    def reset(self, *args, **kwargs):
        self.arm_ctrlr.reset(*args, **kwargs)
        self.grip_ctrlr.reset(*args, **kwargs)

    @property
    def action_space(self):
        return spaces.Dict(
            {
                "arm_ac": self.arm_ctrlr.action_space,
                "grip_ac": self.grip_ctrlr.action_space,
            }
        )

    def step(self, arm_ac, grip_ac, **kwargs):
        self.arm_ctrlr.step(arm_ac, should_step=False)
        if grip_ac is not None and not self.disable_grip:
            self.grip_ctrlr.step(grip_ac, should_step=False)
        return self._sim.step(HabitatSimActions.ARM_ACTION)


@registry.register_task_action
class MagicGraspAction(SimulatorTaskAction):
    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._sim: RearrangeSim = sim
        self.thresh_dist = config.GRASP_THRESH_DIST
        self.snap_markers = config.get("GRASP_MARKERS", False)

    @property
    def action_space(self):
        return spaces.Discrete(1)

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

            if to_target < self.thresh_dist:
                self._sim.set_snapped_obj(closest_obj_idx)

        # Get the marker the EE is closest to.
        marker_name_pos = self._sim.get_marker_positions()
        if self.snap_markers and len(marker_name_pos) > 0:
            marker_name, marker_pos = zip(*marker_name_pos.items())
            marker_pos = np.array(marker_pos)
            closest_marker_idx = np.argmin(
                np.linalg.norm(ee_pos - marker_pos, axis=-1)
            )
            closest_marker_pos = marker_pos[closest_marker_idx]
            to_marker = np.linalg.norm(ee_pos - closest_marker_pos)
            if to_marker < self.thresh_dist:
                self._sim.set_snapped_marker(marker_name[closest_marker_idx])

    def step(self, state, should_step=True, **kwargs):
        return


@registry.register_task_action
class ArmVelAction(SimulatorTaskAction):
    @property
    def action_space(self):
        return spaces.Box(shape=(7,), low=0, high=1, dtype=np.float32)

    def step(self, vel, should_step=True, **kwargs):
        # clip from -1 to 1
        vel = np.clip(vel, -1, 1)
        vel *= self._config.VEL_CTRL_LIM
        # The actual joint positions
        self._sim: RearrangeSim
        self._sim.robot.arm_motor_pos = vel + self._sim.robot.arm_motor_pos
        if should_step:
            return self._sim.step(HabitatSimActions.ARM_VEL)
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

    def _capture_robo_state(self, robot_id, sim):
        forces = sim.get_articulated_object_forces(robot_id)
        vel = sim.get_articulated_object_velocities(robot_id)
        art_pos = sim.get_articulated_object_positions(robot_id)
        return {
            "forces": forces,
            "vel": vel,
            "pos": art_pos,
        }

    def _set_robo_state(self, robot_id, sim: RearrangeSim, set_dat):
        sim.set_articulated_object_forces(robot_id, set_dat["forces"])
        sim.set_articulated_object_velocities(robot_id, set_dat["vel"])
        sim.set_articulated_object_positions(robot_id, set_dat["pos"])

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.does_want_terminate = False

    def update_base(self):
        robot_id = self._sim.use_robo.get_robot_sim_id()
        ctrl_freq = self._sim.ctrl_freq

        before_trans_state = self._capture_robo_state(robot_id, self._sim)

        trans = self._sim.get_articulated_object_root_state(robot_id)
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
        self._sim.set_articulated_object_root_state(robot_id, target_trans)

        if not self._config.get("ALLOW_DYN_SLIDE", True):
            # Check if in the new robot state the arm collides with anything. If so
            # we have to revert back to the previous transform
            self._sim.internal_step(-1)
            colls = self._sim.get_collisions()
            did_coll, _ = rearrang_collision(
                colls, self._sim.snapped_obj_id, False
            )
            if did_coll:
                # Don't allow the step, revert back.
                self._set_robo_state(robot_id, self._sim, before_trans_state)
                self._sim.set_articulated_object_root_state(robot_id, trans)

    def step(self, base_vel, should_step=True, **kwargs):
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
        self.base_vel_ctrl.angular_velocity = mn.Vector3(0, 0, ang_vel)

        if lin_vel != 0.0 or ang_vel != 0.0:
            self.update_base()

        if should_step:
            return self._sim.step(HabitatSimActions.BASE_VEL)
        else:
            return None
