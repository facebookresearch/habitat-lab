#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
import numpy as np
from gym import spaces

from habitat.core.embodied_task import SimulatorTaskAction
from habitat.core.registry import registry
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.utils import coll_name_matches


class GripSimulatorTaskAction(SimulatorTaskAction):
    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._sim: RearrangeSim = sim

    @property
    def requires_action(self):
        return self.action_space is not None


@registry.register_task_action
class MagicGraspAction(GripSimulatorTaskAction):
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
class SuctionGraspAction(GripSimulatorTaskAction):
    """
    Action to automatically grasp when the gripper makes contact with an object. Does not allow for ungrasping.
    """

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._sim: RearrangeSim = sim
        self._suction_offset = mn.Vector3(*config.SUCTION_OFFSET)

    @property
    def action_space(self):
        return None

    def _ungrasp(self):
        self._sim.grasp_mgr.desnap()

    def step(self, _, should_step=True, *args, **kwargs):
        attempt_snap_idx = None
        contacts = self._sim.get_physics_contact_points()

        robot_id = self._sim.robot.sim_obj.object_id
        robot_contacts = [
            c for c in contacts if coll_name_matches(c, robot_id)
        ]

        for scene_obj_id in self._sim.scene_obj_ids:
            has_robot_match = any(
                c for c in robot_contacts if coll_name_matches(c, scene_obj_id)
            )
            if has_robot_match:
                attempt_snap_idx = scene_obj_id

        if attempt_snap_idx is not None and not self._sim.grasp_mgr.is_grasped:
            self._sim.grasp_mgr.snap_to_obj(attempt_snap_idx)
