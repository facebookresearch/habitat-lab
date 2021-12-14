#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from gym import spaces

from habitat.core.embodied_task import SimulatorTaskAction
from habitat.core.registry import registry
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.utils import (
    coll_link_name_matches,
    coll_name_matches,
)


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
        # Get objects we are close to.
        if len(scene_obj_pos) != 0:
            # Get the target the EE is closest to.
            closest_obj_idx = np.argmin(
                np.linalg.norm(scene_obj_pos - ee_pos, ord=2, axis=-1)
            )

            to_target = np.linalg.norm(
                ee_pos - scene_obj_pos[closest_obj_idx], ord=2
            )

            if to_target < self._config.GRASP_THRESH_DIST:
                self._sim.grasp_mgr.snap_to_obj(
                    self._sim.scene_obj_ids[closest_obj_idx]
                )

        # Get markers we are close to.
        markers = self._sim.get_all_markers()
        if len(markers) > 0:
            names = list(markers.keys())
            pos = np.array([markers[k].get_current_position() for k in names])

            closest_idx = np.argmin(
                np.linalg.norm(scene_obj_pos - ee_pos, ord=2, axis=-1)
            )

            to_target = np.linalg.norm(ee_pos - pos[closest_idx], ord=2)

            if to_target < self._config.GRASP_THRESH_DIST:
                self._sim.grasp_mgr.snap_to_marker(names[closest_idx])

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

    @property
    def action_space(self):
        return None

    def _ungrasp(self):
        self._sim.grasp_mgr.desnap()

    def step(self, _, should_step=True, *args, **kwargs):
        if self._sim.grasp_mgr.is_grasped:
            return
        attempt_snap_idx = None
        contacts = self._sim.get_physics_contact_points()

        robot_id = self._sim.robot.sim_obj.object_id
        robot_contacts = [
            c for c in contacts if coll_name_matches(c, robot_id)
        ]

        # Contacted any objects?
        for scene_obj_id in self._sim.scene_obj_ids:
            has_robot_match = any(
                c for c in robot_contacts if coll_name_matches(c, scene_obj_id)
            )
            if has_robot_match:
                attempt_snap_idx = scene_obj_id

        if attempt_snap_idx is not None:
            self._sim.grasp_mgr.snap_to_obj(attempt_snap_idx)
            return

        attempt_marker_snap_name = None
        # Contacted any markers?
        markers = self._sim.get_all_markers()
        for marker_name, marker in markers.items():
            has_match = any(
                c
                for c in robot_contacts
                if coll_name_matches(c, marker.ao_parent.object_id)
                and coll_link_name_matches(c, marker.link_id)
            )
            if has_match:
                attempt_marker_snap_name = marker_name

        if attempt_marker_snap_name is not None:
            self._sim.grasp_mgr.snap_to_marker(attempt_marker_snap_name)
