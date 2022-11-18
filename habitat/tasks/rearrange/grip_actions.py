#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union, Optional

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
        self.wants_grasp = False

    @property
    def requires_action(self):
        return self.action_space is not None


@registry.register_task_action
class MagicGraspAction(GripSimulatorTaskAction):
    @property
    def action_space(self):
        return spaces.Box(shape=(1,), high=1.0, low=-1.0)

    def reset(self, *args, **kwargs):
        self._has_released = False

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
                return

        # Get markers we are close to.
        if not self._config.ALLOW_MARKER_GRASP:
            return
        markers = self._sim.get_all_markers()
        if len(markers) > 0:
            names = list(markers.keys())
            pos = np.array([markers[k].get_current_position() for k in names])

            closest_idx = np.argmin(
                np.linalg.norm(pos - ee_pos, ord=2, axis=-1)
            )

            to_target = np.linalg.norm(ee_pos - pos[closest_idx], ord=2)

            if to_target < self._config.GRASP_THRESH_DIST:
                self._sim.robot.open_gripper()
                self._sim.grasp_mgr.snap_to_marker(names[closest_idx])

    def _ungrasp(self):
        self._sim.grasp_mgr.desnap()

    def step(self, grip_action, should_step=True, *args, **kwargs):
        if grip_action is None:
            return

        if (
            grip_action == 1.0
            and not self._sim.grasp_mgr.is_grasped
            and not self._has_released
        ):
            self._grasp()
        elif grip_action == 0.0 and self._sim.grasp_mgr.is_grasped:
            self._has_released = True
            self._ungrasp()


@registry.register_task_action
class GalaMagicGraspAction(MagicGraspAction):
    def step(self, grip_action, should_step=True, *args, **kwargs):
        if grip_action is None:
            return
        self.wants_grasp = False

        # Check the grip is discrete action.
        grip_action_int = int(grip_action[0])
        if grip_action[0] - float(grip_action_int) > 1e-5:
            raise ValueError(f"Got unexpected grip action {grip_action}")

        if grip_action_int == 1 and not self._sim.grasp_mgr.is_grasped:
            self.wants_grasp = True
            self._grasp()
        elif grip_action_int == 0 and self._sim.grasp_mgr.is_grasped:
            print("UNGRASPING!!!")
            breakpoint()
            raise ValueError()
            self._ungrasp()


@registry.register_task_action
class SuctionGraspAction(MagicGraspAction):
    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._sim: RearrangeSim = sim

    def _grasp(self):
        attempt_snap_entity: Optional[Union[str, int]] = None
        match_coll = None
        contacts = self._sim.get_physics_contact_points()

        robot_id = self._sim.robot.sim_obj.object_id
        all_gripper_links = list(self._sim.robot.params.gripper_joints)
        robot_contacts = [
            c
            for c in contacts
            if coll_name_matches(c, robot_id)
            and any(coll_link_name_matches(c, l) for l in all_gripper_links)
        ]

        if len(robot_contacts) == 0:
            return

        # Contacted any objects?
        for scene_obj_id in self._sim.scene_obj_ids:
            for c in robot_contacts:
                if coll_name_matches(c, scene_obj_id):
                    match_coll = c
                    break
            if match_coll is not None:
                attempt_snap_entity = scene_obj_id
                break

        if attempt_snap_entity is not None:
            rom = self._sim.get_rigid_object_manager()
            ro = rom.get_object_by_id(attempt_snap_entity)

            robot = self._sim.robot
            ee_T = robot.ee_transform
            obj_in_ee_T = ee_T.inverted() @ ro.transformation

            # here we need the link T, not the EE T for the constraint frame
            ee_link_T = robot.sim_obj.get_link_scene_node(
                robot.params.ee_link
            ).absolute_transformation()

            self._sim.grasp_mgr.snap_to_obj(
                int(attempt_snap_entity),
                force=False,
                # rel_pos is the relative position of the object COM in link space
                rel_pos=ee_link_T.inverted().transform_point(ro.translation),
                keep_T=obj_in_ee_T,
                should_open_gripper=False,
            )
            return

        if not self._config.ALLOW_MARKER_GRASP:
            return

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
                attempt_snap_entity = marker_name

        if attempt_snap_entity is not None:
            self._sim.grasp_mgr.snap_to_marker(str(attempt_snap_entity))
