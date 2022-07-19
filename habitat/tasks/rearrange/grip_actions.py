#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

import numpy as np
from gym import spaces

from habitat.core.embodied_task import SimulatorTaskAction
from habitat.core.registry import registry
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.utils import (
    coll_link_name_matches,
    coll_name_matches,
)


class RobotAction(SimulatorTaskAction):
    _sim: RearrangeSim

    @property
    def _ik_helper(self):
        if self._config.AGENT is None:
            return self._sim.robots_mgr[0].ik_helper
        return self._sim.robots_mgr[self._config.AGENT].ik_helper

    @property
    def cur_robot(self):
        if self._config.AGENT is None:
            return self._sim.robots_mgr[0].robot
        return self._sim.robots_mgr[self._config.AGENT].robot

    @property
    def cur_grasp_mgr(self):
        if self._config.AGENT is None:
            return self._sim.robots_mgr[0].grasp_mgr
        return self._sim.robots_mgr[self._config.AGENT].grasp_mgr

    @property
    def _action_arg_prefix(self) -> str:
        if self._config.AGENT is not None:
            return f"AGENT_{self._config.AGENT}_"
        return ""

    @property
    def _robot_prefix(self) -> str:
        if self._config.AGENT is not None:
            return f"ROBOT_{self._config.AGENT}_"
        return "ROBOT_0"


class GripSimulatorTaskAction(RobotAction):
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
        ee_pos = self.cur_robot.ee_transform.translation
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
                self.cur_grasp_mgr.snap_to_obj(
                    self._sim.scene_obj_ids[closest_obj_idx]
                )
                return

        # Get markers we are close to.
        markers = self._sim.get_all_markers()
        if len(markers) > 0:
            names = list(markers.keys())
            pos = np.array([markers[k].get_current_position() for k in names])

            closest_idx = np.argmin(
                np.linalg.norm(pos - ee_pos, ord=2, axis=-1)
            )

            to_target = np.linalg.norm(ee_pos - pos[closest_idx], ord=2)

            if to_target < self._config.GRASP_THRESH_DIST:
                self.cur_robot.open_gripper()
                self.cur_grasp_mgr.snap_to_marker(names[closest_idx])

    def _ungrasp(self):
        self.cur_grasp_mgr.desnap()

    def step(self, grip_action, should_step=True, *args, **kwargs):
        if grip_action is None:
            return

        if grip_action >= 0 and not self.cur_grasp_mgr.is_grasped:
            self._grasp()
        elif grip_action < 0 and self.cur_grasp_mgr.is_grasped:
            self._ungrasp()


@registry.register_task_action
class SuctionGraspAction(MagicGraspAction):
    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._sim: RearrangeSim = sim

    def _grasp(self):
        attempt_snap_entity: Union[str, int] = None
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

            ee_T = self.cur_robot.ee_transform
            obj_in_ee_T = ee_T.inverted() @ ro.transformation

            # here we need the link T, not the EE T for the constraint frame
            ee_link_T = self.cur_robot.sim_obj.get_link_scene_node(
                self.cur_robot.params.ee_link
            ).absolute_transformation()

            self._sim.grasp_mgr.snap_to_obj(
                int(attempt_snap_entity),
                force=False,
                # rel_pos is the relative position of the object COM in link space
                rel_pos=ee_link_T.inverted().transform_point(ro.translation),
                keep_T=obj_in_ee_T,
                should_open_gripper=False,
                gripper_offset=0.0,
            )
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
