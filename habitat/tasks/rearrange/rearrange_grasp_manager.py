#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import magnum as mn
import numpy as np

from habitat.tasks.rearrange.utils import get_aabb
from habitat_sim.physics import (
    CollisionGroupHelper,
    CollisionGroups,
    RigidConstraintSettings,
)


class RearrangeGraspManager:
    """
    Manages the agent grasping onto rigid objects and the links of articulated objects.
    """

    def __init__(self, sim, config):
        self._sim = sim
        self._snapped_obj_id = None
        self._snapped_marker_id = None
        self._snap_constraints = []
        self._leave_info = None
        self._config = config

    def reconfigure(self):
        for constraint_id in self._snap_constraints:
            self._sim.remove_rigid_constraint(constraint_id)
        self._snap_constraints.clear()

    def reset(self):
        # Setup the collision groups. UserGroup7 is the held object group, it
        # can interact with anything except for the robot.
        CollisionGroupHelper.set_mask_for_group(
            CollisionGroups.UserGroup7, ~CollisionGroups.Robot
        )

        self.desnap(True)
        self._leave_info = None

    def is_violating_hold_constraint(self):
        """
        Returns true if the object is too far away from the gripper, meaning
        the agent violated the hold constraint.
        """
        ee_pos = self._sim.robot.ee_transform.translation
        if self._snapped_obj_id is not None:
            obj_pos = self._sim.get_translation(self._snapped_obj_id)
            if np.linalg.norm(ee_pos - obj_pos) >= self._config.HOLD_THRESH:
                return True
        if self._snapped_marker_id is not None:
            marker = self._sim.get_marker(self._snapped_marker_id)
            if (
                np.linalg.norm(ee_pos - marker.get_current_position())
                >= self._config.HOLD_THRESH
            ):
                return True

        return False

    @property
    def is_grasped(self):
        return (
            self._snapped_obj_id is not None
            or self._snapped_marker_id is not None
        )

    def update(self):
        if self._leave_info is not None:
            ee_pos = self._sim.robot.ee_transform.translation
            dist = np.linalg.norm(ee_pos - self._leave_info[0])
            if dist >= self._leave_info[1]:
                rigid_obj = (
                    self._sim.get_rigid_object_manager().get_object_by_id(
                        self._leave_info[2]
                    )
                )
                rigid_obj.override_collision_group(CollisionGroups.Default)
            self._leave_info = None

    def desnap(self, force=False):
        """Removes any hold constraints currently active."""
        if len(self._snap_constraints) == 0:
            # No constraints to unsnap
            self._snapped_obj_id = None
            self._snapped_marker_id = None
            return

        if self._snapped_obj_id is not None:
            obj_bb = get_aabb(self.snap_idx, self._sim)
            if obj_bb is not None:
                if force:
                    self.snap_rigid_obj.override_collision_group(
                        CollisionGroups.Default
                    )
                else:
                    self._leave_info = (
                        self._sim.get_translation(self._snapped_obj_id),
                        max(obj_bb.size_x(), obj_bb.size_y(), obj_bb.size_z()),
                        self._snapped_obj_id,
                    )

        for constraint_id in self._snap_constraints:
            self._sim.remove_rigid_constraint(constraint_id)
        self._snap_constraints = []

        self._snapped_obj_id = None
        self._snapped_marker_id = None

    @property
    def snap_idx(self):
        return self._snapped_obj_id

    @property
    def snapped_marker_id(self):
        return self._snapped_marker_id

    @property
    def snap_rigid_obj(self):
        return self._sim.get_rigid_object_manager().get_object_by_id(
            self._snapped_obj_id
        )

    def snap_to_marker(self, marker_name):
        """
        Create a constraint between the end-effector and the marker on the
        articulated object that is attempted to be grasped.
        """
        if marker_name == self._snapped_marker_id:
            return

        if len(self._snap_constraints) != 0:
            # We were already grabbing something else.
            raise ValueError(
                f"Tried snapping to {marker_name} when already snapped"
            )

        marker = self._sim.get_marker(marker_name)
        self._snap_constraints = [
            self.create_hold_constraint(
                mn.Vector3(0.0, 0.0, 0.0),
                mn.Vector3(*marker.offset_position),
                marker.ao_parent.object_id,
                marker.link_id,
            ),
        ]
        self._snapped_marker_id = marker_name

    def create_hold_constraint(
        self, pivot_in_link, pivot_in_obj, obj_id_b, link_id_b=None
    ):
        c = RigidConstraintSettings()
        c.object_id_a = self._sim.robot.get_robot_sim_id()
        c.link_id_a = self._sim.robot.ee_link_id
        c.object_id_b = obj_id_b
        if link_id_b is not None:
            c.link_id_b = link_id_b
        c.pivot_a = pivot_in_link
        c.pivot_b = pivot_in_obj
        c.max_impulse = self._config.GRASP_IMPULSE
        return self._sim.create_rigid_constraint(c)

    def snap_to_obj(self, snap_obj_id: int, force: bool = True):
        """
        :param snap_obj_id: Simulator object index.
        :param force: Will transform the object to be in the robot's grasp, even if
            the object is already in the grasped state.
        """
        if snap_obj_id == self._snapped_obj_id:
            # Already grasping this object.
            return

        if len(self._snap_constraints) != 0:
            # We were already grabbing something else.
            raise ValueError(
                f"Tried snapping to {snap_obj_id} when already snapped to {self._snapped_obj_id}"
            )

        if force:
            # Set the transformation to be in the robot's hand already.
            self._sim.set_transformation(
                self._sim.robot.ee_transform, snap_obj_id
            )

        self._snapped_obj_id = snap_obj_id

        # Set collision group to GraspedObject so that it doesn't collide
        # with the links of the robot.
        self.snap_rigid_obj.override_collision_group(
            CollisionGroups.UserGroup7
        )

        self._snap_constraints = [
            self.create_hold_constraint(
                mn.Vector3(0.1, 0, 0),
                mn.Vector3(0, 0, 0),
                self._snapped_obj_id,
            ),
            self.create_hold_constraint(
                mn.Vector3(0.0, 0, 0),
                mn.Vector3(-0.1, 0, 0),
                self._snapped_obj_id,
            ),
            self.create_hold_constraint(
                mn.Vector3(0.1, 0.0, 0.1),
                mn.Vector3(0.0, 0.0, 0.1),
                self._snapped_obj_id,
            ),
        ]

        if any((x == -1 for x in self._snap_constraints)):
            raise ValueError("Created bad constraint")
