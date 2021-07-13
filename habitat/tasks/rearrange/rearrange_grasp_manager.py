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
    def __init__(self, sim, config):
        self._sim = sim
        self._snapped_obj_id = None
        self._snap_constraints = []
        self._leave_info = None
        self._config = config

    def reset(self):
        self._leave_info = None

        # Setup the collision groups. UserGroup7 is the held object group, it
        # can interact with anything except for the robot.
        CollisionGroupHelper.set_mask_for_group(
            CollisionGroups.UserGroup7, ~CollisionGroups.Robot
        )

        self.desnap()

    def is_violating_hold_constraint(self):
        if self._config.get("IGNORE_HOLD_VIOLATE", False):
            return False
        # Is the object firmly in the grasp of the robot?
        ee_pos = self._sim.robot.ee_transform.translation
        if self.is_grasped:
            obj_pos = self._sim.get_translation(self._snapped_obj_id)
            if np.linalg.norm(ee_pos - obj_pos) >= self._config.HOLD_THRESH:
                return True

        return False

    @property
    def is_grasped(self):
        return self._snapped_obj_id is not None

    def update(self):
        if self._leave_info is not None:
            ee_pos = self._sim.robot.ee_transform.translation
            dist = np.linalg.norm(ee_pos - self._leave_info[0])
            if dist >= self._leave_info[1]:
                self.snap_rigid_obj.override_collision_group(
                    CollisionGroups.Default
                )

    def desnap(self, force=False):
        """Removes any hold constraints currently active."""
        if len(self._snap_constraints) == 0:
            # No constraints to unsnap
            self._snapped_obj_id = None
            return

        if self._snapped_obj_id is not None:
            obj_bb = get_aabb(self.snap_idx, self._sim)
            if force:
                self.snap_rigid_obj.override_collision_group(
                    CollisionGroups.Default
                )
            else:
                self._leave_info = (
                    self._sim.get_translation(self._snapped_obj_id),
                    max(obj_bb.size_x(), obj_bb.size_y(), obj_bb.size_z()),
                )

        for constraint_id in self._snap_constraints:
            self._sim.remove_rigid_constraint(constraint_id)
        self._snap_constraints = []

        self._snapped_obj_id = None

    @property
    def snap_idx(self):
        return self._snapped_obj_id

    @property
    def snap_rigid_obj(self):
        return self._sim.get_rigid_object_manager().get_object_by_id(
            self._snapped_obj_id
        )

    def snap_to_obj(self, snap_obj_id: int, force: bool = True):
        """
        :param snap_obj_id: Simulator object index.
        :force: Will transform the object to be in the robot's grasp, even if
            the object is already in the grasped state.
        """
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

        if snap_obj_id == self._snapped_obj_id:
            # Already grasping this object.
            return

        self._snapped_obj_id = snap_obj_id

        # Set collision group to GraspedObject so that it doesn't collide
        # with the links of the robot.
        self.snap_rigid_obj.override_collision_group(
            CollisionGroups.UserGroup7
        )

        def create_hold_constraint(pivot_in_link, pivot_in_obj):
            c = RigidConstraintSettings()
            c.object_id_a = self._sim.robot.get_robot_sim_id()
            c.link_id_a = self._sim.robot.ee_link_id
            c.object_id_b = self._snapped_obj_id
            c.pivot_a = pivot_in_link
            c.pivot_b = pivot_in_obj
            c.max_impulse = self._config.GRASP_IMPULSE
            return self._sim.create_rigid_constraint(c)

        self._snap_constraints = [
            create_hold_constraint(mn.Vector3(0.1, 0, 0), mn.Vector3(0, 0, 0)),
            create_hold_constraint(
                mn.Vector3(0.0, 0, 0), mn.Vector3(-0.1, 0, 0)
            ),
            create_hold_constraint(
                mn.Vector3(0.1, 0.0, 0.1), mn.Vector3(0.0, 0.0, 0.1)
            ),
        ]

        if any((x == -1 for x in self._snap_constraints)):
            raise ValueError("Created bad constraint")
