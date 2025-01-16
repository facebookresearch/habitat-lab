#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import magnum as mn
import numpy as np

from habitat_sim.physics import (
    CollisionGroupHelper,
    CollisionGroups,
    ManagedRigidObject,
    RigidConstraintSettings,
    RigidConstraintType,
)

if TYPE_CHECKING:
    from habitat.articulated_agents.manipulator import Manipulator
    from habitat.config.default_structured_configs import SimulatorConfig
    from habitat.tasks.rearrange.rearrange_sim import RearrangeSim


class RearrangeGraspManager:
    """
    Manages the agent grasping onto rigid objects and the links of articulated objects.
    """

    def __init__(
        self,
        sim: "RearrangeSim",
        config: "SimulatorConfig",
        articulated_agent: "Manipulator",
        ee_index=0,
    ) -> None:
        """Initialize a grasp manager for the simulator instance provided.

        :param sim: Pointer to the simulator where the agent is instantiated
        :param config: The task's "simulator" subconfig node. Defines grasping parameters.
        :param articulated_agent: The agent for which we want to manage grasping
        :param ee_index: The index of the end effector of the articulated_agent belonging to this grasp_manager
        """

        self._sim = sim
        self._snapped_obj_id: Optional[int] = None
        self._snapped_marker_id: Optional[str] = None
        self._snap_constraints: List[int] = []
        self._keep_T: Optional[mn.Matrix4] = None
        self._leave_info: Optional[Tuple[mn.Vector3, float]] = None
        self._config = config
        self._managed_articulated_agent = articulated_agent
        self.ee_index = ee_index

        # HACK: This flag sets whether the grasp manager handles moving the grasped object.
        # Turn off in applications that handle grasping themselves.
        self._automatically_update_snapped_object = True

        self._kinematic_mode = self._sim.habitat_config.kinematic_mode

    def reconfigure(self) -> None:
        """Removes any existing constraints managed by this structure."""

        self._snap_constraints.clear()

    def reset(self) -> None:
        """Reset the grasp manager by re-defining the collision group and dropping any grasped object."""

        # Setup the collision groups. UserGroup7 is the held object group, it
        # can interact with anything except for the robot.
        CollisionGroupHelper.set_mask_for_group(
            CollisionGroups.UserGroup7, ~CollisionGroups.Robot
        )

        self.desnap(True)
        self._leave_info = None
        self._vis_info: List[Any] = []

    def is_violating_hold_constraint(self) -> bool:
        """
        Returns true if the object is too far away from the gripper, meaning
        the agent violated the hold constraint.
        """

        ee_pos = self._managed_articulated_agent.ee_transform(
            self.ee_index
        ).translation
        if self._snapped_obj_id is not None and (
            np.linalg.norm(ee_pos - self.snap_rigid_obj.translation)
            >= self._config.hold_thresh
        ):
            return True
        if self._snapped_marker_id is not None:
            marker = self._sim.get_marker(self._snapped_marker_id)
            if (
                np.linalg.norm(ee_pos - marker.get_current_position())
                >= self._config.hold_thresh
            ):
                return True

        return False

    @property
    def is_grasped(self) -> bool:
        """Returns whether or not an object or marker is currently grasped."""

        return (
            self._snapped_obj_id is not None
            or self._snapped_marker_id is not None
        )

    def update(self) -> None:
        """Reset the collision group of the grasped object if its distance to the end effector exceeds a threshold.

        Used to wait for a dropped object to clear the end effector's proximity before re-activating collisions between them.
        """

        if self._leave_info is not None:
            ee_pos = self._managed_articulated_agent.ee_transform(
                self.ee_index
            ).translation
            rigid_obj = self._leave_info[0]
            dist = np.linalg.norm(ee_pos - rigid_obj.translation)
            if dist >= self._leave_info[1]:
                rigid_obj.override_collision_group(CollisionGroups.Default)
                self._leave_info = None
        if self._kinematic_mode and self._snapped_obj_id is not None:
            self.update_object_to_grasp()

    def desnap(self, force=False) -> None:
        """Removes any hold constraints currently active. Removes hold constraints for regular and articulated objects.

        :param force: If True, reset the collision group of the now released object immediately instead of waiting for its distance from the end effector to reach a threshold.
        """

        self._vis_info = []
        if len(self._snap_constraints) == 0:
            # No constraints to unsnap
            self._snapped_obj_id = None
            self._snapped_marker_id = None
            return

        if self._snapped_obj_id is not None:
            obj_bb = self.snap_rigid_obj.aabb
            if obj_bb is not None:
                if force:
                    self.snap_rigid_obj.override_collision_group(
                        CollisionGroups.Default
                    )
                else:
                    self._leave_info = (
                        self.snap_rigid_obj,
                        max(obj_bb.size_x(), obj_bb.size_y(), obj_bb.size_z()),
                    )

        for constraint_id in self._snap_constraints:
            self._sim.remove_rigid_constraint(constraint_id)
        self._snap_constraints = []

        self._snapped_obj_id = None
        self._snapped_marker_id = None
        self._managed_articulated_agent.close_gripper()

    @property
    def snap_idx(self) -> Optional[int]:
        """
        The index of the grasped RigidObject. None if nothing is being grasped.
        """

        return self._snapped_obj_id

    @property
    def snapped_marker_id(self) -> Optional[str]:
        """
        The name of the marker for the grasp. None if nothing is being grasped.
        """

        return self._snapped_marker_id

    @property
    def snap_rigid_obj(self) -> ManagedRigidObject:
        """The grasped object instance."""

        ret_obj = self._sim.get_rigid_object_manager().get_object_by_id(
            self._snapped_obj_id
        )
        if ret_obj is None:
            raise ValueError(
                f"Tried to get non-existence object from ID {self._snapped_obj_id}"
            )
        return ret_obj

    def snap_to_marker(self, marker_name: str) -> None:
        """
        Create a constraint between the end-effector and the marker on the
        articulated object to be grasped.

        :param marker_name: The name/id of the marker.
        """

        if marker_name == self._snapped_marker_id:
            return

        if len(self._snap_constraints) != 0:
            # We were already grabbing something else.
            raise ValueError(
                f"Tried snapping to {marker_name} when already snapped"
            )

        marker = self._sim.get_marker(marker_name)
        self._snapped_marker_id = marker_name
        self._managed_articulated_agent.open_gripper()
        if self._kinematic_mode:
            return

        if self._automatically_update_snapped_object:
            self._snap_constraints = [
                self.create_hold_constraint(
                    RigidConstraintType.PointToPoint,
                    mn.Vector3(0.0, 0.0, 0.0),
                    mn.Vector3(*marker.offset_position),
                    marker.ao_parent.object_id,
                    marker.link_id,
                ),
            ]

    def create_hold_constraint(
        self,
        constraint_type,
        pivot_in_link: mn.Vector3,
        pivot_in_obj: mn.Vector3,
        obj_id_b: int,
        link_id_b: Optional[int] = None,
        rotation_lock_b: Optional[mn.Matrix3] = None,
    ) -> int:
        """Create a new rigid point-to-point (ball joint) constraint between the robot and an object.

        :param pivot_in_link: The origin of the constraint in end effector local space.
        :param pivot_in_obj: The origin of the constraint in object local space.
        :param obj_id_b: The id of the object to be constrained to the end effector.
        :param link_id_b: If the object is articulated, provide the link index for the constraint.

        :return: The id of the newly created constraint or -1 if failed.
        """

        c = RigidConstraintSettings()
        c.object_id_a = self._managed_articulated_agent.get_robot_sim_id()
        c.link_id_a = self._managed_articulated_agent.ee_link_id(self.ee_index)
        c.object_id_b = obj_id_b
        if link_id_b is not None:
            c.link_id_b = link_id_b
        c.pivot_a = pivot_in_link
        c.pivot_b = pivot_in_obj
        c.frame_a = mn.Matrix3.identity_init()
        if rotation_lock_b is not None:
            c.frame_b = rotation_lock_b
        c.max_impulse = self._config.grasp_impulse
        c.constraint_type = constraint_type

        if constraint_type == RigidConstraintType.Fixed:
            # we set the link frame to object rotation in link space (objR -> world -> link)
            link_id = self._managed_articulated_agent.ee_link_id(self.ee_index)
            sim_object = self._managed_articulated_agent.sim_obj
            link_node = sim_object.get_link_scene_node(link_id)
            link_frame_world_space = (
                link_node.absolute_transformation().rotation()
            )

            object_frame_world_space = (
                self._sim.get_rigid_object_manager()
                .get_object_by_id(obj_id_b)
                .transformation.rotation()
            )
            c.frame_a = link_frame_world_space.inverted().__matmul__(
                object_frame_world_space
            )
            # NOTE: object frame is default identity because using it instead is unstable
            self._vis_info.append((pivot_in_obj, obj_id_b))

        return self._sim.create_rigid_constraint(c)

    def update_debug(self) -> None:
        """
        Creates visualizations for grasp points.
        """

        for i, (local_pivot, obj_id) in enumerate(self._vis_info):
            rom = self._sim.get_rigid_object_manager()
            obj = rom.get_object_by_id(obj_id)
            pivot_pos = obj.transformation.transform_point(local_pivot)
            self._sim.viz_ids[i] = self._sim.visualize_position(
                pivot_pos,
                self._sim.viz_ids[i],
                r=0.02,
            )

    def update_object_to_grasp(self) -> None:
        """
        Kinematically update held object to be within robot's grasp. If nothing
        is grasped then nothing will happen.
        """

        if (
            self._snapped_obj_id is None
            or not self._automatically_update_snapped_object
        ):
            # Not grasping anything, so do nothing.
            return

        rel_T = self._keep_T
        if rel_T is None:
            rel_T = mn.Matrix4.identity_init()

        self.snap_rigid_obj.transformation = (
            self._managed_articulated_agent.ee_transform(self.ee_index) @ rel_T
        )

    def snap_to_obj(
        self,
        snap_obj_id: int,
        force: bool = True,
        should_open_gripper=True,
        rel_pos: Optional[mn.Vector3] = None,
        keep_T: Optional[mn.Matrix4] = None,
    ) -> None:
        """Attempt to grasp an object, snapping/constraining it to the robot's
        end effector with 3 ball-joint constraints forming a fixed frame.

        :param snap_obj_id: The id of the object to be constrained to the end effector.
        :param force: Will kinematically snap the object to the robot's end-effector, even if
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

        self._snapped_obj_id = snap_obj_id
        if force:
            # Set the transformation to be in the robot's hand already.
            self.update_object_to_grasp()

        self._managed_articulated_agent.open_gripper()

        if self._kinematic_mode:
            # update the KRM to sever any existing parent relationships for the newly grasped object
            self._sim.kinematic_relationship_manager.relationship_graph.remove_obj_relations(
                snap_obj_id, parents_only=True
            )
            # update root parent transforms so new parent state is registered
            self._sim.kinematic_relationship_manager.prev_root_obj_state = (
                self._sim.kinematic_relationship_manager.get_root_parents_snapshot()
            )
            return

        # Set collision group to GraspedObject so that it doesn't collide
        # with the links of the robot.
        self.snap_rigid_obj.override_collision_group(
            CollisionGroups.UserGroup7
        )

        self._keep_T = keep_T

        # Get object transform in EE frame
        if rel_pos is None:
            rel_pos = mn.Vector3.zero_init()

        if self._automatically_update_snapped_object:
            self._snap_constraints = [
                self.create_hold_constraint(
                    RigidConstraintType.Fixed,
                    # link pivot is the object in link space
                    pivot_in_link=rel_pos,
                    # object pivot is local origin
                    pivot_in_obj=mn.Vector3.zero_init(),
                    obj_id_b=self._snapped_obj_id,
                ),
            ]

        if should_open_gripper:
            self._managed_articulated_agent.open_gripper()

        if any((x == -1 for x in self._snap_constraints)):
            raise ValueError("Created bad constraint")
