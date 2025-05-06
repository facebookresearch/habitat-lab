#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
import numpy as np

from habitat_sim.gfx import DebugLineRender
from scripts.robot import Robot

# NOTE: must be imported after Isaac initialization


class DoFEditor:
    """
    A utility class for manipulating a robot DoF via GUI.
    """

    def __init__(self, robot: Robot, joint_prim_path: str):
        self._robot = robot
        self._isaac_service = self._robot._isaac_service
        self.joint_prim_path = joint_prim_path
        assert joint_prim_path in self._robot._joint_relationships

        self.joint_prim = self._isaac_service.world.stage.GetPrimAtPath(
            joint_prim_path
        )
        self.parent_prim = self._isaac_service.world.stage.GetPrimAtPath(
            self._robot._joint_relationships[joint_prim_path][0]
        )
        self.child_prim = self._isaac_service.world.stage.GetPrimAtPath(
            self._robot._joint_relationships[joint_prim_path][1]
        )
        self.rigid_link_ix = self._robot.get_rigid_prim_ix(
            self._robot._joint_relationships[joint_prim_path][1]
        )

        self.dof_index = self._robot._joint_names_to_dof_ix[joint_prim_path]
        dof_properties = self._robot._robot.dof_properties
        self.lower_limit = dof_properties["lower"][self.dof_index]
        self.upper_limit = dof_properties["upper"][self.dof_index]
        self.dof_target = self._robot._robot.get_joint_positions(
            [self.dof_index]
        )
        # NOTE: PhysX quantities are degrees
        # self.lower_limit = self.joint_prim.GetAttribute('physics:lowerLimit').Get()
        # self.upper_limit = self.joint_prim.GetAttribute('physics:upperLimit').Get()

        # self.dof_target = self.joint_prim.GetAttribute('state:angular:physics:position').Get()

    def update(self, delta: float, set_positions: bool = False) -> None:
        """
        Attempt to increment the dof value by delta.
        """

        from omni.isaac.core.utils.types import ArticulationAction

        self.dof_target += delta
        self.dof_target = min(
            self.upper_limit, max(self.lower_limit, self.dof_target)
        )
        if set_positions:
            self._robot._robot.set_joint_positions(
                positions=np.array([self.dof_target]),
                joint_indices=[self.dof_index],
            )

        action = ArticulationAction(
            joint_positions=np.array([self.dof_target]),
            joint_indices=[self.dof_index],
        )
        self._robot._robot.apply_action(action)

    def debug_draw(self, dblr: DebugLineRender, cam_pos: mn.Vector3) -> None:
        """
        Draw a 1D number line showing the dof state vs. min and max limits.
        target is a green line
        current state is a blue line
        """
        # keep the dof drawn as the mouse moves
        self._robot.draw_dof(dblr, self.rigid_link_ix, cam_pos)

        # transform from the child of the joint
        body_positions, body_rotations = self._robot.get_link_world_poses(
            indices=[self.rigid_link_ix]
        )
        body_pos = body_positions[0]
        # body_rot = body_rotations[0]

        to_camera = (cam_pos - body_pos).normalized()
        left_vec = mn.math.cross(to_camera, mn.Vector3(0, 1, 0))
        rel_up = mn.math.cross(to_camera, left_vec)
        draw_at = body_pos

        line_len = 0.3
        dash_height = 0.05
        frame_color = mn.Color3(0.8, 0.8, 0.4)
        end1 = draw_at - left_vec * (line_len / 2)
        end2 = draw_at + left_vec * (line_len / 2)
        dblr.draw_transformed_line(end1, end2, frame_color)
        dblr.draw_transformed_line(
            end1 + rel_up * dash_height,
            end1 - rel_up * dash_height,
            frame_color,
        )
        dblr.draw_transformed_line(
            end2 + rel_up * dash_height,
            end2 - rel_up * dash_height,
            frame_color,
        )
        cur_dof_value = self._robot._robot.get_joint_positions(
            [self.dof_index]
        )
        # cur_dof_value = self.joint_prim.GetAttribute('state:angular:physics:position').Get()
        for dof_value, color in [
            (self.dof_target, mn.Color3(0.4, 0.8, 0.4)),
            (cur_dof_value, mn.Color3(0.4, 0.4, 0.8)),
        ]:
            # draw the current dof value
            interp_dof = (dof_value - self.lower_limit) / (
                self.upper_limit - self.lower_limit
            )
            cur_dof_pos = end1 + (end2 - end1) * interp_dof
            dblr.draw_transformed_line(
                cur_dof_pos + rel_up * dash_height * 1.05,
                cur_dof_pos - rel_up * dash_height * 1.05,
                color,
            )
