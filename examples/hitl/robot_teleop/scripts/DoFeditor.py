#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn

import habitat.sims.habitat_simulator.sim_utilities as sutils
from habitat_sim.gfx import DebugLineRender
from scripts.robot import Robot


class DoFEditor:
    """
    A utility class for manipulating a robot DoF via GUI.
    """

    def __init__(self, robot: Robot, link_ix: int):
        self.robot = robot
        self.link_ix = link_ix

        self.joint_limits = self.robot.ao.joint_position_limits
        joint_positions = self.robot.ao.joint_positions
        self.dof = self.robot.ao.get_link_joint_pos_offset(
            link_ix
        )  # note this only handle single dof joints
        self.dof_value = joint_positions[self.dof]
        self.min_dof = self.joint_limits[0][self.dof]
        self.max_dof = self.joint_limits[1][self.dof]
        self.motor_id = None
        for motor_id, link_ix in self.robot.motor_ids_to_link_ids.items():
            if link_ix == self.link_ix:
                self.motor_id = motor_id

    def update(self, dt: float, set_positions: bool = False) -> None:
        """
        Attempt to increment the dof value by dt.
        """
        self.dof_value += dt
        # clamp to joint limits
        self.dof_value = min(self.max_dof, max(self.min_dof, self.dof_value))
        if self.motor_id is not None:
            jms = self.robot.ao.get_joint_motor_settings(self.motor_id)
            jms.position_target = self.dof_value
            self.robot.ao.update_joint_motor(self.motor_id, jms)
        if set_positions:
            # this joint has no motor, so directly manipulate the position
            cur_pos = self.robot.ao.joint_positions
            cur_pos[self.dof] = self.dof_value
            self.robot.ao.joint_positions = cur_pos

    def debug_draw(self, dblr: DebugLineRender, cam_pos: mn.Vector3) -> None:
        """
        Draw a 1D number line showing the dof state vs. min and max limits.
        """
        # keep the dof drawn as the mouse moves
        self.robot.draw_dof(dblr, self.link_ix, cam_pos)

        link_obj_id = self.robot.ao.link_ids_to_object_ids[self.link_ix]
        obj_bb, transform = sutils.get_bb_for_object_id(
            self.robot.sim, link_obj_id
        )
        center = transform.transform_point(obj_bb.center())

        to_camera = (cam_pos - center).normalized()
        left_vec = mn.math.cross(to_camera, mn.Vector3(0, 1, 0))
        rel_up = mn.math.cross(to_camera, left_vec)

        size_to_camera, center = sutils.get_obj_size_along(
            self.robot.sim, link_obj_id, to_camera
        )
        draw_at = center + to_camera * (size_to_camera + 0.05)
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
        # draw the current dof value
        interp_dof = (self.dof_value - self.min_dof) / (
            self.max_dof - self.min_dof
        )
        cur_dof_pos = end1 + (end2 - end1) * interp_dof
        dof_color = mn.Color3(0.4, 0.8, 0.4)
        dblr.draw_transformed_line(
            cur_dof_pos + rel_up * dash_height * 1.05,
            cur_dof_pos - rel_up * dash_height * 1.05,
            dof_color,
        )
