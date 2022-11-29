#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
import numpy as np


class MarkerInfo:
    """
    A data structure to track information about markers in the scene. These are
    automatically updated based on the position of the articulated link the
    marker is pinned to.
    """

    def __init__(self, offset_position, link_node, ao_parent, link_id):
        self.offset_position = offset_position
        self.link_node = link_node
        self.link_id = link_id
        self.current_transform = None
        self.ao_parent = ao_parent

        self.joint_idx = ao_parent.get_link_joint_pos_offset(link_id)

        self.update()

    def set_targ_js(self, js):
        js_arr = self.ao_parent.joint_positions[:]
        js_arr[self.joint_idx] = js
        self.ao_parent.joint_positions = js_arr

    def get_targ_js(self):
        return self.ao_parent.joint_positions[self.joint_idx]

    def get_targ_js_vel(self):
        return self.ao_parent.joint_velocities[self.joint_idx]

    def update(self):
        offset_T = mn.Matrix4.translation(mn.Vector3(self.offset_position))
        self.current_transform = self.link_node.transformation @ offset_T

    def get_current_position(self) -> np.ndarray:
        return np.array(self.current_transform.translation)

    def get_current_transform(self) -> mn.Matrix4:
        return self.current_transform
