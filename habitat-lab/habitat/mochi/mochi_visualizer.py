# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
from enum import Enum

# todo: clean up how RenderInstanceHelper is exposed from habitat_sim extension
from habitat_sim._ext.habitat_sim_bindings import RenderInstanceHelper

import numpy as np
import magnum as mn

def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return [w, x, y, z]

def quat_transpose(q):
    w, x, y, z = q
    return [w, -x, -y, -z]

def quat_rotate(q, v):
    q_conj = quat_transpose(q)
    v_quat = [0, *v]
    return quat_multiply(quat_multiply(q, v_quat), q_conj)[1:]

class _InstanceGroup:
    def __init__(self, hab_sim):
        identity_rotation_wxyz = [1.0, 0.0, 0.0, 0.0]
        # self._render_instance_helper = RenderInstanceHelper(hab_sim, identity_rotation_wxyz)
        self._render_instance_helper = RenderInstanceHelper(hab_sim, use_xyzw_orientations=False)
        self._object_handles = {}

    def flush_to_hab_sim(self, mochi):

        if len(self._object_handles) == 0:
            return

        def rotvec_to_quat_wxyz(rotvec):
            theta = np.linalg.norm(rotvec)
            if theta < 1e-8:
                # No rotation, return identity quaternion
                return np.array([1.0, 0.0, 0.0, 0.0])  # [x, y, z, w]
            
            axis = rotvec / theta
            half_theta = theta / 2.0
            sin_half_theta = np.sin(half_theta)
            cos_half_theta = np.cos(half_theta)
            
            q_xyz = axis * sin_half_theta
            q_w = cos_half_theta
            return np.concatenate([[q_w], q_xyz])

        num_objects = len(self._object_handles)

        positions = np.zeros((num_objects, 3), dtype=np.float32)  # perf todo: use np.empty
        wxyz_rotations = np.zeros((num_objects, 4), dtype=np.float32)  # perf todo: use np.empty
        for (i, name) in enumerate(self._object_handles):
            handle = self._object_handles[name]
            pose_com = mochi.get_object_com_transform(handle)
            pose = mochi.get_object_origin_transform(handle)
            positions[i] = pose[0]
            wxyz_rotations[i] = rotvec_to_quat_wxyz(pose[1])

        # todo: coordinate frame conversion?
        # todo: use true object origin instead of CoM

        self._render_instance_helper.set_world_poses(
            np.ascontiguousarray(positions), 
            np.ascontiguousarray(wxyz_rotations))


class _InstanceGroupType(Enum):        
    STATIC = 0
    DYNAMIC = 1

class MochiVisualizer:

    def __init__(self, hab_sim, mochi):

        self._instance_groups = {}
        for group_type in _InstanceGroupType:
            self._instance_groups[group_type] = _InstanceGroup(hab_sim)

        self._object_name_to_render_asset = None
        self._mochi = mochi
        self._is_first_flush = True
        self._next_semantic_id = 1
        
    def add_render_map(self, render_map_filepath):

        object_names = self._mochi.get_actors_names()

        assert self._is_first_flush

        with open(render_map_filepath, "r") as f:
            json_data = json.load(f)
        render_map_json = json_data["render_map"]

        unitScale = mn.Vector3(1.0, 1.0, 1.0)

        for elem_name in render_map_json:

            item_json = render_map_json[elem_name]
            is_dynamic = item_json["is_dynamic"]
            render_asset_filepath = item_json["render_asset_filepath"]
            if render_asset_filepath is None:
                continue
            semantic_id = item_json.get("semantic_id")
            semantic_id = self._next_semantic_id if semantic_id is None else semantic_id
            self._next_semantic_id += 1

            scale = item_json.get("scale")
            scale = mn.Vector3(scale) if scale is not None else unitScale

            if elem_name not in object_names:
                print(f"MochiVisualizer: no object found for {elem_name} named in {render_map_filepath}")
                continue

            mochi_handle = self._mochi.get_actor_handle(elem_name)

            group_type = _InstanceGroupType.DYNAMIC if is_dynamic else _InstanceGroupType.STATIC
            group = self._instance_groups[group_type]
            group._render_instance_helper.add_instance(render_asset_filepath, semantic_id, scale)
            group._object_handles[elem_name] = mochi_handle

    def flush_to_hab_sim(self):

        if self._is_first_flush:
            self._instance_groups[_InstanceGroupType.STATIC].flush_to_hab_sim(self._mochi)
            self._is_first_flush = False

        self._instance_groups[_InstanceGroupType.DYNAMIC].flush_to_hab_sim(self._mochi)


        