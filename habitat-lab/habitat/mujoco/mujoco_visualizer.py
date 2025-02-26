# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
from enum import Enum

# todo: clean up how RenderInstanceHelper is exposed from habitat_sim extension
from habitat_sim._ext.habitat_sim_bindings import RenderInstanceHelper

import numpy as np

from habitat.isaac_sim import isaac_prim_utils

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
        self._render_instance_helper = RenderInstanceHelper(hab_sim, identity_rotation_wxyz)
        self._geom_ids_array = None

    def flush_to_hab_sim(self, mj_model, mj_data):

        if self._geom_ids_array is None:
            return

        positions = np.zeros((len(self._geom_ids_array), 3))  # np.take(mj_data.geom_xpos, self._geom_ids_array, axis=0)

        wxyz_rotations = np.zeros((len(self._geom_ids_array), 4))

        count = 0
        temp_quat = np.zeros((4,1), dtype=np.float64)
        for id in self._geom_ids_array:
            mat3x3 = mj_data.geom_xmat[id].reshape(9, 1)
            # temp lazy import of mujoco
            import mujoco
            mujoco.mju_mat2Quat(temp_quat, mat3x3)  # Convert to quaternion
            rotation = temp_quat.reshape((4,))

            # other_rotation = mj_model.geom_quat[id]

            mesh_id = mj_model.geom_dataid[id]
            other_rotation = mj_model.mesh_quat[mesh_id]

            wxyz_rotations[count] = quat_multiply(rotation, quat_transpose(other_rotation))

            mesh_pos = mj_model.mesh_pos[mesh_id]
            rotated_mesh_pos = quat_rotate(wxyz_rotations[count], mesh_pos)
            positions[count] = mj_data.geom_xpos[id] - rotated_mesh_pos

            count += 1

        # perf todo: pre-allocate converted-orientations and converted-positions arrays
        positions, wxyz_rotations = isaac_prim_utils.isaac_to_habitat(positions, wxyz_rotations)

        self._render_instance_helper.set_world_poses(
            np.ascontiguousarray(positions), 
            np.ascontiguousarray(wxyz_rotations))


class _InstanceGroupType(Enum):        
    STATIC = 0
    DYNAMIC = 1

class MuJoCoVisualizer:

    def __init__(self, hab_sim, mj_model, mj_data):

        self._instance_groups = {}
        for group_type in _InstanceGroupType:
            self._instance_groups[group_type] = _InstanceGroup(hab_sim)

        self._geom_name_to_render_asset = None
        self._mj_model = mj_model
        self._mj_data = mj_data
        self._is_first_flush = True
        
    def on_load_model(self, render_map_filepath):

        with open(render_map_filepath, "r") as f:
            json_data = json.load(f)
        render_map_json = json_data["render_map"]


        geom_ids_list_by_group_type = {_InstanceGroupType.STATIC : [], _InstanceGroupType.DYNAMIC : []}

        found_count = 0
        for geom_name in render_map_json:

            item_json = render_map_json[geom_name]
            is_dynamic = item_json["is_dynamic"]
            render_asset_filepath = item_json["render_asset_filepath"]
            semantic_id = item_json.get("semantic_id")
            semantic_id = found_count if semantic_id is None else semantic_id
            found_count += 1

            import mujoco  # temp sloppy defer import
            geom_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            assert geom_id != -1

            group_type = _InstanceGroupType.DYNAMIC if is_dynamic else _InstanceGroupType.STATIC
            group = self._instance_groups[group_type]
            group._render_instance_helper.add_instance(render_asset_filepath, semantic_id)
            geom_ids_list_by_group_type[group_type].append(geom_id)

        for group_type in self._instance_groups:
            group = self._instance_groups[group_type]
            if len(geom_ids_list_by_group_type[group_type]):
                group._geom_ids_array = np.array(geom_ids_list_by_group_type[group_type])

        self._is_first_flush = True

    def flush_to_hab_sim(self):

        if self._is_first_flush:
            self._instance_groups[_InstanceGroupType.STATIC].flush_to_hab_sim(self._mj_model, self._mj_data)
            self._is_first_flush = False

        self._instance_groups[_InstanceGroupType.DYNAMIC].flush_to_hab_sim(self._mj_model, self._mj_data)


        