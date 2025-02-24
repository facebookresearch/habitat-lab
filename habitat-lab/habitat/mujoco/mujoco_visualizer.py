# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
from enum import Enum

# todo: clean up how RenderInstanceHelper is exposed from habitat_sim extension
from habitat_sim._ext.habitat_sim_bindings import RenderInstanceHelper

import numpy as np

class _InstanceGroup:
    def __init__(self, hab_sim):
        identity_rotation_wxyz = [1.0, 0.0, 0.0, 0.0]
        self._render_instance_helper = RenderInstanceHelper(hab_sim, identity_rotation_wxyz)

    def flush_to_hab_sim(self, mj_data):

        self._positions[:] = mj_data.geom_xpos[self._geom_ids_array]

        orientations = [[1.0, 0.0, 0.0, 0.0]] * self._geom_ids_array.size

        self._render_instance_helper.set_world_poses(
            np.ascontiguousarray(self._positions), 
            np.ascontiguousarray(orientations))


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
            semantic_id = item_json.get("semantic_id", found_count)
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
            group._geom_ids_array = np.array(geom_ids_list_by_group_type[group_type])
            group._positions = self._mj_data.geom_xpos[group._geom_ids_array]

        self._is_first_flush = True

    def flush_to_hab_sim(self):

        if self._is_first_flush:
            self._instance_groups[_InstanceGroupType.STATIC].flush_to_hab_sim(self._mj_data)
            self._is_first_flush = False

        self._instance_groups[_InstanceGroupType.DYNAMIC].flush_to_hab_sim(self._mj_data)


        