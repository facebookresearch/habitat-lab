# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
from enum import Enum

# todo: clean up how RenderInstanceHelper is exposed from habitat_sim extension
from habitat_sim._ext.habitat_sim_bindings import RenderInstanceHelper

import numpy as np
import magnum as mn

# This is needed so that gfx-replay keyframes contain an extra rotation for GLB render assets which is expected by our legacy Unity VR app.
ADD_HACK_ROTATION = True


from habitat.mochi.mochi_utils import (
    quat_multiply,
    rotvec_to_quat_wxyz,
    magnum_quat_to_list_wxyz,
    mochi_to_habitat_position
)

class _InstanceGroup:
    def __init__(self, hab_sim):
        identity_rotation_wxyz = [1.0, 0.0, 0.0, 0.0]
        # self._render_instance_helper = RenderInstanceHelper(hab_sim, identity_rotation_wxyz)
        self._render_instance_helper = RenderInstanceHelper(hab_sim, use_xyzw_orientations=False)
        self._object_handles = {}
        self._hack_rotations = {}

    def flush_to_hab_sim(self, mochi):

        if len(self._object_handles) == 0:
            return

        num_objects = len(self._object_handles)

        positions = np.zeros((num_objects, 3), dtype=np.float32)  # perf todo: use np.empty
        wxyz_rotations = np.zeros((num_objects, 4), dtype=np.float32)  # perf todo: use np.empty
        for (i, name) in enumerate(self._object_handles):
            handle = self._object_handles[name]
            pose = mochi.get_object_origin_transform(handle)
            positions[i] = mochi_to_habitat_position(pose[0])
            wxyz_rotations[i] = rotvec_to_quat_wxyz(pose[1])

            if name in self._hack_rotations:
                hack_rotation = self._hack_rotations[name]
                wxyz_rotations[i] = quat_multiply(wxyz_rotations[i], hack_rotation)

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
            do_legacy_blender_glb_fixup = item_json.get("do_legacy_blender_glb_fixup")
            if render_asset_filepath is None:
                continue
            semantic_id = item_json.get("semantic_id")
            semantic_id = self._next_semantic_id if semantic_id is None else semantic_id
            self._next_semantic_id += 1

            scale = item_json.get("scale")
            scale = mn.Vector3(scale) if scale is not None else unitScale

            if ADD_HACK_ROTATION:
                if do_legacy_blender_glb_fixup:
                    init_rotation = mn.Quaternion()
                    runtime_rotation = mn.Quaternion.rotation(mn.Rad(1.5708), mn.Vector3.x_axis())
                else:
                    init_rotation = mn.Quaternion()
                    runtime_rotation = None
            else:
                if do_legacy_blender_glb_fixup:
                    init_rotation = mn.Quaternion.rotation(mn.Rad(1.5708), mn.Vector3.x_axis())
                    runtime_rotation = None
                else:
                    init_rotation = mn.Quaternion()
                    runtime_rotation = None

            import re
            pattern = re.compile(elem_name)
            matched = False
            for obj_name in object_names:
                m = pattern.fullmatch(obj_name)
                if not m:
                    continue
                matched = True

                resolved_filepath = render_asset_filepath
                for i, g in enumerate(m.groups(), start=1):
                    resolved_filepath = resolved_filepath.replace(f"${i}", g)

                # print(f"mapping {obj_name} to {resolved_filepath}")

                mochi_handle = self._mochi.get_actor_handle(obj_name)
                group_type = _InstanceGroupType.DYNAMIC if is_dynamic else _InstanceGroupType.STATIC
                group = self._instance_groups[group_type]
                group._render_instance_helper.add_instance(
                    resolved_filepath, semantic_id, scale=scale, rotation=init_rotation
                )
                group._object_handles[obj_name] = mochi_handle
                if runtime_rotation:
                    group._hack_rotations[obj_name] = magnum_quat_to_list_wxyz(runtime_rotation)

            if not matched:
                print(f"MochiVisualizer: no object found for regex {elem_name} in {render_map_filepath}")


    def flush_to_hab_sim(self):

        if self._is_first_flush:
            self._instance_groups[_InstanceGroupType.STATIC].flush_to_hab_sim(self._mochi)
            self._is_first_flush = False

        self._instance_groups[_InstanceGroupType.DYNAMIC].flush_to_hab_sim(self._mochi)


        