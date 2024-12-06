# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import json
from dataclasses import dataclass

LOCAL_ROOT_KEY = "[root]"

@dataclass
class RenderAsset:
    """A render asset that can be provided to Habitat-sim ResourceManager::loadAndCreateRenderAssetInstance."""
    filepath: str
    # todo: possible color override


class UsdVisualizer:

    def __init__(self, isaac_stage):

        self._prim_path_to_render_asset = {}
        self._mapping_dicts_by_filepath = {}
        self._stage = isaac_stage
        self._xform_prim_view = None
        self._were_prims_removed = False
        pass

    def _check_load_mapping_file(self, mapping_filepath):

        if mapping_filepath in self._mapping_dicts_by_filepath:
            return

        loaded_assets = {}
        if not os.path.exists(mapping_filepath):
            print(f"UsdVisualizer Error: mapping file [{mapping_filepath}] does not exist.")
        else:
            try:
                with open(mapping_filepath, "r") as file:
                    loaded_data = json.load(file)

                # Convert the loaded data back into a dictionary of RenderAsset objects
                loaded_assets = {key: RenderAsset(**value) for key, value in loaded_data.items()}
                print(f"UsdVisualizer Info: loaded {len(loaded_assets)} entries from [{mapping_filepath}].")

            except json.JSONDecodeError:
                print(f"UsdVisualizer Error: Failed to parse JSON in [{mapping_filepath}].")

        self._mapping_dicts_by_filepath[mapping_filepath] = loaded_assets


    def _map_prim_path_to_render_asset(self, local_prim_path, mapping_filepath):

        mapping_dict = self._mapping_dicts_by_filepath[mapping_filepath]
        return mapping_dict.get(local_prim_path)


    def _get_mapping_filepath_from_usd(self, usd_path):
        return usd_path + ".usd_visualizer.json"

    def on_add_reference_to_stage(self, usd_path, prim_path):

        root_prim_path = prim_path
        root_prim = self._stage.GetPrimAtPath(root_prim_path)
        mapping_filepath = self._get_mapping_filepath_from_usd(usd_path)
        self._check_load_mapping_file(mapping_filepath)

        # Traverse prim hierarchy
        root_prefix_len = len(root_prim_path) + 1
        # lazy import
        from pxr import Usd
        for prim in Usd.PrimRange(root_prim):

            full_prim_path = str(prim.GetPath())
            assert full_prim_path.startswith(root_prim_path)
            if full_prim_path == root_prim_path:
                local_prim_path = LOCAL_ROOT_KEY
            else:
                local_prim_path = full_prim_path[root_prefix_len:]

            assert full_prim_path not in self._prim_path_to_render_asset
            render_asset = self._map_prim_path_to_render_asset(local_prim_path, mapping_filepath)
            if render_asset:
                # todo: Think about how to handle if prim.prim_path is already present in dict.
                self._prim_path_to_render_asset[full_prim_path] = render_asset
                self._set_dirty()

                # todo: inspect prim to decide static or dynamic
            else:
                # todo: can we tell if this prim *should* have a visual mapping? e.g. it is an articulated link or rigid object
                print(f"UsdVisualizer Warning: no render asset found for [{local_prim_path}] in [{mapping_filepath}].")


    def on_add_cube_prim(self, prim_path):
        # todo: inspect cube for dimensions
        pass

    def on_remove_prims(self):
        # 
        self._were_prims_removed = True
        pass

    def _check_were_prims_removed(self): # todo: add underscore and rename
        if not self._were_prims_removed:
            return

        # reset flag    
        self._were_prims_removed = False

        def does_prim_exist(prim_path):
            return self._stage.GetPrimAtPath(prim_path).IsValid()

        keys_to_delete = [path for path in self._prim_path_to_render_asset if not does_prim_exist(path)]

        if len(keys_to_delete) == 0:
            return

        for key in keys_to_delete:
            del self._prim_path_to_render_asset[key]

        self._set_dirty()

    def _set_dirty(self): # todo: add underscore and rename to explain what is dirty
        self._xform_prim_view = None

    def _check_dirty(self): # todo: add underscore
        if self._xform_prim_view is not None:
            return

        # todo: handle case of no prims (empty scene)
        prim_paths = list(self._prim_path_to_render_asset.keys())

        # lazy import
        from omni.isaac.core.prims.xform_prim_view import XFormPrimView
        self._xform_prim_view = XFormPrimView(prim_paths)

        thingy.clear_all_instances()
        for prim_path in prim_paths:
            render_asset = self._prim_path_to_render_asset[prim_path]
            thingy.add_instance(render_asset.filepath)


    def render(self):

        self._check_were_prims_removed()

        self._check_dirty()

        positions, orientations = self._xform_prim_view.get_world_poses()
        thingy.set_world_poses(positions, orientations)

        