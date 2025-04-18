# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum
from typing import Dict

from habitat.isaac_sim import isaac_prim_utils

# todo: clean up how RenderInstanceHelper is exposed from habitat_sim extension
from habitat_sim._ext.habitat_sim_bindings import RenderInstanceHelper

LOCAL_ROOT_KEY = "[root]"


@dataclass
class RenderAsset:
    """A render asset that can be provided to Habitat-sim ResourceManager::loadAndCreateRenderAssetInstance."""

    filepath: str
    # todo: possible color override
    semantic_id: int


import numpy as np


class _InstanceGroup:
    def __init__(self, hab_sim):
        self._prim_path_to_render_asset = {}
        self._xform_prim_view = None
        # isaac_identity_rotation_wxyz = [1.0, 0.0, 0.0, 0.0]
        # NOTE: below bool is use_xyzw_orientations=False, indicating wxyz
        self._render_instance_helper = RenderInstanceHelper(hab_sim, False)

    def set_dirty(
        self,
    ):  # todo: add underscore and rename to explain what is dirty
        self._xform_prim_view = None

    def check_dirty(self):  # todo: add underscore
        if self._xform_prim_view is not None:
            return

        if len(self._prim_path_to_render_asset) == 0:
            return

        # todo: handle case of no prims (empty scene)
        prim_paths = list(self._prim_path_to_render_asset.keys())

        # lazy import
        # TODO: maybe try the RigidPrimView instead to get around slow USD
        from omni.isaac.core.prims.xform_prim_view import XFormPrimView

        self._xform_prim_view = XFormPrimView(prim_paths)

        self._render_instance_helper.clear_all_instances()
        for prim_path in prim_paths:
            render_asset = self._prim_path_to_render_asset[prim_path]
            self._render_instance_helper.add_instance(
                render_asset.filepath, render_asset.semantic_id
            )

    def flush_to_hab_sim(self):
        if len(self._prim_path_to_render_asset) == 0:
            return

        positions, orientations = self._xform_prim_view.get_world_poses()

        positions, orientations = isaac_prim_utils.isaac_to_habitat(
            positions, orientations
        )

        self._render_instance_helper.set_world_poses(
            np.ascontiguousarray(positions), np.ascontiguousarray(orientations)
        )


class _InstanceGroupType(Enum):
    STATIC = 0
    DYNAMIC = 1


class UsdVisualizer:
    def __init__(self, isaac_stage, hab_sim):
        self._stage = isaac_stage
        self._were_prims_removed = False

        self._instance_groups: Dict[_InstanceGroupType, _InstanceGroup] = {}
        for group_type in _InstanceGroupType:
            self._instance_groups[group_type] = _InstanceGroup(hab_sim)

    # def _get_isaac_identity_rotation_quaternion(self):

    #     from pxr import UsdGeom, Sdf
    #     from omni.isaac.core.prims import XFormPrimView

    #     # Get the current USD stage
    #     stage = self._stage

    #     # Define a unique path for the dummy Xform
    #     dummy_xform_path = "/World/DummyXform"
    #     if stage.GetPrimAtPath(dummy_xform_path):
    #         raise RuntimeError(f"Prim already exists at {dummy_xform_path}")

    #     # Create the dummy Xform
    #     UsdGeom.Xform.Define(stage, dummy_xform_path)

    #     # Use XFormPrimView to get the world poses of the dummy Xform
    #     xform_view = XFormPrimView(prim_paths_expr=dummy_xform_path)
    #     positions, rotations = xform_view.get_world_poses()

    #     # Extract the identity rotation (assuming one Xform in the view)
    #     identity_rotation = rotations[0]

    #     # Clean up: Remove the dummy Xform
    #     stage.RemovePrim(Sdf.Path(dummy_xform_path))

    #     return identity_rotation

    # todo: get rid of usd_path or make it optional (only used for error messages)
    def on_add_reference_to_stage(
        self, usd_path, prim_path, semantic_id_offset=0
    ):
        # usd_dir = os.path.dirname(os.path.abspath(usd_path))

        root_prim_path = prim_path
        root_prim = self._stage.GetPrimAtPath(root_prim_path)

        found_count = 0

        # lazy import
        from pxr import Usd, UsdPhysics

        prim_range = Usd.PrimRange(root_prim)
        it = iter(prim_range)
        for prim in it:
            # todo: issue warnings

            prim_path = str(prim.GetPath())

            # Retrieve habitatVisual attributes
            asset_path_attr = prim.GetAttribute("habitatVisual:assetPath")
            if not asset_path_attr or not asset_path_attr.HasAuthoredValue():
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    print(
                        f"UsdVisualizer Warning: no Habitat visual found for RigidBody prim {prim_path} in {usd_path}."
                    )
                continue

            # we found a habitatVisual; it will visualize the entire subtree, so let's ignore children
            it.PruneChildren()

            semantic_id_attr = prim.GetAttribute("habitatVisual:semanticId")
            semantic_id = (
                semantic_id_attr.Get()
                if semantic_id_attr and semantic_id_attr.HasAuthoredValue()
                else None
            )

            if semantic_id is None:
                # if no semantic id is specified, assign a unique instance id (only unique to this USD file)
                semantic_id = found_count

            semantic_id += semantic_id_offset

            found_count += 1

            # asset_path should be relative to the project root, which is hopefully our CWD
            asset_path = asset_path_attr.Get()
            # asset_scale_attr = prim.GetAttribute("habitatVisual:assetScale")
            # asset_scale = (
            #    asset_scale_attr.Get()
            #    if asset_scale_attr and asset_scale_attr.HasAuthoredValue()
            #    else None
            # )

            asset_abs_path = asset_path

            # todo: consider doing this check later
            is_dynamic = prim.HasAPI(UsdPhysics.RigidBodyAPI)

            group_type = (
                _InstanceGroupType.DYNAMIC
                if is_dynamic
                else _InstanceGroupType.STATIC
            )
            group = self._instance_groups[group_type]

            group._prim_path_to_render_asset[prim_path] = RenderAsset(
                filepath=asset_abs_path, semantic_id=semantic_id
            )
            group.set_dirty()

        if not found_count:
            print(
                f"UsdVisualizer Warning: no Habitat visuals found for {usd_path}."
            )

    def on_remove_prims(self):
        #
        self._were_prims_removed = True

    def _check_were_prims_removed(self):  # todo: add underscore and rename
        if not self._were_prims_removed:
            return

        # reset flag
        self._were_prims_removed = False

        for _group_type, group in self._instance_groups.items():

            def does_prim_exist(prim_path):
                return self._stage.GetPrimAtPath(prim_path).IsValid()

            keys_to_delete = [
                path
                for path in group._prim_path_to_render_asset
                if not does_prim_exist(path)
            ]

            if len(keys_to_delete) == 0:
                return

            for key in keys_to_delete:
                del group._prim_path_to_render_asset[key]

            group.set_dirty()

    def flush_to_hab_sim(self):
        self._check_were_prims_removed()

        for group_type in self._instance_groups:
            group = self._instance_groups[group_type]
            was_dirty = group._xform_prim_view is None

            group.check_dirty()

            if was_dirty or group_type == _InstanceGroupType.DYNAMIC:
                group.flush_to_hab_sim()
