# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import magnum as mn
import numpy as np
import omni.physx.scripts.utils
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.utils.stage import add_reference_to_stage
from pxr import UsdPhysics

import habitat_sim
from habitat.isaac_sim import isaac_prim_utils


class IsaacRigidObjectWrapper:
    def __init__(self, object_id: int, prim):
        self._object_id = object_id
        self._prim = prim
        self._rigid_prim = RigidPrim(str(prim.GetPath()))

    @property
    def object_id(self) -> int:
        return self._object_id

    @property
    def transformation(self) -> mn.Matrix4:
        return isaac_prim_utils.get_transformation(self._rigid_prim)

    @transformation.setter
    def transformation(self, transform_mat: mn.Matrix4):
        rotation_quat = mn.Quaternion.from_matrix(
            transform_mat.rotation_normalized()
        )
        translation = transform_mat.translation

        isaac_prim_utils.set_pose(
            self._rigid_prim,
            [*translation],
            isaac_prim_utils.magnum_quat_to_list_wxyz(rotation_quat),
        )

        # todo: consider self._rigid_prim.set_world_pose

    @property
    def translation(self) -> mn.Vector3:
        pos_usd, _ = self._rigid_prim.get_world_pose()
        pos_habitat = isaac_prim_utils.usd_to_habitat_position(pos_usd)
        return mn.Vector3(*pos_habitat)

    @translation.setter
    def translation(self, translation: mn.Vector3):
        translation_usd = isaac_prim_utils.habitat_to_usd_position(
            [translation.x, translation.y, translation.z]
        )
        isaac_prim_utils.set_translation(self._prim, translation_usd)

    @property
    def rotation(self) -> mn.Quaternion:
        _, rot_usd = self._rigid_prim.get_world_pose()
        rot_habitat = isaac_prim_utils.usd_to_habitat_rotation(rot_usd)
        return isaac_prim_utils.rotation_wxyz_to_magnum_quat(rot_habitat)

    @rotation.setter
    def rotation(self, rotation: mn.Quaternion):
        rotation_usd = isaac_prim_utils.habitat_to_usd_rotation(
            isaac_prim_utils.magnum_quat_to_list_wxyz(rotation)
        )
        isaac_prim_utils.set_rotation(self._prim, rotation_usd)

    # todo: implement angular_velocity and linear_velocity
    @property
    def angular_velocity(self) -> mn.Vector3:
        # NOTE: assuming angular vel convention of rotation around each axis which means we can transform the axes as if these were direction vectors
        hab_angles = isaac_prim_utils.usd_to_habitat_position(
            self._rigid_prim.get_angular_velocity()
        )
        return mn.Vector3(*hab_angles)

    @angular_velocity.setter
    def angular_velocity(self, vel: mn.Vector3):
        self._rigid_prim.set_angular_velocity(
            isaac_prim_utils.habitat_to_usd_position(vel)
        )

    @property
    def linear_velocity(self) -> mn.Vector3:
        lin_vel = isaac_prim_utils.usd_to_habitat_position(
            self._rigid_prim.get_linear_velocity()
        )
        return mn.Vector3(*lin_vel)

    @linear_velocity.setter
    def linear_velocity(self, vel: mn.Vector3):
        self._rigid_prim.set_linear_velocity(
            isaac_prim_utils.habitat_to_usd_position(vel)
        )

    @property
    def motion_type(self) -> habitat_sim.physics.MotionType:
        return habitat_sim.physics.MotionType.DYNAMIC

    @motion_type.setter
    def motion_type(self, motion_type: habitat_sim.physics.MotionType):
        # ignore for now and leave objects as dynamic
        pass

    @property
    def awake(self) -> bool:
        # assume awake for now
        return True

    @awake.setter
    def awake(self, a: bool):
        # ignore these requests for now and leave dynamic object awake
        pass

    def get_aabb(self) -> mn.Range3D:
        """
        Returns the global AABB of the object in Habitat coordinate system as Range3D.
        """
        isaac_bounds = isaac_prim_utils.get_bounding_box(self._prim)
        hab_min = mn.Vector3(
            *isaac_prim_utils.usd_to_habitat_position(isaac_bounds[:3])
        )
        hab_max = mn.Vector3(
            *isaac_prim_utils.usd_to_habitat_position(isaac_bounds[3:])
        )
        return mn.Range3D(hab_min, hab_max)

    def clear_dynamics(self) -> None:
        """
        clears velocities for post-teleport stability
        """
        self._rigid_prim.set_angular_velocity(np.ndarray([0, 0, 0]))
        self._rigid_prim.set_linear_velocity(np.ndarray([0, 0, 0]))


class IsaacRigidObjectManager:
    FIRST_OBJECT_ID = 100000
    LAST_OBJECT_ID = 199999

    def __init__(self, isaac_service):
        self._isaac_service = isaac_service

        self._obj_wrapper_by_object_id = {}
        self._obj_wrapper_by_object_handle = {}
        self._obj_wrapper_by_prim_path = {}
        self._next_object_id = IsaacRigidObjectManager.FIRST_OBJECT_ID

    def get_library_has_handle(self, object_handle):
        return object_handle in self._obj_wrapper_by_object_handle

    def _get_usd_filepath_for_object_config(self, object_config_filepath):
        assert object_config_filepath.startswith("data/")
        assert object_config_filepath.endswith(".object_config.json")

        folder, filename = os.path.split(object_config_filepath)

        usd_filepath = os.path.join(
            folder.replace("data/", "data/usd/"),
            f"OBJECT_{filename.removesuffix('.object_config.json')}.usda",
        )

        return os.path.abspath(usd_filepath)

    def _configure_as_dynamic_rigid_object(self, prim):
        omni.physx.scripts.utils.setRigidBody(
            prim, "convexHull", kinematic=False
        )

        mass_api = UsdPhysics.MassAPI.Apply(prim)
        # mass_api.CreateMassAttr(10)
        mass_api.CreateDensityAttr(1000)

    def _add_object(self, object_usd_filepath):
        object_id = self._next_object_id
        self._next_object_id += 1
        assert self._next_object_id <= IsaacRigidObjectManager.LAST_OBJECT_ID

        prim_path = f"/World/rigid_objects/obj_{object_id}"

        add_reference_to_stage(
            usd_path=object_usd_filepath, prim_path=prim_path
        )

        prim = self._isaac_service.world.stage.GetPrimAtPath(prim_path)
        self._configure_as_dynamic_rigid_object(prim)

        # sloppy: must call on_add_reference_to_stage after configuring object as RigidObject via USD
        self._isaac_service.usd_visualizer.on_add_reference_to_stage(
            usd_path=object_usd_filepath, prim_path=prim_path
        )

        obj_wrapper = IsaacRigidObjectWrapper(object_id, prim)
        self._obj_wrapper_by_object_id[object_id] = obj_wrapper
        self._obj_wrapper_by_prim_path[prim_path] = obj_wrapper

        return obj_wrapper

    def post_reset(self):
        physics_sim_view = self._isaac_service.world.physics_sim_view
        assert physics_sim_view
        for obj_wrapper in self._obj_wrapper_by_object_id.values():
            obj_wrapper._rigid_prim.initialize(physics_sim_view)

    def add_object_by_template_handle(self, object_path):
        assert object_path.endswith(".object_config.json")
        object_config_filepath = object_path

        object_usd_filepath = self._get_usd_filepath_for_object_config(
            object_config_filepath
        )
        return self._add_object(object_usd_filepath)

    def get_object_by_id(self, object_id):
        return self._obj_wrapper_by_object_id[object_id]

    def get_objects_by_handle_substring(self):
        return self._obj_wrapper_by_object_handle

    def set_object_handle(self, object_id, object_handle):
        self._obj_wrapper_by_object_handle[
            object_handle
        ] = self.get_object_by_id(object_id)

    def get_object_by_handle(self, object_handle):
        return self._obj_wrapper_by_object_handle[object_handle]

    def get_object_handles(self):
        return self._obj_wrapper_by_object_handle.keys()

    def get_object_by_prim_path(self, prim_path: str):
        return self._obj_wrapper_by_prim_path[prim_path]
