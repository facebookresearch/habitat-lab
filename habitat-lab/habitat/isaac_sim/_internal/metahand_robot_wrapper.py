# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from scipy.spatial.transform import Rotation as R

import magnum as mn

# todo: add guard to ensure SimulatorApp is created, or give nice error message, so we don't get weird import errors here
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.robots import Robot
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.prims.rigid_prim_view import RigidPrimView

from pxr import UsdGeom, Sdf, Usd, UsdPhysics, PhysxSchema
import omni.physx.scripts.utils as physxUtils

from habitat.isaac_sim import isaac_prim_utils

def quaternion_conjugate(q):
    """Compute the conjugate of a quaternion [w, x, y, z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quaternion_multiply(q1, q2):
    """Compute the product of two quaternions [w1, x1, y1, z1] and [w2, x2, y2, z2]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def compute_angular_velocity_with_limit(cur_base_rot_wxyz, target_base_rot_wxyz, step_size, max_ang_vel_mag):
    # Compute the rotation error quaternion
    q_current_conjugate = quaternion_conjugate(cur_base_rot_wxyz)
    q_error = quaternion_multiply(target_base_rot_wxyz, q_current_conjugate)
    
    # Normalize the error quaternion
    q_error /= np.linalg.norm(q_error)
    
    # Extract the axis-angle representation from the quaternion error
    angle = 2 * np.arccos(np.clip(q_error[0], -1.0, 1.0))  # Angle in radians
    if angle > np.pi:  # Ensure the angle is in [0, pi]
        angle -= 2 * np.pi
    axis = q_error[1:] / np.linalg.norm(q_error[1:]) if np.linalg.norm(q_error[1:]) > 0 else np.array([0.0, 0.0, 0.0])
    
    # Compute the angular velocity
    angular_vel = (axis * angle) / step_size
    
    # Enforce the maximum angular velocity magnitude
    ang_vel_mag = np.linalg.norm(angular_vel)
    if ang_vel_mag > max_ang_vel_mag:
        angular_vel = (angular_vel / ang_vel_mag) * max_ang_vel_mag
    
    return angular_vel


class MetahandRobotWrapper:
    def __init__(self, isaac_service, instance_id=0):

        self._isaac_service = isaac_service
        asset_path = "data/usd/robots/allegro_digit360_right_calib_free.usda"
        robot_prim_path = f"/World/env_{instance_id}/Metahand"
        self._robot_prim_path = robot_prim_path

        add_reference_to_stage(usd_path=asset_path, prim_path=robot_prim_path)
        self._isaac_service.usd_visualizer.on_add_reference_to_stage(usd_path=asset_path, prim_path=robot_prim_path)

        robot_prim = self._isaac_service.world.stage.GetPrimAtPath(robot_prim_path)

        if not robot_prim.IsValid():
            raise ValueError(f"Prim at {robot_prim_path} is not valid.")

        # Traverse only the robot's prim hierarchy
        for prim in Usd.PrimRange(robot_prim):

            if prim.HasAPI(PhysxSchema.PhysxJointAPI):
                joint_api = PhysxSchema.PhysxJointAPI(prim)
                joint_api.GetMaxJointVelocityAttr().Set(1000.0)

            if prim.HasAPI(UsdPhysics.DriveAPI):
                # Access the existing DriveAPI
                drive_api = UsdPhysics.DriveAPI(prim, "angular")
                if drive_api:

                    # Modify drive parameters
                    drive_api.GetStiffnessAttr().Set(10.0)  # Position gain
                    drive_api.GetDampingAttr().Set(0.1)     # Velocity gain
                    drive_api.GetMaxForceAttr().Set(1000)  # Maximum force/torque

                drive_api = UsdPhysics.DriveAPI(prim, "linear")
                if drive_api:

                    drive_api = UsdPhysics.DriveAPI.Get(prim, "linear")
                    drive_api.GetStiffnessAttr().Set(1000)  # Example for linear stiffness

            if prim.HasAPI(UsdPhysics.RigidBodyAPI):

                # UsdPhysics.RigidBodyAPI doesn't support damping but PhysxRigidBodyAPI does
                if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                    physx_api = PhysxSchema.PhysxRigidBodyAPI(prim)
                else:
                    physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)

                # todo: decide hard-coded values here
                physx_api.CreateLinearDampingAttr(50.0)
                physx_api.CreateAngularDampingAttr(10.0)

        # todo: investigate if this is needed for kinematic base
        # todo: resolve off-by-100 scale issue
        # self.scale_prim_mass_and_inertia(f"{robot_prim_path}/base", 100.0)

        # todo: unify with robot_prim_path
        self._name = f"metahand_{instance_id}"
        self._robot = self._isaac_service.world.scene.add(Robot(prim_path=robot_prim_path, name=self._name))
        self._robot_controller = self._robot.get_articulation_controller()

        self._instance_id = instance_id
        self._step_count = 0
        self._target_joint_positions = None
        self._target_base_rot = None
        self._target_base_pos = None

    @property
    def robot(self) -> Robot:
        """Get Isaac Sim Robot"""
        return self._robot

    def get_prim_path(self):
        return self._robot_prim_path

    def post_reset(self):
        # todo: just do a single callback
        self._isaac_service.world.add_physics_callback(f"{self._name}_physics_callback", callback_fn=self.physics_callback)

    # def scale_prim_mass_and_inertia(self, path, scale):

    #     prim = self._isaac_service.world.stage.GetPrimAtPath(path)
    #     assert prim.HasAPI(UsdPhysics.MassAPI)
    #     mass_api = UsdPhysics.MassAPI(prim)
    #     mass_api.GetMassAttr().Set(mass_api.GetMassAttr().Get() * scale)
    #     mass_api.GetDiagonalInertiaAttr().Set(mass_api.GetDiagonalInertiaAttr().Get() * scale)


    def set_target_base_position(self, pos_habitat):

        self._target_base_pos = isaac_prim_utils.habitat_to_usd_position(list(pos_habitat))

    def set_target_base_rotation(self, rot_habitat):

        self._target_base_rot = isaac_prim_utils.habitat_to_usd_rotation(isaac_prim_utils.magnum_quat_to_list_wxyz(rot_habitat))

    def drive_root_world_pose(self, step_size):

        # todo: decide "rotation" vs "orientation"
        cur_base_pos, cur_base_rot = self._robot.get_world_pose()

        if self._target_base_pos is not None:

            pos_error = self._target_base_pos - cur_base_pos
            
            # Compute the velocity needed to fully correct the position error in one step
            linear_vel = pos_error / step_size
            
            # Calculate the magnitude of the computed velocity
            vel_mag = np.linalg.norm(linear_vel)
            
            max_linear_vel_mag = 10.0
            # Enforce the maximum velocity magnitude limit
            if vel_mag > max_linear_vel_mag:
                linear_vel = (linear_vel / vel_mag) * max_linear_vel_mag

            self._robot.set_linear_velocity(linear_vel)

            error_mag = np.linalg.norm(pos_error)
            snap_threshold_dist = 0.75
            if error_mag > snap_threshold_dist:
                print("MetahandRobotWrapper: snapped base to target position.")
                self._robot.set_world_pose(self._target_base_pos, cur_base_rot)

        if self._target_base_rot is not None:

            max_ang_vel_mag = 50.0
            ang_vel = compute_angular_velocity_with_limit(cur_base_rot, self._target_base_rot, step_size, max_ang_vel_mag)
            self._robot.set_angular_velocity(ang_vel)

    def drive_hand_pose(self, step_size):

        # if self._target_joint_positions is None or self._step_count % 60 == 0:
        #     # Apply random joint velocities to the robot's first two joints
        #     # velocities = 5 * np.ones((20,))
        #     # robot_controller.apply_action(
        #     #     ArticulationAction(joint_positions=None, joint_efforts=None, joint_velocities=velocities)
        #     # )
        #     num_dofs = 16
        #     self._target_joint_positions = np.random.uniform(-0.7, 0.7, num_dofs)
        #     # random_efforts = 5 * np.ones((20,))
        #     # robot_controller.apply_action(
        #     #     ArticulationAction(joint_positions=None, joint_efforts=random_efforts, joint_velocities=None)
        #     # )

        self._robot_controller.apply_action(
            ArticulationAction(joint_positions=self._target_joint_positions, joint_efforts=None, joint_velocities=None)
        )


    def fix_base(self, step_size):
        pos_usd = isaac_prim_utils.habitat_to_usd_position([-7.0, 1.0, -3.0])
        self._robot.set_world_pose(pos_usd, [1, 0, 0, 0])
        self._robot.set_linear_velocity([0, 0, 0,])
        self._robot.set_angular_velocity([0, 0, 0])

    def physics_callback(self, step_size):
        # self.fix_base(step_size)
        self.drive_root_world_pose(step_size)
        self.drive_hand_pose(step_size)
        self._step_count += 1

