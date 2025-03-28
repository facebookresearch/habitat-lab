# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
import numpy as np
import omni
import omni.physx.scripts.utils as physxUtils

# todo: add guard to ensure SimulatorApp is created, or give nice error message, so we don't get weird import errors here
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.prims.rigid_prim_view import RigidPrimView
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction
from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation as R

from habitat.isaac_sim import isaac_prim_utils


class MurpRobotWrapper:
    """Isaac-internal wrapper for a robot.

    The goal with this wrapper is convenience but not encapsulation. See also (public) IsaacMobileManipulator, which has the goal of exposing a minimal public interface to the rest of Habitat-lab.
    """

    def __init__(self, agent_cfg, isaac_service, instance_id=0):

        self._isaac_service = isaac_service
        asset_path = agent_cfg.articulated_agent_usda
        robot_prim_path = f"/World/env_{instance_id}/Murp"
        self._robot_prim_path = robot_prim_path

        add_reference_to_stage(usd_path=asset_path, prim_path=robot_prim_path)
        self._isaac_service.usd_visualizer.on_add_reference_to_stage(
            usd_path=asset_path, prim_path=robot_prim_path
        )

        robot_prim = self._isaac_service.world.stage.GetPrimAtPath(
            robot_prim_path
        )

        if not robot_prim.IsValid():
            raise ValueError(f"Prim at {robot_prim_path} is not valid.")

        # Traverse only the robot's prim hierarchy
        for prim in Usd.PrimRange(robot_prim):

            if prim.HasAPI(PhysxSchema.PhysxJointAPI):
                joint_api = PhysxSchema.PhysxJointAPI(prim)
                joint_api.GetMaxJointVelocityAttr().Set(200.0)

            if prim.HasAPI(UsdPhysics.DriveAPI):
                # Access the existing DriveAPI
                drive_api = UsdPhysics.DriveAPI(prim, "angular")
                if drive_api:

                    # Modify drive parameters
                    drive_api.GetStiffnessAttr().Set(10.0)  # Position gain
                    drive_api.GetDampingAttr().Set(0.1)  # Velocity gain
                    drive_api.GetMaxForceAttr().Set(50)  # Maximum force/torque

                drive_api = UsdPhysics.DriveAPI(prim, "linear")
                if drive_api:

                    drive_api = UsdPhysics.DriveAPI.Get(prim, "linear")
                    drive_api.GetStiffnessAttr().Set(
                        1000
                    )  # Example for linear stiffness

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

        self._name = f"murp_{instance_id}"
        self._robot = self._isaac_service.world.scene.add(
            Robot(prim_path=robot_prim_path, name=self._name)
        )
        self._robot_controller = self._robot.get_articulation_controller()

        self._step_count = 0
        self._lateral_vel = [0.0, 0.0]
        self._instance_id = instance_id

        self.arm_target_joint_pos = None

        # beware this poses the object
        self._xform_prim_view = self._create_xform_prim_view()

    @property
    def robot(self) -> Robot:
        """Get Isaac Sim Robot"""
        return self._robot

    def get_prim_path(self):
        return self._robot_prim_path

    def set_root_pose(self, pos, rot, convention="hab"):

        rot = [rot.scalar] + list(rot.vector)
        if convention == "hab":
            pos = isaac_prim_utils.habitat_to_usd_position(pos)
            rot = isaac_prim_utils.habitat_to_usd_rotation(rot)
        self._robot.set_world_pose(pos, rot)

    def get_root_pose(self, convention="hab"):

        pos_usd, rot_usd = self._robot.get_world_pose()
        if convention == "hab":
            pos = mn.Vector3(isaac_prim_utils.usd_to_habitat_position(pos_usd))
            rot = isaac_prim_utils.rotation_wxyz_to_magnum_quat(
                isaac_prim_utils.usd_to_habitat_rotation(rot_usd)
            )
        else:
            pos = mn.Vector3(pos_usd)
            rot = isaac_prim_utils.rotation_wxyz_to_magnum_quat(rot_usd)
        return pos, rot

    def get_link_world_poses(self, convention="hab"):

        positions = []
        positions_usd, rotations_usd = self._xform_prim_view.get_world_poses()
        for pos in positions_usd:
            if convention == "hab":
                pos = isaac_prim_utils.usd_to_habitat_position(pos)
            pos = mn.Vector3(pos)
            positions.append(pos)
        rotations = []
        for rot in rotations_usd:
            if convention == "hab":
                rot = isaac_prim_utils.usd_to_habitat_rotation(rot)
            rot = isaac_prim_utils.rotation_wxyz_to_magnum_quat(rot)
            rotations.append(rot)

        # perf todo: consider RobotView.get_body_coms instead

        return positions, rotations

    def _create_xform_prim_view(self):

        root_prim_path = self._robot_prim_path
        root_prim = self._isaac_service.world.stage.GetPrimAtPath(
            root_prim_path
        )

        prim_paths = []

        # lazy import
        from pxr import Usd, UsdPhysics

        prim_range = Usd.PrimRange(root_prim)
        it = iter(prim_range)
        for prim in it:
            prim_path = str(prim.GetPath())

            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue

            # we found a rigid body, so let's ignore children
            it.PruneChildren()
            prim_paths.append(prim_path)

        assert len(prim_paths)
        print("prim_paths: ", prim_paths, len(prim_paths))

        self._body_prim_paths = prim_paths

        from omni.isaac.core.prims.xform_prim_view import XFormPrimView

        return XFormPrimView(prim_paths)

    def reset_arm(self):
        # todo: specify this in isaac_spot_robot.py
        left_arm_joint_names = [
            "left_fr3_joint1",
            "left_fr3_joint2",
            "left_fr3_joint3",
            "left_fr3_joint4",
            "left_fr3_joint5",
            "left_fr3_joint6",
            "left_fr3_joint7",
        ]

        right_arm_joint_names = [
            "right_fr3_joint1",
            "right_fr3_joint2",
            "right_fr3_joint3",
            "right_fr3_joint4",
            "right_fr3_joint5",
            "right_fr3_joint6",
            "right_fr3_joint7",
        ]

        self.ee_link_name = left_arm_joint_names[-1].replace("joint", "link")
        self.right_ee_link_name = right_arm_joint_names[-1].replace(
            "joint", "link"
        )

        self.ee_link_id = self.get_link_id(self.ee_link_name)
        self.right_ee_link_id = self.get_link_id(self.right_ee_link_name)
        left_arm_joint_indices = []
        right_arm_joint_indices = []
        dof_names = self._robot.dof_names
        print("dof names: ", dof_names)
        assert len(dof_names) > 0
        for arm_joint_name in left_arm_joint_names:
            left_arm_joint_indices.append(dof_names.index(arm_joint_name))

        for arm_joint_name in right_arm_joint_names:
            right_arm_joint_indices.append(dof_names.index(arm_joint_name))

        self._arm_joint_indices = np.array(left_arm_joint_indices)
        self._right_arm_joint_indices = np.array(right_arm_joint_indices)
        rest_positions = [
            2.6116285,
            1.5283098,
            1.0930868,
            -0.50559217,
            0.48147443,
            2.628784,
            -1.3962275,
        ]
        self._target_arm_joint_positions = rest_positions
        self._target_right_arm_joint_positions = rest_positions

    def get_link_id(self, link_str):
        return [
            i for i, s in enumerate(self._body_prim_paths) if link_str in s
        ][0]

    def reset_hand(self):
        # todo: specify this in isaac_spot_robot.py
        # only list the revolute joints
        right_hand_joint_names = [
            "joint_0_0",
            "joint_1_0",
            "joint_2_0",
            "joint_3_0",
            "joint_4_0",
            "joint_5_0",
            "joint_6_0",
            "joint_7_0",
            "joint_8_0",
            "joint_9_0",
            "joint_10_0",
            "joint_11_0",
            "joint_12_0",
            "joint_13_0",
            "joint_14_0",
            "joint_15_0",
        ]
        left_hand_joint_names = [
            "joint_l_0_0",
            "joint_l_1_0",
            "joint_l_2_0",
            "joint_l_3_0",
            "joint_l_4_0",
            "joint_l_5_0",
            "joint_l_6_0",
            "joint_l_7_0",
            "joint_l_8_0",
            "joint_l_9_0",
            "joint_l_10_0",
            "joint_l_11_0",
            "joint_l_12_0",
            "joint_l_13_0",
            "joint_l_14_0",
            "joint_l_15_0",
        ]

        left_hand_joint_indices = []
        right_hand_joint_indices = []
        dof_names = self._robot.dof_names
        assert len(dof_names) > 0
        for hand_joint_name in left_hand_joint_names:
            left_hand_joint_indices.append(dof_names.index(hand_joint_name))

        for hand_joint_name in right_hand_joint_names:
            right_hand_joint_indices.append(dof_names.index(hand_joint_name))

        self._hand_joint_indices = np.array(left_hand_joint_indices)
        self._right_hand_joint_indices = np.array(right_hand_joint_indices)

        n_hand_joints = len(left_hand_joint_names)
        closed_positions = np.array([3.14159] * n_hand_joints)
        open_positions = np.zeros(n_hand_joints)
        self._target_hand_joint_positions = open_positions
        self._target_right_hand_joint_positions = open_positions

    def post_reset(self):
        # todo: just do a single callback
        self._isaac_service.world.add_physics_callback(
            f"{self._name}_physics_callback", callback_fn=self.physics_callback
        )
        self.reset_arm()
        self.reset_hand()

    def scale_prim_mass_and_inertia(self, path, scale):

        prim = self._isaac_service.world.stage.GetPrimAtPath(path)
        assert prim.HasAPI(UsdPhysics.MassAPI)
        mass_api = UsdPhysics.MassAPI(prim)
        mass_api.GetMassAttr().Set(mass_api.GetMassAttr().Get() * scale)
        mass_api.GetDiagonalInertiaAttr().Set(
            mass_api.GetDiagonalInertiaAttr().Get() * scale
        )

    def fix_base_orientation_via_angular_vel(
        self, step_size, base_position, base_orientation
    ):

        curr_angular_velocity = self._robot.get_angular_velocity()

        # Constants
        max_angular_velocity = 3.0  # Maximum angular velocity (rad/s)

        # wxyz to xyzw
        base_orientation_xyzw = np.array(
            [
                base_orientation[1],
                base_orientation[2],
                base_orientation[3],
                base_orientation[0],
            ]
        )
        base_rotation = R.from_quat(base_orientation_xyzw)

        # Define the local "up" axis and transform it to world space
        local_up = np.array([0, 0, 1])  # Object's local up axis (hack: -1?)
        world_up = base_rotation.apply(local_up)  # Local up in world space

        # Define the global up axis
        global_up = np.array([0, 0, 1])  # Global up direction

        # Compute the axis of rotation to align world_up to global_up
        rotation_axis = np.cross(world_up, global_up)
        rotation_axis_norm = np.linalg.norm(rotation_axis)

        # Handle special cases where no rotation is needed
        if rotation_axis_norm < 1e-6:  # Already aligned
            desired_angular_velocity = np.array(
                [0, 0, 0]
            )  # No correction needed
        else:
            # Normalize the rotation axis
            rotation_axis /= rotation_axis_norm

            # Compute the angle of rotation using the dot product
            rotation_angle = np.arccos(
                np.clip(np.dot(world_up, global_up), -1.0, 1.0)
            )

            # Calculate the angular velocity to correct the tilt in one step
            hack_scale = 1.0
            tilt_correction_velocity = (
                (rotation_axis * rotation_angle) / step_size
            ) * hack_scale

            # Cap the angular velocity to the maximum allowed value
            angular_velocity_magnitude = np.linalg.norm(
                tilt_correction_velocity
            )
            if angular_velocity_magnitude > max_angular_velocity:
                tilt_correction_velocity *= (
                    max_angular_velocity / angular_velocity_magnitude
                )

            desired_angular_velocity = tilt_correction_velocity

        # don't change rotation about z
        desired_angular_velocity[2] = curr_angular_velocity[2]

        self._robot.set_angular_velocity(desired_angular_velocity)

    def fix_base_height_via_linear_vel_z(
        self, step_size, base_position, base_orientation
    ):

        curr_linear_velocity = self._robot.get_linear_velocity()

        z_target = 0.1  # todo: get from navmesh or assume ground_z==0
        max_linear_vel = 3.0

        # Extract the vertical position and velocity
        z_current = base_position[2]

        # Compute the position error
        position_error = z_target - z_current

        desired_linear_vel_z = position_error / step_size
        desired_linear_vel_z = max(
            -max_linear_vel, min(max_linear_vel, desired_linear_vel_z)
        )

        self._robot.set_linear_velocity(
            [
                curr_linear_velocity[0],
                curr_linear_velocity[1],
                desired_linear_vel_z,
            ]
        )

    def drive_arm(self, step_size):

        if np.array(self._target_arm_joint_positions).any():
            assert len(self._target_arm_joint_positions) == len(
                self._arm_joint_indices
            )
            self._robot_controller.apply_action(
                ArticulationAction(
                    joint_positions=self._target_arm_joint_positions,
                    joint_indices=self._arm_joint_indices,
                )
            )

    def drive_right_arm(self, step_size):

        if np.array(self._target_right_arm_joint_positions).any():
            assert len(self._target_right_arm_joint_positions) == len(
                self._right_arm_joint_indices
            )
            self._robot_controller.apply_action(
                ArticulationAction(
                    joint_positions=self._target_right_arm_joint_positions,
                    joint_indices=self._right_arm_joint_indices,
                )
            )

    def drive_hand(self, step_size):

        if np.array(self._target_hand_joint_positions).any():
            assert len(self._target_hand_joint_positions) == len(
                self._hand_joint_indices
            )
            self._robot_controller.apply_action(
                ArticulationAction(
                    joint_positions=self._target_hand_joint_positions,
                    joint_indices=self._hand_joint_indices,
                )
            )

    def drive_right_hand(self, step_size):

        if np.array(self._target_right_hand_joint_positions).any():
            assert len(self._target_right_hand_joint_positions) == len(
                self._right_hand_joint_indices
            )
            self._robot_controller.apply_action(
                ArticulationAction(
                    joint_positions=self._target_right_hand_joint_positions,
                    joint_indices=self._right_hand_joint_indices,
                )
            )

    def fix_base(self, step_size, base_position, base_orientation):

        self.fix_base_height_via_linear_vel_z(
            step_size, base_position, base_orientation
        )
        self.fix_base_orientation_via_angular_vel(
            step_size, base_position, base_orientation
        )

    def physics_callback(self, step_size):
        base_position, base_orientation = self._robot.get_world_pose()
        self.fix_base(step_size, base_position, base_orientation)
        # self.drive_arm(step_size)
        self.drive_right_arm(step_size)
        # self.drive_hand(step_size)
        self.drive_right_hand(step_size)
        self._step_count += 1

    @property
    def arm_joint_pos(self):
        """Get the current arm joint positions."""
        robot_joint_positions = self._robot.get_joint_positions()
        arm_joint_positions = np.array(
            [robot_joint_positions[i] for i in self._arm_joint_indices],
            dtype=np.float32,
        )

        return arm_joint_positions

    @property
    def right_arm_joint_pos(self):
        """Get the current arm joint positions."""
        robot_joint_positions = self._robot.get_joint_positions()
        arm_joint_positions = np.array(
            [robot_joint_positions[i] for i in self._right_arm_joint_indices],
            dtype=np.float32,
        )

        return arm_joint_positions

    @property
    def hand_joint_pos(self):
        """Get the current arm joint positions."""
        robot_joint_positions = self._robot.get_joint_positions()
        hand_joint_positions = np.array(
            [robot_joint_positions[i] for i in self._hand_joint_indices],
            dtype=np.float32,
        )

        return hand_joint_positions

    @property
    def right_hand_joint_pos(self):
        """Get the current arm joint positions."""
        robot_joint_positions = self._robot.get_joint_positions()
        hand_joint_positions = np.array(
            [robot_joint_positions[i] for i in self._right_hand_joint_indices],
            dtype=np.float32,
        )

        return hand_joint_positions

    def get_fingertip_pose(self, finger_ids, convention="hab"):
        """Get the current ee position and rotation."""
        link_poses = self.get_link_world_poses(convention=convention)
        ee_poses = []
        ee_rots = []
        for finger_id in finger_ids:
            ee_poses.extend([*link_poses[0][finger_id]])
            ee_rots.extend(
                [
                    link_poses[1][finger_id].scalar,
                    *link_poses[1][finger_id].vector,
                ]
            )

        return ee_poses, ee_rots

    def fingertip_pose(self, convention="hab"):
        # index, thumb, middle, ring
        left_finger_ids = [43, 48, 53, 58]
        return self.get_fingertip_pose(left_finger_ids, convention)

    def fingertip_right_pose(self, convention="hab"):
        # index, thumb, middle, ring
        right_finger_ids = [74, 79, 84, 89]
        return self.get_fingertip_pose(right_finger_ids, convention)

    def get_ee_pose(self, link_id, convention="hab"):
        """Get the current ee position and rotation."""
        link_poses = self.get_link_world_poses(convention=convention)

        ee_pos = link_poses[0][link_id]
        ee_rot = link_poses[1][link_id]

        return ee_pos, ee_rot

    def ee_pose(self, convention="hab"):
        return self.get_ee_pose(self.ee_link_id)

    def ee_right_pose(self, convention="hab"):
        return self.get_ee_pose(self.right_ee_link_id)

    def get_articulation_links(self, prim_path: str):
        """Function to get link names which articulation can be applied
        Sample call: get_articulation_links(prim_path="/World/test_scene")"""
        prim = self._isaac_service.world.stage.GetPrimAtPath(prim_path)

        articulation_links = {}

        for child_prim in prim.GetChildren():
            if UsdPhysics.ArticulationRootAPI.CanApply(child_prim):
                articulation_api = UsdPhysics.ArticulationRootAPI(child_prim)

                link_names = []

                for link_prim in child_prim.GetChildren():
                    if link_prim.GetTypeName() == "Xform":
                        link_names.append(link_prim.GetName())

                articulation_links[child_prim.GetName()] = link_names

        return articulation_links

    def hand_pose(self, convention="hab"):
        """Get the current hand base link position and rotation."""
        link_poses = self.get_link_world_poses(convention=convention)
        hand_base_id = self.get_link_id("base_link_hand")

        ee_pos = link_poses[0][hand_base_id]
        ee_rot = link_poses[1][hand_base_id]

        return ee_pos, ee_rot
