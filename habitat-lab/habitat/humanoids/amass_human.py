# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

import magnum as mn
import numpy as np

from fairmotion.core import motion
from fairmotion.data import amass
from fairmotion.ops import motion as motion_ops
from fairmotion.ops import conversions

from habitat_sim.logging import LoggingContext, logger

import habitat_sim.physics as phy

from habitat.humanoids.human_base import Humanoid
from habitat.robots.mobile_manipulator import (
    MobileManipulator,
    MobileManipulatorParams,
    RobotCameraParams,
)

@dataclass
class HumanParams:
    arm_init_params_left: Optional[np.ndarray]
    arm_init_params_right: Optional[np.ndarray]
    cameras: Dict
    gripper_init_params: Optional[np.ndarray]
    ee_link_left: int
    ee_link_right: int
    ee_offset: mn.Vector3
    gripper_joints: List[int]
    arm_joints_right: List[int]
    arm_joints_left: List[int]

@dataclass
class MotionData:
    """
    A class intended to handle precomputations of utilities of the motion we want to
    load into the character.
    """

    # transitive motion is necessary for scenic motion build
    def __init__(
        self,
        motion_: motion.Motion,
    ) -> None:
        logger.info("Loading Motion data...")
        
        ROOT, FIRST, LAST = 0, 0, -1
        # primary
        self.motion = motion_
        self.poses = motion_.poses
        self.fps = motion_.fps
        self.num_of_frames: int = len(motion_.poses)
        self.map_of_total_displacement: float = []
        self.center_of_root_drift: mn.Vector3 = mn.Vector3()
        self.time_length = self.num_of_frames * (1.0 / motion_.fps)

        # intermediates
        self.translation_drifts: List[mn.Vector3] = []
        self.forward_displacements: List[mn.Vector3] = []
        self.root_orientations: List[mn.Quaternion] = []

        # first and last frame root position vectors
        f = motion_.poses[0].get_transform(ROOT, local=False)[0:3, 3]
        f = mn.Vector3(f)
        l = motion_.poses[LAST].get_transform(ROOT, local=False)[0:3, 3]
        l = mn.Vector3(l)

        # axis that motion uses for up and forward
        self.direction_up = mn.Vector3.z_axis()
        forward_V = (l - f) * (mn.Vector3(1.0, 1.0, 1.0) - mn.Vector3.z_axis())
        self.direction_forward = forward_V.normalized()

        ### Fill derived data structures ###
        # fill translation_drifts and forward_displacements
        for i in range(self.num_of_frames):
            j = i + 1
            if j == self.num_of_frames:
                # interpolate forward and drift from nth vectors and 1st vectors and push front
                self.forward_displacements.insert(
                    0,
                    (
                        (
                            self.forward_displacements[LAST]
                            + self.forward_displacements[0]
                        )
                        * 0.5
                    ),
                )
                self.translation_drifts.insert(
                    0,
                    (
                        (
                            self.translation_drifts[LAST]
                            + self.translation_drifts[0]
                        )
                        * 0.5
                    ),
                )
                break

            # root translation
            curr_root_t = motion_.poses[i].get_transform(ROOT, local=False)[
                0:3, 3
            ]
            next_root_t = motion_.poses[j].get_transform(ROOT, local=False)[
                0:3, 3
            ]

            delta_P_vector = mn.Vector3(next_root_t - curr_root_t)
            forward_vector = delta_P_vector.projected(self.direction_forward)
            drift_vector = delta_P_vector - forward_vector

            self.forward_displacements.append(forward_vector)
            self.translation_drifts.append(drift_vector)

        j, summ = 0, 0
        # fill translation_drifts and forward_displacements
        for i in range(self.num_of_frames):
            curr_root_t = motion_.poses[i].get_transform(ROOT, local=False)[
                0:3, 3
            ]
            prev_root_t = motion_.poses[j].get_transform(ROOT, local=False)[
                0:3, 3
            ]

            # fill map_of_total_displacement
            summ += (
                mn.Vector3(curr_root_t - prev_root_t)
                .projected(self.direction_forward)
                .length()
            )
            self.map_of_total_displacement.append(summ)
            j = i

        # fill root_orientations
        for pose in motion_.poses:
            root_T = pose.get_transform(ROOT, local=False)
            root_rotation = mn.Quaternion.from_matrix(
                mn.Matrix3x3(root_T[0:3, 0:3])
            )
            self.root_orientations.append(root_rotation)

        # get center of drift
        summ = mn.Vector3()
        for pose in motion_.poses:
            root_T = mn.Matrix4(pose.get_transform(ROOT, local=False))
            root_T.translation *= (
                mn.Vector3(1.0, 1.0, 1.0) - self.direction_forward
            )
            summ += root_T.translation
        self.center_of_root_drift = summ / self.num_of_frames





class Motions:
    """
    The Motions class is collection of stats that will hold the different movement motions
    for the character to use when following a path. The character is left-footed so that
    is our reference for with step the motions assume first.
    """
    def __init__(self, amass_path, body_model_path):
        self.amass_path = amass_path
        self.body_model_path = body_model_path

        logger.info("Loading Motion data...")
        
        # TODO: add more diversity here
        motion_files = {
            "walk": f"{self.amass_path}/CMU/10/10_04_poses.npz",  # [0] cycle walk
            "run": f"{self.amass_path}/CMU/09/09_01_poses.npz",  # [1] cycle run
        }
        # breakpoint()
        # amass.load("/Users/xavierpuig/Documents/human_sim_data/amass_smpl_h//CMU/10/10_04_poses.npz", bm_path="/Users/xavierpuig/Documents/human_sim_data/smplh/male/model.npz")
        # breakpoint()
        
        motion_data = {key: amass.load(value, bm_path=body_model_path) for key, value in motion_files.items()}
        
        # motion_data = {
        #     "walk": motion_data[0],
        #     "run": motion_data[1]
        # }
        ### TRANSITIVE ###
        # all motions must have same fps for this implementation, so use first motion to set global
        fps = motion_data['walk'].fps

        # Standing pose that must be converted (amass -> habitat joint positions)
        self.standing_pose = motion_data['walk'].poses[0]

        # Walk-to-walk cycle
        self.walk_to_walk = MotionData(
            motion_ops.cut(motion_data['walk'], 300, 430)
        )

        # Run-to-run cycle
        self.run_to_run = MotionData(motion_ops.cut(motion_data['run'], 3, 89)
        )

    def _get_human_params(self):
        return HumanParams(
                gripper_joints=[0,0],
                gripper_init_params=np.array([0.00, 0.00], dtype=np.float32),
                arm_init_params_left=np.array(
                    [-0.45, -1.08, 0.1, 0.935, -0.001, 1.573, 0.005],
                    dtype=np.float32),
                arm_init_params_right=np.array(
                    [-0.45, -1.08, 0.1, 0.935, -0.001, 1.573, 0.005],
                    dtype=np.float32),
                ee_offset=mn.Vector3(),
                ee_link_right=15,
                ee_link_left=9,
                arm_joints_right=[11, 12, 13],
                arm_joints_left=[15, 16, 17],
                
                cameras={
                "robot_arm": RobotCameraParams(
                    cam_offset_pos=mn.Vector3(0, 0.0, 0.1),
                    cam_look_at_pos=mn.Vector3(0.1, 0.0, 0.0),
                    attached_link_id=22,
                    relative_transform=mn.Matrix4.rotation_y(mn.Deg(-90))
                    @ mn.Matrix4.rotation_z(mn.Deg(90)),
                ),
                "robot_head": RobotCameraParams(
                    cam_offset_pos=mn.Vector3(0, 0.5, 0.25),
                    cam_look_at_pos=mn.Vector3(0, 0.5, 0.75),
                    attached_link_id=-1,
                ),
                "robot_third": RobotCameraParams(
                    cam_offset_pos=mn.Vector3(-1.2, 2., -1.2),
                    cam_look_at_pos=mn.Vector3(1, 0.0, 0.75),
                    attached_link_id=-1,
                ),
            }

        )





class AmassHuman(Humanoid):
    def __init__(self, urdf_path, amass_path, body_model_path, sim):
        self.motions = Motions(amass_path, body_model_path)
        self.params = self.motions._get_human_params()
        
        super().__init__(self.params, urdf_path, sim)
        
        self.last_pose = self.motions.standing_pose
        self.urdf_path = urdf_path
        self.amass_path = amass_path
        self.sim = sim
        self.ROOT = 0
        self.arm_joint_pos_left = self.params.arm_init_params_left
        # self.arm_joint_pos = self.params.arm_init_params_left
        self.arm_joint_pos_right = self.params.arm_init_params_right
 
        self.mocap_frame = 0
        self.curr_trans = mn.Vector3([0,0,0])
        self.rotation_offset: Optional[mn.Quaternion] = mn.Quaternion()
        self.translation_offset: Optional[mn.Vector3] = mn.Vector3([2.5, 0.8, 0.7])

        self.prev_orientation = None

        # smoothing_params
        self.frames_to_stop = 10
        self.frames_to_start = 10
        self.draw_fps = 60

        # state variables
        self.time_since_stop = 0 # How many frames since we started stopping
        self.time_since_start = 0 # How many frames since we started walking
        self.fully_started = False 
        self.fully_stopped = False
        self.last_walk_pose = None
        
        
    def close_gripper(self):
        pass

    @property
    def arm_joint_pos(self):
        """Get the current arm joint positions."""

        # deref self vars to cut access in half
        joint_pos_indices = self.joint_pos_indices
        arm_joints = self.params.arm_joints_right
        sim_obj_joint_pos = self.sim_obj.joint_positions

        arm_pos_indices = (joint_pos_indices[x] for x in arm_joints)
        return np.array(
            [sim_obj_joint_pos[i] for i in arm_pos_indices], dtype=np.float32
        )

    @arm_joint_pos.setter
    def arm_joint_pos(self, ctrl: List[float]):
        """Kinematically sets the arm joints and sets the motors to target."""
        # self._validate_arm_ctrl_input(ctrl)
        # breakpoint()
        joint_positions = self.sim_obj.joint_positions

        # breakpoint
        for i, jidx in enumerate(self.params.arm_joints_right):
            # self._set_motor_pos(jidx, ctrl[i])
            joint_positions[self.joint_pos_indices[jidx]] = ctrl[i]
        # self.sim_obj.joint_positions = joint_positions
        # breakpoint()
        self.sim_obj.joint_positions = np.array(ctrl)
        for joint_id in range(len(self.joint_motors)):
            ctrl_quat_id = joint_id*4
            quat = ctrl[ctrl_quat_id:(ctrl_quat_id+4)]
            # breakpoint()
            if joint_id in self.joint_motors:
                self.joint_motors[joint_id][1].spherical_position_target = mn.Quaternion(mn.Vector3(quat[:3]), quat[-1])
                self.sim_obj.update_joint_motor(self.joint_motors[joint_id][0], self.joint_motors[joint_id][1])
            # breakpoint()



    def _update_motor_settings_cache(self):
        """Updates the JointMotorSettings cache for cheaper future updates"""
        self.joint_motors = {}
        # breakpoint()
        for (
            motor_id,
            joint_id,
        ) in self.sim_obj.existing_joint_motor_ids.items():
            self.joint_motors[joint_id] = (
                motor_id,
                self.sim_obj.get_joint_motor_settings(motor_id),
            )

    def reconfigure(self) -> None:
        """Instantiates the human in the scene. Loads the URDF (TODO: complete descr)"""
        #Manipulator.reconfigure(self)
        #RobotBase.reconfigure(self)
        super().reconfigure()
        self.sim_obj.motion_type = phy.MotionType.KINEMATIC
        self.translation_offset = self.sim_obj.translation + mn.Vector3([0,0.90,0])


    def update(self) -> None:
        """Updates the camera transformations
        """
        #Manipulator.update(self)
        #RobotBase.update(self)
        super().update()
        # TODO: Check if this is needed


    def reset(self) -> None:
        """Reset the joints on the human. (Put in rest state)
        """
        super().reset()
        self.last_pose = self.motions.standing_pose

    # TODO: remove the position
    def stop(self, position: mn.Vector3):
        
        new_pose =  self.motions.standing_pose
        new_pose, new_root_trans, root_rotation = self.convert_CMUamass_single_pose(new_pose, self.sim_obj)
       # self.sim_obj.transformation = full_transform
        self.time_since_stop += 1
        
        if self.fully_started: 
            progress = min(self.time_since_stop, self.frames_to_stop)
        else: 
            # if it didn't fully stop it should take us to walk as many 
            # frames as the time we spent stopping
            progress = max(0, self.frames_to_stop - self.time_since_start)

        progress_norm =  progress * 1.0/self.frames_to_stop
        if progress_norm == 1:
            self.fully_stopped = True
        else:
            self.fully_stopped = False

        if self.last_walk_pose is not None:
            interp_pose =  np.array(self.last_walk_pose) * (1-progress_norm) + np.array(new_pose) * progress_norm
            interp_pose = list(interp_pose)
        else:
            interp_pose = new_pose
        if self.time_since_stop >= self.frames_to_stop:
            self.fully_stopped = True
        self.time_since_start = 0

        self.sim_obj.joint_positions = interp_pose
        

        

        

    def walk_path(self, position: List[mn.Vector3]):
        pass

    def walk(self, position: mn.Vector3):
        """ Walks to the desired position. Rotates the character if facing in a different direction """
        step_size = int(self.motions.walk_to_walk.fps / self.draw_fps)
        # breakpoint()
        self.mocap_frame = (self.mocap_frame +  step_size) % self.motions.walk_to_walk.motion.num_frames()
        if self.mocap_frame == 0:
            self.distance_rot = 0
        # curr_pos = self.motions.walk_to_walk[self.mocap_frame]
        new_pose = self.motions.walk_to_walk.poses[self.mocap_frame]
        curr_motion_data = self.motions.walk_to_walk
        new_pose, new_root_trans, root_rotation = self.convert_CMUamass_single_pose(new_pose, self.sim_obj)
                
        char_pos = self.translation_offset

        global_neutral_correction = self.global_correction_quat(
            mn.Vector3.z_axis(), curr_motion_data.direction_forward
        )

        forward_V = position
        
        
        # interpolate facing last margin dist with standing pose
        did_rotate = False
        if self.prev_orientation is not None:
            action_order_facing = self.prev_orientation 
            curr_angle = np.arctan2(forward_V[0], forward_V[2]) * 180./np.pi
            prev_angle = np.arctan2(action_order_facing[0], action_order_facing[2]) * 180./np.pi
            forward_angle = curr_angle - prev_angle
            if np.abs(forward_angle) > 1:                
                # t = forward_angle
                actual_angle_move = 5
                if abs(forward_angle) < actual_angle_move:
                    actual_angle_move = abs(forward_angle)
                new_angle = (prev_angle + actual_angle_move * np.sign(forward_angle)) * np.pi / 180
                did_rotate = True
            else:
                new_angle = curr_angle * np.pi / 180


            forward_V = mn.Vector3(np.sin(new_angle), 0, np.cos(new_angle))



        forward_V[1] = 0.
        forward_V = forward_V.normalized()

        look_at_path_T = mn.Matrix4.look_at(
                    char_pos, char_pos + forward_V.normalized(), mn.Vector3.y_axis()
                )
        full_transform = curr_motion_data.motion.poses[self.mocap_frame].get_transform(
            self.ROOT, local=True
        )
        
        full_transform = mn.Matrix4(full_transform)
        full_transform.translation -= curr_motion_data.center_of_root_drift
        full_transform = (
            mn.Matrix4.from_(
                global_neutral_correction.to_matrix(), mn.Vector3()
            )
            @ full_transform
        )

        # while transform is facing -Z, remove forward displacement
        full_transform.translation *= mn.Vector3.x_axis() + mn.Vector3.y_axis()
        full_transform = look_at_path_T @ full_transform
        
        if self.mocap_frame == 0:
            dist_diff = 0
        else:
            prev_distance = curr_motion_data.map_of_total_displacement[self.mocap_frame - step_size]
            distance_covered = curr_motion_data.map_of_total_displacement[self.mocap_frame];
            dist_diff = max(0, distance_covered - prev_distance)
            if did_rotate:
                dist_diff = 0
            #     self.distance_rot += dist_diff

        self.translation_offset = self.translation_offset + forward_V * dist_diff;
        self.prev_orientation = forward_V
        

        self.time_since_start += 1
        if self.fully_stopped: 
            progress = min(self.time_since_start, self.frames_to_start)
        else: 
            # if it didn't fully stop it should take us to walk as many 
            # frames as the time we spent stopping
            progress = max(0, self.frames_to_start - self.time_since_stop)

        progress_norm = progress * 1.0/self.frames_to_start

        if progress_norm < 1.0:
            standing_pose, _, _ = self.convert_CMUamass_single_pose(self.motions.standing_pose, self.sim_obj)
            # breakpoint()
            interp_pose = (1-progress_norm) * np.array(standing_pose) + progress_norm * np.array(new_pose)
            interp_pose = list(interp_pose)
            self.fully_started = False
        else:
            interp_pose = new_pose
            self.fully_started = True
            # breakpoint()

        if self.time_since_start >= self.frames_to_start:
            self.fully_started = True
        self.time_since_stop = 0
        self.last_walk_pose = new_pose

        self.sim_obj.joint_positions = interp_pose
        self.sim_obj.transformation = full_transform

        

    def global_correction_quat(
        self, up_v: mn.Vector3, forward_v: mn.Vector3
    ) -> mn.Quaternion:
        """
        Given the upward direction and the forward direction of a local space frame, this methd produces
        the correction quaternion to convert the frame to global space (+Y up, -Z forward).
        """
        if up_v.normalized() != mn.Vector3.y_axis():
            angle1 = mn.math.angle(up_v.normalized(), mn.Vector3.y_axis())
            axis1 = mn.math.cross(up_v.normalized(), mn.Vector3.y_axis())
            rotation1 = mn.Quaternion.rotation(angle1, axis1)
            forward_v = rotation1.transform_vector(forward_v)
        else:
            rotation1 = mn.Quaternion()

        forward_v = forward_v * (mn.Vector3(1.0, 1.0, 1.0) - mn.Vector3.y_axis())
        angle2 = mn.math.angle(forward_v.normalized(), -1 * mn.Vector3.z_axis())
        axis2 = mn.Vector3.y_axis()
        rotation2 = mn.Quaternion.rotation(angle2, axis2)

        return rotation2 * rotation1

    def convert_CMUamass_single_pose(
        self, pose, model, raw=False
    ) -> Tuple[List[float], mn.Vector3, mn.Quaternion]:
        """
        This conversion is specific to the datasets from CMU
        """
        new_pose = []

        # Root joint
        root_T = pose.get_transform(self.ROOT, local=False)

        final_rotation_correction = mn.Quaternion()

        if not raw:
            final_rotation_correction = (
                self.global_correction_quat(mn.Vector3.z_axis(), mn.Vector3.x_axis())
                * self.rotation_offset
            )

        root_rotation = final_rotation_correction * mn.Quaternion.from_matrix(
            mn.Matrix3x3(root_T[0:3, 0:3])
        )
        root_translation = (
            self.translation_offset
            + final_rotation_correction.transform_vector(root_T[0:3, 3])
        )

        Q, _ = conversions.T2Qp(root_T)

        # Other joints
        for model_link_id in model.get_link_ids():
            joint_type = model.get_link_joint_type(model_link_id)
            joint_name = model.get_link_name(model_link_id)
            pose_joint_index = pose.skel.index_joint[joint_name]

            # When the target joint do not have dof, we simply ignore it
            if joint_type == phy.JointType.Fixed:
                continue

            # When there is no matching between the given pose and the simulated character,
            # the character just tries to hold its initial pose
            if pose_joint_index is None:
                raise KeyError(
                    "Error: pose data does not have a transform for that joint name"
                )
            if joint_type not in [phy.JointType.Spherical]:
                raise NotImplementedError(
                    f"Error: {joint_type} is not a supported joint type"
                )

            T = pose.get_transform(pose_joint_index, local=True)
            if joint_type == phy.JointType.Spherical:
                Q, _ = conversions.T2Qp(T)

            new_pose += list(Q)
        
        return new_pose, root_translation, root_rotation


    # Probably move somewhere else, maybe make manipualtor an attribute instead?
    def ee_link_id(self, hand=0) -> int:
        """Gets the Habitat Sim link id of the end-effector."""
        if hand == 0:
            return self.params.ee_link_right
        else:
            return self.params.ee_link_left
    
    @property
    def ee_local_offset(self) -> mn.Vector3:
        """Gets the relative offset of the end-effector center from the
        end-effector link.
        """
        return self.params.ee_offset

    def ee_transform(self, hand=0) -> mn.Matrix4:
        """Gets the transformation of the end-effector location. This is offset
        from the end-effector link location.
        """
        # breakpoint()
        ee_link = self.params.ee_link_right
        if hand == 1:
            ee_link = self.params.ee_link_left
        ef_link_transform = self.sim_obj.get_link_scene_node(
            ee_link
        ).transformation
        
        # print(self.sim_obj.get_link_name(
        #     ee_link))
        # breakpoint()
        ef_link_transform.translation = ef_link_transform.transform_point(
            self.ee_local_offset
        )

        return ef_link_transform
    
    def open_gripper(self):
        pass


