# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from os import path as osp
from typing import List, Optional

import pickle as pkl
import magnum as mn
import numpy as np
import pybullet as p
from fairmotion.ops import motion as motion_ops

from habitat.utils.fairmotion_utils import AmassHelper, MotionData
class Pose:
    def __init__(self, joints_quat, root_transform):
        """
        Contains a single humanoid pose
            :param joints_quat: list or array of num_joints * 4 elements, with the rotation quaternions
            :param root_transform: Matrix4 with the root trasnform.
        """
        self.joints = list(joints_quat)
        self.root_transform = root_transform


class Motion:
    """
    Contains a sequential motion, corresponding to a sequence of poses
        :param joints_quat_array: num_poses x num_joints x 4 array, containing the join orientations
        :param transform_array: num_poses x 4 x 4 array, containing the root transform
        :param displacement: on each pose, how much forward displacement was there?
            Used to measure how many poses we should advance to move a cerain amount
        :param fps: the FPS at which the motion was recorded
    """

    def __init__(self, joints_quat_array, transform_array, displacement, fps):
        num_poses = joints_quat_array.shape[0]
        self.num_poses = num_poses
        poses = []
        for index in range(num_poses):
            pose = Pose(
                joints_quat_array[index].reshape(-1),
                mn.Matrix4(transform_array[index]),
            )
            poses.append(pose)

        self.poses = poses
        self.fps = fps
        self.displacement = displacement
        self.map_of_total_displacement = displacement




class AmassHumanController:
    """
    Human Controller, converts high level actions such as walk, or reach into joints that control
    to control a URDF object.
    """

    def __init__(
        self,
        urdf_path,
        amass_path,
        body_model_path,
        grab_path=None,
        obj_translation=None,
        draw_fps=60,
    ):
        self.base_offset = mn.Vector3(0, 0.9, 0)
        self.mocap_frame = 0
        walk_pose_path= "/Users/xavierpuig/Documents/Projects/humans_habitat_3/integrate_hl/test_create_pose/walking_motion_processed_smplx.pkl"
        # walk_pose_path= "/Users/xavierpuig/Documents/Projects/humans_habitat_3/integrate_hl/test_create_pose/walking_motion_processed_smplh.pkl"
        
        with open(walk_pose_path, "rb") as f:
            walk_data = pkl.load(f)
        walk_info = walk_data["walk_motion"]

        self.walk_motion = Motion(
            walk_info["joints_array"],
            walk_info["transform_array"],
            walk_info["displacement"],
            walk_info["fps"],
        )

        self.stop_pose = Pose(
            walk_data["stop_pose"]["joints"].reshape(-1),
            mn.Matrix4(walk_data["stop_pose"]["transform"]),
        )
        self.draw_fps = draw_fps
        self.dist_per_step_size = (
            self.walk_motion.displacement[-1] / self.walk_motion.num_poses
        )

        # These two matrices store the global transformation of the base
        # as well as the transformation caused by the walking gait
        # We initialize them to identity
        self.obj_transform_offset = mn.Matrix4()
        self.obj_transform_base = mn.Matrix4()
        self.joint_pose = []

        self.prev_orientation = None
        self.mocap_frame = 0
        self.obj_transform = mn.Matrix4()

    @property
    def root_pos(self):
        """
        Position of the root (pelvis)
        """
        return self.obj_transform.translation

    @property
    def root_rot(self):
        """
        Position of the root (pelvis)
        """
        return self.obj_transform.rotation()

    @property
    def base_pos(self):
        """
        Position of the base, which is the pelvis projected to the floor
        """
        return self.obj_transform.translation - self.base_offset

    @base_pos.setter
    def base_pos(self, position: mn.Vector3):
        # if self.mocap_frame == 108:
        #     breakpoint()
        self.obj_transform.translation = position + self.base_offset

    def reset(self, position) -> None:
        """Reset the joints on the human. (Put in rest state)"""
        self.base_pos = position
        self.last_pose = self.stop_pose

    def stop(self, progress=None):
        """Sets the human into a standing pose. If progress != None, it interpolates between the previous and stopping pose"""
        new_pose = self.stop_pose


        return new_pose.joints, self.obj_transform


    @property
    def step_distance(self):
        step_size = int(self.walk_motion.fps / self.draw_fps)
        curr_motion_data = self.walk_motion

        prev_distance = curr_motion_data.map_of_total_displacement[
            self.mocap_frame
        ]
        new_pos = self.mocap_frame + step_size
        if new_pos < len(curr_motion_data.map_of_total_displacement):
            distance_covered = curr_motion_data.map_of_total_displacement[
                new_pos
            ]
        else:
            pos_norm = new_pos % len(
                curr_motion_data.map_of_total_displacement
            )
            distance_covered = curr_motion_data.map_of_total_displacement[-1]
            distance_covered += max(
                0,
                (step_size // len(curr_motion_data.map_of_total_displacement))
                - 1,
            )
            distance_covered += curr_motion_data.map_of_total_displacement[
                pos_norm
            ]
        return distance_covered - prev_distance

    def _select_index(self, position: mn.Vector3):
        """
        Given a matrix indexed with 3D coordinates, finds the index of the matrix
        whose key is closest to position. The matrix is indexed such that index i stores coordinate
        (x_{min} + (x_{max}-x_{min}) * xb, y_{min} + (y_{max}-y_{min})*yb, z_{min} + (z_{max}-z_{min}) * zb)
        with yb = N // (bins_x * bins_z), xb = (N // bins_z) % bin_x, zb = N % (bins_x * bins_z) 
        """
        def find_index_quant(minv, maxv, num_bins, value):
            # Find the quantization bin
            value = max(min(value, maxv), minv)
            value_norm = (value - minv) / (maxv - minv)
            # TODO: make sure that this is not round
            index = int(value_norm * num_bins)
            return min(index, num_bins - 1)

        relative_pos = position
        x_diff, y_diff, z_diff = relative_pos.x, relative_pos.y, relative_pos.z

        coord_data = [
            (
                self.vpose["min"][0],
                self.vpose["max"][0],
                self.vpose["bins"][0],
                x_diff,
            ),
            (
                self.vpose["min"][1],
                self.vpose["max"][1],
                self.vpose["bins"][1],
                y_diff,
            ),
            (
                self.vpose["min"][2],
                self.vpose["max"][2],
                self.vpose["bins"][2],
                z_diff,
            ),
        ]
        x_ind, y_ind, z_ind = [find_index_quant(*data) for data in coord_data]
        index = (
            y_ind * self.vpose["bins"][0] * self.vpose["bins"][2]
            + x_ind * self.vpose["bins"][2]
            + z_ind
        )

        return index

    def reach(self, position: mn.Vector3):
        """ Set hand pose to reach a certain position, defined relative to the root """


        if not self.use_ik_grab:
            raise KeyError(
                "Error: reach behavior is not defined when use_ik_grab is off"
            )
        reach_pos = self._select_index(position)
        curr_pose = list(self.grab_quaternions[reach_pos])
        curr_transform = mn.Matrix4(
            self.grab_transform[reach_pos].reshape(4, 4)
        )
        curr_transform.translation = self.obj_transform.translation

        return curr_pose, curr_transform

    def walk(self, position: mn.Vector3):
        """ Walks to the desired position. Rotates the character if facing in a different direction """
        step_size = int(self.walk_motion.fps / self.draw_fps)

        forward_V = position

        forward_V[1] = 0.0
        distance_to_walk = np.linalg.norm(forward_V)

        # interpolate facing last margin dist with standing pose
        did_rotate = False
        if self.prev_orientation is not None:
            action_order_facing = self.prev_orientation
            curr_angle = np.arctan2(forward_V[0], forward_V[2]) * 180.0 / np.pi
            prev_angle = (
                np.arctan2(action_order_facing[0], action_order_facing[2])
                * 180.0
                / np.pi
            )

            forward_angle = curr_angle - prev_angle
            if np.abs(forward_angle) >= 1:
                actual_angle_move = 20
                if abs(forward_angle) < actual_angle_move:
                    actual_angle_move = abs(forward_angle)
                new_angle = prev_angle + actual_angle_move * np.sign(
                    forward_angle
                )
                new_angle *= np.pi / 180
                did_rotate = True
            else:
                new_angle = curr_angle * np.pi / 180
                forward_V2 = mn.Vector3(
                    np.sin(new_angle), 0, np.cos(new_angle)
                )

            forward_V = mn.Vector3(np.sin(new_angle), 0, np.cos(new_angle))
        forward_V = mn.Vector3(forward_V)
        forward_V = forward_V.normalized()

        if did_rotate:
            # print(self.base_pos, forward_V, position)
            distance_to_walk = self.dist_per_step_size * 2
            if np.abs(forward_angle) > 120:
                distance_to_walk *= 0

        step_size = max(
            1, min(step_size, int(distance_to_walk / self.dist_per_step_size))
        )
        self.mocap_frame = (
            self.mocap_frame + step_size
        ) % self.walk_motion.num_poses
        if self.mocap_frame == 0:
            self.distance_rot = 0

        
        new_pose = self.walk_motion.poses[self.mocap_frame]
        # print(new_pose.joints[:8])
        # breakpoint()
        joint_pose, obj_transform = new_pose.joints, new_pose.root_transform

        # How much distance we should have covered in the last step
        prev_distance = self.walk_motion.displacement[
            self.mocap_frame - step_size
        ]
        if (self.mocap_frame - step_size) < 0:
            distance_covered = (
                self.walk_motion.displacement[self.mocap_frame]
                + self.walk_motion.displacement[-1]
            )
        else:
            distance_covered = self.walk_motion.displacement[self.mocap_frame]

        dist_diff = min(
            distance_to_walk, max(0, distance_covered - prev_distance)
        )

        self.prev_orientation = forward_V

        
        full_transform = obj_transform
        full_transform.translation *= 0
        look_at_path_T = mn.Matrix4.look_at(
            self.obj_transform.translation,
            self.obj_transform.translation + forward_V.normalized(),
            mn.Vector3.y_axis(),
        )

        # Remove the forward component, we will set it later ouselves
        full_transform.translation *= mn.Vector3.x_axis() + mn.Vector3.y_axis()
        full_transform = look_at_path_T @ full_transform

        full_transform.translation += forward_V * dist_diff

        # self.time_since_start += 1
        # if self.fully_stopped:
        #     progress = min(self.time_since_start, self.frames_to_start)
        # else:
        #     # if it didn't fully stop it should take us to walk as many
        #     # frames as the time we spent stopping
        #     progress = max(0, self.frames_to_start - self.time_since_stop)

        # Ensure a smooth transition from walking to standing
        # progress_norm = progress * 1.0 / self.frames_to_start
        
        interp_pose = new_pose.joints
        self.fully_started = True

        # if self.time_since_start >= self.frames_to_start:
        #     self.fully_started = True
        self.time_since_stop = 0
        self.last_walk_pose = new_pose

        # breakpoint()

        self.obj_transform = full_transform
        print(full_transform)
        print(interp_pose[:5])

        return interp_pose, full_transform

    def compute_turn(self, rel_pos):
        """Turns a certain angle towards a direction"""
        assert self.prev_orientation is not None
        step_size = int(self.walk_motion.fps / self.draw_fps)
        # breakpoint()
        self.mocap_frame = (
            self.mocap_frame + step_size
        ) % self.walk_motion.num_poses
        if self.mocap_frame == 0:
            self.distance_rot = 0
        # curr_pos = self.motions.walk_to_walk[self.mocap_frame]
        new_pose = self.motions.walk_to_walk.poses[self.mocap_frame]
        # curr_motion_data = self.motions.walk_to_walk

        char_pos = self.base_pos

        forward_V = np.array([rel_pos[0], 0, rel_pos[1]])
        desired_forward = forward_V
        # interpolate facing last margin dist with standing pose
        action_order_facing = self.prev_orientation

        curr_angle = np.arctan2(forward_V[0], forward_V[2]) * 180.0 / np.pi
        prev_angle = (
            np.arctan2(action_order_facing[0], action_order_facing[2])
            * 180.0
            / np.pi
        )
        forward_angle = curr_angle - prev_angle

        # print("angle", forward_angle)
        # breakpoint()

        actual_angle_move = 5
        if abs(forward_angle) < actual_angle_move:
            actual_angle_move = abs(forward_angle)
        new_angle = prev_angle + actual_angle_move * np.sign(forward_angle)
        # breakpoint()
        new_angle = new_angle * np.pi / 180.0
        forward_V = mn.Vector3(np.sin(new_angle), 0, np.cos(new_angle))

        forward_V[1] = 0.0
        forward_V = mn.Vector3(forward_V)
        forward_V = forward_V.normalized()
        # breakpoint()
        look_at_path_T = mn.Matrix4.look_at(
            char_pos, char_pos + forward_V.normalized(), mn.Vector3.y_axis()
        )

        full_transform = new_pose.transform
        # while transform is facing -Z, remove forward displacement
        full_transform.translation *= 0
        full_transform.translation *= mn.Vector3.x_axis() + mn.Vector3.y_axis()

        full_transform = look_at_path_T @ full_transform

        self.prev_orientation = forward_V

        interp_pose = new_pose.joints
        full_transform.translation = self.obj_transform.translation
        self.obj_transform = full_transform
        
        return interp_pose, full_transform

    @classmethod
    def transformAction(cls, pose: List, transform: mn.Matrix4):
        return pose + list(np.asarray(transform.transposed()).flatten())

    def open_gripper(self):
        pass
