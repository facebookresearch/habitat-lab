#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle as pkl
from typing import List

import magnum as mn
import numpy as np


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


MIN_ANGLE_TURN = 5  # If we turn less than this amount, we can just rotate the base and keep walking motion the same as if we had not rotated
TURNING_STEP_AMOUNT = (
    20  # The maximum angle we should be rotating at a given step
)
THRESHOLD_ROTATE_NOT_MOVE = 120  # The rotation angle above which we should only walk as if rotating in place


class HumanoidRearrangeController:
    """
    Humanoid Controller, converts high level actions such as walk, or reach into joints positions
        :param walk_pose_path: file containing the walking poses we care about.
        :param draw_fps: the FPS at which we should be advancing the pose.
        :base_offset: what is the offset between the root of the character and their feet.
    """

    def __init__(
        self,
        walk_pose_path,
        draw_fps=30,
        base_offset=(0, 0.9, 0),
    ):
        self.min_angle_turn = MIN_ANGLE_TURN
        self.turning_step_amount = TURNING_STEP_AMOUNT
        self.threshold_rotate_not_move = THRESHOLD_ROTATE_NOT_MOVE
        self.base_offset = mn.Vector3(base_offset)

        if not os.path.isfile(walk_pose_path):
            raise RuntimeError(
                f"Path does {walk_pose_path} not exist. Reach out to the paper authors to obtain this data."
            )

        with open(walk_pose_path, "rb") as f:
            walk_data = pkl.load(f)
        walk_info = walk_data["walk_motion"]

        self.walk_motion = Motion(
            walk_info["joints_array"],
            walk_info["transform_array2"],
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

        # State variables
        self.obj_transform = mn.Matrix4()
        self.prev_orientation = None
        self.walk_mocap_frame = 0

    def reset(self, position) -> None:
        """Reset the joints on the human. (Put in rest state)"""
        self.obj_transform.translation = position + self.base_offset

    def get_stop_pose(self):
        """
        Returns a stop, standing pose
        """
        joint_pose = self.stop_pose.joints
        obj_transform = (
            self.obj_transform
        )  # the object transform does not change
        return joint_pose, obj_transform

    def compute_turn(self, target_position: mn.Vector3):
        """
        Generate some motion without base transform, just turn
        """
        return self.get_walk_pose(target_position, distance_multiplier=0)

    def get_walk_pose(
        self, target_position: mn.Vector3, distance_multiplier=0
    ):
        """
        Computes a walking pose and transform, so that the humanoid moves to the relative position

        :param position: target position, relative to the character root translation
        :param distance_multiplier: allows to create walk motion while not translating, good for turning
        """

        forward_V = target_position
        if forward_V.length() == 0.0:
            return self.get_stop_pose()
        distance_to_walk = np.linalg.norm(forward_V)
        did_rotate = False

        if self.prev_orientation is not None:
            # If prev orrientation is None, transition to this position directly
            curr_angle = np.arctan2(forward_V[0], forward_V[2]) * 180.0 / np.pi
            prev_orientation = self.prev_orientation
            prev_angle = (
                np.arctan2(prev_orientation[0], prev_orientation[2])
                * 180.0
                / np.pi
            )
            forward_angle = curr_angle - prev_angle
            if np.abs(forward_angle) > self.min_angle_turn:
                actual_angle_move = self.turning_step_amount
                if abs(forward_angle) < actual_angle_move:
                    actual_angle_move = abs(forward_angle)
                new_angle = prev_angle + actual_angle_move * np.sign(
                    forward_angle
                )
                new_angle *= np.pi / 180
                did_rotate = True
            else:
                new_angle = curr_angle * np.pi / 180

            forward_V = mn.Vector3(np.sin(new_angle), 0, np.cos(new_angle))

        forward_V = mn.Vector3(forward_V)
        forward_V = forward_V.normalized()
        self.prev_orientation = forward_V

        # Step size according to the FPS
        step_size = int(self.walk_motion.fps / self.draw_fps)

        if did_rotate:
            # When we rotate, we allow some movement
            distance_to_walk = self.dist_per_step_size * 2
            if np.abs(forward_angle) >= self.threshold_rotate_not_move:
                distance_to_walk *= 0

        # Step size according to how much we moved, this is so that
        # we don't overshoot if the speed of the character would it make
        # it move further than what `position` indicates
        step_size = max(
            1, min(step_size, int(distance_to_walk / self.dist_per_step_size))
        )

        if distance_multiplier == 0.0:
            step_size = 0

        # Advance mocap frame
        prev_mocap_frame = self.walk_mocap_frame
        self.walk_mocap_frame = (
            self.walk_mocap_frame + step_size
        ) % self.walk_motion.num_poses

        # Compute how much distance we covered in this motion
        prev_cum_distance_covered = self.walk_motion.displacement[
            prev_mocap_frame
        ]
        new_cum_distance_covered = self.walk_motion.displacement[
            self.walk_mocap_frame
        ]

        offset = 0
        if self.walk_mocap_frame < prev_mocap_frame:
            # We looped over the motion
            offset = self.walk_motion.displacement[-1]

        distance_covered = max(
            0, new_cum_distance_covered + offset - prev_cum_distance_covered
        )
        dist_diff = min(distance_to_walk, distance_covered)

        new_pose = self.walk_motion.poses[self.walk_mocap_frame]
        joint_pose, obj_transform = new_pose.joints, new_pose.root_transform

        # We correct the object transform
        obj_transform.translation *= 0
        look_at_path_T = mn.Matrix4.look_at(
            self.obj_transform.translation,
            self.obj_transform.translation + forward_V.normalized(),
            mn.Vector3.y_axis(),
        )

        # Remove the forward component, and orient according to forward_V
        obj_transform.translation *= mn.Vector3.x_axis() + mn.Vector3.y_axis()
        obj_transform = look_at_path_T @ obj_transform

        obj_transform.translation += (
            forward_V * dist_diff * distance_multiplier
        )
        self.obj_transform = obj_transform

        return joint_pose, obj_transform

    @classmethod
    def vectorize_pose(cls, pose: List, transform: mn.Matrix4):
        """
        Transforms a pose so that it can be passed as an argument to HumanoidJointAction

        :param pose: a list of 17*4 elements, corresponding to the flattened quaternions
        :param transform: an object transform
        """
        return pose + list(np.asarray(transform.transposed()).flatten())
