#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle as pkl

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
THRESHOLD_ROTATE_NOT_MOVE = 20  # The rotation angle above which we should only walk as if rotating in place
EPS = 1e-5  # Distance at which we should stop

# The frames per second we run the motion at, in relation to the FPS at which the motion was recorded.
# If the motion was recorded at n * DEFAULT_DRAW_FPS, we will be advancing n frames on every env step
DEFAULT_DRAW_FPS = 30


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
        base_offset=(0, 0.9, 0),
    ):
        self.draw_fps = DEFAULT_DRAW_FPS
        self.min_angle_turn = MIN_ANGLE_TURN
        self.turning_step_amount = TURNING_STEP_AMOUNT
        self.threshold_rotate_not_move = TURNING_STEP_AMOUNT
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
            walk_info["transform_array"],
            walk_info["displacement"],
            walk_info["fps"],
        )

        self.stop_pose = Pose(
            walk_data["stop_pose"]["joints"].reshape(-1),
            mn.Matrix4(walk_data["stop_pose"]["transform"]),
        )
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
        self.walk_mocap_frame = 0

    def set_framerate_for_linspeed(self, lin_speed, ang_speed, ctrl_freq):
        seconds_per_step = 1.0 / ctrl_freq
        meters_per_step = lin_speed * seconds_per_step
        frames_per_step = meters_per_step / self.dist_per_step_size
        self.draw_fps = self.walk_motion.fps / frames_per_step
        rotate_amount = ang_speed
        self.turning_step_amount = rotate_amount
        self.threshold_rotate_not_move = rotate_amount

    def reset(self, base_transformation) -> None:
        """Reset the joints on the human. (Put in rest state)"""
        self.obj_transform_offset = mn.Matrix4()
        self.obj_transform_base = base_transformation
        self.prev_orientation = base_transformation.transform_vector(
            mn.Vector3(1.0, 0.0, 0.0)
        )

    def calculate_stop_pose(self):
        """
        Calculates a stop, standing pose
        """
        # the object transform does not change
        self.joint_pose = self.stop_pose.joints

    def calculate_turn_pose(self, target_position: mn.Vector3):
        """
        Generate some motion without base transform, just turn
        """
        self.calculate_walk_pose(target_position, distance_multiplier=0)

    def calculate_walk_pose(
        self, target_position: mn.Vector3, distance_multiplier=1.0
    ):
        """
        Computes a walking pose and transform, so that the humanoid moves to the relative position

        :param position: target position, relative to the character root translation
        :param distance_multiplier: allows to create walk motion while not translating, good for turning
        """
        deg_per_rads = 180.0 / np.pi
        forward_V = target_position
        if forward_V.length() < EPS:
            self.calculate_stop_pose()
            return
        distance_to_walk = np.linalg.norm(forward_V)
        did_rotate = False

        # The angle we initially want to go to
        new_angle = np.arctan2(forward_V[2], forward_V[0]) * deg_per_rads
        if self.prev_orientation is not None:
            # If prev orrientation is None, transition to this position directly
            prev_orientation = self.prev_orientation
            prev_angle = (
                np.arctan2(prev_orientation[2], prev_orientation[0])
                * deg_per_rads
            )
            forward_angle = new_angle - prev_angle
            if forward_angle >= 180:
                forward_angle = forward_angle - 360
            if forward_angle <= -180:
                forward_angle = 360 + forward_angle

            if np.abs(forward_angle) > self.min_angle_turn:
                actual_angle_move = self.turning_step_amount
                if abs(forward_angle) < actual_angle_move:
                    actual_angle_move = abs(forward_angle)
                new_angle = prev_angle + actual_angle_move * np.sign(
                    forward_angle
                )
                new_angle /= deg_per_rads
                did_rotate = True
            else:
                new_angle = new_angle / deg_per_rads
            forward_V = mn.Vector3(np.cos(new_angle), 0, np.sin(new_angle))

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
        forward_V_norm = mn.Vector3(
            [forward_V[2], forward_V[1], -forward_V[0]]
        )
        look_at_path_T = mn.Matrix4.look_at(
            self.obj_transform_base.translation,
            self.obj_transform_base.translation + forward_V_norm.normalized(),
            mn.Vector3.y_axis(),
        )

        # Remove the forward component, and orient according to forward_V
        add_rot = mn.Matrix4.rotation(mn.Rad(np.pi), mn.Vector3(0, 1.0, 0))
        obj_transform = add_rot @ obj_transform
        obj_transform.translation *= mn.Vector3.x_axis() + mn.Vector3.y_axis()

        # This is the rotation and translation caused by the current motion pose
        #  we still need to apply the base_transform to obtain the full transform
        self.obj_transform_offset = obj_transform

        # The base_transform here is independent of transforms caused by the current
        # motion pose.
        obj_transform_base = look_at_path_T
        forward_V_dist = forward_V * dist_diff * distance_multiplier
        obj_transform_base.translation += forward_V_dist

        self.obj_transform_base = obj_transform_base
        self.joint_pose = joint_pose

    def get_pose(self):
        """
        Obtains the controller joints, offset and base transform in a vectorized form so that it can be passed
        as an argument to HumanoidJointAction
        """
        obj_trans_offset = np.asarray(
            self.obj_transform_offset.transposed()
        ).flatten()
        obj_trans_base = np.asarray(
            self.obj_transform_base.transposed()
        ).flatten()
        return self.joint_pose + list(obj_trans_offset) + list(obj_trans_base)
