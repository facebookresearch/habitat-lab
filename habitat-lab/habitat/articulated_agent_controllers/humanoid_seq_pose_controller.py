#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle as pkl

import magnum as mn
import numpy as np

from habitat.tasks.rearrange.utils import euler_to_quat


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




class SeqPoseController:
    """
    Humanoid Controller, converts high level actions such as walk, or reach into joints positions
        :param walk_pose_path: file containing the walking poses we care about.
        :param draw_fps: the FPS at which we should be advancing the pose.
        :base_offset: what is the offset between the root of the character and their feet.
    """

    def __init__(
        self,
        walk_pose_path='/Users/xavierpuig/Desktop/sample03_rep02.npz',
        draw_fps=30,
        rotate_amount=20,
        base_offset=(0, 0.9, 0),
    ):
        self.base_offset = mn.Vector3(base_offset)

        if not os.path.isfile(walk_pose_path):
            raise RuntimeError(
                f"Path does {walk_pose_path} not exist. Reach out to the paper authors to obtain this data."
            )

        
        walk_info = np.load(walk_pose_path, allow_pickle=True)
        walk_info = walk_info["pose_motion"]
        self.walk_motion = Motion(
            walk_info["joints_array"],
            walk_info["transform_array"],
            walk_info["displacement"],
            walk_info["fps"],
        )
        self.draw_fps = draw_fps
        self.obj_transform_offset = mn.Matrix4()
        self.obj_transform_base = mn.Matrix4()
        self.joint_pose = []

    def reset(self, base_transformation) -> None:
        """Reset the joints on the human. (Put in rest state)"""
        self.obj_transform_offset = mn.Matrix4()
        self.obj_transform_base = base_transformation
        self.prev_orientation = base_transformation.transform_vector(
            mn.Vector3(1.0, 0.0, 0.0)
        )
        self.walk_mocap_frame = 0
        self.get_pose_mdm()

    def calculate_stop_pose(self):
        """
        Calculates a stop, standing pose
        """
        # the object transform does not change
        self.joint_pose = self.stop_pose.joints



    def get_pose_mdm(self):
        # print(self.walk_mocap_frame)
        curr_transform = mn.Matrix4(self.walk_motion.poses[self.walk_mocap_frame].root_transform)
        curr_transform.translation = curr_transform.translation - mn.Vector3(0, 0.5, 0)
        curr_poses = self.walk_motion.poses[self.walk_mocap_frame].joints
        self.obj_transform_offset = curr_transform
        rest_pose = []
        # breakpoint()
        self.joint_pose = curr_poses + rest_pose
        self.walk_mocap_frame = (self.walk_mocap_frame + self.walk_motion.fps) % self.walk_motion.num_poses
        


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
        # breakpoint()
        return self.joint_pose + list(obj_trans_offset) + list(obj_trans_base)
