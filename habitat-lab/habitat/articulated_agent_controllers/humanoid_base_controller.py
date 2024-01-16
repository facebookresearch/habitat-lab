#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import magnum as mn
import numpy as np


# TODO: The implementation here assumes a SMPLX representation of humanoids
# where all joints are represented as quaternions. In future PRs we need
# to abstract this class to support other kinds of joints.
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

            Used to measure how many poses we should advance to move a certain amount
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


class HumanoidBaseController:
    """
    Generic class to replay SMPL-X motions

        :param motion_fps: the FPS at which we should be playing the motion.
        :param base_offset: what is the offset between the root of the character and their feet.
    """

    def __init__(
        self,
        motion_fps=30,
        base_offset=(0, 0.9, 0),
    ):
        self.base_offset = mn.Vector3(base_offset)
        self.motion_fps = motion_fps

        # These two matrices store the global transformation of the base
        # as well as the transformation caused by the walking gait
        # We initialize them to identity
        self.obj_transform_offset = mn.Matrix4()
        self.obj_transform_base = mn.Matrix4()
        self.joint_pose = []

    def reset(self, base_transformation: mn.Matrix4) -> None:
        """Reset the joints on the human. (Put in rest state)"""
        self.obj_transform_offset = mn.Matrix4()
        self.obj_transform_base = base_transformation
        self.prev_orientation = base_transformation.transform_vector(
            mn.Vector3(1.0, 0.0, 0.0)
        )

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
