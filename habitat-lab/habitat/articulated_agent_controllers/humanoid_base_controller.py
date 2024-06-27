#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Union

import magnum as mn
import numpy as np

# The default offset from the root of the character and its feet
BASE_HUMANOID_OFFSET: mn.Vector3 = mn.Vector3(0, 0.9, 0)


# TODO: The implementation here assumes a SMPLX representation of humanoids
# where all joints are represented as quaternions. In future PRs we need
# to abstract this class to support other kinds of joints.
class Pose:
    """
    Represents a single humanoid pose: global root transform + joint state in generalized coordinates (dofs).
    NOTE: assumes a SMPLX representation of humanoids where all joints are represented as quaternions.
    """

    def __init__(
        self,
        joints_quat: Union[List[float], np.ndarray],
        root_transform: mn.Matrix4,
    ) -> None:
        """
        :param joints_quat: list or array of num_joints * 4 elements, with the rotation quaternions
        :param root_transform: Matrix4 with the root transform.
        """

        self.joints = list(joints_quat)
        self.root_transform = root_transform


class Motion:
    """
    Represents a sequential motion, corresponding to a sequence of Poses.
    """

    def __init__(
        self,
        joints_quat_array: np.ndarray,
        transform_array: np.ndarray,
        displacement: mn.Vector3,
        fps: int,
    ) -> None:
        """
        :param joints_quat_array: num_poses x num_joints x 4 array, containing the join orientations
        :param transform_array: num_poses x 4 x 4 array, containing the root transform
        :param displacement: on each pose, how much linear displacement was there in the motion? Used to measure how many poses we should advance to linearly translate a certain distance.
        :param fps: the 'frames per second' at which the motion was recorded
        """

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
    Generic class to replay SMPL-X Motions.
    """

    def __init__(
        self,
        motion_fps: int = 30,
        base_offset: mn.Vector3 = BASE_HUMANOID_OFFSET,
    ):
        """
        :param motion_fps: the 'frames per second' at which we should be playing the motion.
        :param base_offset: what is the offset between the root of the character and their feet.
        """

        self.base_offset = base_offset
        self.motion_fps = motion_fps

        # These two matrices store the global transformation of the base
        # as well as the transformation caused by the walking gait
        # We initialize them to identity
        self.obj_transform_offset = mn.Matrix4()
        self.obj_transform_base = mn.Matrix4()
        self.joint_pose: List[np.ndarray] = []

    def reset(self, base_transformation: mn.Matrix4) -> None:
        """Reset the joints on the human. (Put in rest state)"""
        self.obj_transform_offset = mn.Matrix4()
        self.obj_transform_base = base_transformation
        self.prev_orientation = base_transformation.transform_vector(
            mn.Vector3(1.0, 0.0, 0.0)
        )

    def get_pose(self) -> List[float]:
        """
        Obtains the controller joints, offset and base transform in a vectorized form so that it can be passed
        as an argument to HumanoidJointAction.
        """
        obj_trans_offset = np.asarray(
            self.obj_transform_offset.transposed()
        ).flatten()
        obj_trans_base = np.asarray(
            self.obj_transform_base.transposed()
        ).flatten()
        return self.joint_pose + list(obj_trans_offset) + list(obj_trans_base)
