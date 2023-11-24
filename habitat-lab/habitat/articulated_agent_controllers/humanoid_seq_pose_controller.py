#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import magnum as mn
import numpy as np
from humanoid_base_controller import HumanoidBaseController, Motion


class HumanoidSeqPoseController(HumanoidBaseController):
    """
    Humanoid Seq Pose Controller, replays a sequence of humanoid poses.
        :param walk_pose_path: file containing the walking poses we care about.
        :param draw_fps: the FPS at which we should be advancing the pose.
        :base_offset: what is the offset between the root of the character and their feet.
    """

    def __init__(
        self,
        walk_pose_path="/Users/xavierpuig/Desktop/sample03_rep02.npz",
        draw_fps=30,
        base_offset=(0, 0.9, 0),
    ):
        super.__init__(draw_fps, base_offset)

        if not os.path.isfile(walk_pose_path):
            raise RuntimeError(
                f"Path does {walk_pose_path} not exist. Reach out to the paper authors to obtain this data."
            )

        motion_info = np.load(walk_pose_path, allow_pickle=True)
        motion_info = motion_info["pose_motion"]
        self.humanoid_motion = Motion(
            motion_info["joints_array"],
            motion_info["transform_array"],
            motion_info["displacement"],
            motion_info["fps"],
        )

    def reset(self, base_transformation) -> None:
        """Reset the joints on the human. (Put in rest state)"""
        super().reset(base_transformation)
        self.walk_mocap_frame = 0
        self.get_pose_mdm()

    def next_pose(self):
        self.walk_mocap_frame = min(
            self.walk_mocap_frame + 1, self.humanoid_motion.num_poses
        )

    def prev_pose(self):
        self.walk_mocap_frame = max(0, self.walk_mocap_frame - 1)

    def get_pose_mdm(self):
        curr_transform = mn.Matrix4(
            self.humanoid_motion.poses[self.walk_mocap_frame].root_transform
        )
        curr_transform.translation = curr_transform.translation - mn.Vector3(
            0, 0.5, 0
        )
        curr_poses = self.walk_motion.poses[self.walk_mocap_frame].joints
        self.obj_transform_offset = curr_transform
        self.joint_pose = curr_poses
        self.walk_mocap_frame = (
            self.walk_mocap_frame + self.walk_motion.fps
        ) % self.walk_motion.num_poses
