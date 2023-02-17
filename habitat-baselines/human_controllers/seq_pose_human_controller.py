# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import magnum as mn
import numpy as np
import pybullet as p
from fairmotion.ops import motion as motion_ops

from habitat.utils.fairmotion_utils import AmassHelper, MotionData


class SeqPoseHumanController:
    def __init__(
        self, urdf_path, bm_path, pose_paths, obj_translation=None, draw_fps=60
    ):
        self.urdf_path = urdf_path
        self.curr_trans = mn.Vector3([0, 0, 0.3])
        self.rotation_offset: Optional[mn.Quaternion] = mn.Quaternion()
        self.translation_offset: Optional[mn.Vector3] = mn.Vector3([0, 0, 0])
        # self.obj_transform = self.obtain_root_transform_at_frame(0)

        # pose_path = f'{pose_path}/results.npy'
        file_contents = []
        for pose_path in pose_paths:
            file_contents.append(np.load(pose_path, allow_pickle=True))

        self.pose_data = file_contents
        self.num_vids = len(
            self.pose_data
        )  # self.pose_data['num_samples'] * self.pose_data['num_repetitions']

        # state variables
        self.current_frame = 0
        self.current_video_index = 0
        self.num_frames = self.pose_data[self.current_video_index][
            "trans"
        ].shape[0]
        # self.num_frames = 0 # self.pose_data['lengths'][self.current_video_index]

        if obj_translation is not None:
            self.translation_offset = obj_translation + mn.Vector3(
                [0, 0.90, 0]
            )
        self.pc_id = p.connect(p.DIRECT)
        self.human_bullet_id = p.loadURDF(urdf_path)

        self.link_ids = list(range(p.getNumJoints(self.human_bullet_id)))

        # TODO: There is a mismatch between the indices we get here and ht eones from model.get_link_ids. Would be good to resolve it
        link_indices = [
            0,
            1,
            2,
            6,
            7,
            8,
            14,
            11,
            12,
            13,
            9,
            10,
            15,
            16,
            17,
            18,
            3,
            4,
            5,
        ]

        # Joint INFO
        # https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstart_guide/PyBulletQuickstartGuide.md.html#getjointinfo

        self.joint_info = [
            p.getJointInfo(self.human_bullet_id, index)
            for index in link_indices
        ]

        self.amass_helper = AmassHelper(bm_path)

    def reset(self, position) -> None:
        """Reset the joints on the human. (Put in rest state)"""
        self.translation_offset = position

    def next_video(self):
        self.current_video_index += 1
        self.current_video_index = (
            self.current_video_index + self.num_vids
        ) % self.num_vids
        self.current_frame = 0
        self.num_frames = self.pose_data[self.current_video_index][
            "trans"
        ].shape[0]

    def prev_video(self):
        self.current_video_index -= 1
        self.current_video_index = (
            self.current_video_index + self.num_vids
        ) % self.num_vids
        self.current_frame = 0
        self.num_frames = self.pose_data[self.current_video_index][
            "trans"
        ].shape[0]

    def next_frame(self):
        self.current_frame += 1
        self.current_frame = (
            self.current_frame + self.num_frames
        ) % self.num_frames

    def prev_frame(self):
        self.current_frame -= 1
        self.current_frame = (
            self.current_frame + self.num_frames
        ) % self.num_frames

    def pose_per_frame(self):
        # 22 x 3 pose
        # breakpoint()
        # current_pose = self.pose_data['motion'][self.current_video_index, :, :, self.current_frame]
        root_trans = (
            self.pose_data[self.current_video_index]["trans"]
            + self.translation_offset
        )
        root_orient = self.pose_data[self.current_video_index]["root_orient"]
        poses = self.pose_data[self.current_video_index]["pose_body"]
        self.num_frames = self.pose_data[self.current_video_index][
            "trans"
        ].shape[0]
        pose_info = {
            "pose": poses,
            "root_orient": root_orient,
            "trans": root_trans,
        }
        pose = MotionData.obtain_pose(
            self.amass_helper.skeleton, pose_info, self.current_frame
        )
        pose, root_trans, root_rot = AmassHelper.convert_CMUamass_single_pose(
            pose, self.joint_info, raw=False
        )
        full_transform = mn.Matrix4.from_(root_rot.to_matrix(), root_trans)
        return pose, full_transform

    @classmethod
    def transformAction(cls, pose: List, transform: mn.Matrix4):
        return pose + list(np.asarray(transform.transposed()).flatten())

    def open_gripper(self):
        pass
