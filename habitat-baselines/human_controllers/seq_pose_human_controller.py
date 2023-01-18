# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

import magnum as mn
import numpy as np
import os
from os import path as osp

from fairmotion.ops import motion as motion_ops

from habitat.utils.fairmotion_utils import (
    MotionData,
    AmassHelper
)

import pybullet as p





class SeqPoseHumanController:
    def __init__(self, urdf_path, bm_path, pose_path, obj_translation=None, draw_fps=60):
        self.urdf_path = urdf_path
        self.curr_trans = mn.Vector3([0,0,0.3])
        self.rotation_offset: Optional[mn.Quaternion] = mn.Quaternion()
        self.translation_offset: Optional[mn.Vector3] = mn.Vector3([0, 0, 0])
        self.obj_transform = self.obtain_root_transform_at_frame(0)

        pose_path = f'{}/results.npy'
        file_content = np.load(pose_path, allow_pickle=True)
        self.pose_data = file_content[None][0]
        self.num_vids = self.pose_data['num_samples'] * self.pose_data['num_repetitions'] 

        # state variables
        self.current_frame = 0
        self.current_video_index = 0
        self.num_frames = self.pose_data['lengths'][self.current_video_index]

        if obj_translation is not None:
            self.translation_offset = obj_translation + mn.Vector3([0,0.90,0])
        self.pc_id = p.connect(p.DIRECT)
        self.human_bullet_id = p.loadURDF(urdf_path)

        self.link_ids = list(range(p.getNumJoints(self.human_bullet_id)))

        # TODO: There is a mismatch between the indices we get here and ht eones from model.get_link_ids. Would be good to resolve it
        link_indices = [0, 1, 2, 6, 7, 8, 14, 11, 12, 13, 9, 10, 15, 16, 17, 18, 3, 4, 5]

        # Joint INFO
        # https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstart_guide/PyBulletQuickstartGuide.md.html#getjointinfo

        self.joint_info = [p.getJointInfo(self.human_bullet_id, index) for index in link_indices]

        self.amass_helper = AmassHelper(bm_path)
        

    def reset(self, position) -> None:
        """Reset the joints on the human. (Put in rest state)
        """
        self.translation_offset = position
        
    def next_video(self):
        self.current_video_index += 1
        self.current_video_index = (self.current_video_index + self.num_vids) % self.num_vids
        self.current_frame = 0
        self.num_frames = self.pose_data['lengths'][self.current_video_index]

    def prev_video(self):
        self.current_video_index -= 1
        self.current_video_index = (self.current_video_index + self.num_vids) % self.num_vids
        self.current_frame = 0
        self.num_frames = self.pose_data['lengths'][self.current_video_index]

    def next_pose(self):
        self.current_frame += 1
        self.current_frame = (self.current_frame + self.num_frames) % self.num_frames

    def prev_pose(self):
        self.current_frame -= 1
        self.current_frame = (self.current_frame + self.num_frames) % self.num_frames

    def pose_per_frame(self):
        # 22 x 3 pose
        current_pose = self.pose_data['motion'][self.current_video_index, :, :, self.current_frame]
        root_trans = current_pose[self.current_video_index, -1, :, self.current_frame]
         # check why this is the case
        # TODO: probably we can stop relying on this POSE structure
        pose_info = {
            'pose': current_pose[1:, :][None, :]
            'root_orient': current_pose[:1, :][None, :]
            'root_trans': root_trans[None, :]
        }
        pose = MotionData.obtain_pose(self.amass_helper.skeleton, pose_info, 0)
        pose, root_trans, root_rot = AmassHelper.convert_CMUamass_single_pose(pose, self.joint_info, raw=True)
        full_transform = mn.Matrix4.from_(root_rot.to_matrix(), root_trans)
        return new_pose, full_transform


    @classmethod
    def transformAction(cls, pose: List, transform: mn.Matrix4):
        return pose + list(np.asarray(transform.transposed()).flatten())


    def open_gripper(self):
        pass
