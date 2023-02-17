# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import List, Tuple

import magnum as mn
import numpy as np
import pybullet as p
import torch
from fairmotion.core import motion
from fairmotion.data import amass
from fairmotion.ops import conversions
from fairmotion.ops import motion as motion_ops
from human_body_prior.body_model.body_model import BodyModel


@dataclass
class MotionData:
    """
    A class intended to handle precomputations of utilities of the motion we want to
    load into a character.
    """

    # transitive motion is necessary for scenic motion build
    def __init__(
        self,
        motion_: motion.Motion,
    ) -> None:
        # logger.info("Loading Motion data...")

        ROOT, FIRST, LAST = 0, 0, -1
        # primary
        self.motion = motion_
        self.poses = motion_.poses
        self.fps = motion_.fps
        self.num_of_frames: int = len(motion_.poses)
        self.map_of_total_displacement: List[float] = []
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

    @classmethod
    def obtain_pose(
        cls, skel: motion.Skeleton, body_info, index
    ) -> motion.Pose:
        pose_body = body_info["pose"][index]
        root_orient = body_info["root_orient"][index]
        root_trans = body_info["trans"][index]
        num_joints = (pose_body.shape[0] // 3) + 1
        # breakpoint()
        pose_data = []
        for j in range(num_joints):
            if j == 0:
                T = conversions.Rp2T(conversions.A2R(root_orient), root_trans)
            else:
                T = conversions.R2T(
                    conversions.A2R(pose_body[(j - 1) * 3 : (j - 1) * 3 + 3])
                )
            pose_data.append(T)
        # if len(pose_data) == 8:
        # breakpoint()
        return motion.Pose(skel, pose_data)


class AmassHelper:
    """
    This class contains methods to load and convert AmassData into a format for Habitat
    """

    def __init__(self, bm_path):
        self.bm_path = bm_path
        self.bm = BodyModel(
            bm_fname=bm_path,
            num_betas=10,
            # model_type=model_type
        )

        # Get skeleton
        # bdata = np.load(bm_path)
        betas = np.array(
            [
                -0.0312,
                -0.1780,
                -0.0597,
                0.0200,
                0.0168,
                0.0530,
                -0.0313,
                -0.0084,
                0.0375,
                0.0034,
            ]
        )
        betas = torch.tensor(betas[np.newaxis]).float()
        # betas = bdata["betas"][:10][np.newaxis]
        joint_names = amass.joint_names

        num_joints = len(joint_names)

        self.skeleton = amass.create_skeleton_from_amass_bodymodel(
            self.bm, betas, num_joints, joint_names
        )

    def get_skeleton(self):
        return self.skeleton

    @classmethod
    def load_amass_file(cls, filename, **kwargs):
        return amass.load(filename, **kwargs)

    @classmethod
    def global_correction_quat(
        cls, up_v: mn.Vector3, forward_v: mn.Vector3
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

        forward_v = forward_v * (
            mn.Vector3(1.0, 1.0, 1.0) - mn.Vector3.y_axis()
        )
        angle2 = mn.math.angle(
            forward_v.normalized(), -1 * mn.Vector3.z_axis()
        )
        axis2 = mn.Vector3.y_axis()
        rotation2 = mn.Quaternion.rotation(angle2, axis2)

        return rotation2 * rotation1

    @classmethod
    def convert_CMUamass_single_pose(
        cls,
        pose,
        joint_info,
        raw=False,
        translation_offset=mn.Vector3(),
        rotation_offset=mn.Quaternion(),
        root_index=0,
    ) -> Tuple[List[float], mn.Vector3, mn.Quaternion]:
        """
        Converts a pose from CMU format into a format that can be processed in habitat:
        This conversion is specific to the datasets from CMU.
        Args:
            pose: FairMotion pose
            joint_info: a list of dictionaries, containing information of every joint
            raw: if False, convert pose so that it looks at -Z and up is Y
            translation_offset: how much to translate the final pose
            rotation_offset: how much to rotate the final pose
        Returns:
            new_pose: list of Quaternions (excluding root)
            root_translation: translation of the joint, after applying offset
            root_rotation: rotation of the root joint, after applying offset
        """
        new_pose = []
        ROOT = root_index

        # Root joint
        root_T = pose.get_transform(ROOT, local=False)

        final_rotation_correction = mn.Quaternion()

        if not raw:
            # Rotate so that up is in Y and front is in -Z
            final_rotation_correction = (
                cls.global_correction_quat(
                    mn.Vector3.z_axis(), mn.Vector3.x_axis()
                )
                * rotation_offset
            )

        root_rotation = final_rotation_correction * mn.Quaternion.from_matrix(
            mn.Matrix3x3(root_T[0:3, 0:3])
        )
        root_translation = (
            translation_offset
            + final_rotation_correction.transform_vector(root_T[0:3, 3])
        )

        # This should not be used
        Q, _ = conversions.T2Qp(root_T)

        # Other joints
        joint_map = {}
        joint_map["root"] = "m_avg_Pelvis"
        joint_map["lhip"] = "m_avg_L_Hip"
        joint_map["lknee"] = "m_avg_L_Knee"
        joint_map["lankle"] = "m_avg_L_Ankle"
        joint_map["rhip"] = "m_avg_R_Hip"
        joint_map["rknee"] = "m_avg_R_Knee"
        joint_map["rankle"] = "m_avg_R_Ankle"
        joint_map["lowerback"] = "m_avg_Spine1"
        joint_map["upperback"] = "m_avg_Spine2"
        joint_map["chest"] = "m_avg_Spine3"
        joint_map["lowerneck"] = "m_avg_Neck"
        joint_map["upperneck"] = "m_avg_Head"
        joint_map["lclavicle"] = "m_avg_L_Collar"
        joint_map["lshoulder"] = "m_avg_L_Shoulder"
        joint_map["lelbow"] = "m_avg_L_Elbow"
        joint_map["lwrist"] = "m_avg_L_Wrist"
        joint_map["rclavicle"] = "m_avg_R_Collar"
        joint_map["rshoulder"] = "m_avg_R_Shoulder"
        joint_map["relbow"] = "m_avg_R_Elbow"
        joint_map["rwrist"] = "m_avg_R_Wrist"
        inv_map = {v: k for k, v in joint_map.items()}
        
        for model_link_id in range(len(joint_info)):
            joint_type = joint_info[model_link_id][2]
            joint_name = joint_info[model_link_id][1].decode("UTF-8")
            pose_joint_index = pose.skel.index_joint[joint_name]
            # When the target joint do not have dof, we simply ignore it
            if joint_type == p.JOINT_FIXED:
                continue

            # When there is no matching between the given pose and the simulated character,
            # the character just tries to hold its initial pose
            if pose_joint_index is None:
                raise KeyError(
                    "Error: pose data does not have a transform for that joint name"
                )
            if joint_type not in [p.JOINT_SPHERICAL]:
                raise NotImplementedError(
                    f"Error: {joint_type} is not a supported joint type"
                )

            T = pose.get_transform(pose_joint_index, local=True)

            if joint_type == p.JOINT_SPHERICAL:
                Q, _ = conversions.T2Qp(T)

            new_pose += list(Q)
            # if debug:
            #     print(joint_name, Q)
        # breakpoint()
        return new_pose, root_translation, root_rotation
