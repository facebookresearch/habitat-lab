#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import pickle as pkl

import magnum as mn
import numpy as np
import pybullet as p

smplx_body_joint_names = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]


def global_correction_quat(
    up_v: mn.Vector3, forward_v: mn.Vector3
) -> mn.Quaternion:
    """
    Given the upward direction and the forward direction of a local space frame, this method produces
    the correction quaternion to convert the frame to global space (+Y up, -Z forward).
    """
    if up_v.normalized() != mn.Vector3.y_axis():
        angle1 = mn.math.angle(up_v.normalized(), mn.Vector3.y_axis())
        axis1 = mn.math.cross(up_v.normalized(), mn.Vector3.y_axis())
        rotation1 = mn.Quaternion.rotation(angle1, axis1)
        forward_v = rotation1.transform_vector(forward_v)
    else:
        rotation1 = mn.Quaternion()

    forward_v = forward_v * (mn.Vector3(1.0, 1.0, 1.0) - mn.Vector3.y_axis())
    angle2 = mn.math.angle(forward_v.normalized(), -1 * mn.Vector3.z_axis())
    axis2 = mn.Vector3.y_axis()
    rotation2 = mn.Quaternion.rotation(angle2, axis2)

    return rotation2 * rotation1


class MotionConverterSMPLX:
    """
    Human Controller, converts high level actions such as walk, or reach into joints
    to control a URDF object.
    """

    def __init__(self, urdf_path):
        self.index_joint_map = {
            joint_name: index_joint
            for index_joint, joint_name in enumerate(smplx_body_joint_names)
        }
        self.pc_id = p.connect(p.DIRECT)
        self.human_bullet_id = p.loadURDF(urdf_path)

        self.link_ids = list(range(p.getNumJoints(self.human_bullet_id)))
        self.joint_info = [
            p.getJointInfo(self.human_bullet_id, index)
            for index in self.link_ids
        ]
        self.final_rotation_correction = global_correction_quat(
            mn.Vector3.z_axis(), mn.Vector3.x_axis()
        )

    def convert_pose_to_rotation(
        self, root_translation, root_orientation, pose_joints
    ):
        """
        Converts a single pose from SMPL-X format to Habitat format. The input pose assumes that
        character faces +X and Z is up.
        The output pose is converted so that the character faces -Z and +Y is up.
            :param root_translation: The global pose translation, measured as the position of the pelvis
            :param root_orientation: The global root orientation in axis-angle format
            :param pose_joints: Array of num_joints * 3 elements where pose_joints[i*3:(i+1)*3] contains the rotation of the
                         ith joint, in axis-angle format.

            :return: root_translation
            :return: root_rotation
            :return: new_pose
        """

        axis_angle_root_rotation_vec = mn.Vector3(root_orientation)
        root_trans = mn.Vector3(root_translation)
        axis_angle_root_rotation_angl = mn.Rad(
            axis_angle_root_rotation_vec.length()
        )
        root_T = self.final_rotation_correction * mn.Quaternion.rotation(
            axis_angle_root_rotation_angl,
            axis_angle_root_rotation_vec.normalized(),
        )
        root_rotation = root_T.to_matrix()
        root_translation = self.final_rotation_correction.transform_vector(
            root_trans
        )
        new_pose = []
        for model_link_id in range(len(self.joint_info)):
            joint_type = self.joint_info[model_link_id][2]
            joint_name = self.joint_info[model_link_id][1].decode("UTF-8")
            if joint_name not in self.index_joint_map:
                pose_joint_index = None
            else:
                # We subtract 1 cause pose does not contain the root rotation
                pose_joint_index = self.index_joint_map[joint_name] - 1

            # When the target joint do not have dof, we simply ignore it
            if joint_type == p.JOINT_FIXED:
                continue

            if joint_type not in [p.JOINT_SPHERICAL]:
                raise NotImplementedError(
                    f"Error: {joint_type} is not a supported joint type"
                )

            if pose_joint_index is None:
                Ql = [0, 0, 0, 1]
            else:
                pose_joint_indices = slice(
                    pose_joint_index * 3, pose_joint_index * 3 + 3
                )
                # get_transform(pose_joint_index, local=True)

                if joint_type == p.JOINT_SPHERICAL:
                    axis_angle_rotation = mn.Vector3(
                        pose_joints[pose_joint_indices]
                    )
                    axis_angle_rotation_ang = mn.Rad(
                        axis_angle_rotation.length()
                    )
                    Q = mn.Quaternion.rotation(
                        axis_angle_rotation_ang,
                        axis_angle_rotation.normalized(),
                    )
                    Ql = list(Q.vector) + [float(Q.scalar)]

            new_pose += list(Ql)
        return root_translation, root_rotation, new_pose

    def convert_motion_file(self, motion_path, output_path="output_motion"):
        """
        Convert a npz file containing a SMPL-X motion into a file of rotations
        that can be played in Habitat
            :param motion_path: path to the npz file containing a SMPL-X motion.
            :param output_path: output path where to save the pkl file with the converted motion
        """
        content_motion = np.load(motion_path, allow_pickle=True)

        pose_info = {
            "trans": content_motion["trans"],
            "root_orient": content_motion["root_orient"],
            "pose": content_motion["poses"][:, 3:66],
        }
        num_poses = content_motion["poses"].shape[0]
        transform_array = []
        joints_array = []
        for index in range(num_poses):
            root_trans, root_rot, pose_quat = self.convert_pose_to_rotation(
                pose_info["trans"][index],
                pose_info["root_orient"][index],
                pose_info["pose"][index],
            )
            transform_as_mat = np.array(mn.Matrix4.from_(root_rot, root_trans))
            transform_array.append(transform_as_mat[None, :])
            joints_array.append(np.array(pose_quat)[None, :])

        transform_array = np.concatenate(transform_array)
        joints_array = np.concatenate(joints_array)

        walk_motion = {
            "joints_array": joints_array,
            "transform_array": transform_array,
            "displacement": None,
            "fps": 1,
        }
        content_motion = {
            "pose_motion": walk_motion,
        }
        with open(f"{output_path}.pkl", "wb+") as ff:
            pkl.dump(content_motion, ff)


if __name__ == "__main__":
    # Converts a npz motion file into a pkl file that can be processed in habitat. Note that the
    # resulting pkl file will have the same format as the ones used to move the character in
    # Humanoid RearrangeController
    # The npz file should contain:
    # trans: N x 3, specifying the root translation on each of the N poses
    # poses: N x (J*3 + 1) * 3: containing the root rotation, as well as the rotation for each
    # of the 21 joints
    motion_file = "data/humanoids/humanoid_data/walk_motion/CMU_10_04_stageii.npz"  # motion_to_convert (npz)
    files = glob.glob(motion_file)
    for in_path in files:
        convert_helper = MotionConverterSMPLX(
            urdf_path="data/humanoids/humanoid_data/female_2/female_2.urdf"
        )
        convert_helper.convert_motion_file(
            motion_path=in_path,
            output_path=in_path.replace(".npz", ""),
        )
