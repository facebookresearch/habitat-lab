# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import numpy as np
from human_body_prior.body_model.body_model import BodyModel
from . import motion as motion_class,conversions,utils

"""
Structure of npz file in AMASS dataset is as follows.
- trans (num_frames, 3):  translation (x, y, z) of root joint
- gender str: Gender of actor
- mocap_framerate int: Framerate in Hz
- betas (16): Shape parameters of body. See https://smpl.is.tue.mpg.de/
- dmpls (num_frames, 8): DMPL parameters
- poses (num_frames, 156): Pose data. Each pose is represented as 156-sized
    array. The mapping of indices encoding data is as follows:
    0-2 Root orientation
    3-65 Body joint orientations
    66-155 Finger articulations
"""

# Custom names for 22 joints in AMASS data
joint_names = [
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


def create_skeleton_from_amass_bodymodel(bm, betas, num_joints, joint_names):
    pose_body_zeros = torch.zeros((1, 3 * (num_joints - 1)))
    body = bm(pose_body=pose_body_zeros, betas=betas)
    base_position = body.Jtr.detach().numpy()[0, 0:num_joints]
    parents = bm.kintree_table[0].long()[:num_joints]

    joints = []
    for i in range(num_joints):
        joint = motion_class.Joint(name=joint_names[i])
        if i == 0:
            joint.info["dof"] = 6
            joint.xform_from_parent_joint = conversions.p2T(np.zeros(3))
        else:
            joint.info["dof"] = 3
            joint.xform_from_parent_joint = conversions.p2T(
                base_position[i] - base_position[parents[i]]
            )
        joints.append(joint)

    parent_joints = []
    for i in range(num_joints):
        parent_joint = None if parents[i] < 0 else joints[parents[i]]
        parent_joints.append(parent_joint)

    skel = motion_class.Skeleton()
    for i in range(num_joints):
        skel.add_joint(joints[i], parent_joints[i])

    return skel


def create_motion_from_amass_data(filename, bm, override_betas=None):
    bdata = np.load(filename)

    if override_betas is not None:
        betas = torch.Tensor(override_betas[:10][np.newaxis]).to("cpu")
    else:
        betas = torch.Tensor(bdata["betas"][:10][np.newaxis]).to("cpu")
    
    skel = create_skeleton_from_amass_bodymodel(
        bm, betas, len(joint_names), joint_names,
    )
    if "mocap_framerate" in bdata:
        fps = float(bdata["mocap_framerate"])
    elif "mocap_frame_rate" in bdata:
        fps = float(bdata["mocap_frame_rate"])
    root_orient = bdata["poses"][:, :3]  # controls the global root orientation
    pose_body = bdata["poses"][:, 3:66]  # controls body joint angles
    trans = bdata["trans"][:, :3]  # controls the finger articulation

    motion = motion_class.Motion(skel=skel, fps=fps)

    num_joints = skel.num_joints()
    parents = bm.kintree_table[0].long()[:num_joints]

    for frame in range(pose_body.shape[0]):
        pose_body_frame = pose_body[frame]
        root_orient_frame = root_orient[frame]
        root_trans_frame = trans[frame]
        pose_data = []
        for j in range(num_joints):
            if j == 0:
                T = conversions.Rp2T(
                    conversions.A2R(root_orient_frame), root_trans_frame
                )
            else:
                T = conversions.R2T(
                    conversions.A2R(
                        pose_body_frame[(j - 1) * 3 : (j - 1) * 3 + 3]
                    )
                )
            pose_data.append(T)
        motion.add_one_frame(pose_data)

    return motion


def load_body_model(bm_path, num_betas=10, model_type="smplh"):
    comp_device = torch.device("cpu")
    bm = BodyModel(
        bm_fname=bm_path, 
        num_betas=num_betas, 
        # model_type=model_type
    ).to(comp_device)
    return bm


def load(file, bm=None, bm_path=None, num_betas=10, model_type="smplh", override_betas=None):
    if bm is None:
        # Download the required body model. For SMPL-H download it from
        # http://mano.is.tue.mpg.de/.
        assert bm_path is not None, "Please provide SMPL body model path"
        bm = load_body_model(bm_path, num_betas, model_type)
    return create_motion_from_amass_data(
        filename=file, bm=bm, override_betas=override_betas)


def save():
    raise NotImplementedError("Using bvh.save() is recommended")


def load_parallel(files, cpus=20, **kwargs):
    return utils.run_parallel(load, files, num_cpus=cpus, **kwargs)
