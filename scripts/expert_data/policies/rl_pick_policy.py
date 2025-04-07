import os
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from habitat.utils.gum_utils import (
    quat_to_axis_angle,
    sample_point_cloud_from_urdf,
    to_world_frame,
)
from scripts.expert_data.utils.utils import create_T_matrix

third_party_ckpt_root_folder = (
    "/opt/hpcaas/.mounts/fs-03ee9f8c6dddfba21/jtruong/repos/gum_ws/src/GUM"
)
sys.path.append(third_party_ckpt_root_folder)
from gum.planning.rl.ppo import PPO

from habitat.isaac_sim import isaac_prim_utils


class RLPickPolicy:
    def __init__(self, murp_env):
        self.murp_wrapper = murp_env
        self.env = murp_env.env
        self.config = murp_env.config

        self.convention = "isaac"
        self.debug = murp_env.config.debug
        if self.debug:
            self.traj = self.load_traj()

        self.reset()
        self.checkpoint_path = self.config.pick_ckpt_path
        self.policy = self.load_checkpoint()

    def load_traj(self):
        data = torch.load(self.config.traj_path, map_location="cpu")
        return data

    def load_checkpoint(self):
        return PPO.from_checkpoint(self.checkpoint_path)

    def convert_to_ref_frame(
        self,
        translation,
        rotation,
        global_T_ref_frame,
        rotation_convention="wxyz",
    ):
        if rotation_convention == "wxyz":
            # convert from wxyz to xyzw
            rotation = self.wxyz_to_xyzw(rotation)

        self.global_T_asset = create_T_matrix(translation, rotation)
        # invert and multiply
        ref_T_asset = np.linalg.inv(global_T_ref_frame) @ self.global_T_asset
        return ref_T_asset[:3, -1], ref_T_asset[:3, :3]

    def convert_position_conventions(self, position):
        x, y, z = position
        return np.array([-x, z, y])

    def map_joints(self, joints, from_isaac=True):
        # map the joints from isaac convention to habitat convention
        # habitat convention is [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        # isaac convention is [ 0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 10, 11]
        # map the joints from habitat convention to isaac convention
        if from_isaac:
            joints_hab = np.zeros(16)
            joints_hab[:4] = joints[:4]
            joints_hab[4:8] = joints[8:12]
            joints_hab[8:12] = joints[12:]
            joints_hab[12:] = joints[4:8]
            return joints_hab
        else:
            joints_isaac = np.zeros(16)
            joints_isaac[:4] = joints[:4]
            joints_isaac[4:8] = joints[8:12]
            joints_isaac[8:12] = joints[12:]
            joints_isaac[12:] = joints[4:8]
            return joints_isaac

    def get_priv_info(self):
        self.private_info = {}
        # randomly sample the object scale, mass, and friction
        self.private_info["object_scale"] = torch.tensor(
            [np.random.choice([0.77, 0.79, 0.81, 0.83, 0.84])]
        )
        self.private_info["object_mass"] = torch.tensor(
            [np.random.uniform(0.04, 0.08)]
        )
        self.private_info["object_friction"] = torch.tensor(
            [np.random.uniform(0.3, 1.0)]
        )
        self.private_info["object_center_of_mass"] = torch.tensor(
            [
                np.random.uniform(-0.01, 0.01),
                np.random.uniform(-0.01, 0.01),
                np.random.uniform(-0.01, 0.01),
            ]
        )
        self.private_info["object_restitution"] = torch.tensor(
            [np.random.uniform(0, 1.0)]
        )

        # initialize object and fingertips to zeros
        self.private_info["object_trans"] = torch.zeros(3)
        self.private_info["object_rot"] = torch.zeros(4)
        self.private_info["object_angvel"] = torch.zeros(3)

        self.private_info["fingertip_trans"] = torch.zeros(12)

    def xyzw_to_wxyz(self, quat_xyzw):
        # convert from xyzw to wxyz
        return np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        )

    def wxyz_to_xyzw(self, quat_wxyz):
        # convert from wxyz to xyzw
        return np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        )

    def get_obj_T_matrix(self):
        # quat in wxyz format
        hab_global_T_obj_trans, hab_global_T_obj_rot_xyzw = (
            self.env.sim.get_prim_transform(
                self.obj_prim_path, convention="quat"
            )
        )
        hab_global_T_obj_rot_wxyz = self.xyzw_to_wxyz(
            hab_global_T_obj_rot_xyzw
        )
        # if self.convention == "isaac":
        #     isaac_global_T_obj_trans = (
        #         isaac_prim_utils.habitat_to_usd_position(
        #             hab_global_T_obj_trans
        #         )
        #     )
        #     hab_global_T_obj_rot_wxyz = self.xyzw_to_wxyz(
        #         hab_global_T_obj_rot_xyzw
        #     )
        #     isaac_global_T_obj_rot_wxyz = (
        #         isaac_prim_utils.habitat_to_usd_rotation(
        #             hab_global_T_obj_rot_wxyz
        #         )
        #     )
        #     isaac_global_T_obj_rot_xyzw = self.wxyz_to_xyzw(
        #         isaac_global_T_obj_rot_wxyz
        #     )

        isaac_global_T_obj_trans, isaac_global_T_obj_rot_wxyz = (
            self.env.sim.get_prim_transform(
                self.obj_prim_path, coord_convention="isaac", convention="quat"
            )
        )
        self.isaac_global_T_obj = create_T_matrix(
            isaac_global_T_obj_trans, isaac_global_T_obj_rot_wxyz
        )
        self.hab_global_T_obj = create_T_matrix(
            hab_global_T_obj_trans, hab_global_T_obj_rot_wxyz
        )

    def habitat_to_isaac(self, hab_trans, hab_quat, convention="xyzw"):
        isaac_trans = isaac_prim_utils.habitat_to_usd_position(hab_trans)
        if convention == "xyzw":
            hab_quat = self.xyzw_to_wxyz(hab_quat)
        isaac_quat_wyxz = isaac_prim_utils.habitat_to_usd_rotation(hab_quat)
        isaac_quat_xyzw = self.wxyz_to_xyzw(isaac_quat_wyxz)
        return isaac_trans, isaac_quat_xyzw

    def get_hand_pose(self):
        isaac_global_T_ee_trans, isaac_global_T_ee_rot_quat_xyzw = (
            self.murp_wrapper.get_curr_ee_pose(
                coord_convention="usd", convention="quat"
            )
        )
        # hab_global_T_ee_trans, hab_global_T_ee_rot_quat_xyzw = (
        #     self.murp_wrapper.get_curr_ee_pose(
        #         coord_convention="usd", convention="quat"
        #     )
        # )
        # isaac_global_T_ee_trans, isaac_global_T_ee_rot_quat_xyzw = (
        #     self.habitat_to_isaac(
        #         hab_global_T_ee_trans,
        #         hab_global_T_ee_rot_quat_xyzw,
        #         convention="xyzw",
        #     )
        # )

        isaac_obj_T_ee_trans, isaac_obj_T_ee_rot_mat = (
            self.convert_to_ref_frame(
                isaac_global_T_ee_trans,
                isaac_global_T_ee_rot_quat_xyzw,
                self.isaac_global_T_obj,
                rotation_convention="xyzw",
            )
        )
        isaac_obj_T_ee_rot_quat_xyzw = R.from_matrix(
            isaac_obj_T_ee_rot_mat
        ).as_quat()

        self.isaac_global_T_hand = create_T_matrix(
            isaac_obj_T_ee_trans, isaac_obj_T_ee_rot_quat_xyzw
        )
        return isaac_obj_T_ee_trans, isaac_obj_T_ee_rot_quat_xyzw

    def get_obj_transform(self):
        # hab_global_T_obj_trans, hab_global_T_obj_rot_quat_wxyz = (
        #     self.env.sim.get_prim_transform(
        #         self.obj_prim_path, convention="quat"
        #     )
        # )
        # isaac_global_T_obj_trans, isaac_global_T_obj_rot_quat_wxyz = (
        #     self.habitat_to_isaac(
        #         hab_global_T_obj_trans,
        #         hab_global_T_obj_rot_quat_wxyz,
        #         convention="wxyz",
        #     )
        # )
        isaac_global_T_obj_trans, isaac_global_T_obj_rot_quat_wxyz = (
            self.env.sim.get_prim_transform(
                self.obj_prim_path, coord_convention="isaac", convention="quat"
            )
        )

        isaac_obj_T_hand_trans, isaac_obj_T_hand_rot_mat = (
            self.convert_to_ref_frame(
                isaac_global_T_obj_trans,
                isaac_global_T_obj_rot_quat_wxyz,
                self.isaac_global_T_hand,
            )
        )
        isaac_obj_T_hand_rot_quat_xyzw = R.from_matrix(
            isaac_obj_T_hand_rot_mat
        ).as_quat()
        return isaac_obj_T_hand_trans, isaac_obj_T_hand_rot_quat_xyzw

    def get_fingertip_poses(self):
        # hab_global_T_finger_trans, hab_global_T_finger_rot_ = (
        #     self.env.sim.articulated_agent._robot_wrapper.fingertip_right_pose()
        # )
        isaac_global_T_finger_trans, isaac_global_T_finger_rot_wxyz = (
            self.env.sim.articulated_agent._robot_wrapper.fingertip_right_pose(
                convention="isaac"
            )
        )

        isaac_global_T_finger_trans_split = (
            np.array(isaac_global_T_finger_trans).reshape(-1, 3).tolist()
        )
        isaac_global_T_finger_rot_wxyz_split = (
            np.array(isaac_global_T_finger_rot_wxyz).reshape(-1, 4).tolist()
        )
        # get position and rotation for each finger
        isaac_finger_T_obj_translations, isaac_finger_T_obj_rotations = [], []
        num_fingers = 4
        for i in range(len(isaac_global_T_finger_trans_split)):
            isaac_finger_T_obj_trans, isaac_finger_T_obj_rot_mat = (
                self.convert_to_ref_frame(
                    isaac_global_T_finger_trans_split[i],
                    isaac_global_T_finger_rot_wxyz_split[i],
                    self.isaac_global_T_obj,
                )
            )

            isaac_finger_T_obj_translations.extend(isaac_finger_T_obj_trans)
            # convert from 3x3 rotation matrix to 4-dim quaternion
            isaac_finger_T_obj_rot_quat_xyzw = R.from_matrix(
                isaac_finger_T_obj_rot_mat
            ).as_quat()
            isaac_finger_T_obj_rotations.extend(
                isaac_finger_T_obj_rot_quat_xyzw
            )
            # increment by 3 for each finger
        return isaac_finger_T_obj_translations, isaac_finger_T_obj_rotations

    def get_curr_targets(self):
        isaac_obj_T_ee_pos, isaac_obj_T_ee_rot_quat_xyzw = self.get_hand_pose()
        isaac_obj_T_ee_rot_quat_rpy = R.from_quat(
            isaac_obj_T_ee_rot_quat_xyzw
        ).as_euler("xyz", degrees=False)

        right_hand_joints = torch.tensor(
            self.env.sim.articulated_agent._robot_wrapper.right_hand_joint_pos
        )
        return torch.cat(
            [
                torch.tensor(isaac_obj_T_ee_pos),
                torch.tensor(isaac_obj_T_ee_rot_quat_rpy),
                right_hand_joints,
            ]
        )

    def reset(self):
        self.target_fingers = np.zeros(16)
        self.progress_ctr = 0
        self.obj_prim_path = (
            str(self.env.sim._rigid_objects[0]._prim)
            .split("<")[1]
            .split(">")[0]
        )
        self.object_name = self.env.current_episode.rigid_objs[0][0]
        pc, normals = sample_point_cloud_from_urdf(
            os.path.abspath("data/assets"),
            f"dexgraspnet2/meshdata/{self.object_name}/simplified_sdf.urdf",
            100,
            seed=4,
        )
        self.gt_object_point_clouds__object = torch.tensor(pc).unsqueeze(0)

        self.get_priv_info()
        self.get_obj_T_matrix()
        targets = self.get_curr_targets()

        # initialize to values from beginning, not zero
        self.obs_buf_lag_history = torch.tensor(
            np.repeat(targets, 6)
        )  # joints = 22-dim, target = 22-dim, history of 3
        self.prev_targets = torch.tensor(targets)
        self.behavior_action = torch.zeros(22).unsqueeze(0)

    def get_obs_dict(self, convention="hab"):
        obs_dict = {}
        self.get_obj_T_matrix()

        obs_dict["behavior_action"] = self.behavior_action

        ## get hand_base_pose, relative to world (under object)

        isaac_obj_T_ee_pos, isaac_obj_T_ee_rot_quat_xyzw = self.get_hand_pose()
        obs_dict["hand_base_pose"] = torch.cat(
            [
                torch.tensor(isaac_obj_T_ee_pos),
                torch.tensor(isaac_obj_T_ee_rot_quat_xyzw),
            ]
        ).unsqueeze(0)

        ## get gt object point cloud in hand frame
        object_pc__world = to_world_frame(
            self.gt_object_point_clouds__object,
            torch.tensor(isaac_obj_T_ee_rot_quat_xyzw).unsqueeze(0),
            torch.tensor(isaac_obj_T_ee_pos).unsqueeze(0),
        )
        obs_dict["gt_object_point_cloud"] = object_pc__world

        ## get object translation, rotation and angular velocity in hand frame
        isaac_obj_T_hand_trans, isaac_obj_T_hand_rot_quat_xyzw = (
            self.get_obj_transform()
        )
        self.private_info["object_trans"] = torch.tensor(
            isaac_obj_T_hand_trans
        )
        self.private_info["object_rot"] = torch.tensor(
            isaac_obj_T_hand_rot_quat_xyzw
        )
        self.private_info["object_angvel"] = torch.tensor(
            self.env.sim._rigid_objects[0]._rigid_prim.get_angular_velocity()
        )

        ## get fingertip translation in hand frame
        isaac_finger_T_obj_translations, isaac_finger_T_obj_rotations = (
            self.get_fingertip_poses()
        )
        self.private_info["fingertip_trans"] = torch.tensor(
            isaac_finger_T_obj_translations
        )

        prev_obs_buf = self.obs_buf_lag_history[44:]

        curr_targets = self.get_curr_targets()
        print("curr_targets: ", curr_targets)
        print("self.prev_targets: ", self.prev_targets)
        cur_obs_buf = torch.cat([curr_targets, self.prev_targets])
        obs_buf = torch.cat([prev_obs_buf, cur_obs_buf]).unsqueeze(0)

        clip_obs = 5.0
        obs_dict["obs"] = torch.clamp(obs_buf, -clip_obs, clip_obs)

        ## set priv info
        obs_dict["priv_info"] = torch.cat(
            [
                self.private_info["object_trans"],
                self.private_info["object_scale"],
                self.private_info["object_mass"],
                self.private_info["object_friction"],
                self.private_info["object_center_of_mass"],
                self.private_info["object_rot"],
                self.private_info["object_angvel"],
                self.private_info["fingertip_trans"],
                self.private_info["object_restitution"],
            ]
        ).unsqueeze(0)

        # add current observation to obs_buf_lag_history
        self.obs_buf_lag_history = torch.cat([prev_obs_buf, cur_obs_buf])

        obs_dict["progress_float"] = torch.tensor(
            [self.progress_ctr]
        ).unsqueeze(0)

        if self.debug:
            traj_obs = self.traj[self.progress_ctr + 1]["obs"]
            keep_keys = [
                "gt_object_point_cloud",
                "progress_float",
                "priv_info",
                "obs",
                "hand_base_pose",
                "behavior_action",
            ]
            traj_obs["gt_object_point_cloud"] = obs_dict[
                "gt_object_point_cloud"
            ]
            traj_obs["progress_float"] = obs_dict["progress_float"]
            # traj_obs["priv_info"] = obs_dict["priv_info"]
            # traj_obs["hand_base_pose"] = obs_dict["hand_base_pose"]
            obs_dict = {k: v for k, v in traj_obs.items() if k in keep_keys}
        for k, v in obs_dict.items():
            obs_dict[k] = v.float().to("cuda:0")

        self.obs_dict = obs_dict
        return obs_dict

    def step(self, action):
        # if self.debug:
        # traj_action = self.traj[self.progress_ctr + 1]["act"]
        # action = traj_action
        wrist_scale = 0.2
        finger_scale = 0.05  # 0.005
        # delta_wrist_trans, delta_wrist_axis_angle, delta_fingers
        curr_ee_pos, _ = self.murp_wrapper.get_curr_ee_pose(convention="rpy")
        _, curr_ee_rot = self.murp_wrapper.get_curr_ee_pose(
            convention="rpy", use_global=False
        )

        if self.progress_ctr == 0:
            # self.rest_ori = curr_ee_rot
            self.open_loop_rot = np.array(
                [-1.57079633, 1.57079633, -1.57079621]
            )

        base_T_wrist = create_T_matrix(curr_ee_pos, self.open_loop_rot)

        action[0, 3:6] = 0
        delta_wrist_trans = action[0, :3]
        delta_wrist_axis_angle = action[0, 3:6]

        # delta_wrist_mat = R.from_rotvec(delta_wrist_axis_angle.cpu().numpy())
        delta_fingers = action[0, 6:]

        # apply the deltas to the current wrist pose
        delta_wrist_trans = action[0, :3]

        print("action: ", action)
        print("delta_wrist_trans: ", delta_wrist_trans)
        # convert to habitat conventions
        delta_wrist_trans_hab = (
            np.array(
                isaac_prim_utils.habitat_to_usd_position(
                    delta_wrist_trans.cpu().numpy()
                )
            )
            * wrist_scale
        )
        base_T_delta = create_T_matrix(
            delta_wrist_trans_hab,
            delta_wrist_axis_angle.cpu().numpy(),
            use_rotvec=True,
        )

        new_wrist_T = base_T_wrist @ base_T_delta
        new_wrist_pos = new_wrist_T[:3, -1]
        new_wrist_rot_mat = R.from_matrix(new_wrist_T[:3, :3])
        new_wrist_rot_rpy = new_wrist_rot_mat.as_euler("xyz")
        self.open_loop_rot = new_wrist_rot_rpy

        new_wrist_axis_angle = new_wrist_rot_mat.as_rotvec()
        curr_hand_pos = self.murp_wrapper.get_curr_hand_pose()

        delta_fingers_hab = self.map_joints(
            delta_fingers.cpu().numpy(), from_isaac=True
        )
        new_fingers = curr_hand_pos + delta_fingers_hab * finger_scale
        new_wrist_pos_v2 = curr_ee_pos + delta_wrist_trans_hab

        # move the robot to the new wrist pose
        if self.debug:
            new_wrist_pos_v2 = self.murp_wrapper.env.sim._rigid_objects[
                0
            ].translation
            new_wrist_pos_v2[1] += 0.2
        self.murp_wrapper.move_ee_and_hand(
            new_wrist_pos_v2,
            self.open_loop_rot,
            new_fingers,
            timeout=1,
            text="using RL pick controller",
        )
        self.prev_targets = action.squeeze(0).cpu()
        self.behavior_action = action
