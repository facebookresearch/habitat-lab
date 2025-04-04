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

    def convert_to_obj_frame(
        self, translation, rotation, rotation_convention="wxyz"
    ):
        if rotation_convention == "wxyz":
            rotation = np.array(
                [rotation[3], rotation[0], rotation[1], rotation[2]]
            )

        self.global_T_asset = create_T_matrix(translation, rotation)
        # invert and multiply
        obj_T_asset = np.linalg.inv(self.global_T_obj) @ self.global_T_asset
        return obj_T_asset[:3, -1], obj_T_asset[:3, :3]

    def get_fingertip_poses(self, convention="local"):
        finger_poses, finger_rots = (
            self.env.sim.articulated_agent._robot_wrapper.fingertip_right_pose()
        )

        if convention != "local":
            return finger_poses, finger_rots
        else:
            finger_pos_split = np.array(finger_poses).reshape(-1, 3).tolist()
            finger_rot_split = np.array(finger_rots).reshape(-1, 4).tolist()
            # get position and rotation for each finger
            obj_T_ee_poses, obj_T_ee_rots = [], []
            num_fingers = 4
            for i in range(len(finger_pos_split)):
                if self.convention == "isaac":
                    finger_pos_split[i] = self.convert_position_conventions(
                        finger_pos_split[i]
                    )

                ee_pos, ee_rot = self.convert_to_obj_frame(
                    finger_pos_split[i], finger_rot_split[i]
                )

                obj_T_ee_poses.extend(ee_pos)
                # convert from 3x3 rotation matrix to 4-dim quaternion
                ee_rot = R.from_matrix(ee_rot).as_quat()
                obj_T_ee_rots.extend(ee_rot)
                # increment by 3 for each finger
            return obj_T_ee_poses, obj_T_ee_rots

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

    def reset(self):
        self.progress_ctr = 0
        self.obj_prim_path = (
            str(self.env.sim._rigid_objects[0]._prim)
            .split("<")[1]
            .split(">")[0]
        )
        self.object_name = self.env.current_episode.rigid_objs[0][0]
        self.target_fingers = np.zeros(16)
        pc, normals = sample_point_cloud_from_urdf(
            os.path.abspath("data/assets"),
            f"dexgraspnet2/meshdata/{self.object_name}/simplified_sdf.urdf",
            100,
            seed=4,
        )
        self.gt_object_point_clouds__object = torch.tensor(pc).unsqueeze(0)
        self.private_info = {}

        # quat in wxyz format
        obj_trans, obj_rot = self.env.sim.get_prim_transform(
            self.obj_prim_path, convention="quat"
        )
        if self.convention == "isaac":
            obj_trans = self.convert_position_conventions(obj_trans)
        obj_rot_wxyz = np.array(
            [obj_rot[3], obj_rot[0], obj_rot[1], obj_rot[2]]
        )
        self.global_T_obj = create_T_matrix(obj_trans, obj_rot_wxyz)
        obj_T_obj_pos, obj_T_obj_rot = self.convert_to_obj_frame(
            obj_trans, obj_rot
        )
        self.private_info["object_trans"] = obj_T_obj_pos

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
        self.private_info["object_rot"] = obj_T_obj_rot.flatten()
        self.private_info["object_angvel"] = torch.tensor(
            self.env.sim._rigid_objects[0]._rigid_prim.get_angular_velocity()
        )

        obj_T_ee_poses, obj_T_ee_rots = self.get_fingertip_poses(
            convention="local"
        )
        self.private_info["fingertip_trans"] = torch.tensor(obj_T_ee_poses)
        self.private_info["object_restitution"] = torch.tensor(
            [np.random.uniform(0, 1.0)]
        )

        curr_ee_pos, curr_ee_rot_quat = self.murp_wrapper.get_curr_ee_pose(
            convention="quat"
        )
        if self.convention == "isaac":
            curr_ee_pos = self.convert_position_conventions(curr_ee_pos)
        wrist_axis_angle = quat_to_axis_angle(torch.tensor(curr_ee_rot_quat))

        curr_hand_pos = self.murp_wrapper.get_curr_hand_pose()
        if self.convention == "isaac":
            curr_hand_pos = self.map_joints(curr_hand_pos, from_isaac=True)

        targets = np.concatenate(
            [curr_ee_pos, wrist_axis_angle.cpu().numpy(), curr_hand_pos]
        )

        # initialize to values from beginning, not zero
        self.obs_buf_lag_history = torch.tensor(
            np.repeat(targets, 6)
        )  # joints = 22-dim, target = 22-dim, history of 3
        self.target_obs = torch.tensor(targets)
        self.behavior_action = torch.zeros(22).unsqueeze(0)

    def get_obs_dict(self, convention="hab"):
        obs_dict = {}
        obj_trans, obj_rot = self.env.sim.get_prim_transform(
            self.obj_prim_path, convention="quat"
        )
        if self.convention == "isaac":
            obj_trans = self.convert_position_conventions(obj_trans)
        # obj_trans = torch.tensor(obj_trans).unsqueeze(0)
        # obj_rot = torch.tensor(obj_rot).unsqueeze(0)

        obj_T_obj_pos, obj_T_obj_rot = self.convert_to_obj_frame(
            obj_trans, obj_rot
        )
        obj_T_obj_rot_quat = R.from_matrix(obj_T_obj_rot).as_quat()
        obj_T_obj_pos = torch.tensor(obj_T_obj_pos)
        obj_T_obj_rot_quat = torch.tensor(obj_T_obj_rot_quat)

        object_pc__world = to_world_frame(
            self.gt_object_point_clouds__object,
            obj_T_obj_rot_quat.unsqueeze(0),
            obj_T_obj_pos.unsqueeze(0),
        )
        obs_dict["gt_object_point_cloud"] = object_pc__world
        self.private_info["object_trans"] = obj_T_obj_pos
        self.private_info["object_rot"] = obj_T_obj_rot_quat
        obj_T_ee_poses, obj_T_ee_rots = self.get_fingertip_poses(
            convention="local"
        )
        self.private_info["fingertip_trans"] = torch.tensor(obj_T_ee_poses)

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
        clip_obs = 5.0
        prev_obs_buf = self.obs_buf_lag_history[44:]
        joints_obs = torch.tensor(
            self.env.sim.articulated_agent._robot_wrapper.right_hand_joint_pos
        )
        curr_ee_pos, curr_ee_rot_rpy = self.murp_wrapper.get_curr_ee_pose()
        if self.convention == "isaac":
            curr_ee_pos = self.convert_position_conventions(curr_ee_pos)
        _, curr_ee_rot_quat = self.murp_wrapper.get_curr_ee_pose(
            convention="quat"
        )
        obj_T_ee_pos, obj_T_ee_rot = self.convert_to_obj_frame(
            curr_ee_pos, curr_ee_rot_quat, rotation_convention="xyzw"
        )
        obj_T_ee_rot_rpy = R.from_matrix(obj_T_ee_rot).as_euler(
            "xyz", degrees=False
        )
        obj_T_ee_rot_quat = R.from_matrix(obj_T_ee_rot).as_quat()
        # convert to torch and concatenate
        wrist_pose = torch.cat(
            [torch.tensor(obj_T_ee_pos), torch.tensor(obj_T_ee_rot_rpy)]
        )
        cur_obs_buf = torch.cat([joints_obs, wrist_pose, self.target_obs])

        obs_buf = torch.cat([prev_obs_buf, cur_obs_buf]).unsqueeze(0)
        obs_dict["obs"] = torch.clamp(obs_buf, -clip_obs, clip_obs)
        # add current observation to obs_buf_lag_history

        self.obs_buf_lag_history = torch.cat([prev_obs_buf, cur_obs_buf])

        obs_dict["hand_base_pose"] = torch.cat(
            [torch.tensor(obj_T_ee_pos), torch.tensor(obj_T_ee_rot_quat)]
        ).unsqueeze(0)
        obs_dict["behavior_action"] = self.behavior_action
        obs_dict["progress_float"] = torch.tensor(
            [self.progress_ctr]
        ).unsqueeze(0)
        for k, v in obs_dict.items():
            obs_dict[k] = v.float().to("cuda:0")
            # print(k, v.shape)

        # print("obs_dict: ", obs_dict)
        return obs_dict

    def step(self, action):
        if self.debug:
            action = self.traj[self.progress_ctr + 1]["act"]
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
        # convert to habitat conventions
        delta_wrist_trans_hab = (
            self.convert_position_conventions(delta_wrist_trans.cpu().numpy())
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
        self.behavior_action = action
