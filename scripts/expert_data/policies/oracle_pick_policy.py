import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

from habitat.utils.gum_utils import quat_to_axis_angle

third_party_ckpt_root_folder = (
    "/opt/hpcaas/.mounts/fs-03ee9f8c6dddfba21/jtruong/repos/gum_ws/src/GUM"
)
sys.path.append(third_party_ckpt_root_folder)
from gum.planning.grasp_planning.grasp_oracle import (
    SimpleApproachLiftOraclePolicy,
)

from scripts.expert_data.utils.utils import create_T_matrix


class OraclePickPolicy:
    def __init__(self, murp_env):
        self.murp_wrapper = murp_env
        self.env = murp_env.env
        self.config = murp_env.config

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device: ", self.device)
        self.grasp_cache_path = self.config.grasp_cache_path
        self.policy = self.load_checkpoint()
        self.data_cache = self.load_grasp_cache()
        self.reset()

    def load_checkpoint(self):
        cfg = OmegaConf.create(
            {
                "wait_duration": 0,
                "approach_duration": 50,
                "pre_grasp_duration": 50,
                "grasp_duration": 100,
                "add_episode_noise": False,
                "episode_length": 200,
                "num_envs": 1,
                "num_actions": 22,
                "device": self.device,
                "seed": 0,
            }
        )
        return SimpleApproachLiftOraclePolicy(cfg, device=self.device)

    def load_grasp_cache(self):
        data = torch.load(self.grasp_cache_path, map_location="cpu")
        data_cache = data["cache"]
        return data_cache

    def map_joints(self, joints, from_isaac=True):
        # map the joints from isaac convention to habitat convention
        # habitat convention is [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        # isaac convention is [ 0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7]
        # map the joints from habitat convention to isaac convention
        if from_isaac:
            joints_hab = np.zeros(16)
            joints_hab[:4] = joints[:4]
            joints_hab[4:8] = joints[12:]
            joints_hab[8:12] = joints[4:8]
            joints_hab[12:] = joints[8:12]
            return joints_hab
        else:
            joints_isaac = np.zeros(16)
            joints_isaac[:4] = joints[:4]
            joints_isaac[4:8] = joints[8:12]
            joints_isaac[8:12] = joints[12:]
            joints_isaac[12:] = joints[4:8]
            return joints_isaac

    def convert_position_conventions(self, position):
        x, y, z = position
        return np.array([-x, z, y])

    def reset(self):
        self.progress_ctr = 0
        self.obj_prim_path = (
            str(self.env.sim._rigid_objects[0]._prim)
            .split("<")[1]
            .split(">")[0]
        )

        self.target_fingers = self.map_joints(
            self.data_cache["048_@_1.0"]["robot_dof_pos"][0, :],
            from_isaac=True,
        )
        self.obs_dict = None

    def get_obs_dict(self, convention="hab"):
        # {
        #         "hand_trans":  the hand xyx position in world frame
        #         "hand_rot":   the hand quaternion in world frame
        #         "target_wrist":   this can be zeros shape (B, 7)
        #         "target_fingers":  the final target xyz where to put the fingers, this can be obtained from grasp cache
        #         "progress":  the time step
        #         "prev_targets":  the previous 22-dim targets that were given to the environment
        #     }
        obs_dict = {}
        obj_trans, obj_rot = self.env.sim.get_prim_transform(
            self.obj_prim_path, convention="quat"
        )
        if convention == "isaac":
            obj_trans = self.convert_position_conventions(obj_trans)
        obs_dict["object_trans"] = torch.tensor(obj_trans).unsqueeze(0)

        # quat in xyzw format
        curr_ee_pos, curr_ee_rot_quat = self.murp_wrapper.get_curr_ee_pose(
            convention="quat"
        )
        if convention == "isaac":
            curr_ee_pos = self.convert_position_conventions(curr_ee_pos)
        obs_dict["hand_trans"] = torch.tensor(curr_ee_pos).unsqueeze(0)
        obs_dict["hand_rot"] = torch.tensor(curr_ee_rot_quat).unsqueeze(0)

        wrist_axis_angle = quat_to_axis_angle(torch.tensor(curr_ee_rot_quat))
        # obs_dict["target_wrist"] = torch.zeros(7).unsqueeze(0)
        obs_dict["target_wrist"] = torch.cat(
            [torch.tensor(curr_ee_pos), wrist_axis_angle]
        ).unsqueeze(0)
        obs_dict["target_fingers"] = self.data_cache["048_@_1.0"][
            "robot_dof_pos"
        ][0, :].unsqueeze(0)
        obs_dict["progress"] = torch.tensor([self.progress_ctr])

        curr_hand_pos = self.murp_wrapper.get_curr_hand_pose()
        if convention == "isaac":
            self.map_joints(curr_hand_pos, from_isaac=True)

        self.prev_targets = np.concatenate(
            [curr_ee_pos, wrist_axis_angle.cpu().numpy(), curr_hand_pos]
        )

        obs_dict["prev_targets"] = torch.tensor(self.prev_targets).unsqueeze(0)

        for k, v in obs_dict.items():
            obs_dict[k] = v.float().to("cuda:0")

        self.obs_dict = obs_dict
        return obs_dict

    def step(self, action):

        wrist_scale = 0.2
        finger_scale = 0.1
        # delta_wrist_trans, delta_wrist_axis_angle, delta_fingers
        curr_ee_pos, curr_ee_rot_global = self.murp_wrapper.get_curr_ee_pose(
            convention="rpy"
        )
        curr_ee_pos_local, curr_ee_rot = self.murp_wrapper.get_curr_ee_pose(
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

        print("curr_ee_pos: ", curr_ee_pos)
        print(
            "delta_wrist_trans_hab: ",
            delta_wrist_trans_hab,
        )
        print("new_wrist_pos: ", new_wrist_pos)
        print("new_wrist_pos_v2: ", new_wrist_pos_v2)
        print("")
        print("curr_ee_rot: ", curr_ee_rot)
        print(
            "delta_wrist_rot: ",
            R.from_rotvec(delta_wrist_axis_angle.cpu().numpy()).as_euler(
                "xyz"
            ),
        )
        print("new_wrist_rot_rpy: ", new_wrist_rot_rpy)
        # breakpoint()

        # move the robot to the new wrist pose
        if self.progress_ctr != 0:
            self.murp_wrapper.move_ee_and_hand(
                new_wrist_pos_v2,
                self.open_loop_rot,
                curr_hand_pos,
                timeout=20,
                text="using oracle pick",
            )
        # self.murp_wrapper.visualize_pos(new_wrist_pos)
        # self.prev_targets = np.concatenate(
        #     [new_wrist_pos, new_wrist_axis_angle, new_fingers]
        # )
