import sys

import numpy as np
import torch
from omegaconf import OmegaConf

third_party_ckpt_root_folder = (
    "/opt/hpcaas/.mounts/fs-03ee9f8c6dddfba21/jtruong/repos/gum_ws/src/GUM"
)
sys.path.append(third_party_ckpt_root_folder)
from gum.planning.grasp_planning.grasp_oracle import (
    SimpleApproachLiftOraclePolicy,
)


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
                "wait_duration": 25,
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

    def reset(self):
        self.progress_ctr = 0
        self.obj_prim_path = (
            str(self.env.sim._rigid_objects[0]._prim)
            .split("<")[1]
            .split(">")[0]
        )
        self.prev_targets = torch.zeros(22).unsqueeze(0)

    def get_obs_dict(self):
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
        obj_trans = torch.tensor(obj_trans).unsqueeze(0)
        obs_dict["object_trans"] = obj_trans
        hand_pos, hand_rot = (
            self.env.sim.articulated_agent._robot_wrapper.hand_pose()
        )
        hand_rot_npy = np.array([hand_rot.scalar, *hand_rot.vector])
        obs_dict["hand_trans"] = torch.tensor(hand_pos).unsqueeze(0)
        obs_dict["hand_rot"] = torch.tensor(hand_rot_npy).unsqueeze(0)

        curr_ee_pos_vec, curr_ee_rot = (
            self.env.sim.articulated_agent._robot_wrapper.ee_right_pose()
        )
        curr_ee_pos = torch.tensor([*curr_ee_pos_vec]).unsqueeze(0)
        curr_ee_rot_quat = torch.tensor(
            [curr_ee_rot.scalar, *curr_ee_rot.vector]
        ).unsqueeze(0)

        # obs_dict["target_wrist"] = torch.zeros(7).unsqueeze(0)
        obs_dict["target_wrist"] = torch.cat(
            [curr_ee_pos, curr_ee_rot_quat], dim=1
        )
        obs_dict["target_fingers"] = self.data_cache["048_@_1.0"][
            "robot_dof_pos"
        ][0, :].unsqueeze(0)
        obs_dict["progress"] = torch.tensor(self.progress_ctr)
        obs_dict["prev_targets"] = self.prev_targets
        for k, v in obs_dict.items():
            obs_dict[k] = v.float().to("cuda:0")
        return obs_dict
