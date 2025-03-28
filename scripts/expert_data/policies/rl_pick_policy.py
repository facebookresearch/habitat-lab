import sys

import numpy as np
import torch

from habitat.utils.gum_utils import (
    sample_point_cloud_from_urdf,
    to_world_frame,
)

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

        self.reset()
        self.checkpoint_path = self.config.pick_ckpt_path
        self.policy = self.load_checkpoint()

    def load_checkpoint(self):
        return PPO.from_checkpoint(self.checkpoint_path)

    def reset(self):
        self.obj_prim_path = (
            str(self.env.sim._rigid_objects[0]._prim)
            .split("<")[1]
            .split(">")[0]
        )
        self.object_name = self.env.current_episode.rigid_objs[0][0]

        pc, normals = sample_point_cloud_from_urdf(
            os.path.abspath("data/assets"),
            f"dexgraspnet2/meshdata/{object_name}/simplified_sdf.urdf",
            100,
            seed=4,
        )
        self.gt_object_point_clouds__object = torch.tensor(pc).unsqueeze(0)
        self.private_info = {}

        obj_trans, obj_rot = self.env.sim.get_prim_transform(
            self.obj_prim_path, convention="quat"
        )

        self.private_info["object_trans"] = obj_trans.flatten()
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
        self.private_info["object_rot"] = obj_rot.flatten()
        self.private_info["object_angvel"] = torch.tensor(
            self.env.sim._rigid_objects[0]._rigid_prim.get_angular_velocity()
        )

        ee_poses, ee_rots = (
            self.env.sim.articulated_agent._robot_wrapper.fingertip_right_pose()
        )
        self.private_info["fingertip_trans"] = torch.tensor(ee_poses)
        self.private_info["object_restitution"] = torch.tensor(
            [np.random.uniform(0, 1.0)]
        )
        self.obs_buf_lag_history = torch.zeros(
            22 * 2 * 3
        )  # joints = 22-dim, target = 22-dim, history of 3
        self.target_obs = torch.zeros(22)

    def get_obs_dict(self):
        obs_dict = {}
        obj_trans, obj_rot = self.env.sim.get_prim_transform(
            self.obj_prim_path, convention="quat"
        )
        obj_trans = torch.tensor(obj_trans).unsqueeze(0)
        obj_rot = torch.tensor(obj_rot).unsqueeze(0)

        object_pc__world = to_world_frame(
            self.gt_object_point_clouds__object, obj_rot, obj_trans
        )
        obs_dict["gt_object_point_cloud"] = object_pc__world
        self.private_info["object_trans"] = obj_trans.flatten()
        self.private_info["object_rot"] = obj_rot.flatten()
        ee_poses, ee_rots = (
            self.env.sim.articulated_agent._robot_wrapper.fingertip_right_pose()
        )
        self.private_info["fingertip_trans"] = torch.tensor(ee_poses)

        obs_dict["priv_info"] = torch.cat(
            list(self.private_info.values())
        ).unsqueeze(0)
        clip_obs = 5.0
        prev_obs_buf = self.obs_buf_lag_history[44:]
        joints_obs = torch.tensor(
            self.env.sim.articulated_agent._robot_wrapper.right_hand_joint_pos
        )
        curr_ee_pos, curr_ee_rot_rpy = self.get_curr_ee_pose()
        # convert to torch and concatenate
        wrist_pose = torch.cat(
            [torch.tensor(curr_ee_pos), torch.tensor(curr_ee_rot_rpy)]
        )
        cur_obs_buf = torch.cat([joints_obs, wrist_pose, self.target_obs])
        obs_buf = torch.cat([prev_obs_buf, cur_obs_buf])
        obs_dict["obs"] = torch.clamp(obs_buf, -clip_obs, clip_obs)
        # add current observation to obs_buf_lag_history
        self.obs_buf_lag_history = torch.cat(
            [prev_obs_buf, cur_obs_buf]
        ).unsqueeze(0)

        for k, v in obs_dict.items():
            obs_dict[k] = v.to("cuda:0")

        return obs_dict
