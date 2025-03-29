import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

from scripts.expert_data.envs.murp_env import MurpEnv
from scripts.expert_data.utils.utils import import_fn


def init_arm_and_hand(murp_env, policy_env):
    target = murp_env.env.sim._rigid_objects[0].transformation
    rotation_matrix = np.array(
        [
            [target[0].x, target[0].y, target[0].z],
            [target[1].x, target[1].y, target[1].z],
            [target[2].x, target[2].y, target[2].z],
        ]
    )
    rpy = R.from_matrix(rotation_matrix).as_euler("xyz", degrees=True)
    ee_rot = rpy + np.array([0, 90, 0])
    print("target_fingers: ", policy_env.target_fingers)
    murp_env.move_ee_and_hand(
        murp_env.env.sim._rigid_objects[0].translation,
        ee_rot,
        policy_env.target_fingers,
        timeout=300,
        text="using arm controller",
    )


def main(config):
    config = OmegaConf.load(config)
    murp_env = MurpEnv(config)

    if config.policy_cls == "OraclePickPolicy":
        from scripts.expert_data.policies.oracle_pick_policy import (
            OraclePickPolicy,
        )
    elif config.policy_cls == "RLPickPolicy":
        from scripts.expert_data.policies.rl_pick_policy import RLPickPolicy
    policy_env = eval(config.policy_cls)(murp_env)

    hand = config.hand

    murp_env.reset_robot(murp_env.env.current_episode.action_target[0])

    # arm control
    init_arm_and_hand(murp_env, policy_env)
    # grasp control
    for i in range(100):
        obs_dict = policy_env.get_obs_dict(convention="isaac")
        action = policy_env.policy.act(obs_dict)
        policy_env.step(action)
        print("action: ", action)
        policy_env.progress_ctr += 1
        # policy_env.prev_targets = action
    print("saved video to: ", murp_env.save_path)


if __name__ == "__main__":
    config = "scripts/expert_data/configs/config.yaml"
    main(config)
