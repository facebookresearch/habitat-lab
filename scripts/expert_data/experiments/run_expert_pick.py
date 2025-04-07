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
    # ee_rot_rad = np.array([1.70754709e-05, 9.00000171e01, -8.99999829e01])
    ee_rot = np.array([0, 90, 0])
    ee_rot_rad = np.deg2rad(ee_rot)
    pick_location = murp_env.env.sim._rigid_objects[0].translation
    pick_location[1] += 0.22  # add z offset
    # murp_env.visualize_pos(pick_location)
    murp_env.env.sim.articulated_agent._robot_wrapper.teleport = True
    murp_env.move_ee_and_hand(
        pick_location,
        ee_rot_rad,
        policy_env.target_fingers,
        timeout=1,
        text="using arm controller",
    )
    _, curr_ee_rot = murp_env.get_curr_ee_pose(
        convention="rpy", use_global=False
    )
    print("STARTING EE ROTATION: ", curr_ee_rot)


def lift_arm_and_hand(murp_env, policy_env):
    curr_ee_pos, _ = murp_env.get_curr_ee_pose(convention="rpy")
    _, curr_ee_rot = murp_env.get_curr_ee_pose(
        convention="rpy", use_global=False
    )
    curr_hand_pos = murp_env.get_curr_hand_pose()
    target_ee_pos = curr_ee_pos.copy()
    target_ee_pos[1] += 0.3
    murp_env.move_ee_and_hand(
        target_ee_pos,
        policy_env.open_loop_rot,
        curr_hand_pos,
        timeout=10,
        text="using arm controller",
    )


def heurestic_step(murp_env, config, policy):
    hand = config.hand

    murp_env.reset_robot(config.target_name)

    grip_iters, open_iters, move_iters, _ = policy.TARGET_CONFIG[
        config.target_name
    ]
    if config.hand == "left":
        move_iters = None
    policy.execute_grasp_sequence(
        config.hand, grip_iters, open_iters, move_iters
    )
    print("saved video to: ", murp_env.save_path)


def main(config):
    config = OmegaConf.load(config)
    murp_env = MurpEnv(config)

    if config.policy_cls == "OraclePickPolicy":
        from scripts.expert_data.policies.oracle_pick_policy import (
            OraclePickPolicy,
        )
    elif config.policy_cls == "RLPickPolicy":
        from scripts.expert_data.policies.rl_pick_policy import RLPickPolicy
    elif config.policy_cls == "HeuristicPickPolicy":
        from scripts.expert_data.policies.heurestic_pick_policy import (
            HeuristicPickPolicy,
        )

    policy_env = eval(config.policy_cls)(murp_env)
    if config.policy_cls == "HeuristicPickPolicy":
        heurestic_step(murp_env, config, policy_env)
    else:
        murp_env.reset_robot(murp_env.env.current_episode.action_target[0])
        # arm control
        init_arm_and_hand(murp_env, policy_env)
        # grasp control
        murp_env.env.sim.articulated_agent._robot_wrapper.teleport = False

        max_steps = 99
        if policy_env.debug:
            max_steps = len(policy_env.traj)
        for i in range(99):
            obs_dict = policy_env.get_obs_dict(convention="isaac")
            action = policy_env.policy.act(obs_dict)
            policy_env.step(action)
            # print("action: ", action)
            policy_env.progress_ctr += 1
            # policy_env.prev_targets = action
        lift_arm_and_hand(murp_env, policy_env)
        print("saved video to: ", murp_env.save_path)


if __name__ == "__main__":
    config = "scripts/expert_data/configs/config.yaml"
    main(config)
