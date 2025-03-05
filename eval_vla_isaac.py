import warnings

import magnum as mn

import habitat_sim
from habitat.tasks.rearrange.isaac_rearrange_sim import IsaacRearrangeSim

warnings.filterwarnings("ignore")
import argparse
import gzip
import json
import os
import random
import time

import cv2
import imageio
import numpy as np
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from yacs.config import CfgNode as CN

import habitat
from habitat.articulated_agents.robots import FetchRobot
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    ActionConfig,
    AgentConfig,
    ArmActionConfig,
    ArmDepthSensorConfig,
    ArmRGBSensorConfig,
    BaseVelocityActionConfig,
    DatasetConfig,
    EnvironmentConfig,
    HabitatConfig,
    HabitatSimV0Config,
    HeadPanopticSensorConfig,
    HeadRGBSensorConfig,
    OracleNavActionConfig,
    SimulatorConfig,
    TaskConfig,
    ThirdRGBSensorConfig,
)
from habitat.core.env import Env
from habitat.isaac_sim import actions, isaac_prim_utils
from habitat.isaac_sim.isaac_app_wrapper import IsaacAppWrapper
from habitat_sim.physics import JointMotorSettings, MotionType
from habitat_sim.utils import viz_utils as vut
from habitat_sim.utils.settings import make_cfg
from vla_policy import VLAPolicy

data_path = "/opt/hpcaas/.mounts/fs-03ee9f8c6dddfba21/jtruong/data"


def read_dataset(dataset_file):
    if dataset_file.endswith(".json.gz"):
        with gzip.open(dataset_file, "rt", encoding="utf-8") as f:
            data = json.load(f)
    elif dataset_file.endswith(".json"):
        with open(dataset_file, "r") as file:
            data = json.load(file)
    return data


from habitat.datasets.rearrange.navmesh_utils import compute_turn
from habitat.tasks.utils import get_angle


class OracleNavSkill:
    def __init__(self, env, target_pos):
        self.env = env
        self.target_pos = target_pos
        self.target_base_pos = target_pos
        self.dist_thresh = 0.5
        self.turn_velocity = 2

        self.forward_velocity = 8
        self.turn_thresh = 0.2
        self.articulated_agent = self.env.sim.articulated_agent

    def _path_to_point(self, point):
        """
        Obtain path to reach the coordinate point. If agent_pos is not given
        the path starts at the agent base pos, otherwise it starts at the agent_pos
        value
        :param point: Vector3 indicating the target point
        """
        agent_pos = self.articulated_agent.base_pos

        path = habitat_sim.ShortestPath()
        path.requested_start = agent_pos
        path.requested_end = point
        found_path = self.env.sim.pathfinder.find_path(path)
        if not found_path:
            return [agent_pos, point]
        return path.points

    def get_step(self):

        obj_targ_pos = np.array(self.target_pos)
        base_T = self.articulated_agent.base_transformation

        curr_path_points = self._path_to_point(self.target_base_pos)
        robot_pos = np.array(self.articulated_agent.base_pos)
        if len(curr_path_points) == 1:
            curr_path_points += curr_path_points

        cur_nav_targ = np.array(curr_path_points[1])
        forward = np.array([1.0, 0, 0])
        robot_forward = np.array(base_T.transform_vector(forward))

        # Compute relative target
        rel_targ = cur_nav_targ - robot_pos
        # Compute heading angle (2D calculation)
        robot_forward = robot_forward[[0, 2]]
        rel_targ = rel_targ[[0, 2]]
        rel_pos = (obj_targ_pos - robot_pos)[[0, 2]]
        dist_to_final_nav_targ = np.linalg.norm(
            (np.array(self.target_base_pos) - robot_pos)[[0, 2]],
        )
        angle_to_target = get_angle(robot_forward, rel_targ)
        angle_to_obj = get_angle(robot_forward, rel_pos)

        # Compute the distance
        at_goal = (
            dist_to_final_nav_targ < self.dist_thresh
            and angle_to_obj < self.turn_thresh
        )

        # Planning to see if the robot needs to do back-up

        if not at_goal:
            if dist_to_final_nav_targ < self.dist_thresh:
                # TODO: this does not account for the sampled pose's final rotation
                # Look at the object target position when getting close
                vel = compute_turn(
                    rel_pos,
                    self.turn_velocity,
                    robot_forward,
                )
            elif angle_to_target < self.turn_thresh:
                # Move forward towards the target
                vel = [self.forward_velocity, 0]
            else:
                # Look at the target waypoint
                vel = compute_turn(
                    rel_targ,
                    self.turn_velocity,
                    robot_forward,
                )
        else:
            vel = [0, 0]
        vel2 = compute_turn(
            rel_targ,
            self.turn_velocity,
            robot_forward,
        )
        vel2 = vel

        # print(vel, dist_to_final_nav_targ, angle_to_obj, angle_to_target)
        action = {
            "action": "base_velocity_action",
            "action_args": {
                "base_vel": np.array([vel[0], vel[1]], dtype=np.float32)
            },
        }
        return action


class VLAEvaluator:
    def __init__(self, ep_id, exp_name, ckpt):
        self.exp_name = exp_name
        self.json_path = "data/datasets/fremont_hlab_isaac/fremont_900.json.gz"

        # Define the agent configuration
        main_agent_config = AgentConfig()

        urdf_path = os.path.join(
            data_path, "robots/hab_spot_arm/urdf/hab_spot_arm.urdf"
        )
        ik_arm_urdf_path = os.path.join(
            data_path, "robots/hab_spot_arm/urdf/spot_onlyarm.urdf"
        )

        main_agent_config.articulated_agent_urdf = urdf_path
        main_agent_config.articulated_agent_type = "SpotRobot"
        main_agent_config.ik_arm_urdf = ik_arm_urdf_path

        # Define sensors that will be attached to this agent, here a third_rgb sensor and a head_rgb.
        # We will later talk about why we are giving the sensors these names
        main_agent_config.sim_sensors = {
            "third_rgb": ThirdRGBSensorConfig(),
            "articulated_agent_arm_rgb": ArmRGBSensorConfig(),
            "articulated_agent_arm_depth": ArmDepthSensorConfig(max_depth=3.0),
        }

        # We create a dictionary with names of agents and their corresponding agent configuration
        agent_dict = {"main_agent": main_agent_config}

        action_dict = {
            "base_velocity_action": BaseVelocityActionConfig(
                type="BaseVelIsaacAction"
            ),
            "arm_reach_action": ActionConfig(type="ArmReachAction"),
            "arm_reach_ee_action": ActionConfig(type="ArmReachEEAction"),
        }
        self.env = self.init_rearrange_env(agent_dict, action_dict)
        _ = self.env.reset()
        self.ep_id = ep_id

        data = read_dataset(self.json_path)
        self.episode_info = data["episodes"][self.ep_id]

        video_dir = f"videos_vla_eval/{self.exp_name}/ckpt_{ckpt}"
        os.makedirs(video_dir, exist_ok=True)
        self.video_path = f"{video_dir}/eval_{ep_id}.mp4"
        self.writer = imageio.get_writer(
            self.video_path,
            fps=30,
        )

        self.arm_rest_pose = [
            0.0,
            -2.0943951,
            0.0,
            1.04719755,
            0.0,
            1.53588974,
            0.0,
            -1.57,
        ]

        vla_configs = CN()
        vla_configs.VLA_CKPT = f"/fsx-siro/jtruong/repos/robot-skills/results/{self.exp_name}/train_lr_5e-05_seed_42/checkpoint/step{ckpt}.pt"
        vla_configs.VLA_CONFIG_FILE = f"/fsx-siro/jtruong/repos/robot-skills/results/{self.exp_name}/vla_cfg.yaml"
        vla_configs.LANGUAGE_INSTRUCTION = (
            "Navigate to the dresser and pick up the avocado plush toy"
        )
        self.policy = VLAPolicy(vla_configs, "cuda:0")

    def make_sim_cfg(self, agent_dict):
        # Start the scene config
        sim_cfg = SimulatorConfig(type="IsaacRearrangeSim-v0")

        # This is for better graphics
        sim_cfg.habitat_sim_v0.enable_hbao = True
        sim_cfg.habitat_sim_v0.enable_physics = False

        # TODO: disable this, causes performance issues
        sim_cfg.habitat_sim_v0.frustum_culling = False

        # Set up an example scene
        sim_cfg.scene = "NONE"  #
        # sim_cfg.scene = os.path.join(data_path, "hab3_bench_assets/hab3-hssd/scenes/103997919_171031233.scene_instance.json")

        sim_cfg.ctrl_freq = 180.0  # match isaac freq
        cfg = OmegaConf.create(sim_cfg)

        # Set the scene agents
        cfg.agents = agent_dict
        cfg.agents_order = list(cfg.agents.keys())
        return cfg

    def make_hab_cfg(self, agent_dict, action_dict):
        sim_cfg = self.make_sim_cfg(agent_dict)
        task_cfg = TaskConfig(type="RearrangeEmptyTask-v0")
        task_cfg.actions = action_dict
        env_cfg = EnvironmentConfig()
        dataset_cfg = DatasetConfig(
            type="RearrangeDataset-v0",
            data_path=self.json_path,  # "data/hab3_bench_assets/episode_datasets/small_large.json.gz",
        )

        hab_cfg = HabitatConfig()
        hab_cfg.environment = env_cfg
        hab_cfg.task = task_cfg

        hab_cfg.dataset = dataset_cfg
        hab_cfg.simulator = sim_cfg
        hab_cfg.simulator.seed = hab_cfg.seed
        hab_cfg.environment.max_episode_steps = 2000
        hab_cfg.environment.max_episode_seconds = 20000000
        return hab_cfg

    def init_rearrange_env(self, agent_dict, action_dict):
        hab_cfg = self.make_hab_cfg(agent_dict, action_dict)
        res_cfg = OmegaConf.create(hab_cfg)
        print("hab_cfg: ", hab_cfg)
        print("res_cfg: ", res_cfg)
        return Env(res_cfg)

    def process_obs_img(self, obs):
        im = obs["third_rgb"]
        im2 = obs["articulated_agent_arm_rgb"]
        im3 = (255 * obs["articulated_agent_arm_depth"]).astype(np.uint8)
        imt = np.zeros(im.shape, dtype=np.uint8)
        imt[: im2.shape[0], : im2.shape[1], :] = im2
        imt[im2.shape[0] :, : im2.shape[1], 0] = im3[:, :, 0]
        imt[im2.shape[0] :, : im2.shape[1], 1] = im3[:, :, 0]
        imt[im2.shape[0] :, : im2.shape[1], 2] = im3[:, :, 0]
        im = np.concatenate([im, imt], 1)
        return im

    def visualize_pos(self, target_pos, r=0.05):
        self.env.sim.viz_ids["target"] = self.env.sim.visualize_position(
            target_pos, self.env.sim.viz_ids["target"], r=r
        )

    def move_to_joint(
        self, target_joint_pos, timeout=200, visualize=False, save_info=True
    ):
        ctr = 0
        curr_joint_pos = (
            self.env.sim.articulated_agent._robot_wrapper.arm_joint_pos
        )

        while not np.allclose(
            curr_joint_pos,
            target_joint_pos,
            atol=0.003,
        ):
            self.env.sim.articulated_agent._robot_wrapper._target_arm_joint_positions = (
                target_joint_pos
            )
            action = {
                "action": "base_velocity_action",
                "action_args": {
                    "base_vel": np.array([0.0, 0.0], dtype=np.float32)
                },
            }
            ctr += 1
            obs = self.env.step(action)
            if ctr > timeout:
                break

    def reset_robot_pose(self):
        start_position = self.episode_info["start_position"]
        start_rotation = np.rad2deg(self.episode_info["start_rotation"][-1])
        # start_position = np.array([2.0, 0.7, -1.64570129])
        # start_rotation = -90

        position = mn.Vector3(start_position)
        rotation = mn.Quaternion.rotation(
            mn.Deg(start_rotation), mn.Vector3.y_axis()
        )
        trans = mn.Matrix4.from_(rotation.to_matrix(), position)
        self.env.sim.articulated_agent.base_transformation = trans
        print("set base: ", start_position, start_rotation)
        self.move_to_joint(
            self.arm_rest_pose, visualize=False, save_info=False
        )

    def replay_data(self):
        self.reset_robot_pose()

        episode_path = "/fsx-siro/jtruong/data/sim_robot_data/heuristic_expert/nav_pick_fremont_physics_v2/fremont/test_npy/ep_0.npy"
        data = np.load(episode_path, allow_pickle=True)

        forward_velocity = 8.0
        turn_velocity = 2.0

        target_joint_pos = (
            self.env.sim.articulated_agent._robot_wrapper.arm_joint_pos
        )
        for i in range(len(data)):
            action = data[i]["joint_action"]
            joint_action = data[i]["joint_action"][:7]
            # print("data action: ", action)
            action_mask = np.array([1, 1, 0, 1, 0, 1, 0])
            masked_joint_action = joint_action[action_mask == 1]
            target_joint_pos[0] += masked_joint_action[0]
            target_joint_pos[1] += masked_joint_action[1]
            target_joint_pos[3] += masked_joint_action[2]
            target_joint_pos[4] += joint_action[4]
            target_joint_pos[5] += masked_joint_action[3]
            # target_joint_pos[6] += joint_action[6]
            # target_joint_pos[:7] += data[i]["joint_action"][:7]
            target_joint_pos[-1] = 0 if action[-1] > 0 else -1.57
            target_base_action = [
                action[7] * forward_velocity,
                action[8] * turn_velocity,
            ]
            print("base_velocity_action: ", target_base_action)

            self.env.sim.articulated_agent._robot_wrapper._target_arm_joint_positions = (
                target_joint_pos
            )
            action_dict = {
                "action": "base_velocity_action",
                "action_args": {
                    "base_vel": np.array(
                        target_base_action,
                        dtype=np.float32,
                    )
                },
            }
            obs = self.env.step(action_dict)
            im = self.process_obs_img(obs)
            self.writer.append_data(im)

        self.writer.close()
        print("saved video at: ", self.video_path)
        return

    def run(self):
        self.reset_robot_pose()

        self.policy.reset()

        action_dict = {
            "action": "base_velocity_action",
            "action_args": {
                "base_vel": np.array([0.0, 0.0], dtype=np.float32)
            },
        }
        obs = self.env.step(action_dict)
        obs["arm_rgb"] = obs["articulated_agent_arm_rgb"]
        curr_arm_joints = (
            self.env.sim.articulated_agent._robot_wrapper.arm_joint_pos
        )
        # mask = np.array([1, 1, 1, 0, 1, 0, 0, 0])
        mask = np.array([1, 1, 0, 1, 0, 1, 0, 0])
        print("curr_arm_joints: ", curr_arm_joints)
        obs["joint"] = curr_arm_joints[mask == 1]
        print("obs: ", obs.keys())
        done = False
        ctr = 0
        forward_velocity = 8.0
        turn_velocity = 2.0

        max_steps = 250
        print("max_steps: ", max_steps)
        target_joint_pos = (
            self.env.sim.articulated_agent._robot_wrapper.arm_joint_pos
        )
        for i in range(max_steps):
            # print(f"=================== step # {ctr} ===================")
            action = self.policy.act(obs)[0]

            # n_repeat = 6 if "30" in self.exp_name else 1
            # print("action: ", action)
            target_joint_pos = (
                self.env.sim.articulated_agent._robot_wrapper.arm_joint_pos
            )
            target_joint_pos[0] = action[0]
            target_joint_pos[1] = action[1]
            target_joint_pos[2] = action[2]
            target_joint_pos[4] = action[3]
            target_joint_pos[-1] = 0 if action[-1] > 0 else -1.57
            target_base_action = [
                action[4] * forward_velocity,
                action[5] * turn_velocity,
            ]

            # action_mask = np.array([1, 1, 0, 1, 0, 1, 0])
            # masked_joint_action = joint_action[action_mask == 1]

            # target_joint_pos[:7] = data[i]["joint"][:7]
            # target_joint_pos[:7] += data[i]["joint_action"][:7]

            # target_joint_pos[0] += masked_joint_action[0]
            # target_joint_pos[1] += masked_joint_action[1]
            # target_joint_pos[3] += masked_joint_action[2]
            # target_joint_pos[5] += masked_joint_action[3]
            # target_joint_pos[-1] = 0 if action[-1] > 0 else -1.57
            # target_base_action = [
            #     action[7] * forward_velocity,
            #     action[8] * turn_velocity,
            # ]

            self.env.sim.articulated_agent._robot_wrapper._target_arm_joint_positions = (
                target_joint_pos
            )
            action_dict = {
                "action": "base_velocity_action",
                "action_args": {
                    "base_vel": np.array(
                        target_base_action,
                        dtype=np.float32,
                    )
                },
            }
            obs = self.env.step(action_dict)
            im = self.process_obs_img(obs)
            self.writer.append_data(im)

            obs["arm_rgb"] = obs["articulated_agent_arm_rgb"]
            curr_arm_joints = (
                self.env.sim.articulated_agent._robot_wrapper.arm_joint_pos
            )

            obs["joint"] = curr_arm_joints[mask == 1]
        self.writer.close()
        print("saved video at: ", self.video_path)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Your program description")

    # Add arguments
    parser.add_argument("--ep-id", type=int, default=0, help="Episode id")
    parser.add_argument("--exp", default="", help="Episode id")
    parser.add_argument("--ckpt", type=int, default=10000, help="Episode id")
    parser.add_argument("--replay", action="store_true")

    args = parser.parse_args()

    datagen = VLAEvaluator(args.ep_id, args.exp, args.ckpt)
    if args.replay:
        datagen.replay_data()
    else:
        datagen.run()
