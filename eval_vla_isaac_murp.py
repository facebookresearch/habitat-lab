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
from viz_utils import add_text_to_image
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


class VLAEvaluator:
    def __init__(self, ep_id, exp_name, ckpt, task, replay):
        self.exp_name = exp_name
        self.json_path = "data/datasets/fremont_hlab_isaac/fremont_100.json.gz"
        self.task = task

        # Define the agent configuration
        main_agent_config = AgentConfig()

        urdf_path = os.path.join(
            data_path,
            "murp/murp/platforms/franka_tmr/franka_description_tmr/urdf/franka_with_hand.urdf",
        )
        ik_arm_urdf_path = os.path.join(
            data_path,
            "murp/murp/platforms/franka_tmr/franka_description_tmr/urdf/franka_tmr_left_arm_only.urdf",
        )

        main_agent_config.articulated_agent_urdf = urdf_path
        main_agent_config.articulated_agent_type = "MurpRobot"
        main_agent_config.ik_arm_urdf = ik_arm_urdf_path

        # Define sensors that will be attached to this agent, here a third_rgb sensor and a head_rgb.
        # We will later talk about why we are giving the sensors these names
        main_agent_config.sim_sensors = {
            "third_rgb": ThirdRGBSensorConfig(),
            "articulated_agent_arm_rgb": ArmRGBSensorConfig(),
            "articulated_agent_arm_depth": ArmDepthSensorConfig(),
        }

        # We create a dictionary with names of agents and their corresponding agent configuration
        agent_dict = {"main_agent": main_agent_config}

        action_dict = {
            "base_velocity_action": BaseVelocityActionConfig(
                type="BaseVelIsaacAction"
                # type="BaseVelKinematicIsaacAction"
            ),
            "arm_reach_ee_action": ActionConfig(type="ArmReachEEAction"),
        }
        self.env = self.init_rearrange_env(agent_dict, action_dict)
        _ = self.env.reset()
        self.ep_id = ep_id

        data = read_dataset(self.json_path)
        self.episode_info = data["episodes"][self.ep_id]

        video_dir = f"videos_vla_eval/{self.exp_name}/ckpt_{ckpt}"
        os.makedirs(video_dir, exist_ok=True)
        if replay:
            self.video_path = f"{video_dir}/eval_{ep_id}_replay.mp4"
        else:
            self.video_path = f"{video_dir}/eval_{ep_id}.mp4"
        self.results_path = f"{video_dir}/results.txt"
        self.writer = imageio.get_writer(
            self.video_path,
            fps=30,
        )

        self.arm_rest_pose = np.zeros(7)

        vla_configs = CN()
        vla_configs.VLA_CKPT = f"/fsx-siro/jtruong/repos/robot-skills/results/{self.exp_name}/train_lr_5e-05_seed_42/checkpoint/step{ckpt}.pt"
        vla_configs.VLA_CONFIG_FILE = f"/fsx-siro/jtruong/repos/robot-skills/results/{self.exp_name}/vla_cfg.yaml"
        vla_configs.LANGUAGE_INSTRUCTION = "Open the refrigerator"
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
        hab_cfg.environment.max_episode_steps = 20000000
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

    def get_poses(self, name, pose_type="base"):
        poses = {
            "fridge": {
                "base_pos": np.array([-4.7, 0.0015, 0.8]),
                "base_rot": 180,
                "ee_pos": np.array([-6.2, 1.2, 2.4]),
                "ee_rot": np.deg2rad([120, 0, 0]),
            },
        }
        return (
            poses[name][f"{pose_type}_pos"],
            poses[name][f"{pose_type}_rot"],
        )

    def reset_robot_pose(self):
        start_position, start_rotation = self.get_poses(
            "fridge", pose_type="base"
        )

        position = mn.Vector3(start_position)
        rotation = mn.Quaternion.rotation(
            mn.Deg(start_rotation), mn.Vector3.y_axis()
        )
        self.base_trans = mn.Matrix4.from_(rotation.to_matrix(), position)
        self.env.sim.articulated_agent.base_transformation = self.base_trans

        # for _ in range(20):
        #     action = {
        #         "action": "base_velocity_action",
        #         "action_args": {
        #             "base_vel": np.array([0.0, 0.0], dtype=np.float32),
        #         },
        #     }
        #     self.env.sim.articulated_agent.base_transformation = (
        #         self.base_trans
        #     )
        #     obs = self.env.step(action)
        #     self.base_trans = (
        #         self.env.sim.articulated_agent.base_transformation
        #     )
        # self.move_hand("pre_grasp")

        print(f"set base to: {start_position}, {start_rotation}")

    def get_grasp_mode(self, name):
        # num_hand_joints = 10
        num_hand_joints = 16
        grasp_joints = {
            "open": np.zeros(num_hand_joints),
            "pre_grasp": np.concatenate(
                (np.full(12, 0.7), [-0.785], np.full(3, 0.7))
            ),
            "close": np.concatenate((np.full(12, 0.90), np.zeros(4))),
            "close_thumb": np.concatenate(
                (np.full(12, 0.90), np.full(4, 0.4))
            ),
            # "close": np.array([1.0] * num_hand_joints),
        }
        return grasp_joints[name]

    def get_arm_mode(self, name):
        arm_joints = {
            "rest": np.zeros(7),
            "side": np.array(
                [
                    2.6116285,
                    1.5283098,
                    1.0930868,
                    -0.50559217,
                    0.48147443,
                    2.628784,
                    -1.3962275,
                ]
            ),
        }
        return arm_joints[name]

    def pin_right_arm(self):
        self.env.sim.articulated_agent._robot_wrapper._target_right_arm_joint_positions = self.get_arm_mode(
            "rest"
        )
        self.env.sim.articulated_agent._robot_wrapper._target_right_hand_joint_positions = self.get_grasp_mode(
            "open"
        )

    def move_hand(self, mode="close", timeout=100):
        target_hand_pos = self.get_grasp_mode(mode)
        print(f"moving hand to: {mode}")
        self.move_hand_joints(target_hand_pos, timeout)

    def get_curr_hand_pose(self, arm="left"):
        if arm == "left":
            return self.env.sim.articulated_agent._robot_wrapper.hand_joint_pos
        elif arm == "right":
            return (
                self.env.sim.articulated_agent._robot_wrapper.right_hand_joint_pos
            )

    def get_curr_joint_pose(self, arm="left"):
        if arm == "left":
            return self.env.sim.articulated_agent._robot_wrapper.arm_joint_pos
        elif arm == "right":
            return (
                self.env.sim.articulated_agent._robot_wrapper.right_arm_joint_pos
            )

    def move_hand_joints(self, target_hand_pos, timeout=100):
        curr_hand_pos = self.get_curr_hand_pose()
        ctr = 0
        while not np.allclose(curr_hand_pos, target_hand_pos, atol=0.1):
            action = {
                "action": "base_velocity_action",
                "action_args": {
                    "base_vel": np.array([0.0, 0.0], dtype=np.float32)
                },
            }
            self.env.sim.articulated_agent.base_transformation = (
                self.base_trans
            )
            self.env.sim.articulated_agent._robot_wrapper._target_arm_joint_positions = (
                self.get_curr_joint_pose()
            )
            self.env.sim.articulated_agent._robot_wrapper._target_hand_joint_positions = (
                target_hand_pos
            )
            self.pin_right_arm()
            obs = self.env.step(action)
            im = self.process_obs_img(obs)
            self.writer.append_data(im)
            curr_hand_pos = self.get_curr_hand_pose()
            ctr += 1
            if ctr > timeout:
                print(
                    f"Exceeded time limit. {np.abs(curr_hand_pos-target_hand_pos)} away from target. Curr hand pos: {curr_hand_pos}"
                )
                break

    def move_base_ee_and_hand(
        self,
        base_lin_vel,
        base_ang_vel,
        target_arm_joints,
        target_hand_joints=None,
        timeout=300,
    ):
        ctr = 0
        curr_arm_joint_pos = (
            self.env.sim.articulated_agent._robot_wrapper.arm_joint_pos
        )
        while not np.allclose(
            curr_arm_joint_pos, target_arm_joints, atol=0.02
        ):
            action = {
                "action": "base_velocity_action",
                "action_args": {
                    "base_vel": np.array(
                        [base_lin_vel, base_ang_vel], dtype=np.float32
                    ),
                },
            }
            self.env.sim.articulated_agent.base_transformation = (
                self.base_trans
            )
            self.env.sim.articulated_agent._robot_wrapper._target_arm_joint_positions = (
                target_arm_joints
            )
            self.env.sim.articulated_agent._robot_wrapper._target_hand_joint_positions = (
                target_hand_joints
            )
            self.env.sim.articulated_agent._robot_wrapper._target_right_arm_joint_positions = self.get_arm_mode(
                "rest"
            )
            self.env.sim.articulated_agent._robot_wrapper._target_right_hand_joint_positions = self.get_grasp_mode(
                "open"
            )
            # self.pin_right_arm()

            obs = self.env.step(action)
            self.base_trans = (
                self.env.sim.articulated_agent.base_transformation
            )
            im = self.process_obs_img(obs)
            self.writer.append_data(im)

            curr_arm_joint_pos = (
                self.env.sim.articulated_agent._robot_wrapper.arm_joint_pos
            )
            ctr += 1
            if ctr > timeout:
                print(
                    f"Exceeded time limit. {np.abs(curr_arm_joint_pos-target_arm_joints)} away from target"
                )
                break
            self.writer.append_data(im)

    def replay_data(self):
        self.reset_robot_pose()

        episode_path = "/fsx-siro/jtruong/repos/vla-physics/habitat-lab/data/sim_robot_data/heuristic_expert_open/fremont_fridge_v1/train_npy/ep_0.npy"
        data = np.load(episode_path, allow_pickle=True)

        forward_velocity = 1.0
        turn_velocity = 1.0

        target_arm_joint_pos = (
            self.env.sim.articulated_agent._robot_wrapper.arm_joint_pos
        )
        target_hand_joint_pos = (
            self.env.sim.articulated_agent._robot_wrapper.hand_joint_pos
        )
        for i in range(len(data)):
            print("base_action: ", data[i]["delta_joint_action"][:2])
            print(
                "arm_action: ",
                np.round(data[i]["delta_joint_action"][2 : 2 + 7], 4),
            )
            print(
                "hand_action: ",
                np.round(data[i]["delta_joint_action"][2 + 7 :], 4),
            )
            target_base_action = data[i]["delta_joint_action"][:2]
            target_arm_joint_pos += data[i]["delta_joint_action"][2 : 2 + 7]
            target_hand_joint_pos += data[i]["delta_joint_action"][2 + 7 :]
            self.move_base_ee_and_hand(
                target_base_action[0],
                target_base_action[1],
                target_arm_joint_pos,
                target_hand_joint_pos,
                timeout=1,
            )
            curr_base_pos, curr_base_rot = (
                self.env.sim.articulated_agent._robot_wrapper.get_root_pose()
            )
            # target_arm_joint_pos = (
            #     self.env.sim.articulated_agent._robot_wrapper.arm_joint_pos
            # )
            # target_hand_joint_pos = (
            #     self.env.sim.articulated_agent._robot_wrapper.hand_joint_pos
            # )
            print("base_pos: ", curr_base_pos)

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
        self.pin_right_arm()

        obs = self.env.step(action_dict)
        obs["arm_rgb"] = obs["articulated_agent_arm_rgb"]
        curr_base_pos, curr_base_rot = (
            self.env.sim.articulated_agent._robot_wrapper.get_root_pose()
        )
        curr_arm_joints = (
            self.env.sim.articulated_agent._robot_wrapper.arm_joint_pos
        )
        curr_hand_pos = (
            self.env.sim.articulated_agent._robot_wrapper.hand_joint_pos
        )

        obs["joint"] = np.concatenate(
            [curr_base_pos, curr_arm_joints, curr_hand_pos]
        )
        done = False
        ctr = 0
        forward_velocity = 1.0
        turn_velocity = 1.0

        max_steps = 250
        print("max_steps: ", max_steps)
        grasp_ctr = 0
        episode_path = "/fsx-siro/jtruong/repos/vla-physics/habitat-lab/data/sim_robot_data/heuristic_expert_open/fremont_fridge_v1/train_npy/ep_0.npy"
        data = np.load(episode_path, allow_pickle=True)

        target_arm_pos = (
            self.env.sim.articulated_agent._robot_wrapper.arm_joint_pos
        )
        target_hand_pos = (
            self.env.sim.articulated_agent._robot_wrapper.hand_joint_pos
        )

        for i in range(max_steps):
            # print(f"=================== step # {ctr} ===================")
            obs["arm_rgb"] = data[i]["rgb"]
            obs["joint"] = np.concatenate(
                [
                    data[i]["base_pos"],
                    data[i]["arm_joint"],
                    data[i]["hand_joint"],
                ]
            )
            action = self.policy.act(obs)[0]
            action = data[i]["delta_joint_action"]
            print("action: ", action)
            print("GT action: ", data[i]["delta_joint_action"])

            print("obs joint: ", obs["joint"])
            print(
                "GT obs: ",
                np.concatenate(
                    [
                        data[i]["base_pos"],
                        data[i]["arm_joint"],
                        data[i]["hand_joint"],
                    ]
                ),
            )

            target_arm_pos += action[2 : 2 + 7]
            target_hand_pos += action[2 + 7 :]
            self.pin_right_arm()
            self.move_base_ee_and_hand(
                0.0,
                0.0,
                target_arm_pos,
                target_hand_pos,
                timeout=1,
            )

            obs["arm_rgb"] = obs["articulated_agent_arm_rgb"]
            curr_base_pos, curr_base_rot = (
                self.env.sim.articulated_agent._robot_wrapper.get_root_pose()
            )
            curr_arm_joints = (
                self.env.sim.articulated_agent._robot_wrapper.arm_joint_pos
            )
            curr_hand_pos = (
                self.env.sim.articulated_agent._robot_wrapper.hand_joint_pos
            )
            # target_arm_pos = curr_arm_joints
            # target_hand_pos = curr_hand_pos
            obs["joint"] = np.concatenate(
                [curr_base_pos, curr_arm_joints, curr_hand_pos]
            )
        self.writer.close()
        print("saved video at: ", self.video_path)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Your program description")

    # Add arguments
    parser.add_argument("--ep-id", type=int, default=0, help="Episode id")
    parser.add_argument("--exp", default="", help="Episode id")
    parser.add_argument("--ckpt", type=int, default=10000, help="Episode id")
    parser.add_argument("--task", default="task", help="Task")
    parser.add_argument("--replay", action="store_true")

    args = parser.parse_args()

    datagen = VLAEvaluator(
        args.ep_id, args.exp, args.ckpt, args.task, args.replay
    )
    if args.replay:
        datagen.replay_data()
    else:
        datagen.run()
