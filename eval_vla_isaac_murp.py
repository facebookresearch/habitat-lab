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
    def __init__(
        self,
        ep_id,
        exp_name,
        ckpt,
        task,
        grasp_mode,
        replay,
        debug,
        action_type,
        action_space,
    ):
        np.set_printoptions(precision=4, suppress=True)
        self.exp_name = exp_name
        self.json_path = "data/datasets/fremont_hlab_isaac/fremont_100.json.gz"
        self.task = task
        self.debug = debug
        self.arm_type = "left"
        self.grasp_mode = grasp_mode
        self.action_type = action_type
        self.action_space = action_space

        # Define the agent configuration
        main_agent_config = AgentConfig()

        urdf_path = os.path.join(
            data_path,
            "murp/murp/platforms/franka_tmr/franka_description_tmr/urdf/franka_with_hand.urdf",
        )
        if self.arm_type == "left":
            ik_arm_urdf_path = os.path.join(
                data_path,
                "murp/murp/platforms/franka_tmr/franka_description_tmr/urdf/franka_tmr_left_arm_only.urdf",
            )
        else:
            ik_arm_urdf_path = os.path.join(
                data_path,
                "murp/murp/platforms/franka_tmr/franka_description_tmr/urdf/franka_tmr_right_arm_only.urdf",
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

        video_dir = f"videos_vla_eval/{self.exp_name}"
        os.makedirs(video_dir, exist_ok=True)
        base_video_path = f"{self.action_type}_eval_{ep_id}_ckpt_{ckpt}"
        if replay:
            self.video_path = f"{video_dir}/replay_{base_video_path}.mp4"
        if debug:
            self.video_path = f"{video_dir}/debug_{base_video_path}.mp4"
        else:
            self.video_path = f"{video_dir}/{base_video_path}.mp4"
        self.results_path = f"{video_dir}/results.txt"
        self.writer = imageio.get_writer(
            self.video_path,
            fps=30,
        )

        self.arm_rest_pose = np.zeros(7)

        vla_configs = CN()
        vla_configs.VLA_CKPT = f"/fsx-siro/jtruong/repos/robot-skills/results/{self.exp_name}/train_lr_5e-05_seed_42/checkpoint/step{ckpt}.pt"
        vla_configs.VLA_CONFIG_FILE = f"/fsx-siro/jtruong/repos/robot-skills/results/{self.exp_name}/vla_cfg.yaml"
        vla_configs.LANGUAGE_INSTRUCTION = "Open the fridge"
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
                "base_pos": np.array([-4.7, 0.1, 0.8]),
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
            "close_thumb": np.concatenate((np.full(12, 0.90), np.zeros(4))),
            "close": np.concatenate((np.full(12, 0.90), np.full(4, 0.4))),
            # "close": np.array([1.0] * num_hand_joints),
        }
        return grasp_joints[name]

    def get_grasp_mode_idx(self, name):
        grasp_joints = {
            "open": np.array([0]),
            "pre_grasp": np.array([0]),
            "close": np.array([1]),
            "close_thumb": np.array([1]),
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

    def pin_left_arm(self):
        self.env.sim.articulated_agent._robot_wrapper._target_arm_joint_positions = self.get_arm_mode(
            "rest"
        )
        self.env.sim.articulated_agent._robot_wrapper._target_hand_joint_positions = self.get_grasp_mode(
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
        if self.arm_type == "left":
            curr_arm_joint_pos = (
                self.env.sim.articulated_agent._robot_wrapper.arm_joint_pos
            )
        elif self.arm_type == "right":
            curr_arm_joint_pos = (
                self.env.sim.articulated_agent._robot_wrapper.right_arm_joint_pos
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
            if self.arm_type == "left":
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
            elif self.arm_type == "right":
                self.env.sim.articulated_agent._robot_wrapper._target_right_arm_joint_positions = (
                    target_arm_joints
                )
                self.env.sim.articulated_agent._robot_wrapper._target_right_hand_joint_positions = (
                    target_hand_joints
                )
                self.env.sim.articulated_agent._robot_wrapper._target_arm_joint_positions = self.get_arm_mode(
                    "rest"
                )
                self.env.sim.articulated_agent._robot_wrapper._target_hand_joint_positions = self.get_grasp_mode(
                    "open"
                )

            # self.pin_right_arm()

            obs = self.env.step(action)
            self.base_trans = (
                self.env.sim.articulated_agent.base_transformation
            )
            im = self.process_obs_img(obs)
            self.writer.append_data(im)

            if self.arm_type == "left":
                curr_arm_joint_pos = (
                    self.env.sim.articulated_agent._robot_wrapper.arm_joint_pos
                )
            elif self.arm_type == "right":
                curr_arm_joint_pos = (
                    self.env.sim.articulated_agent._robot_wrapper.right_arm_joint_pos
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

        episode_path = "/fsx-siro/jtruong/repos/vla-physics/habitat-lab/data/sim_robot_data/heuristic_expert_open/fremont_fridge_base_arm_hand_v1/train_npy/ep_0.npy"
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
        print("saved video at: ", os.path.abspath(self.video_path))
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
        if self.action_space == "base_arm_hand":
            if self.grasp_mode:
                curr_grasp_mode = np.array([0])
                obs["joint"] = np.concatenate(
                    [curr_base_pos, curr_arm_joints, curr_grasp_mode]
                )
            else:
                obs["joint"] = np.concatenate(
                    [curr_base_pos, curr_arm_joints, curr_hand_pos]
                )
        elif self.action_space == "arm":
            obs["joint"] = np.array(curr_arm_joints)

        done = False
        ctr = 0
        forward_velocity = 1.0
        turn_velocity = 1.0

        max_steps = 500
        grasp_ctr = 0
        episode_path = "/fsx-siro/jtruong/repos/vla-physics/habitat-lab/data/sim_robot_data/heuristic_expert_open/fremont_fridge_left_v2/train_fix_grasp_npy/ep_0.npy"
        data_v1 = np.load(episode_path, allow_pickle=True)
        data = data_v1.copy()
        # data = data_v1[::18]
        if self.debug:
            max_steps = len(data)
        print("max_steps: ", max_steps)

        target_arm_pos = (
            self.env.sim.articulated_agent._robot_wrapper.arm_joint_pos
        )
        target_hand_pos = (
            self.env.sim.articulated_agent._robot_wrapper.hand_joint_pos
        )

        # grasp_mapping = {"0": "open", "1": "pre_grasp", "2": "close"}
        grasp_mapping = {"0": "pre_grasp", "1": "close"}
        for i in range(max_steps):
            # print(f"=================== step # {ctr} ===================")
            if self.debug:
                obs["arm_rgb"] = data[i]["rgb"]
                if self.grasp_mode:
                    grasp_obs = data[i]["grasp_mode"]
                else:
                    grasp_obs = data[i]["hand_joint"]
                gt_obs = np.concatenate(
                    [
                        data[i]["base_pos"],
                        data[i]["arm_joint"],
                        grasp_obs,
                    ]
                )
                obs["joint"] = gt_obs
            # if self.debug:
            #     if self.grasp_mode:
            #         action = data[i]["delta_joint_action_grasp_mode"]
            #     else:
            #         action = data[i]["delta_joint_action"]
            #     print("action: ", action)
            #     print("GT action: ", data[i]["delta_joint_action_grasp_mode"])

            #     print("obs joint: ", obs["joint"])
            #     print("GT obs: ", gt_obs)
            # else:
            # action = self.policy.act(obs)[0]
            action_chunk = self.policy.act(obs)
            if self.action_type == "all":
                for i in range(len(action_chunk)):
                    action = action_chunk[i][0]
                    target_arm_pos += action[2 : 2 + 7]
                    if self.action_space == "base_arm_hand":
                        if self.grasp_mode:
                            grasp_mode_name = grasp_mapping[
                                str(int(action[-1]))
                            ]
                            target_hand_pos = self.get_grasp_mode(
                                grasp_mode_name
                            )
                            curr_grasp_mode = self.get_grasp_mode_idx(
                                grasp_mode_name
                            )
                        else:
                            target_hand_pos += action[2 + 7 :]
                    else:
                        target_hand_pos = self.get_grasp_mode("pre_grasp")
                    if self.arm_type == "left":
                        self.pin_right_arm()
                    elif self.arm_type == "right":
                        self.pin_left_arm()

                    if "base" in self.action_space:
                        base_linear = action[0]
                        base_angular = action[1]
                        if np.abs(base_linear) < 0.2:
                            base_linear = 0
                        if np.abs(base_angular) < 0.2:
                            base_angular = 0
                        base_linear = 0
                        base_angular = 0
                    else:
                        base_linear = 0
                        base_angular = 0
                    self.move_base_ee_and_hand(
                        base_linear,
                        base_angular,
                        target_arm_pos,
                        target_hand_pos,
                        timeout=1,
                    )
            else:
                if self.action_type == "first":
                    action = action_chunk.pop(0)[0]
                elif self.action_type == "mean":
                    action = np.mean(action_chunk, axis=0)[0]
                if self.action_space == "base_arm_hand":
                    target_arm_pos += action[2 : 2 + 7]
                elif self.action_space == "arm":
                    target_arm_pos += action

                if "hand" in self.action_space:
                    if self.grasp_mode:
                        grasp_mode_name = grasp_mapping[str(int(action[-1]))]
                        target_hand_pos = self.get_grasp_mode(grasp_mode_name)

                    else:
                        target_hand_pos += action[2 + 7 :]
                else:
                    target_hand_pos = self.get_grasp_mode("pre_grasp")
                if self.arm_type == "left":
                    self.pin_right_arm()
                elif self.arm_type == "right":
                    self.pin_left_arm()
                if "base" in self.action_space:
                    base_linear = action[0]
                    base_angular = action[1]
                    if np.abs(base_linear) < 0.2:
                        base_linear = 0
                    if np.abs(base_angular) < 0.2:
                        base_angular = 0
                    base_linear = 0
                    base_angular = 0
                else:
                    base_linear = 0
                    base_angular = 0
                self.move_base_ee_and_hand(
                    base_linear,
                    base_angular,
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

            if not self.debug:
                target_arm_pos = curr_arm_joints
                target_hand_pos = curr_hand_pos

            if self.action_space == "base_arm_hand":
                if self.grasp_mode:
                    curr_grasp_mode = self.get_grasp_mode_idx(grasp_mode_name)
                    grasp_obs = curr_grasp_mode
                else:
                    grasp_obs = curr_hand_pos

                obs["joint"] = np.concatenate(
                    [curr_base_pos, curr_arm_joints, grasp_obs]
                )
            elif self.action_space == "arm":
                obs["joint"] = np.array(curr_arm_joints)
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
    parser.add_argument("--grasp_mode", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--action_type", default="first", help="first, mean, all"
    )
    parser.add_argument(
        "--action_space",
        default="base_arm_hand",
        help="base_arm_hand, base_arm, arm, arm_hand",
    )

    args = parser.parse_args()

    datagen = VLAEvaluator(
        args.ep_id,
        args.exp,
        args.ckpt,
        args.task,
        args.grasp_mode,
        args.replay,
        args.debug,
        args.action_type,
        args.action_space,
    )
    if args.replay:
        datagen.replay_data()
    else:
        datagen.run()
        # else:
        # datagen.run()
