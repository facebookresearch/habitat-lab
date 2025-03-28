import time
import warnings

import magnum as mn

import habitat_sim
from habitat.tasks.rearrange.isaac_rearrange_sim import IsaacRearrangeSim

warnings.filterwarnings("ignore")
import argparse
import math
import os
import random

import imageio
import numpy as np
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation as R

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
from habitat.utils.gum_utils import (
    sample_point_cloud_from_urdf,
    to_world_frame,
)
from habitat_sim.physics import JointMotorSettings, MotionType
from habitat_sim.utils import viz_utils as vut
from habitat_sim.utils.settings import make_cfg
from scripts.expert_data.utils.utils import process_obs_img
from scripts.expert_data.utils.viz_utils import add_text_to_image


class MurpEnv:
    def __init__(self, config):
        self.config = config

        agent_dict = self.make_agent_dict()
        action_dict = self.make_action_dict()

        self.env = self.init_rearrange_env(agent_dict, action_dict)

        aux = self.env.reset()
        self.target_name = self.env.current_episode.action_target[0]
        self.skill = self.config.skill
        self.save_path = (
            f"results/output_env_murp_{self.skill}_{self.target_name}.mp4"
        )
        self.writer = imageio.get_writer(
            self.save_path,
            fps=30,
        )
        self.base_trans = None

    def make_agent_dict(self):
        main_agent_config = AgentConfig()

        main_agent_config.articulated_agent_urdf = os.path.join(
            self.config.data_path, self.config.urdf_path
        )
        main_agent_config.articulated_agent_usda = os.path.join(
            self.config.data_path, self.config.usda_path
        )
        main_agent_config.ik_arm_urdf = os.path.join(
            self.config.data_path, self.config.arm_urdf_path
        )
        main_agent_config.articulated_agent_type = "MurpRobot"

        # Define sensors that will be attached to this agent, here a third_rgb sensor and a head_rgb.
        # We will later talk about why we are giving the sensors these names
        main_agent_config.sim_sensors = {
            "third_rgb": ThirdRGBSensorConfig(),
            "articulated_agent_arm_rgb": ArmRGBSensorConfig(),
            "articulated_agent_arm_depth": ArmDepthSensorConfig(),
        }

        # We create a dictionary with names of agents and their corresponding agent configuration
        agent_dict = {"main_agent": main_agent_config}
        return agent_dict

    def make_action_dict(self):
        action_dict = {
            "base_velocity_action": BaseVelocityActionConfig(
                type="BaseVelIsaacAction"
            ),
            "arm_reach_ee_action": ActionConfig(type="ArmReachEEAction"),
        }
        return action_dict

    def make_sim_cfg(self, agent_dict):
        # Start the scene config
        sim_cfg = SimulatorConfig(type="IsaacRearrangeSim-v0")

        # This is for better graphics
        sim_cfg.habitat_sim_v0.enable_hbao = True
        sim_cfg.habitat_sim_v0.enable_physics = False

        # TODO: disable this, causes performance issues
        sim_cfg.habitat_sim_v0.frustum_culling = False

        # Set up an example scene
        sim_cfg.scene = "NONE"  # os.path.join(data_path, "hab3_bench_assets/hab3-hssd/scenes/103997919_171031233.scene_instance.json")

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
            data_path=self.config.episode_json,
        )

        hab_cfg = HabitatConfig()
        hab_cfg.environment = env_cfg
        hab_cfg.task = task_cfg

        hab_cfg.dataset = dataset_cfg
        hab_cfg.simulator = sim_cfg
        hab_cfg.simulator.seed = hab_cfg.seed

        return hab_cfg

    def init_rearrange_env(self, agent_dict, action_dict):
        hab_cfg = self.make_hab_cfg(agent_dict, action_dict)
        res_cfg = OmegaConf.create(hab_cfg)
        res_cfg.environment.max_episode_steps = 100000000
        return Env(res_cfg)

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

    def reset_robot(self, name):
        print("start pose: ", self.env.current_episode.start_position)
        start_position, start_rotation = (
            np.array(self.env.current_episode.start_position),
            self.env.current_episode.start_rotation,
        )

        position = mn.Vector3(start_position)
        # rotation = mn.Quaternion.rotation(
        #     mn.Deg(start_rotation), mn.Vector3.y_axis()
        # )
        rotation = mn.Quaternion(
            mn.Vector3(start_rotation[:3]), start_rotation[3]
        )
        self.base_trans = mn.Matrix4.from_(rotation.to_matrix(), position)
        self.env.sim.articulated_agent.base_transformation = self.base_trans
        print(f"set base to {name}: {start_position}, {start_rotation}")

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

    def get_curr_ee_pose(self, arm="right"):
        if arm == "left":
            curr_ee_pos_vec, curr_ee_rot = (
                self.env.sim.articulated_agent._robot_wrapper.ee_pose()
            )
        elif arm == "right":
            curr_ee_pos_vec, curr_ee_rot = (
                self.env.sim.articulated_agent._robot_wrapper.ee_right_pose()
            )

        curr_ee_rot_quat = R.from_quat(
            [*curr_ee_rot.vector, curr_ee_rot.scalar]
        )
        curr_ee_rot_rpy = curr_ee_rot_quat.as_euler("xyz", degrees=True)
        curr_ee_pos = np.array([*curr_ee_pos_vec])
        return curr_ee_pos, curr_ee_rot_rpy

    def get_curr_joint_pose(self, arm="right"):
        if arm == "left":
            return self.env.sim.articulated_agent._robot_wrapper.arm_joint_pos
        elif arm == "right":
            return (
                self.env.sim.articulated_agent._robot_wrapper.right_arm_joint_pos
            )

    def get_curr_hand_pose(self, arm="right"):
        if arm == "left":
            return self.env.sim.articulated_agent._robot_wrapper.hand_joint_pos
        elif arm == "right":
            return (
                self.env.sim.articulated_agent._robot_wrapper.right_hand_joint_pos
            )

    def move_to_ee(
        self, target_ee_pos, target_ee_rot=None, grasp=None, timeout=1000
    ):
        print(f"moving arm to: {target_ee_pos}, with hand {grasp}")
        ctr = 0
        curr_ee_pos, curr_ee_rot_rpy = self.get_curr_ee_pose()
        while not np.allclose(curr_ee_pos, target_ee_pos, atol=0.02):
            action = {
                "action": "arm_reach_ee_action",
                "action_args": {
                    "target_pos": np.array(target_ee_pos, dtype=np.float32),
                    "target_rot": target_ee_rot,
                },
            }
            self.env.sim.articulated_agent.base_transformation = (
                self.base_trans
            )
            if grasp is None:
                self.env.sim.articulated_agent._robot_wrapper._target_right_hand_joint_positions = (
                    self.get_curr_hand_pose()
                )
            else:
                self.env.sim.articulated_agent._robot_wrapper._target_right_hand_joint_positions = self.get_grasp_mode(
                    grasp
                )
            self.pin_left_arm()

            obs = self.env.step(action)
            im = process_obs_img(obs)
            im = add_text_to_image(im, "moving ee")
            self.writer.append_data(im)

            curr_ee_pos, curr_ee_rot_rpy = self.get_curr_ee_pose()
            ctr += 1
            if ctr > timeout:
                print(
                    f"Exceeded time limit. {np.abs(curr_ee_pos-target_ee_pos)} away from target"
                )
                break

    def move_hand(self, mode="close", timeout=100):
        target_hand_pos = self.get_grasp_mode(mode)
        print(f"moving hand to: {mode}")
        self.move_hand_joints(target_hand_pos, timeout)

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
            self.env.sim.articulated_agent._robot_wrapper._target_right_arm_joint_positions = (
                self.get_curr_joint_pose()
            )
            self.env.sim.articulated_agent._robot_wrapper._target_right_hand_joint_positions = (
                target_hand_pos
            )
            self.pin_left_arm()
            obs = self.env.step(action)
            im = process_obs_img(obs)
            self.writer.append_data(im)
            curr_hand_pos = self.get_curr_hand_pose()
            ctr += 1
            if ctr > timeout:
                print(
                    f"Exceeded time limit. {np.abs(curr_hand_pos-target_hand_pos)} away from target. Curr hand pos: {curr_hand_pos}"
                )
                break

    def move_ee_and_hand(
        self, target_ee_pos, target_ee_rot=None, hand_joints=None, timeout=300
    ):
        ctr = 0
        curr_ee_pos, curr_ee_rot_rpy = self.get_curr_ee_pose()
        while not np.allclose(curr_ee_pos, target_ee_pos, atol=0.02):
            action = {
                "action": "arm_reach_ee_action",
                "action_args": {
                    "target_pos": np.array(target_ee_pos, dtype=np.float32),
                    "target_rot": target_ee_rot,
                },
            }
            self.env.sim.articulated_agent.base_transformation = (
                self.base_trans
            )
            self.env.sim.articulated_agent._robot_wrapper._target_right_hand_joint_positions = (
                hand_joints
            )
            self.pin_left_arm()

            obs = self.env.step(action)
            im = process_obs_img(obs)
            im = add_text_to_image(im, "using grasp controller")
            self.writer.append_data(im)

            curr_ee_pos, curr_ee_rot_rpy = self.get_curr_ee_pose()
            ctr += 1
            if ctr > timeout:
                print(
                    f"Exceeded time limit. {np.abs(curr_ee_pos-target_ee_pos)} away from target"
                )
                break

    def move_base(self, base_lin_vel, base_ang_vel):
        action = {
            "action": "base_velocity_action",
            "action_args": {
                "base_vel": np.array(
                    [base_lin_vel, base_ang_vel], dtype=np.float32
                )
            },
        }

        obs = self.env.step(action)
        self.base_trans = self.env.sim.articulated_agent.base_transformation
        im = process_obs_img(obs)
        im = add_text_to_image(im, "using base controller")
        self.writer.append_data(im)

    def move_base_ee_and_hand(
        self,
        base_lin_vel,
        base_ang_vel,
        target_ee_pos,
        target_ee_rot=None,
        hand_joints=None,
        timeout=300,
    ):
        ctr = 0
        curr_ee_pos, curr_ee_rot_rpy = self.get_curr_ee_pose()
        while not np.allclose(curr_ee_pos, target_ee_pos, atol=0.02):
            action = {
                "action": ("arm_reach_ee_action", "base_velocity_action"),
                "action_args": {
                    "target_pos": np.array(target_ee_pos, dtype=np.float32),
                    "target_rot": target_ee_rot,
                    "base_vel": np.array(
                        [base_lin_vel, base_ang_vel], dtype=np.float32
                    ),
                },
            }
            self.env.sim.articulated_agent.base_transformation = (
                self.base_trans
            )
            self.env.sim.articulated_agent._robot_wrapper._target_hand_joint_positions = (
                hand_joints
            )
            self.pin_right_arm()

            obs = self.env.step(action)
            self.base_trans = (
                self.env.sim.articulated_agent.base_transformation
            )
            im = process_obs_img(obs)
            im = add_text_to_image(im, "using grasp controller")
            self.writer.append_data(im)

            curr_ee_pos, curr_ee_rot_rpy = self.get_curr_ee_pose()
            ctr += 1
            if ctr > timeout:
                print(
                    f"Exceeded time limit. {np.abs(curr_ee_pos-target_ee_pos)} away from target"
                )
                break
            self.writer.append_data(im)

    def visualize_pos(self, pos, name="target_ee_pos"):
        self.env.sim.viz_ids[name] = self.env.sim.visualize_position(
            pos, self.env.sim.viz_ids[name]
        )

    def get_poses(self, name, pose_type="base"):
        poses = {
            "cabinet": {
                "base_pos": np.array([1.7, 0.1, -0.2]),
                "base_rot": -90,
                "ee_pos": self.env.sim._rigid_objects[0].translation,
                "ee_rot": np.deg2rad([0, 80, -30]),
            },
            "shelf": {
                "base_pos": np.array([-4.4, 0.1, -3.5]),
                "base_rot": 180,
                "ee_pos": np.array([-5.6, 1.0, -3.9]),
                "ee_rot": np.deg2rad([-60, 0, 0]),
            },
            "island": {
                "base_pos": np.array([-5.3, 0.1, -1.6]),
                "base_rot": 0,
                "ee_pos": np.array([-4.4, 0.5, -2.0]),
                "ee_rot": np.deg2rad([0, 80, -30]),
            },
            "oven": {
                "base_pos": np.array([-4.75, 0.1, -3.3]),
                "base_rot": 180,
                "ee_pos": np.array([-5.5, 1.6, -2.7]),
                "ee_rot": np.deg2rad([0, 80, -30]),
            },
            # "fridge": {
            #     "base_pos": np.array([-4.4, 0.1, 0.7]),
            #     "base_rot": 180,
            #     "ee_pos": np.array([-5.4, 1.4, 1.3]),
            #     "ee_rot": np.deg2rad([0, 0, 0]),
            # },
            "fridge": {
                "base_pos": np.array([-4.7, 0.1, 0.8]),
                "base_rot": 180,
                "ee_pos": np.array([-6.2, 1.2, 2.4]),
                "ee_rot": np.deg2rad([120, 0, 0]),
            },
            "fridge2": {
                "base_pos": np.array([-4.0, 0.1, 1.28]),
                "base_rot": 180,
                "ee_pos": np.array([-6.3, 1.2, 1.3]),
                "ee_rot": np.deg2rad([-60, 0, 0]),
            },
            "freezer": {
                "base_pos": np.array([-4.9, 0.1, 0.7]),
                "base_rot": 180,
                "ee_pos": np.array([-5.7, 0.5, 1.34531]),
                "ee_rot": np.deg2rad([0, 80, -30]),
            },
        }
        return (
            poses[name][f"{pose_type}_pos"],
            poses[name][f"{pose_type}_rot"],
        )
