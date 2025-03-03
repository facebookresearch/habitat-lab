import warnings

import magnum as mn

import habitat_sim
from habitat.tasks.rearrange.isaac_rearrange_sim import IsaacRearrangeSim

warnings.filterwarnings("ignore")
import os
import random

import imageio
import numpy as np
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
from habitat_sim.physics import JointMotorSettings, MotionType
from habitat_sim.utils import viz_utils as vut
from habitat_sim.utils.settings import make_cfg

data_path = "/fsx-siro/jtruong/repos/vla-physics/habitat-lab/data/"


def make_sim_cfg(agent_dict):
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


def make_hab_cfg(agent_dict, action_dict):
    sim_cfg = make_sim_cfg(agent_dict)
    task_cfg = TaskConfig(type="RearrangeEmptyTask-v0")
    task_cfg.actions = action_dict
    env_cfg = EnvironmentConfig()
    dataset_cfg = DatasetConfig(
        type="RearrangeDataset-v0",
        data_path="data/hab3_bench_assets/episode_datasets/small_large.json.gz",
    )

    hab_cfg = HabitatConfig()
    hab_cfg.environment = env_cfg
    hab_cfg.task = task_cfg

    hab_cfg.dataset = dataset_cfg
    hab_cfg.simulator = sim_cfg
    hab_cfg.simulator.seed = hab_cfg.seed

    return hab_cfg


def init_rearrange_env(agent_dict, action_dict):
    hab_cfg = make_hab_cfg(agent_dict, action_dict)
    res_cfg = OmegaConf.create(hab_cfg)
    res_cfg.environment.max_episode_steps = 100000000
    print("hab_cfg: ", hab_cfg)
    print("res_cfg: ", res_cfg)
    return Env(res_cfg)


from habitat.datasets.rearrange.navmesh_utils import compute_turn
from habitat.tasks.utils import get_angle


def process_obs_img(obs):
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


class ExpertDatagen:
    def __init__(self, target_name="cabinet", skill="pick"):
        # Define the agent configuration
        main_agent_config = AgentConfig()

        urdf_path = os.path.join(
            data_path,
            "hab_murp/murp_tmr_franka/murp_tmr_franka_metahand_obj.urdf",
        )
        arm_urdf_path = os.path.join(
            data_path,
            "hab_murp/murp_tmr_franka/murp_tmr_franka_metahand_left_arm_obj.urdf",
        )
        main_agent_config.articulated_agent_urdf = urdf_path
        main_agent_config.articulated_agent_type = "MurpRobot"
        main_agent_config.ik_arm_urdf = arm_urdf_path

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
                type="BaseVelKinematicIsaacAction"
            ),
            "arm_reach_ee_action": ActionConfig(type="ArmReachEEAction"),
        }
        self.env = init_rearrange_env(agent_dict, action_dict)

        aux = self.env.reset()
        self.target_name = target_name
        self.skill = skill
        self.save_path = f"output_env_murp_{self.skill}_{self.target_name}.mp4"
        self.writer = imageio.get_writer(
            self.save_path,
            fps=30,
        )
        self.base_trans = None

    def get_grasp_mode(self, name):
        grasp_joints = {
            "open": np.zeros(10),
            "close": np.array([1.57] * 10),
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
        start_position, start_rotation = self.get_poses(name, pose_type="base")

        position = mn.Vector3(start_position)
        rotation = mn.Quaternion.rotation(
            mn.Deg(start_rotation), mn.Vector3.y_axis()
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

    def get_curr_ee_pose(self):
        curr_ee_pos, curr_ee_rot = (
            self.env.sim.articulated_agent._robot_wrapper.ee_pose()
        )

        curr_ee_rot_quat = R.from_quat(
            [*curr_ee_rot.vector, curr_ee_rot.scalar]
        )
        curr_ee_rot_rpy = curr_ee_rot_quat.as_euler("xyz", degrees=True)
        return curr_ee_pos, curr_ee_rot_rpy

    def get_curr_joint_pose(self, arm="left"):
        if arm == "left":
            return self.env.sim.articulated_agent._robot_wrapper.arm_joint_pos
        elif arm == "right":
            self.env.sim.articulated_agent._robot_wrapper.right_arm_joint_pos

    def get_curr_hand_pose(self, arm="left"):
        if arm == "left":
            return self.env.sim.articulated_agent._robot_wrapper.hand_joint_pos
        elif arm == "right":
            self.env.sim.articulated_agent._robot_wrapper.right_hand_joint_pos

    def move_to_ee(
        self, target_ee_pos, target_ee_rot=None, grasp="open", timeout=1000
    ):
        print(f"moving arm to: {target_ee_pos}, with hand {grasp}")
        ctr = 0
        curr_ee_pos, curr_ee_rot_rpy = self.get_curr_ee_pose()
        while not np.allclose(curr_ee_pos, target_ee_pos, atol=0.08):
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
            self.env.sim.articulated_agent._robot_wrapper._target_hand_joint_positions = self.get_grasp_mode(
                grasp
            )
            self.pin_right_arm()

            obs = self.env.step(action)
            im = process_obs_img(obs)
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
        curr_hand_pos = self.get_curr_hand_pose()
        print(f"moving hand to: {mode}")
        ctr = 0
        while not np.allclose(curr_hand_pos, target_hand_pos, atol=0.08):
            action = {
                "action": "base_velocity_action",
                "action_args": {
                    "base_vel": np.array([0.0, 0.0], dtype=np.float32)
                },
            }
            self.env.sim.articulated_agent.base_transformation = (
                self.base_trans
            )
            self.env.sim.articulated_agent._robot_wrapper._target_hand_joint_positions = (
                target_hand_pos
            )
            self.pin_right_arm()

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

    def visualize_pos(self, pos):
        self.env.sim.viz_ids["target_ee_pos"] = (
            self.env.sim.visualize_position(
                pos, self.env.sim.viz_ids["target_ee_pos"]
            )
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
                "base_pos": np.array([-4.5, 0.1, -3.5]),
                "base_rot": 180,
                "ee_pos": self.env.sim._rigid_objects[0].translation,
                "ee_rot": np.deg2rad([0, 80, -30]),
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
                "base_pos": np.array([-4.97, 0.1, 0.74]),
                "base_rot": 180,
                "ee_pos": np.array([-5.60977, 0.95913, 1.51181]),
                "ee_rot": np.deg2rad([0, 0, 0]),
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

    def run_expert(self):
        self.reset_robot(self.target_name)

        target_ee_pos, target_ee_rot = self.get_poses(
            self.target_name, pose_type="ee"
        )
        self.visualize_pos(target_ee_pos)

        # target_ee_pos[1] += 0.05
        self.move_to_ee(
            target_ee_pos, target_ee_rot, grasp="open", timeout=700
        )

        # grasp object
        self.move_hand("close")

        if self.skill == "pick":
            # move hand up
            target_ee_pos, _ = self.get_curr_ee_pose()
            target_ee_pos[1] += 0.3
        elif self.skill == "open":
            # move hand backwards
            target_ee_pos, _ = self.get_curr_ee_pose()
            target_ee_pos[0] += 0.2
            target_ee_pos[1] -= 0.2

        self.move_to_ee(
            target_ee_pos, target_ee_rot, grasp="close", timeout=300
        )

        self.writer.close()
        print(f"saved file to: {self.save_path}")


if __name__ == "__main__":
    target_name = "freezer"
    skill = "open"
    datagen = ExpertDatagen(target_name, skill)
    datagen.run_expert()
