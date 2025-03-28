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
import rotation_conversions
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
from viz_utils import add_text_to_image

user = "joanne"
if user == "joanne":
    data_path = "/fsx-siro/jtruong/repos/vla-physics/habitat-lab/data/"
else:
    data_path = "/home/joanne/habitat-lab/data/"


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
        data_path="/fsx-siro/jtruong/repos/vla-physics/habitat-lab/habitat-lab/habitat/tasks/rearrange/task_pick.json.gz",
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
    def __init__(self, target_name="cabinet", skill="pick", replay=False):
        # Define the agent configuration
        self.replay = replay
        main_agent_config = AgentConfig()

        if user == "joanne":
            urdf_path = os.path.join(
                data_path,
                "murp/murp/platforms/franka_tmr/franka_description_tmr/urdf/franka_with_hand_2_vyshnav.urdf",
            )
            arm_urdf_path = os.path.join(
                data_path,
                # "murp/murp/platforms/franka_tmr/franka_description_tmr/urdf/franka_tmr_left_arm_only.urdf",
                "murp/murp/platforms/franka_tmr/franka_description_tmr/urdf/franka_right_arm.urdf",
            )
            usda_path = os.path.join(
                data_path, "usd/robots/franka_with_hand_right.usda"
            )
        else:
            urdf_path = os.path.join(
                data_path,
                "franka_tmr/franka_description_tmr/urdf/franka_with_hand_2.urdf",  # Lambda Change
            )
            arm_urdf_path = os.path.join(
                data_path,
                "franka_tmr/franka_description_tmr/urdf/franka_right_arm.urdf",  # Lambda Change
            )
            usda_path = os.path.join(
                data_path, "usd/robots/franka_with_hand_right.usda"
            )
        main_agent_config.articulated_agent_urdf = urdf_path
        main_agent_config.articulated_agent_usda = usda_path
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
                # type="BaseVelKinematicIsaacAction"
                type="BaseVelIsaacAction"
            ),
            "arm_reach_ee_action": ActionConfig(type="ArmReachEEAction"),
        }
        self.env = init_rearrange_env(agent_dict, action_dict)

        aux = self.env.reset()
        self.target_name = self.env.current_episode.action_target[0]
        self.skill = skill
        self.save_path = f"output_env_murp_{self.skill}_{self.target_name}.mp4"
        self.writer = imageio.get_writer(
            self.save_path,
            fps=30,
        )
        self.base_trans = None
        self.TARGET_CONFIG = {
            "cabinet": [
                30,
                15,
                25,
                "_urdf_kitchen_FREMONT_KITCHENSET_FREMONT_KITCHENSET_CLEANED_urdf/kitchenset_cabinet",
            ],
            "shelf": [
                25,
                20,
                1,
                "_urdf_kitchen_FREMONT_KITCHENSET_FREMONT_KITCHENSET_CLEANED_urdf/kitchenset_door18",
            ],
            "island": [
                20,
                12,
                18,
                "_urdf_kitchen_FREMONT_KITCHENSET_FREMONT_KITCHENSET_CLEANED_urdf/kitchenset_island",
            ],
            "oven": [
                35,
                18,
                30,
                "_urdf_kitchen_FREMONT_KITCHENSET_FREMONT_KITCHENSET_CLEANED_urdf/kitchenset_ovendoor2",
            ],
            "fridge": [
                30,
                14,
                28,
                "_urdf_kitchen_FREMONT_KITCHENSET_FREMONT_KITCHENSET_CLEANED_urdf/kitchenset_fridgedoor1",
            ],
            "fridge_2": [
                30,
                20,
                19,
                "_urdf_kitchen_FREMONT_KITCHENSET_FREMONT_KITCHENSET_CLEANED_urdf/kitchenset_fridgedoor2",
            ],
            "freezer": [
                28,
                12,
                24,
                "_urdf_kitchen_FREMONT_KITCHENSET_FREMONT_KITCHENSET_CLEANED_urdf/kitchenset_freezer",
            ],
        }

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

    def set_targets(
        self, target_w_xyz, target_w_quat, target_joints, hand="left"
    ):
        self.target_w_xyz = target_w_xyz
        self.target_w_quat = target_w_quat
        target_quat = R.from_quat(target_w_quat, scalar_first=True)
        # Assumes (wxyz), isaac outputs (xyzw). If direct from isaac remember to swap.
        # self.target_w_rotmat = rotation_conversions.quaternion_to_matrix( self.target_w_quat )
        self.target_w_rotmat = target_quat.as_matrix()
        self.target_joints = target_joints

        # XYZ
        self.open_xyz = target_w_xyz.copy()
        if hand == "left":
            self.open_xyz[2] -= 0.1
            self.open_xyz[0] += 0.1

            # Pre-Grasp Targets
            OPEN_JOINTS = [1, 5, 9, 14]
            # OPEN_JOINTS = [0, 4, 6, 9]
            # Grasp fingers
            self.grasp_fingers = self.target_joints.copy()
            self.close_fingers = self.target_joints.copy()
            self.close_fingers[OPEN_JOINTS] += 0.2
        else:
            self.open_xyz[2] -= 0.1
            self.open_xyz[0] += 0.1
            SECONDARY_JOINTS = [2, 6, 10, 15]
            TERTIARY_JOINTS = [3, 7, 11]
            OPEN_JOINTS = [1, 5, 9]
            CURVE_JOINTS = [13]
            BASE_THUMB_JOINT = [12]
            self.grasp_fingers = self.target_joints.copy()
            self.close_fingers = self.target_joints.copy()
            self.close_fingers[BASE_THUMB_JOINT] += 1.1
            # self.close_fingers[CURVE_JOINTS] -=0.5
            self.close_fingers[SECONDARY_JOINTS] += 0.7
            self.close_fingers[TERTIARY_JOINTS] += 1.0
            self.close_fingers[OPEN_JOINTS] += 0.7

    def get_targets(self, name="target", hand="right"):
        # Lambda Changes
        if name == "target":
            return (
                torch.tensor(self.grasp_fingers, device="cuda:0"),
                torch.tensor(self.target_w_xyz, device="cuda:0"),
                torch.tensor(self.target_w_rotmat, device="cuda:0"),
            )

        elif name == "target_grip":
            return (
                torch.tensor(self.close_fingers),
                # torch.tensor(self.target_w_xyz),
                torch.tensor(self.get_curr_ee_pose()[0]),
                torch.tensor(self.target_w_rotmat),
            )

        elif name == "open":
            self.open_xyz = self.get_curr_ee_pose()[0]
            if hand == "right":
                # self.open_xyz[1] -= 0.1
                self.open_xyz[0] += 0.1

            return (
                torch.tensor(self.close_fingers, device="cuda:0"),
                torch.tensor(self.open_xyz, device="cuda:0"),
                torch.tensor(self.target_w_rotmat, device="cuda:0"),
            )

    def generate_action(self, cur_obs, name):
        cur_tar_wrist_xyz = cur_obs["wrist_tar_xyz"]
        cur_tar_fingers = cur_obs["tar_joints"][:16]

        # Phase 2 go to pick
        delta_t = 0.1
        self.step += 1
        print(self.step)
        # if self.step < 600:
        #     name = "target"
        # elif self.step < 1200:
        #     name = "target_grip"
        # else:
        #     name = "open"
        # if self.step < 600:
        #     name = "target_grip"
        # else:
        #     name = "open"

        tar_joints, tar_xyz, tar_rot = self.get_targets(name=name)

        delta_xyz = -(cur_tar_wrist_xyz - tar_xyz.to(cur_tar_wrist_xyz))
        print("cur_tar_wrist_xyz: ", cur_tar_wrist_xyz)
        print("tar_xyz: ", tar_xyz)
        print("delta_xyz: ", delta_xyz)
        delta_joints = tar_joints.to(cur_tar_fingers) - cur_tar_fingers

        tar_xyz = cur_tar_wrist_xyz + delta_xyz * delta_t
        tar_joint = cur_tar_fingers + delta_joints * delta_t
        # Orientation
        door_orientation = cur_obs["orientation_door"]
        door_orientation = torch.tensor(
            door_orientation, dtype=torch.float32, device="cuda:0"
        )  # Lambda Change
        door_rot = rotation_conversions.quaternion_to_matrix(door_orientation)

        rot_y = rotation_conversions.euler_angles_to_matrix(
            torch.tensor([math.pi, -math.pi, 0.0], device="cuda:0"), "XYZ"
        )
        target_rot = torch.einsum("ij,jk->ik", door_rot, rot_y)
        tar_rot = target_rot

        act = {
            "tar_xyz": tar_xyz.cpu().numpy(),
            "tar_rot": tar_rot.cpu().numpy(),
            "tar_fingers": tar_joint.cpu().numpy(),
        }

        return act

    def apply_rotation(self, quat_door):
        hab_T_door = R.from_quat(quat_door)
        isaac_T_hab_list = [-90, 0, 0]
        isaac_T_hab = R.from_euler("xyz", isaac_T_hab_list, degrees=True)
        isaac_T_door_mat = R.from_matrix(
            isaac_T_hab.as_matrix() @ hab_T_door.as_matrix()
        )
        isaac_T_door_quat = isaac_T_door_mat.as_quat()
        return isaac_T_door_quat

    def get_door_quat(self):
        path = self.TARGET_CONFIG[self.target_name][3]
        door_trans, door_orientation_quat = self.env.sim.get_prim_transform(
            os.path.join("/World/test_scene/", path), convention="quat"
        )
        self.visualize_pos(door_trans, "door")

        isaac_T_door_quat = self.apply_rotation(door_orientation_quat)
        print("door_quat: ", isaac_T_door_quat)
        return isaac_T_door_quat

    def create_T_matrix(self, pos, rot):
        T_mat = np.eye(4)
        rot_quat = R.from_quat(np.array([rot.scalar, *rot.vector]))
        T_mat[:3, :3] = rot_quat.as_matrix()
        T_mat[:, -1] = np.array([*pos, 1])
        return T_mat

    def grasp_obj(self, name):
        quat_door = self.get_door_quat()

        cur_obs = {
            "wrist_tar_xyz": torch.tensor(self.current_target_xyz),
            "tar_joints": torch.tensor(self.current_target_fingers),
            "orientation_door": torch.tensor(quat_door),
        }
        act = self.generate_action(cur_obs, name)

        base_T_hand = act["tar_xyz"]

        ee_pos, ee_rot = (
            self.env.sim.articulated_agent._robot_wrapper.hand_pose()
        )
        base_T_ee = self.create_T_matrix(ee_pos, ee_rot)

        hand_pos, hand_rot = (
            self.env.sim.articulated_agent._robot_wrapper.hand_pose()
        )
        base_T_hand = self.create_T_matrix(hand_pos, hand_rot)

        ee_T_hand = np.linalg.inv(base_T_ee) @ base_T_hand
        base_T_hand = base_T_ee @ ee_T_hand
        print("base_T_hand: ", base_T_hand[:, -1])

        self.visualize_pos(act["tar_xyz"], "hand")
        # self.move_hand_joints(act["tar_fingers"], timeout=10)
        target_rot_mat = R.from_matrix(act["tar_rot"])
        target_rot_rpy = target_rot_mat.as_euler("xyz", degrees=False)

        print(
            f"Policy XYZ {act['tar_xyz']},  Rot {np.rad2deg(target_rot_rpy)}"
        )
        curr_xyz, curr_ori = self.get_curr_ee_pose()
        print(f"Curr XYZ {curr_xyz}, Rot {curr_ori}")
        target_rot_rpy = self.target_ee_rot
        # if name == "open" and self.step > 10:
        #     self.move_base_ee_and_hand(
        #         -0.1,
        #         0.0,
        #         act["tar_xyz"],
        #         target_rot_rpy,
        #         act["tar_fingers"],
        #         timeout=10,
        #     )
        #     # self.move_base(
        #     #     -0.1,
        #     #     0.0,
        #     # )
        # else:
        self.move_ee_and_hand(
            act["tar_xyz"], target_rot_rpy, act["tar_fingers"], timeout=10
        )
        self.current_target_fingers = act["tar_fingers"]
        self.current_target_xyz = act["tar_xyz"]
        _current_target_rotmat = R.from_matrix(act["tar_rot"])
        self.current_target_quat = _current_target_rotmat.as_quat()

    def replay_grasp_obj(self):
        quat_door = self.get_door_quat()
        print("quat_door: ", quat_door)
        obs_data = np.load("input.npy", allow_pickle=True)
        act_data = np.load("actions.npy", allow_pickle=True)
        for idx in range(30):
            cur_obs = {
                "wrist_tar_xyz": torch.tensor(
                    obs_data[idx]["wrist_tar_xyz"][0, :], device="cuda:0"
                ),  # Lambda Change
                "tar_joints": torch.tensor(obs_data[idx]["tar_joints"][0, :]),
                "orientation_door": torch.tensor(
                    obs_data[idx]["orientation_door"][0, :], device="cuda:0"
                ),  # Lambda Change
            }
            act = self.generate_action(cur_obs)
            saved_act = act_data[idx]
            print("orientation_door: ", cur_obs["orientation_door"], idx)
            print("act: ", act["tar_fingers"], idx)
            print(
                "saved_act tar_fingers: ", saved_act["tar_fingers"][0, :], idx
            )
            print("saved_act tar_xyz: ", saved_act["tar_xyz"][0, :], idx)
            self.move_hand_joints(saved_act["tar_fingers"][0, :], timeout=5)

    def execute_grasp_sequence(
        self, hand, grip_iters, open_iters, move_iters=None
    ):
        self.move_to_ee(
            self.target_ee_pos,
            self.target_ee_rot,
            grasp="pre_grasp" if hand == "left" else "open",
            timeout=300 if hand == "left" else 200,
        )

        self.step = 0
        self.current_target_fingers = (
            self.env.sim.articulated_agent._robot_wrapper.right_hand_joint_pos
        )
        self.current_target_xyz = self.target_ee_pos
        target_xyz, target_ori = self.get_curr_ee_pose()
        target_xyz[1] -= 0.52
        target_ori_rpy = R.from_euler("xyz", target_ori, degrees=True)
        target_quaternion = target_ori_rpy.as_quat(scalar_first=True)  # wxzy
        target_joints = (
            self.env.sim.articulated_agent._robot_wrapper.right_hand_joint_pos
        )
        self.set_targets(
            target_w_xyz=target_xyz,
            target_w_quat=target_quaternion,
            target_joints=target_joints,
            hand=hand,
        )
        if move_iters:
            for _ in range(move_iters):
                self.move_base(1.0, 0.0)

        # Grasp and open object
        if self.replay:
            self.replay_grasp_obj()
        else:
            for _ in range(grip_iters):
                self.grasp_obj(name="target_grip")

        for _ in range(open_iters):
            self.grasp_obj(name="open")

        # Move robot back
        for _ in range(10):
            self.move_base(-1.0, 0.0)

    def run_expert_w_grasp(self, hand="left"):
        self.target_name = self.env.current_episode.action_target[0]
        self.reset_robot(self.target_name)
        print("TARGET_NAME", self.target_name)
        self.target_ee_pos, self.target_ee_rot = (
            self.env.current_episode.action_target[1],
            self.env.current_episode.action_target[2],
        )
        self.visualize_pos(self.target_ee_pos)
        grip_iters, open_iters, move_iters, _ = self.TARGET_CONFIG[
            self.target_name
        ]
        if hand == "left":
            self.execute_grasp_sequence(hand, grip_iters, open_iters)
        elif hand == "right":
            self.execute_grasp_sequence(
                hand, grip_iters, open_iters, move_iters
            )

    def rl_reset(self):
        # self.object_asset_files_dict = {
        #     "simple_tennis_ball": "ball.urdf",
        #     "simple_cylin4cube": "cylinder4cube.urdf",
        #     "000": "dexgraspnet2/meshdata/000/simplified_sdf.urdf",
        #     "048": "dexgraspnet2/meshdata/048/simplified_sdf.urdf",
        # } # self.object_asset_files_dict[object_name]
        # object_name = "048"
        object_name = self.env.current_episode.rigid_objs[0][0]

        pc, normals = sample_point_cloud_from_urdf(
            os.path.abspath("data/assets"),
            f"dexgraspnet2/meshdata/{object_name}/simplified_sdf.urdf",
            100,
            seed=4,
        )
        self.gt_object_point_clouds__object = torch.tensor(pc).unsqueeze(0)
        self.private_info = {}
        path = (
            str(self.env.sim._rigid_objects[0]._prim)
            .split("<")[1]
            .split(">")[0]
        )
        obj_trans, obj_rot = self.env.sim.get_prim_transform(
            path, convention="quat"
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
        self.private_info["fingertip_trans"] = torch.tensor(
            ee_poses
        )  # check finger tip ordering
        self.private_info["object_restitution"] = torch.tensor(
            [np.random.uniform(0, 1.0)]
        )
        self.obs_buf_lag_history = torch.zeros(
            22 * 2 * 3
        )  # joints = 22-dim, target = 22-dim, history of 3
        self.target_obs = torch.zeros(22)
        self.obj_prim_path = (
            str(self.env.sim._rigid_objects[0]._prim)
            .split("<")[1]
            .split(">")[0]
        )

    def get_obs_dict(self):
        # {'obs': torch.Size([16384, 132]), 'priv_info': torch.Size([16384, 29]), 'proprio_hist': torch.Size([16384, 30, 44]), 'gt_object_point_cloud': torch.Size([16384, 100, 3]), 'progress': torch.Size([16384]), 'init_hand_dof': torch.Size([16384, 16]), 'init_fingertip_trans': torch.Size([16384, 12]), 'hand_base_pose': torch.Size([16384, 7]), 'cur_dof_pos': torch.Size([16384, 16]), 'fingertip_trans': torch.Size([16384, 12]), 'fingertip_rot': torch.Size([16384, 16]), 'fine_contact': torch.Size([16384, 30, 30, 3]), 'robot_dof_pos': torch.Size([16384, 16]), 'prev_targets': torch.Size([16384, 22]), 'last_action': torch.Size([16384, 22]), 'progress_float': torch.Size([16384, 1]), 'behavior_action': torch.Size([16384, 22]), 'behavior_masks': torch.Size([16384, 22])}
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

    def get_oracle_obs_dict(self):
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

        obs_dict["target_wrist"] = torch.zeros(7).unsqueeze(0)
        obs_dict["target_fingers"] = self.data_cache["048_@_1.0"][
            "robot_dof_pos"
        ][0, :].unsqueeze(0)
        obs_dict["progress"] = torch.tensor(self.progress_ctr)
        obs_dict["prev_targets"] = self.prev_targets
        for k, v in obs_dict.items():
            obs_dict[k] = v.float().to("cuda:0")
        return obs_dict

    def load_grasp_cache(self):
        load_file = "/fsx-siro/jtruong/repos/vla-physics/habitat-lab/data/cache/initial_grasps/misc/dexgraspnet2.pth"
        data = torch.load(load_file, map_location="cpu")
        data_cache = data["cache"]
        cases = list(data_cache.keys())
        return data_cache

    def test_pick(self):
        print(
            "self.env.current_episode: ", self.env.current_episode.rigid_objs
        )
        # self.load_gum_policy()
        self.target_name = self.env.current_episode.action_target[0]
        self.reset_robot(self.target_name)

        # arm control
        target = self.env.sim._rigid_objects[0].transformation
        rotation_matrix = np.array(
            [
                [target[0].x, target[0].y, target[0].z],
                [target[1].x, target[1].y, target[1].z],
                [target[2].x, target[2].y, target[2].z],
            ]
        )
        rpy = R.from_matrix(rotation_matrix).as_euler("xyz", degrees=True)
        ee_rot = rpy + np.array([0, 90, 0])

        # grap control
        self.load_gum_oracle_policy()
        self.data_cache = self.load_grasp_cache()
        self.progress_ctr = 0
        self.obj_prim_path = (
            str(self.env.sim._rigid_objects[0]._prim)
            .split("<")[1]
            .split(">")[0]
        )

        self.move_to_ee(
            self.env.sim._rigid_objects[0].translation,
            ee_rot,
            timeout=500,
        )

        self.prev_targets = torch.zeros(22).unsqueeze(0)
        for i in range(100):
            obs_dict = self.get_oracle_obs_dict()
            action = self.policy.act(obs_dict)
            print("action: ", action)
            self.progress_ctr += 1
            self.prev_targets = action

        # mu = self.policy.model.act(obs_dict)["mus"]
        # mu = torch.clamp(mu, -1.0, 1.0)
        # print("mu: ", mu)

    def load_gum_policy(self):
        import sys

        third_party_ckpt_root_folder = "/opt/hpcaas/.mounts/fs-03ee9f8c6dddfba21/jtruong/repos/gum_ws/src/GUM"
        sys.path.append(third_party_ckpt_root_folder)
        from gum.planning.rl.ppo import PPO

        self.policy = PPO.from_checkpoint(
            "/opt/hpcaas/.mounts/fs-03ee9f8c6dddfba21/jtruong/repos/gum_ws/src/GUM/results/experiment_8/run5/best_reward_3340.29.pth"
        )
        print("loaded policy!")

    def load_gum_oracle_policy(self):
        from gum.planning.grasp_planning.grasp_oracle import (
            SimpleApproachLiftOraclePolicy,
        )
        from omegaconf import OmegaConf

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
                "device": "cuda:0",
                "seed": 0,
            }
        )
        self.policy = SimpleApproachLiftOraclePolicy(cfg, device="cuda:0")

    def main(self):
        self.test_pick()
        # for _ in range(self.env.number_of_episodes):
        #     if self.skill == "pick":
        #         self.test_pick()
        #     elif self.skill == "open":
        #         self.run_expert_w_grasp(hand="right")
        #     self.env.reset()
        print(f"saved video to: {self.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your program description")

    # Add arguments
    parser.add_argument(
        "--target-name", default="shelf", help="target object name"
    )
    parser.add_argument("--skill", default="open", help="open, pick")
    parser.add_argument("--replay", action="store_true")

    args = parser.parse_args()
    datagen = ExpertDatagen(args.target_name, args.skill, args.replay)

    datagen.main()
