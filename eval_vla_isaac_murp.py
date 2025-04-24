import time
import warnings

import magnum as mn

import habitat_sim
from habitat.tasks.rearrange.isaac_rearrange_sim import IsaacRearrangeSim

warnings.filterwarnings("ignore")
import argparse
import json
import math
import os
import random
import shutil

import cv2
import imageio
import numpy as np
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from PIL import Image
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
from habitat_sim.physics import JointMotorSettings, MotionType
from habitat_sim.utils import viz_utils as vut
from habitat_sim.utils.settings import make_cfg
from viz_utils import add_text_to_image
from vla_utils import (  # ACTION_P01,; ACTION_P99,
    SUCCESS_METRIC,
    dennormalize,
    get_statistics,
    infer_action_open_murp,
    load_vla_skill,
    normalize,
    performance_metric,
)

user = "joanne"
if user == "joanne":
    data_path = "/fsx-siro/jtruong/repos/vla-physics/habitat-lab/data/"
else:
    data_path = "/home/joanne/habitat-lab/data/"

# remove scientific notation
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)
os.environ["VLA_DATA_DIR"] = "/fsx-siro/jtruong/data/vla_data"
# third_party_ckpt_root_folder = "/fsx-siro/jtruong/repos/robot-skills"
# sys.path.append(third_party_ckpt_root_folder)
# from src.agent.dataset import TorchRLDSInterleavedDataset
# from torch.utils.data import DataLoader


class VLAEvaluator:
    def __init__(
        self, target_name="cabinet", skill="pick", replay=False, seed=1
    ):
        # Define the agent configuration
        self.seed = seed
        print("using seed: ", self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.replay = replay
        main_agent_config = AgentConfig()

        urdf_path = os.path.join(
            data_path,
            "murp/murp/platforms/franka_tmr/franka_description_tmr/urdf/franka_right_digit.urdf",
        )
        arm_urdf_path = os.path.join(
            data_path,
            # "murp/murp/platforms/franka_tmr/franka_description_tmr/urdf/franka_tmr_left_arm_only.urdf",
            "murp/murp/platforms/franka_tmr/franka_description_tmr/urdf/franka_tmr_right_arm_only.urdf",
        )
        main_agent_config.articulated_agent_urdf = urdf_path
        main_agent_config.articulated_agent_type = "MurpRobot"
        main_agent_config.ik_arm_urdf = arm_urdf_path

        # Define sensors that will be attached to this agent, here a third_rgb sensor and a head_rgb.
        # We will later talk about why we are giving the sensors these names
        main_agent_config.sim_sensors = {
            "third_rgb": ThirdRGBSensorConfig(),
            "articulated_agent_arm_rgb": ArmRGBSensorConfig(hfov=110),
            "articulated_agent_arm_depth": ArmDepthSensorConfig(
                min_depth=0.1, max_depth=8.0, hfov=110
            ),
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
        self.env = self.init_rearrange_env(agent_dict, action_dict)

        self.ep_id = self.seed
        self.episode_json = {
            "episode_id": 0,
            "scene_id": "Fremont",
            "language_instruction": "Open the cabinet door",
            "episode_data": [],
            "success": True,
        }
        self.save_keys = [
            "articulated_agent_arm_rgb",
            "articulated_agent_arm_depth",
        ]
        mode = "train"
        save_pth_base = f"eval/jsons"
        self.json_save_path = f"{save_pth_base}/{mode}"
        os.makedirs(self.json_save_path, exist_ok=True)

        aux = self.env.reset()
        self.env.sim.seed(self.seed)
        self.target_name = target_name
        self.skill = skill
        # self.video_save_path = f"{save_pth_base}/videos"
        self.video_save_path = f"eval/videos"
        os.makedirs(self.video_save_path, exist_ok=True)
        self.save_path = f"{self.video_save_path}/eval_output_env_murp_{self.skill}_{self.target_name}_{self.seed}_replay={self.replay}.mp4"
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
        }

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
            data_path="data/hab3_bench_assets/episode_datasets/small_large.json.gz",
        )

        hab_cfg = HabitatConfig()
        hab_cfg.environment = env_cfg
        hab_cfg.task = task_cfg
        hab_cfg.seed = self.seed

        hab_cfg.dataset = dataset_cfg
        hab_cfg.simulator = sim_cfg
        hab_cfg.simulator.seed = self.seed

        return hab_cfg

    def init_rearrange_env(self, agent_dict, action_dict):
        hab_cfg = self.make_hab_cfg(agent_dict, action_dict)
        res_cfg = OmegaConf.create(hab_cfg)
        res_cfg.environment.max_episode_steps = 100000000
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

    def save_img(self, observations, img_key, ep_data):
        if img_key in observations.keys():
            robot_img = observations[img_key]
            if "depth" in img_key:
                robot_img_tmp = np.concatenate(
                    [robot_img, robot_img, robot_img], axis=-1
                )
                robot_img = robot_img_tmp
                robot_img *= 255
            elif "rgb" in img_key:
                robot_img = cv2.cvtColor(robot_img, cv2.COLOR_BGR2RGB)
            img_dir = f"{self.json_save_path}/imgs"
            os.makedirs(img_dir, exist_ok=True)
            img_pth = f"{img_dir}/{img_key}_img_{int(time.time()*1000)}.png"
            cv2.imwrite(f"{img_pth}", robot_img)
            ep_data[img_key] = img_pth
        return ep_data

    def convert_frame(self, orig_pos, orig_rot, frame_pos, frame_rot):
        local_pos = frame_rot.inverted().transform_vector(orig_pos - frame_pos)
        # Convert to local rotation
        local_rot = frame_rot.inverted() * orig_rot
        return local_pos, local_rot

    def get_proprio_data(self):
        curr_arm_joints = (
            self.env.sim.articulated_agent._robot_wrapper.right_arm_joint_pos
        )
        curr_hand_joints = (
            self.env.sim.articulated_agent._robot_wrapper.right_hand_joint_pos
        )
        global_T_ee_pos, global_T_ee_rot = (
            self.env.sim.articulated_agent._robot_wrapper.ee_right_pose(
                convention="usd"
            )
        )
        curr_base_pos, curr_base_rot = (
            self.env.sim.articulated_agent._robot_wrapper.get_root_pose(
                convention="usd"
            )
        )
        right_arm_base_pos, right_arm_base_rot = (
            self.env.sim.articulated_agent._robot_wrapper.get_link_world_pose(
                self.env.sim.articulated_agent._robot_wrapper.right_base_ee_link_id,
                "usd",
            )
        )
        base_T_ee_pos, base_T_ee_rot = self.convert_frame(
            global_T_ee_pos,
            global_T_ee_rot,
            right_arm_base_pos,
            right_arm_base_rot,
        )
        return (
            curr_arm_joints,
            curr_hand_joints,
            global_T_ee_pos,
            global_T_ee_rot,
            base_T_ee_pos,
            base_T_ee_rot,
            curr_base_pos,
            curr_base_rot,
        )

    def add_episode_data(self, obs, action):
        frame = "arm_base"
        (
            curr_arm_joints,
            curr_hand_joints,
            global_T_ee_pos,
            global_T_ee_rot,
            base_T_ee_pos,
            base_T_ee_rot,
            curr_base_pos,
            curr_base_rot,
        ) = self.get_proprio_data()

        ep_data = {
            "arm_joints": np.array(curr_arm_joints).tolist(),
            "hand_joints": np.array(curr_hand_joints).tolist(),
            "global_T_ee_pos": np.array(global_T_ee_pos).tolist(),
            "global_T_ee_rot": np.array(
                [*global_T_ee_rot.vector, global_T_ee_rot.scalar]
            ).tolist(),
            "base_T_ee_pos": np.array(base_T_ee_pos).tolist(),
            "base_T_ee_rot": np.array(
                [*base_T_ee_rot.vector, base_T_ee_rot.scalar]
            ).tolist(),
            "global_T_base_pos": np.array(curr_base_pos).tolist(),
            "global_T_base_rot": np.array(
                [*curr_base_rot.vector, curr_base_rot.scalar]
            ).tolist(),
        }

        for save_key in self.save_keys:
            ep_data = self.save_img(obs, save_key, ep_data)

        ep_data["arm_action"] = action["arm_action"]
        ep_data["base_action"] = action["base_action"]
        ep_data["empty_action"] = action["empty_action"]
        ep_data["hand_action"] = action["hand_action"]
        ep_data["grasp_mode"] = action["grasp_mode"]

        self.episode_json["episode_data"].append(ep_data)

    def save_json(self):
        ep_path = f"{self.json_save_path}/ep_{self.ep_id}_50_horizon_actions_reset_arm.json"
        if self.episode_json["success"]:
            print(f"success! saving episode: {ep_path}")
            with open(ep_path, "w") as outfile:
                json.dump(self.episode_json, outfile)
        else:
            if os.path.exists(self.json_save_path):
                shutil.rmtree(self.json_save_path)

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
        start_position, start_rotation = self.get_poses(name, pose_type="base")

        position = mn.Vector3(start_position)
        rotation = mn.Quaternion.rotation(
            mn.Deg(start_rotation), mn.Vector3.y_axis()
        )
        self.base_trans = mn.Matrix4.from_(rotation.to_matrix(), position)
        self.env.sim.articulated_agent.base_transformation = self.base_trans
        murp_joint_limits_lower = np.deg2rad(
            np.array([-157, -102, -166, -174, -160, 31, -172])
        )
        murp_joint_limits_upper = np.deg2rad(
            np.array([157, 102, 166, -8, 160, 258, 172])
        )
        start_joint_pos = np.array(
            [
                0.14936262,
                -0.65780519,
                -0.26952777,
                -2.65130757,
                0.6578265,
                2.40055512,
                -0.56525831,
            ]
        )
        # start_joint_pos += np.random.normal(-0.1, 0.1, start_joint_pos.shape)
        start_joint_pos = np.clip(
            start_joint_pos, murp_joint_limits_lower, murp_joint_limits_upper
        )
        print("start_joint_pos: ", np.rad2deg(start_joint_pos))
        self.env.sim.articulated_agent._robot_wrapper.teleport_right_arm(
            start_joint_pos
        )

        curr_arm_joints = self.get_curr_joint_pose()
        print("curr_arm_joints: ", curr_arm_joints)

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

    def get_grasp_mode_idx(self, name):
        grasp_modes = {
            "open": 0,
            "pre_grasp": 0,
            "close": 1,
        }
        return grasp_modes[name]

    def visualize_pos(self, pos, name="target_ee_pos"):
        self.env.sim.viz_ids[name] = self.env.sim.visualize_position(
            pos, self.env.sim.viz_ids[name]
        )
        return

    def get_poses(self, name, pose_type="base"):
        poses = {
            "cabinet": {
                "base_pos": np.array([1.7, 0.1, -0.2]),
                "base_rot": -90,
                "ee_pos": self.env.sim._rigid_objects[0].translation,
                "ee_rot": np.deg2rad([0, 80, -30]),
            },
            "shelf": {
                # "base_pos": np.array([-4.4, 0.1, -3.5]),
                "base_pos": np.array([-4.5, 0.1, -3.5]),
                "base_rot": 180,
                "ee_pos": np.array([-5.0, 1.0, -3.9]),
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
        rand_offset = np.array(
            [
                np.random.normal(-0.01, 0.01),
                np.random.normal(-0.01, 0.01),
                np.random.normal(-0.01, 0.01),
            ]
        )
        print("rand_offset: ", rand_offset)
        return (
            poses[name][f"{pose_type}_pos"] + rand_offset,
            poses[name][f"{pose_type}_rot"],
        )

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
        door_trans, door_orientation_rpy = (
            self.env.sim.articulated_agent._robot_wrapper.get_prim_transform(
                path
            )
        )
        self.visualize_pos(door_trans, "door")
        quat_door = door_orientation_rpy.GetQuaternion()
        # Getting Quaternion Val to Array
        scalar = quat_door.GetReal()
        vector = quat_door.GetImaginary()
        quat_door = [scalar, vector[0], vector[1], vector[2]]
        isaac_T_door_quat = self.apply_rotation(quat_door)
        print("door_quat: ", isaac_T_door_quat)
        return isaac_T_door_quat

    def create_T_matrix(self, pos, rot):
        T_mat = np.eye(4)
        # if rot is magnum.Quaternion:
        if isinstance(rot, mn.Quaternion):
            rot_np = np.array([*rot.vector, rot.scalar])
        else:
            rot_np = np.array(rot)
        rot_quat = R.from_quat(rot_np, scalar_first=False)
        T_mat[:3, :3] = rot_quat.as_matrix()
        T_mat[:, -1] = np.array([*pos, 1])
        return T_mat

    def setup_policy(self):
        # exp_name = "open_cabinet_abs_joints_arm_fingers_60hz_v2"
        # ckpt = "step165000_0.024.pt"
        # exp_name = (
        #     "ft_open_cabinet_joints_arm_fingers_60hz_oxe_magic_soup_lang_only"
        # )
        # ckpt = "step300000_0.018.pt"
        # exp_name = "open_cabinet_isaac_right_bottom_joints_fingers_60hz"
        # ckpt = "step80000_0.047.pt"
        # exp_name = "open_cabinet_abs_local_ee_fingers_usd_60hz_v2"
        # ckpt = "step135000_0.027.pt"
        # exp_name = "open_cabinet_joints_fingers_60hz_horizon_50"
        # ckpt = "step125000_0.033.pt"
        # exp_name = "open_cabinet_joints_fingers_60hz_horizon_50"
        # ckpt = "step125000_0.033.pt"
        exp_name = "open_cabinet_joints_fingers_60hz_horizon_50_reset_arm"
        ckpt = "step85000_0.060.pt"
        # vla_skill_path = f"/opt/hpcaas/.mounts/fs-03ee9f8c6dddfba21/jtruong/repos/robot-skills/results/{exp_name}/train_lr_5e-05_seed_42/checkpoint/{ckpt}"
        vla_skill_path = f"/opt/hpcaas/.mounts/fs-03ee9f8c6dddfba21/jtruong/repos/robot-skills/results/{exp_name}/good_ckpt/{ckpt}"
        main_config = {
            "cond_steps": 2,
            # "horizon_steps": 4,  # 50
            "horizon_steps": 50,  # 50
            "num_images_to_attend_in_the_past": 1000000,
            "train_with_random_offset_position_ids": False,
            "use_flex": False,
            # "load_depth": False,
            "load_depth": True,
            "distribute_model_in_gpus": False,
            "action_history": False,
            "max_proprio_dim": 23,
            "only_use_depth": False,
            "num_images_per_step": 2,
            # "num_images_per_step": 1,
            "interleave_rgb_depth": False,
            "put_goal_in_action_model": False,
            "is_point_nav_training": False,
        }
        print("using main config: ", main_config)
        VLA_PATH = "/opt/hpcaas/.mounts/fs-03ee9f8c6dddfba21/jtruong/repos/robot-skills"
        config = OmegaConf.load(
            os.path.join(VLA_PATH, "results", exp_name, "vla_cfg.yaml")
        )
        global PROPRIO_P01, PROPRIO_P99, ACTION_P01, ACTION_P99
        PROPRIO_P01, PROPRIO_P99, ACTION_P01, ACTION_P99 = get_statistics(
            config
        )

        vla_skill, vla_processor, vla_config = load_vla_skill(
            vla_skill_path, exp_name, main_config
        )
        ckpt_name = vla_skill_path.split("/")[-1].split(".pth")[0]

        print("Successfully load the ckpt!!")

        traj_len = 400
        vla_config.data.train.window_size = traj_len
        vla_config.data.train.shuffle_buffer_size = 10

        batch_i = 0
        batch_size = 0
        return vla_skill, vla_processor, vla_config

    def _resize_numpy(self, arr, scale_depth=False):
        if arr.shape[-1] == 3:
            # RGB
            image = Image.fromarray(np.array(arr))
            image = image.resize((224, 224))
            image = np.array(image, dtype=np.uint8)
        else:
            # Depth
            if scale_depth:
                arr = (255 * np.array(arr)).astype(np.uint8)
            image = Image.fromarray(arr)
            image = image.resize((224, 224))
            image = np.expand_dims(image, axis=-1)
            image = np.array(image, dtype=np.float32)
        return image

    def run_policy_replay(self):
        self.reset_robot(self.target_name)
        ee_pos, ee_rot = self.get_poses(name, pose_type="ee")
        self.visualize_pos(ee_pos, "ee_pos")
        vla_skill, vla_processor, vla_config = self.setup_policy()
        filepath = f"/fsx-siro/jtruong/data/sim_robot_data/heuristic_expert_open/fremont_open_cabinet_right_usd/train_60hz_npy/ep_0.npy"
        data = np.load(filepath, allow_pickle=True)

        action_type = "joints"
        for step_data in data:
            image_raw = torch.tensor(
                self._resize_numpy(step_data["rgb"])
            ).unsqueeze(0)
            image_depth = torch.tensor(
                self._resize_numpy(step_data["depth"][..., 0])
            ).unsqueeze(0)
            if action_type == "joints":
                proprios = torch.tensor(
                    [*step_data["arm_joint"], *step_data["hand_joint"]]
                ).unsqueeze(0)
            elif action_type == "ee":
                proprios = torch.tensor(
                    [
                        *step_data["ee_pos"],
                        *step_data["ee_rot"],
                        *step_data["hand_joint"],
                    ]
                ).unsqueeze(0)
            proprios = normalize(proprios, PROPRIO_P01, PROPRIO_P99)
            obs = {
                "image_raw": image_raw,
                "image_depth": image_depth,
                "proprio": proprios,
                "text": "Open the cabinet door",
            }
            start_time = time.time()
            vla_action = infer_action_open_murp(
                vla_skill,
                vla_processor,
                vla_config,
                obs,
                scale_input=False,
                descale_output=True,
            )  # [batch size, prediction horizen, action dim]
            end_time = time.time()
            vla_action = vla_action[0, 0].cpu().detach().numpy()

            ctr = 0
            timeout = 1
            if action_type == "joints":
                target_arm_joints = vla_action[:7]
                target_hand_pos = vla_action[7:]
                base_lin_vel = 0
                base_ang_vel = 0
                action = {
                    "action": ("base_velocity_action"),
                    "action_args": {
                        "base_vel": np.array(
                            [base_lin_vel, base_ang_vel], dtype=np.float32
                        ),
                    },
                }
                self.env.sim.articulated_agent.base_transformation = (
                    self.base_trans
                )
                self.env.sim.articulated_agent._robot_wrapper._target_right_hand_joint_positions = (
                    target_hand_pos
                )
                self.env.sim.articulated_agent._robot_wrapper._target_right_arm_joint_positions = (
                    target_arm_joints
                )
                self.pin_left_arm()

                obs = self.env.step(action)
                # self.base_trans = (
                #     self.env.sim.articulated_agent.base_transformation
                # )
                im = self.process_obs_img(obs)
                im = add_text_to_image(im, "using vla policy (GT obs)")
                self.writer.append_data(im)

                curr_ee_pos, curr_ee_rot_rpy = self.get_curr_ee_pose()
                ctr += 1
                if ctr > timeout:
                    print(
                        f"Exceeded time limit. {np.abs(curr_ee_pos-target_ee_pos)} away from target"
                    )
                    break
                self.writer.append_data(im)

            elif action_type == "ee":
                (
                    target_ee_pos,
                    target_ee_rot,
                    target_hand_pos,
                    base_lin_vel,
                    base_ang_vel,
                ) = self.process_vla_action(vla_action)
                curr_ee_pos, curr_ee_rot_rpy = self.get_curr_ee_pose()

                while not np.allclose(curr_ee_pos, target_ee_pos, atol=0.02):
                    action = {
                        "action": (
                            "arm_reach_ee_action",
                            "base_velocity_action",
                        ),
                        "action_args": {
                            "target_pos": np.array(
                                target_ee_pos, dtype=np.float32
                            ),
                            "target_rot": target_ee_rot,
                            "base_vel": np.array(
                                [base_lin_vel, base_ang_vel], dtype=np.float32
                            ),
                        },
                    }
                    self.env.sim.articulated_agent.base_transformation = (
                        self.base_trans
                    )
                    self.env.sim.articulated_agent._robot_wrapper._target_right_hand_joint_positions = (
                        target_hand_pos
                    )
                    self.pin_left_arm()

                    obs = self.env.step(action)

                    self.base_trans = (
                        self.env.sim.articulated_agent.base_transformation
                    )
                    im = self.process_obs_img(obs)
                    im = add_text_to_image(im, "using vla policy (GT obs)")
                    self.writer.append_data(im)

                    curr_ee_pos, curr_ee_rot_rpy = self.get_curr_ee_pose()
                    ctr += 1
                    if ctr > timeout:
                        print(
                            f"Exceeded time limit. {np.abs(curr_ee_pos-target_ee_pos)} away from target"
                        )
                        break
                    self.writer.append_data(im)

        self.writer.close()
        print("saved video at: ", self.save_path)

    def process_vla_action(self, vla_action):
        print("vla_action: ", vla_action)
        base_T_ee_pos_action = vla_action[:3]
        base_T_ee_rot_action = vla_action[3:7]

        global_T_base_pos, global_T_base_rot = (
            self.env.sim.articulated_agent._robot_wrapper.get_root_pose(
                convention="usd"
            )
        )
        # create transformation matrix
        global_T_base = self.create_T_matrix(
            global_T_base_pos, global_T_base_rot
        )

        base_T_ee_action = self.create_T_matrix(
            base_T_ee_pos_action, base_T_ee_rot_action
        )
        # multiply transformation matrices
        global_T_ee_action = global_T_base @ base_T_ee_action
        # extract position and rotation from transformation matrix
        target_ee_pos_usd = global_T_ee_action[:3, -1]
        target_ee_rot = global_T_ee_action[:3, :3]
        # convert rotation matrix to quaternion
        target_ee_rot_R = R.from_matrix(target_ee_rot)
        target_ee_rot_usd_wxyz = target_ee_rot_R.as_quat(scalar_first=True)
        # convert quaternion to euler angles
        target_hand_pos = vla_action[7:]
        base_lin_vel = 0
        base_ang_vel = 0
        # convert pos and rot to habitat
        target_ee_pos = isaac_prim_utils.usd_to_habitat_position(
            target_ee_pos_usd
        )
        target_ee_rot_wyxz = isaac_prim_utils.usd_to_habitat_rotation(
            target_ee_rot_usd_wxyz
        )
        # convert to eulers
        target_ee_rot_R = R.from_quat(target_ee_rot_wyxz, scalar_first=True)
        target_ee_rot = target_ee_rot_R.as_euler("xyz", degrees=False)
        print("action: ", target_ee_pos, target_ee_rot, target_hand_pos)
        return (
            target_ee_pos,
            target_ee_rot,
            target_hand_pos,
            base_lin_vel,
            base_ang_vel,
        )

    def run_policy(self):
        self.reset_robot(self.target_name)
        curr_arm_joints = (
            self.env.sim.articulated_agent._robot_wrapper.right_arm_joint_pos
        )
        curr_hand_joints = (
            self.env.sim.articulated_agent._robot_wrapper.right_hand_joint_pos
        )
        print("starting with arm joints: ", curr_arm_joints)
        print("starting with hand joints: ", curr_hand_joints)
        vla_skill, vla_processor, vla_config = self.setup_policy()
        max_steps = 20

        # get first observation
        action_dict = {
            "action": "base_velocity_action",
            "action_args": {
                "base_vel": np.array([0.0, 0.0], dtype=np.float32)
            },
        }
        self.pin_left_arm()

        obs = self.env.step(action_dict)
        print("obs keys: ", obs.keys())
        action_type = "joints"
        for i in range(max_steps):
            image_raw = torch.tensor(
                obs["articulated_agent_arm_rgb"]
            ).unsqueeze(0)
            image_depth = torch.tensor(
                obs["articulated_agent_arm_depth"]
            ).unsqueeze(0)
            (
                curr_arm_joints,
                curr_hand_joints,
                global_T_ee_pos,
                global_T_ee_rot,
                base_T_ee_pos,
                base_T_ee_rot,
                curr_base_pos,
                curr_base_rot,
            ) = self.get_proprio_data()
            # sim_step_data = {}
            # sim_step_data["arm_joints"] = np.array([*curr_arm_joints]).tolist()
            # sim_step_data["hand_joints"] = np.array(
            #     [*curr_hand_joints]
            # ).tolist()
            # sim_step_data["base_T_ee_pos"] = np.array(
            #     [*base_T_ee_pos]
            # ).tolist()
            # sim_step_data["base_T_ee_rot"] = np.array(
            #     [*base_T_ee_rot.vector, base_T_ee_rot.scalar]
            # ).tolist()
            # sim_step_data["curr_base_pos"] = np.array(
            #     [*curr_base_rot.vector, curr_base_rot.scalar]
            # ).tolist()
            # sim_step_data["global_T_ee_pos"] = np.array(
            #     [*global_T_ee_pos]
            # ).tolist()
            # sim_step_data["global_T_ee_rot"] = np.array(
            #     [*global_T_ee_rot.vector, global_T_ee_rot.scalar]
            # ).tolist()
            # for save_key in self.save_keys:
            #     sim_step_data = self.save_img(obs, save_key, sim_step_data)
            # self.episode_json["episode_data"].append(sim_step_data)
            base_T_ee_rot_np = np.array(
                [*base_T_ee_rot.vector, base_T_ee_rot.scalar]
            )
            base_T_ee_pos = np.array([*base_T_ee_pos])
            if action_type == "joints":
                curr_arm_joints = self.get_curr_joint_pose("right")
                proprios = torch.tensor(
                    [*curr_arm_joints, *curr_hand_joints]
                ).unsqueeze(0)
            elif action_type == "ee":
                proprios = torch.tensor(
                    [*base_T_ee_pos, *base_T_ee_rot_np, *curr_hand_joints]
                ).unsqueeze(0)
            proprios = normalize(proprios, PROPRIO_P01, PROPRIO_P99)

            # print("image_raw: ", image_raw)
            # print("proprios: ", proprios)

            obs = {
                "image_raw": image_raw,
                "image_depth": image_depth,
                "proprio": proprios,
                "text": "Open the cabinet door",
            }
            start_time = time.time()
            vla_actions = infer_action_open_murp(
                vla_skill,
                vla_processor,
                vla_config,
                obs,
                scale_input=False,
                descale_output=True,
            )  # [batch size, prediction horizen, action dim]
            end_time = time.time()
            vla_actions = vla_actions[0].cpu().detach().numpy()
            for vla_action in vla_actions:
                (
                    curr_arm_joints,
                    curr_hand_joints,
                    global_T_ee_pos,
                    global_T_ee_rot,
                    base_T_ee_pos,
                    base_T_ee_rot,
                    curr_base_pos,
                    curr_base_rot,
                ) = self.get_proprio_data()
                sim_step_data = {}
                sim_step_data["arm_joints"] = np.array(
                    [*curr_arm_joints]
                ).tolist()
                sim_step_data["hand_joints"] = np.array(
                    [*curr_hand_joints]
                ).tolist()
                sim_step_data["base_T_ee_pos"] = np.array(
                    [*base_T_ee_pos]
                ).tolist()
                sim_step_data["base_T_ee_rot"] = np.array(
                    [*base_T_ee_rot.vector, base_T_ee_rot.scalar]
                ).tolist()
                sim_step_data["curr_base_pos"] = np.array(
                    [*curr_base_rot.vector, curr_base_rot.scalar]
                ).tolist()
                sim_step_data["global_T_ee_pos"] = np.array(
                    [*global_T_ee_pos]
                ).tolist()
                sim_step_data["global_T_ee_rot"] = np.array(
                    [*global_T_ee_rot.vector, global_T_ee_rot.scalar]
                ).tolist()
                # for save_key in self.save_keys:
                # sim_step_data = self.save_img(obs, save_key, sim_step_data)
                ctr = 0
                timeout = 1
                if action_type == "joints":
                    target_arm_joints = vla_action[:7]
                    target_hand_pos = vla_action[7:]
                    sim_step_data["arm_joints_cmd"] = np.array(
                        target_arm_joints
                    ).tolist()
                    sim_step_data["hand_joints_cmd"] = np.array(
                        target_hand_pos
                    ).tolist()

                    base_lin_vel = 0
                    base_ang_vel = 0
                    action = {
                        "action": ("base_velocity_action"),
                        "action_args": {
                            "base_vel": np.array(
                                [base_lin_vel, base_ang_vel], dtype=np.float32
                            ),
                        },
                    }
                    self.env.sim.articulated_agent.base_transformation = (
                        self.base_trans
                    )
                    self.env.sim.articulated_agent._robot_wrapper._target_right_hand_joint_positions = (
                        target_hand_pos
                    )
                    self.env.sim.articulated_agent._robot_wrapper._target_right_arm_joint_positions = (
                        target_arm_joints
                    )
                    self.pin_left_arm()

                    obs = self.env.step(action)
                    im = self.process_obs_img(obs)
                    im = add_text_to_image(im, "using vla policy")
                    self.writer.append_data(im)

                    curr_ee_pos, curr_ee_rot_rpy = self.get_curr_ee_pose()
                    ctr += 1
                    if ctr > timeout:
                        print(
                            f"Exceeded time limit. {np.abs(curr_ee_pos-target_ee_pos)} away from target"
                        )
                        break
                    self.writer.append_data(im)

                elif action_type == "ee":
                    (
                        target_ee_pos,
                        target_ee_rot,
                        target_hand_pos,
                        base_lin_vel,
                        base_ang_vel,
                    ) = self.process_vla_action(vla_action)
                    curr_ee_pos, curr_ee_rot_rpy = self.get_curr_ee_pose()

                    while not np.allclose(
                        curr_ee_pos, target_ee_pos, atol=0.02
                    ):
                        action = {
                            "action": (
                                "arm_reach_ee_action",
                                "base_velocity_action",
                            ),
                            "action_args": {
                                "target_pos": np.array(
                                    target_ee_pos, dtype=np.float32
                                ),
                                "target_rot": target_ee_rot,
                                "base_vel": np.array(
                                    [base_lin_vel, base_ang_vel],
                                    dtype=np.float32,
                                ),
                            },
                        }
                        self.env.sim.articulated_agent.base_transformation = (
                            self.base_trans
                        )
                        self.env.sim.articulated_agent._robot_wrapper._target_right_hand_joint_positions = (
                            target_hand_pos
                        )
                        self.pin_left_arm()

                        obs = self.env.step(action)

                        im = self.process_obs_img(obs)
                        im = add_text_to_image(im, "using vla policy")
                        self.writer.append_data(im)

                        curr_ee_pos, curr_ee_rot_rpy = self.get_curr_ee_pose()
                        ctr += 1
                        if ctr > timeout:
                            print(
                                f"Exceeded time limit. {np.abs(curr_ee_pos-target_ee_pos)} away from target"
                            )
                            break
                        self.writer.append_data(im)
                self.episode_json["episode_data"].append(sim_step_data)

        # self.save_json()
        self.writer.close()
        print("saved video at: ", self.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your program description")

    # Add arguments
    parser.add_argument(
        "--target-name", default="shelf", help="target object name"
    )
    parser.add_argument("--skill", default="open", help="open, pick")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="seed")

    args = parser.parse_args()
    evaluator = VLAEvaluator(
        args.target_name, args.skill, args.replay, args.seed
    )

    if args.replay:
        evaluator.run_policy_replay()
    else:
        evaluator.run_policy()
