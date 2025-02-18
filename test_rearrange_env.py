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


class ExpertDatagen:
    def __init__(self, ep_id=0, mode="train"):
        if mode == "train":
            self.json_path = (
                "data/datasets/fremont_hlab_isaac/fremont_900.json.gz"
            )
        elif mode == "val":
            self.json_path = (
                "data/datasets/fremont_hlab_isaac/fremont_100.json.gz"
            )

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
                type="BaseVelKinematicIsaacAction"
            ),
            "arm_reach_action": ActionConfig(type="ArmReachAction"),
            "arm_reach_ee_action": ActionConfig(type="ArmReachEEAction"),
        }
        self.env = self.init_rearrange_env(agent_dict, action_dict)
        _ = self.env.reset()
        self.ep_id = ep_id

        data = read_dataset(self.json_path)
        self.episode_info = data["episodes"][self.ep_id]

        video_dir = f"videos_no_retract/{mode}"
        os.makedirs(video_dir, exist_ok=True)

        self.writer = imageio.get_writer(
            f"{video_dir}/output_env_{ep_id}.mp4",
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

        self.episode_json = {
            "episode_id": self.episode_info["episode_id"],
            "scene_id": self.episode_info["scene_id"],
            "language_instruction": self.episode_info["language_instruction"],
            "episode_data": [],
            "success": True,
        }
        self.obj_id = 0
        self.obj_id_dist_dict = {"0": 0.10, "2": 0.14, "10": 0.17}
        self.save_path = f"/opt/hpcaas/.mounts/fs-03ee9f8c6dddfba21/jtruong/repos/vla-physics/habitat-lab/sim_data/fremont_no_retract/{mode}"
        self.save_keys = [
            "articulated_agent_arm_rgb",
            "articulated_agent_arm_depth",
        ]

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
            img_dir = (
                f'{self.save_path}/ep_{self.episode_json["episode_id"]}/imgs'
            )
            os.makedirs(img_dir, exist_ok=True)
            img_pth = f"{img_dir}/{img_key}_img_{int(time.time()*1000)}.png"
            cv2.imwrite(f"{img_pth}", robot_img)
            ep_data[img_key] = img_pth
        return ep_data

    def add_episode_data(self, obs, action):
        curr_joints = (
            self.env.sim.articulated_agent._robot_wrapper.arm_joint_pos
        )
        curr_ee_pos, curr_ee_rot = (
            self.env.sim.articulated_agent._robot_wrapper.ee_pose()
        )
        curr_base_pos, curr_base_rot = (
            self.env.sim.articulated_agent._robot_wrapper.get_root_pose()
        )
        target_position_YZX = self.env.sim._rigid_objects[
            self.obj_id
        ].translation

        ep_data = {
            "joints": np.array(curr_joints).tolist(),
            "ee_pos": np.array(curr_ee_pos).tolist(),
            "ee_rot": np.array(
                [*curr_ee_rot.vector, curr_ee_rot.scalar]
            ).tolist(),
            "base_pos": np.array(curr_base_pos).tolist(),
            "base_rot": np.array(
                [*curr_base_rot.vector, curr_base_rot.scalar]
            ).tolist(),
            "target_position": np.array(target_position_YZX).tolist(),
        }

        for save_key in self.save_keys:
            ep_data = self.save_img(obs, save_key, ep_data)

        ep_data["arm_action"] = action["arm_action"]
        ep_data["base_action"] = action["base_action"]
        ep_data["empty_action"] = action["empty_action"]
        ep_data["grip_action"] = action["grip_action"]

        self.episode_json["episode_data"].append(ep_data)

    def save_json(self):
        if self.episode_json["success"]:
            print("success! saving episode")
            os.makedirs(self.save_path, exist_ok=True)
            ep_save_path = f"{self.save_path}/ep_{self.ep_id}"
            with open(f"{ep_save_path}/ep_{self.ep_id}.json", "w") as outfile:
                json.dump(self.episode_json, outfile)

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

    def move_to_ee(self, visualize=False):
        target_ee_pos = self.env.sim._rigid_objects[self.obj_id].translation
        offset = mn.Vector3(0, 0.4, 0)
        ctr = 0
        curr_ee_pos, _ = (
            self.env.sim.articulated_agent._robot_wrapper.ee_pose()
        )
        while not np.allclose(curr_ee_pos, target_ee_pos, atol=0.16):
            target_ee_pos_shift = target_ee_pos - offset
            arm_reach = {
                "action": "arm_reach_ee_action",
                "action_args": {
                    "target_pos": np.array(
                        target_ee_pos_shift, dtype=np.float32
                    )
                },
            }
            if visualize:
                self.env.sim.viz_ids["place_tar"] = (
                    self.env.sim.visualize_position(
                        target_ee_pos, self.env.sim.viz_ids["place_tar"]
                    )
                )
                self.env.sim.viz_ids["place_tar_shift"] = (
                    self.env.sim.visualize_position(
                        target_ee_pos_shift,
                        self.env.sim.viz_ids["place_tar_shift"],
                    )
                )

            obs = self.env.step(arm_reach)
            curr_ee_pos, _ = (
                self.env.sim.articulated_agent._robot_wrapper.ee_pose()
            )
            save_action = {
                "arm_action": np.array(curr_ee_pos).tolist(),
                "base_action": [0, 0],
                "empty_action": [0],
                "grip_action": [0],
            }

            self.add_episode_data(obs, save_action)

            im = self.process_obs_img(obs)
            self.writer.append_data(im)
            curr_ee_pos, _ = (
                self.env.sim.articulated_agent._robot_wrapper.ee_pose()
            )
            ctr += 1
            if ctr > 200:
                break

    def visualize_pos(self, target_pos, r=0.05):
        self.env.sim.viz_ids["target"] = self.env.sim.visualize_position(
            target_pos, self.env.sim.viz_ids["target"], r=r
        )

    def move_to_joint(
        self, target_joint_pos, timeout=200, visualize=False, save_info=True
    ):
        # joint_pos = [0.0, -2.0943951, 0.0, 1.04719755, 0.0, 1.53588974, 0.0, -1.57]
        # -1.57 is open, 0 is closed
        ctr = 0
        curr_joint_pos = (
            self.env.sim.articulated_agent._robot_wrapper.arm_joint_pos
        )

        while not np.allclose(
            curr_joint_pos,
            target_joint_pos,
            atol=0.003,
        ):
            if visualize:
                curr_ee_pos, _ = (
                    self.env.sim.articulated_agent._robot_wrapper.ee_pose()
                )

                self.env.sim.viz_ids["place_tar"] = (
                    self.env.sim.visualize_position(
                        curr_ee_pos, self.env.sim.viz_ids["place_tar"]
                    )
                )
            self.env.sim.articulated_agent._robot_wrapper._target_arm_joint_positions = (
                target_joint_pos
            )
            action = {
                "action": "base_velocity_action",
                "action_args": {
                    "base_vel": np.array([0.0, 0.0], dtype=np.float32)
                },
            }
            obs = self.env.step(action)
            im = self.process_obs_img(obs)
            self.writer.append_data(im)
            ctr += 1
            curr_joint_pos = (
                self.env.sim.articulated_agent._robot_wrapper.arm_joint_pos
            )
            curr_ee_pos, _ = (
                self.env.sim.articulated_agent._robot_wrapper.ee_pose()
            )
            save_action = {
                "arm_action": np.array(curr_ee_pos).tolist(),
                "base_action": [0, 0],
                "empty_action": [0],
                "grip_action": [1] if curr_joint_pos[-1] > -1.57 else [0],
            }
            if save_info:
                self.add_episode_data(obs, save_action)
            if ctr > timeout:
                break

    def nav_to_obj(self):
        target_obj = self.env.sim._rigid_objects[
            self.obj_id
        ].translation  # first_obj in habitat conventions
        nav_point = self.env.sim.pathfinder.get_random_navigable_point_near(
            circle_center=target_obj, radius=1
        )
        curr_pos = self.env.sim.articulated_agent.base_pos

        dist = np.linalg.norm(
            (np.array(curr_pos) - nav_point) * np.array([1, 0, 1])
        )
        nav_planner = OracleNavSkill(self.env, nav_point)
        i = 0
        while dist > 0.5 and i < 200:
            i += 1
            action_planner = nav_planner.get_step()
            obs = self.env.step(action_planner)
            curr_ee_pos, _ = (
                self.env.sim.articulated_agent._robot_wrapper.ee_pose()
            )
            save_action = {
                "arm_action": np.array(curr_ee_pos).tolist(),
                "base_action": action_planner["action_args"][
                    "base_vel"
                ].tolist(),
                "empty_action": [0],
                "grip_action": [0],
            }

            self.add_episode_data(obs, save_action)
            im = self.process_obs_img(obs)
            self.writer.append_data(im)
            curr_pos = self.env.sim.articulated_agent.base_pos
            dist = np.linalg.norm(
                (np.array(curr_pos) - nav_point) * np.array([1, 0, 1])
            )
        print(
            "root_pose 2: ",
            self.env.sim.articulated_agent._robot_wrapper.get_root_pose(),
        )

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

    def retract_arm(self, save_info=True):
        print("Translating up")
        for _ in range(1):
            target_joint_pos = (
                self.env.sim.articulated_agent._robot_wrapper.arm_joint_pos
            )
            target_joint_pos[3] -= 0.1
            target_joint_pos[-1] = 0.0
            self.move_to_joint(
                target_joint_pos,
                timeout=100,
                visualize=False,
                save_info=save_info,
            )

        # 12 rigid objects
        # visualize_pos(env, env.sim._rigid_objects[3].translation, r=0.05)
        # visualize_pos(env, env.sim._rigid_objects[4].translation, r=0.1)
        # visualize_pos(env, env.sim._rigid_objects[5].translation, r=0.15)

        # target_joint_pos = env.sim.articulated_agent._robot_wrapper.arm_joint_pos
        # target_joint_pos[-1] = 0.0
        # move_to_joint(env, writer, target_joint_pos, timeout=100, visualize=False)

        target_joint_pos = (
            self.env.sim.articulated_agent._robot_wrapper.arm_joint_pos
        )
        target_joint_pos[1] = -2.0943951
        target_joint_pos[3] = 1.04719755
        target_joint_pos[-1] = 0.0
        self.move_to_joint(
            target_joint_pos, timeout=100, visualize=False, save_info=save_info
        )

        # print("moving arm to rest")
        target_joint_pos = [
            0.0,
            -2.0943951,
            0.0,
            1.04719755,
            0.0,
            1.53588974,
            0.0,
            0.0,
        ]
        self.move_to_joint(
            target_joint_pos, visualize=False, save_info=save_info
        )
        print(
            "final gripper: ",
            np.round(
                self.env.sim.articulated_agent._robot_wrapper.arm_joint_pos[
                    -1
                ],
                2,
            ),
        )

    def grasp_object(self):
        first_obj = self.env.sim._rigid_objects[
            self.obj_id
        ].translation  # first_obj in habitat conventions
        curr_ee_pos, _ = (
            self.env.sim.articulated_agent._robot_wrapper.ee_pose()
        )
        ee_to_obj_dist = curr_ee_pos[1] - first_obj[1]
        target_joint_pos = (
            self.env.sim.articulated_agent._robot_wrapper.arm_joint_pos
        )
        # visualize_pos(env, first_obj)
        print("ee_to_obj_dist: ", ee_to_obj_dist)
        # while np.abs(ee_to_obj_dist) > 0.17:

        while np.abs(ee_to_obj_dist) > self.obj_id_dist_dict[str(self.obj_id)]:
            print("Translating down: ", ee_to_obj_dist)
            target_joint_pos = (
                self.env.sim.articulated_agent._robot_wrapper.arm_joint_pos
            )
            target_joint_pos[3] += 0.05

            self.move_to_joint(target_joint_pos, timeout=200, visualize=False)
            curr_ee_pos, _ = (
                self.env.sim.articulated_agent._robot_wrapper.ee_pose()
            )
            first_obj = self.env.sim._rigid_objects[self.obj_id].translation
            ee_to_obj_dist = curr_ee_pos[1] - first_obj[1]
            print("ee_to_obj_dist: ", ee_to_obj_dist)

        print("closing gripper")
        target_joint_pos = (
            self.env.sim.articulated_agent._robot_wrapper.arm_joint_pos
        )
        target_joint_pos[-1] = 0.0
        self.move_to_joint(target_joint_pos, timeout=100, visualize=False)
        print(
            "finished closing: ",
            np.round(
                self.env.sim.articulated_agent._robot_wrapper.arm_joint_pos[
                    -1
                ],
                2,
            ),
        )

    def run_expert(self):
        # Joints
        # dof names:  ['arm0_sh0', 'fl_hx', 'fr_hx', 'hl_hx', 'hr_hx', 'arm0_sh1', 'fl_hy',
        # 'fr_hy', 'hl_hy', 'hr_hy', 'arm0_hr0', 'fl_kn', 'fr_kn',
        # 'hl_kn', 'hr_kn', 'arm0_el0', 'arm0_el1', 'arm0_wr0', 'arm0_wr1', 'arm0_f1x']
        self.reset_robot_pose()

        self.nav_to_obj()

        self.move_to_ee(visualize=False)

        self.grasp_object()

        self.retract_arm(save_info=False)

        gripper_state = (
            self.env.sim.articulated_agent._robot_wrapper.arm_joint_pos[-1]
        )
        if np.allclose(gripper_state, 0.0, atol=0.03):
            self.episode_json["success"] = False

        self.save_json()
        self.writer.close()


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Your program description")

    # Add arguments
    parser.add_argument("--ep-id", type=int, default=0, help="Episode id")
    parser.add_argument("--mode", default="train", help="Mode")

    args = parser.parse_args()

    datagen = ExpertDatagen(args.ep_id, args.mode)
    datagen.run_expert()
