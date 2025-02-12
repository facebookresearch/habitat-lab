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

# data_path = "/fsx-siro/xavierpuig/projects/habitat_isaac/habitat-lab/data/"
data_path = "/opt/hpcaas/.mounts/fs-03ee9f8c6dddfba21/jtruong/data"


def make_sim_cfg(agent_dict):
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


def make_hab_cfg(agent_dict, action_dict):
    sim_cfg = make_sim_cfg(agent_dict)
    task_cfg = TaskConfig(type="RearrangeEmptyTask-v0")
    task_cfg.actions = action_dict
    env_cfg = EnvironmentConfig()
    dataset_cfg = DatasetConfig(
        type="RearrangeDataset-v0",
        data_path="data/datasets/hssd/rearrange/test/rearrange_ep_dataset_55k.json.gz",  # "data/hab3_bench_assets/episode_datasets/small_large.json.gz",
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


def init_rearrange_env(agent_dict, action_dict):
    hab_cfg = make_hab_cfg(agent_dict, action_dict)
    res_cfg = OmegaConf.create(hab_cfg)
    print("hab_cfg: ", hab_cfg)
    print("res_cfg: ", res_cfg)
    return Env(res_cfg)


from habitat.datasets.rearrange.navmesh_utils import compute_turn
from habitat.tasks.utils import get_angle


class OracleNavSkill:
    def __init__(self, env, target_pos):
        self.env = env
        self.target_pos = target_pos
        self.target_base_pos = target_pos
        self.dist_thresh = 0.1
        self.turn_velocity = 2

        self.forward_velocity = 10
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


def get_point_along_vector(
    curr: np.ndarray, goal: np.ndarray, step_size: float = 0.15
) -> np.ndarray:
    vector = goal - curr
    distance = np.linalg.norm(vector)
    if distance <= step_size:
        return goal

    # Normalize the vector and multiply by step_size
    unit_vector = vector / distance
    new_point = curr + (unit_vector * step_size)

    return np.round(new_point, 6)


def move_to_ee(env, writer, target_ee_pos, visualize=False):
    offset = mn.Vector3(0, 0.4, 0)
    ctr = 0
    curr_ee_pos, _ = env.sim.articulated_agent._robot_wrapper.ee_pose()
    while not np.allclose(curr_ee_pos, target_ee_pos, atol=0.16):
        target_ee_pos_shift = target_ee_pos - offset
        arm_reach = {
            "action": "arm_reach_ee_action",
            "action_args": {
                "target_pos": np.array(target_ee_pos_shift, dtype=np.float32)
            },
        }
        if visualize:
            env.sim.viz_ids["place_tar"] = env.sim.visualize_position(
                target_ee_pos, env.sim.viz_ids["place_tar"]
            )
            env.sim.viz_ids["place_tar_shift"] = env.sim.visualize_position(
                target_ee_pos_shift, env.sim.viz_ids["place_tar_shift"]
            )

        obs = env.step(arm_reach)

        im = process_obs_img(obs)
        writer.append_data(im)
        curr_ee_pos, _ = env.sim.articulated_agent._robot_wrapper.ee_pose()
        ctr += 1
        # print(
        #     "ee ctr: ",
        #     ctr,
        #     curr_ee_pos,
        #     target_ee_pos,
        #     np.absolute(curr_ee_pos - target_ee_pos),
        # )
        if ctr > 200:
            break


def visualize_pos(env, target_pos, r=0.05):
    env.sim.viz_ids["target"] = env.sim.visualize_position(
        target_pos, env.sim.viz_ids["target"], r=r
    )


def move_to_joint(env, writer, target_joint_pos, timeout=200, visualize=False):
    # joint_pos = [0.0, -2.0943951, 0.0, 1.04719755, 0.0, 1.53588974, 0.0, -1.57]
    # -1.57 is open, 0 is closed
    ctr = 0
    curr_joint_pos = env.sim.articulated_agent._robot_wrapper.arm_joint_pos

    while not np.allclose(
        curr_joint_pos,
        target_joint_pos,
        atol=0.003,
    ):
        if visualize:
            curr_ee_pos, _ = env.sim.articulated_agent._robot_wrapper.ee_pose()

            env.sim.viz_ids["place_tar"] = env.sim.visualize_position(
                curr_ee_pos, env.sim.viz_ids["place_tar"]
            )
        env.sim.articulated_agent._robot_wrapper._target_arm_joint_positions = (
            target_joint_pos
        )
        action = {
            "action": "base_velocity_action",
            "action_args": {
                "base_vel": np.array([0.0, 0.0], dtype=np.float32)
            },
        }
        obs = env.step(action)
        im = process_obs_img(obs)
        writer.append_data(im)
        ctr += 1
        curr_joint_pos = env.sim.articulated_agent._robot_wrapper.arm_joint_pos
        # print(
        #     "joint ctr: ",
        #     ctr,
        #     curr_joint_pos,
        #     target_joint_pos,
        #     np.absolute(curr_joint_pos - target_joint_pos),
        # )
        if ctr > timeout:
            break


def nav_to_obj(env, writer, target_obj):
    nav_point = env.sim.pathfinder.get_random_navigable_point_near(
        circle_center=target_obj, radius=1
    )
    curr_pos = env.sim.articulated_agent.base_pos

    dist = np.linalg.norm(
        (np.array(curr_pos) - nav_point) * np.array([1, 0, 1])
    )
    nav_planner = OracleNavSkill(env, nav_point)
    i = 0
    while dist > 0.10 and i < 200:

        i += 1
        action_planner = nav_planner.get_step()
        obs = env.step(action_planner)
        im = process_obs_img(obs)
        writer.append_data(im)
        curr_pos = env.sim.articulated_agent.base_pos
        dist = np.linalg.norm(
            (np.array(curr_pos) - nav_point) * np.array([1, 0, 1])
        )
    print(
        "root_pose 2: ",
        env.sim.articulated_agent._robot_wrapper.get_root_pose(),
    )


def main():
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
        "articulated_agent_arm_depth": ArmDepthSensorConfig(),
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
    env = init_rearrange_env(agent_dict, action_dict)

    aux = env.reset()
    writer = imageio.get_writer(
        "output_env.mp4",
        fps=30,
    )

    action_example = {
        "action": "base_velocity_action",
        "action_args": {"base_vel": np.array([5.0, 0], dtype=np.float32)},
    }

    # Joints
    # dof names:  ['arm0_sh0', 'fl_hx', 'fr_hx', 'hl_hx', 'hr_hx', 'arm0_sh1', 'fl_hy',
    # 'fr_hy', 'hl_hy', 'hr_hy', 'arm0_hr0', 'fl_kn', 'fr_kn',
    # 'hl_kn', 'hr_kn', 'arm0_el0', 'arm0_el1', 'arm0_wr0', 'arm0_wr1', 'arm0_f1x']

    print(
        "root_pose 1: ",
        env.sim.articulated_agent._robot_wrapper.get_root_pose(),
    )

    # -1.57 is open, 0 is closed
    target_joint_pos = [
        0.0,
        -2.0943951,
        0.0,
        1.04719755,
        0.0,
        1.53588974,
        0.0,
        -1.57,
    ]
    move_to_joint(env, writer, target_joint_pos, visualize=False)
    print(
        "env.sim._rigid_objects: ",
        env.sim._rigid_objects,
        len(env.sim._rigid_objects),
    )
    first_obj = env.sim._rigid_objects[
        2
    ].translation  # first_obj in habitat conventions
    visualize_pos(env, first_obj)
    nav_to_obj(env, writer, first_obj)

    move_to_ee(env, writer, first_obj, visualize=False)

    curr_ee_pos, _ = env.sim.articulated_agent._robot_wrapper.ee_pose()
    ee_to_obj_dist = curr_ee_pos[1] - first_obj[1]
    target_joint_pos = env.sim.articulated_agent._robot_wrapper.arm_joint_pos
    # target_joint_pos[1] += 0.1
    # target_joint_pos[3] += ee_to_obj_dist - 0.14

    # move_to_joint(env, writer, target_joint_pos, timeout=200, visualize=False)

    while np.abs(ee_to_obj_dist) > 0.17:
        print("Translating down: ", ee_to_obj_dist)
        target_joint_pos = (
            env.sim.articulated_agent._robot_wrapper.arm_joint_pos
        )
        # target_joint_pos[1] += 0.1
        target_joint_pos[3] += 0.05

        move_to_joint(
            env, writer, target_joint_pos, timeout=200, visualize=False
        )
        curr_ee_pos, _ = env.sim.articulated_agent._robot_wrapper.ee_pose()
        ee_to_obj_dist = curr_ee_pos[1] - first_obj[1]
        print("ee_to_obj_dist: ", ee_to_obj_dist)

    print("closing gripper")
    target_joint_pos = env.sim.articulated_agent._robot_wrapper.arm_joint_pos
    target_joint_pos[-1] = 0.0
    move_to_joint(env, writer, target_joint_pos, timeout=100, visualize=False)
    print(
        "finished closing: ",
        env.sim.articulated_agent._robot_wrapper.arm_joint_pos,
    )

    print("Translating up")
    target_joint_pos = env.sim.articulated_agent._robot_wrapper.arm_joint_pos
    target_joint_pos[3] += 0.1
    move_to_joint(env, writer, target_joint_pos, timeout=200, visualize=False)

    # 12 rigid objects
    # visualize_pos(env, env.sim._rigid_objects[3].translation, r=0.05)
    # visualize_pos(env, env.sim._rigid_objects[4].translation, r=0.1)
    # visualize_pos(env, env.sim._rigid_objects[5].translation, r=0.15)

    print("moving arm to rest")
    # first_obj in habitat conventions
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
    move_to_joint(env, writer, target_joint_pos, visualize=False)

    writer.close()


if __name__ == "__main__":
    main()
