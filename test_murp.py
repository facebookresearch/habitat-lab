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


def get_point_along_vector(
    curr: np.ndarray, goal: np.ndarray, step_size: float = 0.15
) -> np.ndarray:
    """
    Calculate a point that lies on the vector from curr to goal,
    at a specified distance (step_size) from curr.

    Args:
        curr: numpy array [x, y, z] representing current position
        goal: numpy array [x, y, z] representing goal position
        step_size: Distance the returned point should be from curr (default 0.15)

    Returns:
        numpy array [x, y, z] representing the calculated point
        If distance between curr and goal is less than step_size, returns goal
    """
    # Calculate vector from curr to goal
    vector = goal - curr

    # Calculate distance between points
    distance = np.linalg.norm(vector)

    # If distance is less than step_size, return goal
    if distance <= step_size:
        return goal

    # Normalize the vector and multiply by step_size
    unit_vector = vector / distance
    new_point = curr + (unit_vector * step_size)

    # Round to reduce floating point errors
    return np.round(new_point, 6)


def main():
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
    env = init_rearrange_env(agent_dict, action_dict)

    aux = env.reset()
    writer = imageio.get_writer(
        "output_env_murp.mp4",
        fps=30,
    )

    # start_position = np.array([2.0, 0.1, -1.0])
    # # living room cabinet
    start_position = np.array([1.7, 0.1, -0.2])
    start_rotation = -90

    # # kitchen shelf
    # start_position = np.array([-4.5, 0.1, -3.5])
    # start_rotation = 180

    # kitchen island
    # start_position = np.array([-5.25, 0.1, -1.6])
    # start_rotation = 0

    position = mn.Vector3(start_position)
    rotation = mn.Quaternion.rotation(
        mn.Deg(start_rotation), mn.Vector3.y_axis()
    )
    trans = mn.Matrix4.from_(rotation.to_matrix(), position)
    env.sim.articulated_agent.base_transformation = trans
    print("set base: ", start_position, start_rotation)

    ctr = 0
    timeout = 500
    curr_left_joint_pos = (
        env.sim.articulated_agent._robot_wrapper.arm_joint_pos
    )
    print("curr_left_joint_pos: ", curr_left_joint_pos)
    target_joint_pos = np.array(
        [
            2.6116285,
            1.5283098,
            1.0930868,
            -0.50559217,
            0.48147443,
            2.628784,
            -1.3962275,
        ]
    )
    open_positions = np.zeros(10)

    target_ee_pos = env.sim._rigid_objects[0].translation
    ctr = 0
    curr_ee_pos, _ = env.sim.articulated_agent._robot_wrapper.ee_pose()

    rest_positions = np.array(
        [
            2.6116285,
            1.5283098,
            1.0930868,
            -0.50559217,
            0.48147443,
            2.628784,
            -1.3962275,
        ]
    )
    rest_positions = np.zeros(7)
    while not np.allclose(curr_ee_pos, target_ee_pos, atol=0.08):
        curr_ee_pos, curr_ee_rot = (
            env.sim.articulated_agent._robot_wrapper.ee_pose()
        )
        # curr_target_ee = get_point_along_vector(
        #     curr_ee_pos, target_ee_pos, 0.15
        # )
        # env.sim.viz_ids["curr_ee_pos"] = env.sim.visualize_position(
        #     curr_ee_pos, env.sim.viz_ids["curr_ee_pos"]
        # )
        # env.sim.viz_ids["curr_target_ee"] = env.sim.visualize_position(
        #     curr_target_ee, env.sim.viz_ids["curr_target_ee"]
        # )
        # print(f"{curr_ee_pos=}, {curr_target_ee=}, {target_ee_pos=}")

        arm_reach = {
            "action": "arm_reach_ee_action",
            "action_args": {
                "target_pos": np.array(target_ee_pos, dtype=np.float32)
            },
        }
        env.sim.articulated_agent.base_transformation = trans
        env.sim.articulated_agent._robot_wrapper._target_right_arm_joint_positions = (
            rest_positions
        )
        env.sim.articulated_agent._robot_wrapper._target_hand_joint_positions = (
            open_positions
        )
        env.sim.articulated_agent._robot_wrapper._target_right_hand_joint_positions = (
            open_positions
        )

        obs = env.step(arm_reach)
        curr_joint_pos = env.sim.articulated_agent._robot_wrapper.arm_joint_pos
        curr_right_joint_pos = (
            env.sim.articulated_agent._robot_wrapper.right_arm_joint_pos
        )
        curr_ee_pos, curr_ee_rot = (
            env.sim.articulated_agent._robot_wrapper.ee_pose()
        )

        curr_ee_rot_quat = R.from_quat(
            [*curr_ee_rot.vector, curr_ee_rot.scalar]
        )
        curr_ee_rot_rpy = curr_ee_rot_quat.as_euler("xyz", degrees=True)
        print("curr_ee_pos: ", curr_ee_pos, "curr_ee_rot: ", curr_ee_rot_rpy)
        print(f"{curr_joint_pos=}")
        # print(f"{curr_right_joint_pos=}")
        im = process_obs_img(obs)
        writer.append_data(im)
        ctr += 1
        print("ctr: ", ctr)
        if ctr > 1000:
            break

    closed_positions = np.array([3.14159] * 10)
    # hand_positions = np.array()
    for _ in range(100):
        action = {
            "action": "base_velocity_action",
            "action_args": {
                "base_vel": np.array([0.0, 0.0], dtype=np.float32)
            },
        }
        env.sim.articulated_agent.base_transformation = trans
        env.sim.articulated_agent._robot_wrapper._target_hand_joint_positions = (
            closed_positions
        )
        env.sim.articulated_agent._robot_wrapper._target_right_hand_joint_positions = (
            closed_positions
        )
        curr_ee_pos, curr_ee_rot = (
            env.sim.articulated_agent._robot_wrapper.ee_pose()
        )
        print("curr_ee_pos: ", curr_ee_pos, "curr_ee_rot: ", curr_ee_rot)

        print("closing gripper")
        obs = env.step(arm_reach)
        im = process_obs_img(obs)
        writer.append_data(im)

    # # raise arm
    # target_joint_pos = np.array(
    #     [
    #         -1.5707963268,
    #         -1.6567535,
    #         -0.59521294,
    #         -1.51944518,
    #         -1.0471975512,
    #         1.8663365,
    #         -0.5235987756,
    #     ]
    # )
    # curr_joint_pos = env.sim.articulated_agent._robot_wrapper.arm_joint_pos
    # ctr = 0
    # for _ in range(300):
    #     env.sim.articulated_agent._robot_wrapper._target_arm_joint_positions = (
    #         target_joint_pos
    #     )
    #     env.sim.articulated_agent._robot_wrapper._target_right_arm_joint_positions = (
    #         rest_positions
    #     )
    #     env.sim.articulated_agent._robot_wrapper._target_hand_joint_positions = (
    #         closed_positions
    #     )
    #     env.sim.articulated_agent._robot_wrapper._target_right_hand_joint_positions = (
    #         closed_positions
    #     )
    #     env.sim.articulated_agent.base_transformation = trans
    #     action = {
    #         "action": "base_velocity_action",
    #         "action_args": {
    #             "base_vel": np.array([0.0, 0.0], dtype=np.float32)
    #         },
    #     }
    #     curr_joint_pos = env.sim.articulated_agent._robot_wrapper.arm_joint_pos
    #     curr_ee_pos, curr_ee_rot = (
    #         env.sim.articulated_agent._robot_wrapper.ee_pose()
    #     )

    #     curr_ee_rot_quat = R.from_quat(
    #         [*curr_ee_rot.vector, curr_ee_rot.scalar]
    #     )
    #     curr_ee_rot_rpy = curr_ee_rot_quat.as_euler("xyz", degrees=True)

    #     # print(
    #     #     "hand_joint_pos: ",
    #     #     env.sim.articulated_agent._robot_wrapper.hand_joint_pos,
    #     # )
    #     print("curr_joint_pos: ", curr_joint_pos)
    #     print("curr_ee_pos: ", curr_ee_pos, "curr_ee_rot: ", curr_ee_rot_rpy)
    #     ctr += 1
    #     print("raising arm")
    #     obs = env.step(action)
    #     im = process_obs_img(obs)
    #     writer.append_data(im)
    #     if ctr > 500:
    #         break

    writer.close()


if __name__ == "__main__":
    main()
