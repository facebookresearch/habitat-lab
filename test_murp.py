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


def main():
    # Define the agent configuration
    main_agent_config = AgentConfig()

    urdf_path = os.path.join(
        data_path,
        "hab_murp/murp_tmr_franka/murp_tmr_franka_metahand_obj.urdf",
    )
    # arm_urdf_path = os.path.join(
    #     data_path,
    #     "hab_murp/murp_tmr_franka/murp_tmr_franka_metahand_right_arm_obj.urdf",
    # )
    main_agent_config.urdf = urdf_path
    main_agent_config.articulated_agent_type = "MurpRobot"
    # main_agent_config.articulated_agent_type = "SpotRobot"
    # main_agent_config.ik_arm_urdf = arm_urdf_path

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

    start_position = np.array([2.0, 0.7, -1.64570129])
    start_rotation = -90

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
    while not np.allclose(
        curr_left_joint_pos,
        target_joint_pos,
        atol=0.003,
    ):
        env.sim.articulated_agent._robot_wrapper._target_arm_joint_positions = (
            target_joint_pos
        )
        env.sim.articulated_agent._robot_wrapper._target_right_arm_joint_positions = (
            target_joint_pos
        )
        action = {
            "action": "base_velocity_action",
            "action_args": {
                "base_vel": np.array([0.0, 0.0], dtype=np.float32)
            },
        }
        curr_left_joint_pos = (
            env.sim.articulated_agent._robot_wrapper.arm_joint_pos
        )
        print("curr_left_joint_pos: ", curr_left_joint_pos)
        print(
            "curr_right_joint_pos: ",
            env.sim.articulated_agent._robot_wrapper.right_arm_joint_pos,
        )
        print("ctr: ", ctr)
        ctr += 1
        obs = env.step(action)
        im = process_obs_img(obs)
        writer.append_data(im)
        if ctr > 2000:
            break

    writer.close()


if __name__ == "__main__":
    main()
