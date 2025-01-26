import habitat_sim
import magnum as mn
import warnings
from habitat.tasks.rearrange.isaac_rearrange_sim import IsaacRearrangeSim
warnings.filterwarnings('ignore')
from habitat_sim.utils.settings import make_cfg
from matplotlib import pyplot as plt
from habitat_sim.utils import viz_utils as vut
from omegaconf import DictConfig
import numpy as np
from habitat.articulated_agents.robots import FetchRobot
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import ThirdRGBSensorConfig, HeadRGBSensorConfig, HeadPanopticSensorConfig
from habitat.config.default_structured_configs import SimulatorConfig, HabitatSimV0Config, AgentConfig
from habitat.config.default import get_agent_config
import habitat
from habitat_sim.physics import JointMotorSettings, MotionType
from omegaconf import OmegaConf
import os
from habitat.isaac_sim.isaac_app_wrapper import IsaacAppWrapper
from habitat.isaac_sim import isaac_prim_utils
import random
from habitat.config.default_structured_configs import TaskConfig, EnvironmentConfig, DatasetConfig, HabitatConfig
from habitat.config.default_structured_configs import ArmActionConfig, BaseVelocityActionConfig, OracleNavActionConfig, ActionConfig
import imageio
from habitat.core.env import Env

data_path = "/fsx-siro/xavierpuig/projects/habitat_isaac/habitat-lab/data/"


def make_sim_cfg(agent_dict):
    # Start the scene config
    sim_cfg = SimulatorConfig(type="IsaacRearrangeSim-v0")
    
    # This is for better graphics
    sim_cfg.habitat_sim_v0.enable_hbao = True
    sim_cfg.habitat_sim_v0.enable_physics = False

    
    # Set up an example scene
    sim_cfg.scene = "NONE" # os.path.join(data_path, "hab3_bench_assets/hab3-hssd/scenes/103997919_171031233.scene_instance.json")
    # sim_cfg.scene_dataset = os.path.join(data_path, "hab3_bench_assets/hab3-hssd/hab3-hssd.scene_dataset_config.json")
    # sim_cfg.additional_object_paths = [os.path.join(data_path, 'objects/ycb/configs/')]

    
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
    dataset_cfg = DatasetConfig(type="RearrangeDataset-v0", data_path="data/hab3_bench_assets/episode_datasets/small_large.json.gz")
    
    
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
    return Env(res_cfg)



def main():
    # Define the agent configuration
    main_agent_config = AgentConfig()
    
    urdf_path = os.path.join(data_path, "robots/hab_spot_arm/urdf/hab_spot_arm.urdf")
    main_agent_config.articulated_agent_urdf = urdf_path
    main_agent_config.articulated_agent_type = "SpotRobot"

    # Define sensors that will be attached to this agent, here a third_rgb sensor and a head_rgb.
    # We will later talk about why we are giving the sensors these names
    main_agent_config.sim_sensors = {
        "third_rgb": ThirdRGBSensorConfig(),
        "head_rgb": HeadRGBSensorConfig(),
    }

    # We create a dictionary with names of agents and their corresponding agent configuration
    agent_dict = {"main_agent": main_agent_config}

    action_dict = {
        "base_velocity_action": BaseVelocityActionConfig(type="BaseVelIsaacAction"),
    }
    env = init_rearrange_env(agent_dict, action_dict)
    aux = env.reset()

    writer = imageio.get_writer(
        "output_env.mp4",
        fps=30,
    )
    action = {'action': 'base_velocity_action', 'action_args': {'base_vel': np.array([ 10.0, 0], dtype=np.float32)}}
    for i in range(100):
        
        obs = env.step(action)
        im = obs["third_rgb"]
        writer.append_data(im)
    writer.close()
    breakpoint()
    # def get_pick_target_pos():
    #     ro = isaac_viewer._rigid_objects[isaac_viewer._pick_target_rigid_object_idx]
    #     com_world = isaac_prim_utils.get_com_world(ro._rigid_prim)
    #     # self.draw_axis(0.05, mn.Matrix4.translation(com_world))
    #     return com_world

    # isaac_viewer._spot_state_machine.set_pick_target(get_pick_target_pos)

    # # breakpoint()
    # for it in range(100):
    #     isaac_viewer.update_isaac({})
    #     look_up = mn.Vector3(0,1,0)
    #     isaac_viewer.update_spot_pre_step(0.01)

    #     look_at = sim.agents_mgr._all_agent_data[0].articulated_agent.base_pos
    #     print(look_at)
    #     camera_pos = look_at + mn.Vector3(-0.7, 1.5, -0.7)
        
    #     sim._sensors["third_rgb"]._sensor_object.node.rotation =  mn.Quaternion.from_matrix(
    #         mn.Matrix4.look_at(camera_pos, look_at, look_up).rotation()
    #     )
        
    #     sim._sensors["third_rgb"]._sensor_object.node.translation = camera_pos
    #     # import cv2
    #     sim.reset()
    
    #     res = sim.get_sensor_observations()
        
    #     im = res["third_rgb"][:,:,[0,1,2]]
    #     import cv2
    #     cv2.imwrite("third2.png", im)
    #     breakpoint()
    #     writer.append_data(im)
        
    # writer.close()
    # breakpoint()
        

if __name__ == "__main__":
    main()