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
from habitat.config.default_structured_configs import ThirdRGBSensorConfig, HeadRGBSensorConfig, ArmDepthSensorConfig, HeadPanopticSensorConfig
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
    sim_cfg.habitat_sim_v0.frustum_culling = False

    
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


from habitat.tasks.utils import get_angle
from habitat.datasets.rearrange.navmesh_utils import compute_turn
class OracleNavSkill():
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
        action = {'action': 'base_velocity_action', 'action_args': {'base_vel': np.array([ vel[0], vel[1]], dtype=np.float32)}}
        return action


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
        "articulated_agent_arm_depth": ArmDepthSensorConfig(),
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

    writer2 = imageio.get_writer(
        "output_env_head.mp4",
        fps=30,
    )
    action1 = {'action': 'base_velocity_action', 'action_args': {'base_vel': np.array([ 5.0, 0], dtype=np.float32)}}
    action2 = {'action': 'base_velocity_action', 'action_args': {'base_vel': np.array([ 0, 5], dtype=np.float32)}}
    action3 = {'action': 'base_velocity_action', 'action_args': {'base_vel': np.array([ 0, 0], dtype=np.float32)}}
    
    first_obj = env.sim._rigid_objects[0].translation
    nav_point = env.sim.pathfinder.get_random_navigable_point_near(circle_center=first_obj, radius=1)
    curr_pos = env.sim.articulated_agent.base_pos
    dist = np.linalg.norm((np.array(curr_pos) - nav_point) * np.array([1,0,1]))
    nav_planner = OracleNavSkill(env, nav_point)
    i = 0
    while dist > 0.10 or i < 200:
        
        i += 1
        action_planner = nav_planner.get_step()
        obs = env.step(action_planner)
        im = obs["third_rgb"]
        
        writer.append_data(im)
    
        curr_pos = env.sim.articulated_agent.base_pos
        dist = np.linalg.norm((np.array(curr_pos) - nav_point) * np.array([1,0,1]))
        print(dist)
    writer.close()
    breakpoint()

    writer2.close()

        

if __name__ == "__main__":
    main()