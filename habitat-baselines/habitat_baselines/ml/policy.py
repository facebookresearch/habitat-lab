from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.nn import DataParallel

import habitat_baselines.ml.utils.pose_utils as pu
from habitat.core.simulator import Observations
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.ml.mapping.dense.semantic.categorical_2d_semantic_map_state import (
    Categorical2DSemanticMapState,
)
from habitat_baselines.ml.navigation_planner.discrete_planner import (
    DiscretePlanner,
)
from habitat_baselines.ml.obs_preprocessor.obs_preprocessor import (
    ObsPreprocessor,
)
from habitat_baselines.ml.visualizer.constants import (
    FloorplannertoSriramObjects,
)
from habitat_baselines.ml.visualizer.visualizer import Visualizer
from habitat_baselines.rl.ppo.policy import Policy

from .objectnav_agent_module import ObjectNavAgentModule


@baseline_registry.register_policy
class ModularFrontierExplorationBaselinePolicy(Policy):
    def __init__(self, config, envs, device):
        self.config = config
        self.device = device
        self.map_resolution = self.config.habitat.ml_environment.map_resolution
        self.map_size_cm = self.config.habitat.ml_environment.map_size_cm
        self.global_downscaling = (
            self.config.habitat.ml_environment.global_downscaling
        )
        self.num_sem_categories = (
            self.config.habitat.ml_environment.num_sem_categories
        )

        self.collision_threshold = (
            self.config.habitat.ml_environment.collision_threshold
        )
        self.obs_dilation_selem_radius = (
            self.config.habitat.ml_environment.obs_dilation_selem_radius
        )
        self.goal_dilation_selem_radius = (
            self.config.habitat.ml_environment.goal_dilation_selem_radius
        )

        self.max_steps = self.config.habitat.ml_environment.max_steps

        self.dump_location = self.config.habitat.ml_environment.dump_location
        self.exp_name = self.config.habitat.ml_environment.exp_name

        self.num_environments = envs.num_envs
        if self.config.habitat.ml_environment.panorama_start:
            self.panorama_start_steps = int(
                360 / self.config.habitat.simulator.turn_angle
            )
        else:
            self.panorama_start_steps = 0

        self._module = ObjectNavAgentModule(self.config)
        self._module = self._module.to(self.device)

        self.module = DataParallel(self._module, device_ids=[self.device])

        self.obs_preprocessor = ObsPreprocessor(
            config.habitat, self.num_environments, self.device
        )
        self.semantic_map = Categorical2DSemanticMapState(
            device=self.device,
            num_environments=self.num_environments,
            num_sem_categories=self.num_sem_categories,
            map_resolution=self.map_resolution,
            map_size_cm=self.map_size_cm,
            global_downscaling=self.global_downscaling,
        )
        self.planner = DiscretePlanner(
            turn_angle=self.config.habitat.simulator.turn_angle,
            collision_threshold=self.collision_threshold,
            obs_dilation_selem_radius=self.obs_dilation_selem_radius,
            goal_dilation_selem_radius=self.goal_dilation_selem_radius,
            map_size_cm=self.map_size_cm,
            map_resolution=self.map_resolution,
            visualize=False,
            print_images=False,
            dump_location=self.dump_location,
            exp_name=self.exp_name,
        )

        self.goal_update_steps = self._module.goal_update_steps

        self.reset_vectorized()
        print("Policy initialized.")

    def reset_vectorized(self):
        """Initialize agent state."""
        self.timesteps = [0] * self.num_environments
        self.timesteps_before_goal_update = [0] * self.num_environments
        self.semantic_map.init_map_and_pose()

    def reset_vectorized_for_env(self, e: int):
        """Initialize agent state for a specific environment."""
        self.timesteps[e] = 0
        self.timesteps_before_goal_update[e] = 0
        self.semantic_map.init_map_and_pose_for_env(e)

    def prepare_planner_inputs(self, obs, pose_delta, object_goal_category=None, recep_goal_category=None):
        dones = torch.tensor([False] * self.num_environments)
        update_global = torch.tensor(
            [
                self.timesteps_before_goal_update[e] == 0
                for e in range(self.num_environments)
            ]
        )

        if object_goal_category is not None:
            object_goal_category =  object_goal_category.unsqueeze(1)
        if recep_goal_category is not None:
            recep_goal_category = recep_goal_category.unsqueeze(1)
        (
            goal_map,
            found_goal,
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        ) = self.module(
            obs.unsqueeze(1),
            pose_delta.unsqueeze(1),
            dones.unsqueeze(1),
            update_global.unsqueeze(1),
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            self.semantic_map.local_pose,
            self.semantic_map.global_pose,
            self.semantic_map.lmb,
            self.semantic_map.origins,
            seq_object_goal_category=object_goal_category,
            seq_recep_goal_category=recep_goal_category
        )

        self.semantic_map.local_pose = seq_local_pose[:, -1]
        self.semantic_map.global_pose = seq_global_pose[:, -1]
        self.semantic_map.lmb = seq_lmb[:, -1]
        self.semantic_map.origins = seq_origins[:, -1]

        goal_map = goal_map.squeeze(1).cpu().numpy()
        found_goal = found_goal.squeeze(1).cpu()

        for e in range(self.num_environments):
            if found_goal[e]:
                self.semantic_map.update_global_goal_for_env(e, goal_map[e])
            elif self.timesteps_before_goal_update[e] == 0:
                self.semantic_map.update_global_goal_for_env(e, goal_map[e])
                self.timesteps_before_goal_update[e] = self.goal_update_steps

        self.timesteps = [
            self.timesteps[e] + 1 for e in range(self.num_environments)
        ]
        self.timesteps_before_goal_update = [
            self.timesteps_before_goal_update[e] - 1
            for e in range(self.num_environments)
        ]
        planner_inputs = [
            {
                "obstacle_map": self.semantic_map.get_obstacle_map(e),
                "goal_map": self.semantic_map.get_goal_map(e),
                "sensor_pose": self.semantic_map.get_planner_pose_inputs(e),
                "found_goal": found_goal[e].item(),
            }
            for e in range(self.num_environments)
        ]
        vis_inputs = [
            {
                "explored_map": self.semantic_map.get_explored_map(e),
                "semantic_map": self.semantic_map.get_semantic_map(e),
                "been_close_map": self.semantic_map.get_been_close_map(e),
                "timestep": self.timesteps[e],
            }
            for e in range(self.num_environments)
        ]

        return planner_inputs, vis_inputs

    def act(self, observations, infos, envs):
        obs = torch.cat([ob.to(self.device) for ob in observations])
        pose_delta = torch.cat([info["pose_delta"] for info in infos])
        object_goal_category =  None
        if infos[0]["object_goal_category"] is not None:
            object_goal_category = torch.cat([info["object_goal_category"] for info in infos])
        recep_goal_category =  None
        if infos[0]["recep_goal_category"] is not None:
            recep_goal_category = torch.cat([info["recep_goal_category"] for info in infos])

        planner_inputs, vis_inputs = self.prepare_planner_inputs(
            obs, pose_delta, object_goal_category=object_goal_category, recep_goal_category=recep_goal_category
        )
        actions = [
            *envs.call(
                ["plan_and_step"] * envs.num_envs,
                [
                    {"planner_inputs": p_in, "vis_inputs": v_in}
                    for p_in, v_in in zip(planner_inputs, vis_inputs)
                ],
            )
        ]
        # Map the discrete actions to continuous actions
        action_map  = {HabitatSimActions.turn_right:[0, 0, -1, -1], HabitatSimActions.move_forward:[0, 1, 0, -1], HabitatSimActions.turn_left:[0, 0, 1, -1], HabitatSimActions.stop:[0, 0, 0, 1]}
        waypoints = [action_map[action] for action in actions]

        return np.array(waypoints, dtype=np.float32)

    @classmethod
    def from_config(cls, config: "DictConfig", envs, device):
        return cls(config, envs, device)
