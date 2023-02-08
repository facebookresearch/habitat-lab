from typing import TYPE_CHECKING, Optional, Tuple, Type

import torch

import habitat
from habitat import Dataset
from habitat.core.environments import GymHabitatEnv
from habitat.core.simulator import Observations
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.ml.navigation_planner.discrete_planner import (
    DiscretePlanner,
)
from habitat_baselines.ml.obs_preprocessor.obs_preprocessor import (
    ObsPreprocessor,
)
from habitat_baselines.ml.visualizer.visualizer import Visualizer


@habitat.registry.register_env(name="MLEnv")
class MLEnv(GymHabitatEnv):
    def __init__(
        self, config: "DictConfig", dataset: Optional[Dataset] = None
    ):
        super().__init__(config)
        self.config = config

        self.device = torch.device(
            f"cuda:{config.simulator.habitat_sim_v0.gpu_device_id}"
        )

        self.max_steps = config.ml_environment.max_steps
        self.num_sem_categories = config.ml_environment.num_sem_categories
        if config.ml_environment.panorama_start:
            self.panorama_start_steps = int(360 / config.simulator.turn_angle)
        else:
            self.panorama_start_steps = 0

        self.episode_idx = 0

        self.planner = DiscretePlanner(
            turn_angle=config.simulator.turn_angle,
            collision_threshold=config.ml_environment.collision_threshold,
            obs_dilation_selem_radius=config.ml_environment.obs_dilation_selem_radius,
            goal_dilation_selem_radius=config.ml_environment.goal_dilation_selem_radius,
            map_size_cm=config.ml_environment.map_size_cm,
            map_resolution=config.ml_environment.map_resolution,
            visualize=False,
            print_images=False,
            dump_location=config.ml_environment.dump_location,
            exp_name=config.ml_environment.exp_name,
        )
        self.visualizer = Visualizer(config)
        self.obs_preprocessor = ObsPreprocessor(config, 1, self.device)

        self.scene_id = None
        self.episode_id = None
        self.last_semantic_frame = None
        self.last_goal_name = None
        self.last_closest_goal_map = None
        self.episode_panorama_start_steps = None

    def reset(self):
        print("------------")
        print("Resetting environment")
        obs = super().reset()

        self.episode_idx += 1
        self.episode_panorama_start_steps = self.panorama_start_steps

        self.obs_preprocessor.reset(self)
        self.planner.reset()
        self.visualizer.reset()
        self.scene_id = (
            self.current_episode().scene_id.split("/")[-1].split(".")[0]
        )
        self.episode_id = self.current_episode().episode_id
        # print(self.scene_id, self.episode_id)
        self._set_vis_dir(self.scene_id, self.episode_id)

        return obs

    def plan_and_step(
        self, planner_inputs: dict, vis_inputs: dict
    ) -> Tuple[Observations, bool, dict]:
        # 1 - Visualization of previous timestep - now that we have
        #  all necessary components
        vis_inputs["semantic_frame"] = self.last_semantic_frame
        vis_inputs["goal_name"] = self.last_goal_name
        vis_inputs["closest_goal_map"] = self.last_closest_goal_map
        vis_inputs["third_person_rgb_frame"] = self.last_third_person_rgb_frame
        self.visualizer.visualize(**planner_inputs, **vis_inputs)

        # 2 - Planning
        self.last_closest_goal_map = None

        if planner_inputs["found_goal"]:
            self.episode_panorama_start_steps = 0
        if vis_inputs["timestep"] < self.episode_panorama_start_steps:
            action = HabitatSimActions.turn_right
        elif vis_inputs["timestep"] > self.max_steps:
            action = HabitatSimActions.stop
        else:
            action, self.last_closest_goal_map = self.planner.plan(
                **planner_inputs
            )
        return action

        # 4 - Preprocess obs - if done, record episode metrics and
        #  reset environment
        # done = self.episode_over
        # if done:
        #     done_info = {
        #         "last_episode_scene_id": self.scene_id,
        #         "last_episode_id": self.episode_id,
        #         "last_goal_name": self.last_goal_name,
        #         "last_episode_metrics": self.get_metrics(),
        #     }
        #     obs_preprocessed, info = self.reset()

    def preprocess_obs(self, obs: Observations) -> Tuple[torch.Tensor, dict]:
        (
            obs_preprocessed,
            semantic_frame,
            third_person_rgb_frame,
            pose_delta,
            object_goal_category,
            recep_goal_category,
            goal_name,
        ) = self.obs_preprocessor.preprocess([obs])

        self.last_semantic_frame = semantic_frame[0]
        self.last_goal_name = goal_name[0]
        self.last_third_person_rgb_frame = third_person_rgb_frame[0]

        info = {"pose_delta": pose_delta, "object_goal_category": object_goal_category,  "recep_goal_category": recep_goal_category}

        return obs_preprocessed, info

    def _set_vis_dir(self, scene_id: str, episode_id: str):
        """Reset visualization directory."""
        self.planner.set_vis_dir(scene_id, episode_id)
        self.visualizer.set_vis_dir(scene_id, episode_id)

    def _disable_print_images(self):
        self.planner.disable_print_images()
        self.visualizer.disable_print_images()
