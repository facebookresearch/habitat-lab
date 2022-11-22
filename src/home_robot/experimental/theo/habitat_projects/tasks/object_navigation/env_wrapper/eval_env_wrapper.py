import torch
import json
import os
from typing import Tuple, Optional, List

from habitat import Config
from habitat.core.env import Env
from habitat.core.simulator import Observations
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.core.dataset import EpisodeIterator

from home_robot.agent.navigation_planner.discrete_planner import DiscretePlanner
from home_robot.agent.visualization.object_navigation.objectnav_visualizer import (
    ObjectNavVisualizer,
)
from home_robot.experimental.theo.habitat_projects.tasks.object_navigation.obs_preprocessor.obs_preprocessor import (
    ObsPreprocessor,
)


class EvalEnvWrapper(Env):
    """
    This environment wrapper is used for evaluation. It contains stepping
    the underlying environment, preprocessing observations, and planning
    given a high-level goal predicted by the policy. It is complemented by
    a semantic map state, update, and high-level goal policy in the agent.
    """

    def __init__(self, config: Config, episode_ids: Optional[List[str]] = None):
        """
        Arguments:
            episode_ids: if specified, force the environment to iterate
             through these episodes before others - this is useful to
             debug behavior on specific episodes
        """
        super().__init__(config=config.TASK_CONFIG)
        os.environ["MAGNUM_LOG"] = "quiet"
        os.environ["HABITAT_SIM_LOG"] = "quiet"

        self.device = (
            torch.device("cpu")
            if config.NO_GPU
            else torch.device(f"cuda:{self.sim.gpu_device}")
        )
        self.max_steps = config.AGENT.max_steps
        if config.AGENT.panorama_start:
            self.panorama_start_steps = int(360 / config.ENVIRONMENT.turn_angle)
        else:
            self.panorama_start_steps = 0

        self.forced_episode_ids = episode_ids if episode_ids else []
        self.episode_idx = 0

        # Keep only episodes with a goal on the same floor as the
        #  starting position
        if config.EVAL_VECTORIZED.goal_on_same_floor:
            new_episodes = []
            for episode in self._dataset.episodes:
                scene_dir = "/".join(episode.scene_id.split("/")[:-1])
                map_dir = scene_dir + "/floor_semantic_maps_annotations_top_down"
                scene_id = episode.scene_id.split("/")[-1].split(".")[0]
                with open(f"{map_dir}/{scene_id}_info.json", "r") as f:
                    scene_info = json.load(f)
                start_on_first_floor = (
                    abs(
                        episode.start_position[1] * 100.0
                        - scene_info["floor_heights_cm"][0]
                    )
                    < 50
                )
                goal_on_same_floor = (
                    len(
                        [
                            goal
                            for goal in episode.goals
                            if episode.start_position[1] - 0.25
                            < goal.position[1]
                            < episode.start_position[1] + 1.5
                        ]
                    )
                    > 0
                )
                if start_on_first_floor and goal_on_same_floor:
                    new_episodes.append(episode)

            # TODO - Keep at least one episode to avoid environment crashing,
            #  there's probably a cleaner way to do this
            if len(new_episodes) == 0:
                new_episodes = [self._dataset.episodes[0]]

            print(
                f"From {len(self._dataset.episodes)} total episodes for this "
                f"environment to {len(new_episodes)} on the same floor"
            )
            self._dataset.episodes = new_episodes
            self.episode_iterator = EpisodeIterator(
                new_episodes,
                shuffle=False,
                group_by_scene=False,
            )
            self._current_episode = None

        self.planner = DiscretePlanner(
            turn_angle=config.ENVIRONMENT.turn_angle,
            collision_threshold=config.AGENT.PLANNER.collision_threshold,
            obs_dilation_selem_radius=config.AGENT.PLANNER.obs_dilation_selem_radius,
            goal_dilation_selem_radius=config.AGENT.PLANNER.goal_dilation_selem_radius,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            visualize=False,
            print_images=False,
            dump_location=config.DUMP_LOCATION,
            exp_name=config.EXP_NAME,
        )
        self.visualizer = ObjectNavVisualizer(
            num_sem_categories=config.ENVIRONMENT.num_sem_categories,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            show_images=config.VISUALIZE,
            print_images=config.PRINT_IMAGES,
            dump_location=config.DUMP_LOCATION,
            exp_name=config.EXP_NAME,
        )
        self.obs_preprocessor = ObsPreprocessor(config, 1, self.device)

        self.scene_id = None
        self.episode_id = None
        self.last_semantic_frame = None
        self.last_goal_name = None
        self.last_closest_goal_map = None
        self.episode_panorama_start_steps = None

    def reset(self) -> Tuple[torch.Tensor, dict]:
        if self.episode_idx < len(self.forced_episode_ids):
            obs = self._reset_to_episode(self.forced_episode_ids[self.episode_idx])
        else:
            obs = super().reset()

        self.episode_idx += 1
        self.episode_panorama_start_steps = self.panorama_start_steps

        self.obs_preprocessor.reset()
        self.planner.reset()
        self.visualizer.reset()

        self.scene_id = self.current_episode.scene_id.split("/")[-1].split(".")[0]
        self.episode_id = self.current_episode.episode_id
        self._set_vis_dir(self.scene_id, self.episode_id)
        if (
            len(self.forced_episode_ids) > 0
            and self.episode_id not in self.forced_episode_ids
        ):
            self._disable_print_images()

        obs_preprocessed, info = self._preprocess_obs(obs)

        return obs_preprocessed, info

    def _reset_to_episode(self, episode_id: str) -> Observations:
        """
        Reset the environment to a specific episode ID
        Adapted from:
        https://github.com/facebookresearch/habitat-lab/blob/main/habitat/core/env.py
        """
        self._reset_stats()

        episode = [e for e in self.episodes if e.episode_id == episode_id][0]
        self._current_episode = episode

        self._episode_from_iter_on_reset = True
        self._episode_force_changed = False

        self.reconfigure(self._config)

        observations = self.task.reset(episode=self.current_episode)
        self._task.measurements.reset_measures(
            episode=self.current_episode,
            task=self.task,
            observations=observations,
        )
        return observations

    def _preprocess_obs(self, obs: Observations) -> Tuple[torch.Tensor, dict]:
        (
            obs_preprocessed,
            semantic_frame,
            pose_delta,
            goal_category,
            goal_name,
        ) = self.obs_preprocessor.preprocess([obs])

        self.last_semantic_frame = semantic_frame[0]
        self.last_goal_name = goal_name[0]

        info = {"pose_delta": pose_delta, "goal_category": goal_category}

        return obs_preprocessed, info

    def plan_and_step(
        self, planner_inputs: dict, vis_inputs: dict
    ) -> Tuple[Observations, bool, dict]:
        # 1 - Visualization of previous timestep - now that we have
        #  all necessary components
        vis_inputs["semantic_frame"] = self.last_semantic_frame
        vis_inputs["goal_name"] = self.last_goal_name
        vis_inputs["closest_goal_map"] = self.last_closest_goal_map
        self.visualizer.visualize(**planner_inputs, **vis_inputs)

        # 2 - Planning
        self.last_closest_goal_map = None
        if planner_inputs["found_goal"]:
            self.episode_panorama_start_steps = 0
        if vis_inputs["timestep"] < self.episode_panorama_start_steps:
            action = HabitatSimActions.TURN_RIGHT
        elif vis_inputs["timestep"] > self.max_steps:
            action = HabitatSimActions.STOP
        else:
            action, self.last_closest_goal_map = self.planner.plan(**planner_inputs)

        # 3 - Step
        obs = self.step(action)

        # 4 - Preprocess obs - if done, record episode metrics and
        #  reset environment
        done = self.episode_over
        if done:
            done_info = {
                "last_episode_scene_id": self.scene_id,
                "last_episode_id": self.episode_id,
                "last_goal_name": self.last_goal_name,
                "last_episode_metrics": self.get_metrics(),
            }
            obs_preprocessed, info = self.reset()
            info.update(done_info)

        else:
            obs_preprocessed, info = self._preprocess_obs(obs)

        return obs_preprocessed, done, info

    def _set_vis_dir(self, scene_id: str, episode_id: str):
        """Reset visualization directory."""
        self.planner.set_vis_dir(scene_id, episode_id)
        self.visualizer.set_vis_dir(scene_id, episode_id)

    def _disable_print_images(self):
        self.planner.disable_print_images()
        self.visualizer.disable_print_images()
