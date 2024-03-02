#!/usr/bin/env python3


import os.path as osp
import random
import time
from collections import defaultdict

try:
    from collections import Sequence
except ImportError:
    from collections.abc import Sequence

from typing import Dict, Generator, List, Optional, Sequence, Tuple, Union

import magnum as mn
import numpy as np
from tqdm import tqdm

import habitat.datasets.rearrange.samplers as samplers
import habitat.sims.habitat_simulator.sim_utilities as sutils
import habitat_sim
from habitat.config import DictConfig
from habitat.core.logging import logger
from habitat.datasets.rearrange.navmesh_utils import (
    get_largest_island_index,
    path_is_navigable_given_robot,
)

from habitat.core.simulator import ShortestPathPoint
from habitat.datasets.utils import get_action_shortest_path
from habitat.datasets.social_nav.social_nav_dataset import AgentEpisode, SocialNavigationEpisode
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal

from habitat.sims.habitat_simulator.debug_visualizer import DebugVisualizer
from habitat.utils.common import cull_string_list_by_substrings
from habitat_sim.nav import NavMeshSettings

try:
    from habitat_sim.errors import GreedyFollowerError
except ImportError:
    GreedyFollower = BaseException
try:
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
except ImportError:
    habitat_sim = BaseException
ISLAND_RADIUS_LIMIT = 1.5

class ShortestPathFollowerAgent(Agent):
    r"""Implementation of the :ref:`habitat.core.agent.Agent` interface that
    uses :ref`habitat.tasks.nav.shortest_path_follower.ShortestPathFollower` utility class
    for extracting the action on the shortest path to the goal.
    """

    def __init__(self, env: habitat.Env, goal_radius: float, goals: List):
        self.env = env
        self.shortest_path_follower = ShortestPathFollower(
            sim=cast("HabitatSim", env.sim),
            goal_radius=goal_radius,
            return_one_hot=False,
        )
        self.goals = List

    def act(self, observations: "Observations") -> Union[int, np.ndarray]:
        return self.shortest_path_follower.get_next_action(
            cast(NavigationEpisode, self.env.current_episode).goals[0].position
        )

    def reset(self) -> None:
        pass


class SocialNavEpisodeGenerator:
    """Generate class for sampling individual episodes for social nav (point nav) task"""
    def __enter__(self) -> "RearrangeEpisodeGenerator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.sim != None:
            self.sim.close(destroy=True)
            del self.sim

    def __init__(
        self,
        cfg: DictConfig,
        debug_visualization: bool = False,
        limit_scene_set: Optional[str] = None,
        num_episodes: int = 1,
    ) -> None:
        """
        Initialize the generator object for a particular configuration.
        Loads yaml, sets up samplers and debug visualization settings.

        :param cfg: The SocialNavEpisodeGeneratorConfig.
        :param debug_visualization: Whether or not to generate debug images and videos.
        :param limit_scene_set: Option to limit all generation to a single scene set.
        :param num_episodes: The number of episodes which this RearrangeEpisodeGenerator will generate. Accuracy required only for BalancedSceneSampler.
        """
        # load and cache the config
        self.cfg = cfg
        self.start_cfg = self.cfg.copy()
        self._limit_scene_set = limit_scene_set

        # debug visualization settings
        self._render_debug_obs = self._make_debug_video = debug_visualization
        self.vdb: DebugVisualizer = (
            None  # visual debugger initialized with sim
        )

        # hold a habitat Simulator object for efficient re-use
        self.sim: habitat_sim.Simulator = None
        # initialize an empty scene and load the SceneDataset
        self.initialize_sim("NONE", self.cfg.dataset_path)

        # Setup the sampler caches from config
        self._get_scene_sampler(num_episodes)
        self._get_ao_state_samplers()

        self.num_ep_generated = 0
        self.goals = []
    
#################################################     

    def get_door_start_goal(self, ):
        sem_scene = self.sim.semantic_annotations() #TODO: check inside semantic
        #TODO
        

    def _create_episode(self,
        episode_id: Union[int, str],
        scene_id: str,
        start_position: List[float],
        start_rotation: List[float],
        target_position: List[float],
        shortest_paths: Optional[List[List[ShortestPathPoint]]] = None,
        radius: Optional[float] = None,
        info: Optional[Dict[str, float]] = None,
        agent: List[AgentEpisode],
    ) -> Optional[SocialNavigationEpisode]:
        goals = [SocialNavigationGoal(position=target_position, radius=radius)]
        return SocialNavigationEpisode(
            episode_id=str(episode_id),
            goals=goals,
            scene_id=scene_id,
            start_position=start_position,
            start_rotation=start_rotation,
            shortest_paths=shortest_paths,
            info=info,
            agent=agent,
        )
        
        
    def generate_social_nav_episode(self,
        num_episodes: int = 1,
        verbose: bool = False,
    ) -> Generator[SocialNavigationEpisode, None, None]:
        generated_episodes: List[SocialNavEpisode] = []
        
        episode_count = 0

        if verbose:
            pbar = tqdm(total=num_episodes)
        while episode_count < num_episodes or num_episodes < 0:
            try:
                new_episode = self.generate_single_episode()
            except Exception:
                new_episode = None
                logger.error("Generation failed with exception...")
            if new_episode is None:
                failed_episodes += 1
                continue
            generated_episodes.append(new_episode)
            if verbose:
                pbar.update(1)
            episode_count = episode_count + 1
        if verbose:
            pbar.close()

        logger.info(
            f"Generated {num_episodes} episodes in {num_episodes+failed_episodes} tries."
        )

        return generated_episodes


    def is_compatible_episode(self,
        s: Sequence[float],
        t: Sequence[float],
        sim: "HabitatSim",
        near_dist: float,
        far_dist: float,
        geodesic_to_euclid_ratio: float,
    ) -> Union[Tuple[bool, float], Tuple[bool, int]]:
        euclid_dist = np.power(np.power(np.array(s) - np.array(t), 2).sum(0), 0.5)
        if np.abs(s[1] - t[1]) > 0.5:  # check height difference to assure s and
            #  t are from same floor
            return False, 0
        d_separation = sim.geodesic_distance(s, [t])
        if d_separation == np.inf:
            return False, 0
        if not near_dist <= d_separation <= far_dist:
            return False, 0
        distances_ratio = d_separation / euclid_dist
        if distances_ratio < geodesic_to_euclid_ratio and (
            np.random.rand()
            > _ratio_sample_rate(distances_ratio, geodesic_to_euclid_ratio)
        ):
            return False, 0
        if sim.island_radius(s) < ISLAND_RADIUS_LIMIT:
            return False, 0
        return True, d_separation
         

    def generate_single_episode(self,
        number_retries_per_target: int=10,
        closest_dist_limit: float = 1,
        furthest_dist_limit: float = 30,
        geodesic_to_euclid_min_ratio: float = 1.1,
        ) -> Optional[SocialNavigationEpisode]:
        # agent = ShortestPathFollowerAgent(
        #     goals = self.goals
        #     goal_radius=config.habitat.task.measurements.success.success_distance,
        # )
        # agent.reset()
        
        for _retry in range(number_retries_per_target):
            target_position, target_position = self.get_door_start_goal()
            is_compatible, dist = self.is_compatible_episode(
                source_position,
                target_position,
                sim,
                near_dist=closest_dist_limit,
                far_dist=furthest_dist_limit,
                geodesic_to_euclid_ratio=geodesic_to_euclid_min_ratio,
            )
            if is_compatible:
                break
        if is_compatible:
            angle = np.random.uniform(0, 2 * np.pi)
            source_rotation = [0.0, np.sin(angle / 2), 0, np.cos(angle / 2)]

            shortest_paths = None
            if is_gen_shortest_path:
                try:
                    shortest_paths = [
                        get_action_shortest_path(
                            sim,
                            source_position=source_position,
                            source_rotation=source_rotation,
                            goal_position=target_position,
                            success_distance=shortest_path_success_distance,
                            max_episode_steps=shortest_path_max_steps,
                        )
                    ]
                # Throws an error when it can't find a path
                except GreedyFollowerError:
                    continue
            # generate random agent path
            agent = AgentEpisode().get_door_start_goal()
            
            episode = self._create_episode(
                episode_id=episode_count,
                scene_id=sim.habitat_config.scene,
                start_position=source_position,
                start_rotation=source_rotation,
                target_position=target_position,
                shortest_paths=shortest_paths,
                radius=shortest_path_success_distance,
                info={"geodesic_distance": dist},
                agent=[agent]
            )

        return episode