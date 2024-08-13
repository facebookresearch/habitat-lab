#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
import random
import time
from collections import defaultdict

try:
    from collections import Sequence
except ImportError:
    from collections.abc import Sequence

from typing import Any, Dict, List, Optional, Tuple

import magnum as mn
import numpy as np
from tqdm import tqdm

import habitat.datasets.rearrange.samplers as samplers
import habitat_sim
from habitat.config import DictConfig
from habitat.core.logging import logger
from habitat.datasets.rearrange.navmesh_utils import (
    get_largest_island_index,
    path_is_navigable_given_robot,
)
from habitat.datasets.rearrange.rearrange_dataset import RearrangeEpisode
from habitat.datasets.rearrange.samplers.receptacle import (
    OnTopOfReceptacle,
    Receptacle,
    ReceptacleSet,
    ReceptacleTracker,
    find_receptacles,
    get_navigable_receptacles,
)
from habitat.sims.habitat_simulator.debug_visualizer import DebugVisualizer
from habitat.utils.common import cull_string_list_by_substrings
from habitat_sim.nav import NavMeshSettings


def get_sample_region_ratios(load_dict) -> Dict[str, float]:
    sample_region_ratios: Dict[str, float] = defaultdict(lambda: 1.0)
    sample_region_ratios.update(
        load_dict["params"].get("sample_region_ratio", {})
    )
    return sample_region_ratios


class RearrangeEpisodeGenerator:
    """Generator class encapsulating logic for procedurally sampling individual episodes for general rearrangement task.

    Initialized from a provided configuration file defining dataset paths, object,scene,and receptacle sets, and state sampler parameters.

    See rearrange_dataset.py for details on the RearrangeDataset and RearrangeEpisodes produced by this generator.
    See this file's main executable function below for details on running the generator.
    See `test_rearrange_episode_generator()` in test/test_rearrange_task.py for unit test example.
    """

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

        :param cfg: The RearrangeEpisodeGeneratorConfig.
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
        self.dbv: DebugVisualizer = (
            None  # visual debugger initialized with sim
        )

        # hold a habitat Simulator object for efficient re-use
        self.sim: habitat_sim.Simulator = None
        # initialize an empty scene and load the SceneDataset
        self.initialize_sim("NONE", self.cfg.dataset_path)

        # Setup the sampler caches from config
        self._get_resource_sets()
        self._get_scene_sampler(num_episodes)
        self._get_obj_samplers()
        self._get_ao_state_samplers()

        # cache objects sampled by this generator for the most recent episode
        self.ep_sampled_objects: List[
            habitat_sim.physics.ManagedRigidObject
        ] = []
        self.num_ep_generated = 0

    def _get_resource_sets(self) -> None:
        """
        Extracts and validates scene, object, and receptacle sets from the config and fills internal datastructures for later reference.
        Assumes the Simulator (self.sim) is already initialized.
        """
        # {scene set name -> [scene handles]}
        self._scene_sets: Dict[str, List[str]] = {}

        # {object set name -> [object handles]}
        self._obj_sets: Dict[str, List[str]] = {}

        # {receptacle set name -> ([included object handles], [excluded object handles], [included receptacle name substrings], [excluded receptacle name substrings])}
        self._receptacle_sets: Dict[str, ReceptacleSet] = {}

        expected_list_keys = ["included_substrings", "excluded_substrings"]
        # scene sets
        for scene_set in self.cfg.scene_sets:
            assert "name" in scene_set
            assert (
                scene_set["name"] not in self._scene_sets
            ), f"cfg.scene_sets - Duplicate name ('{scene_set['name']}') detected."
            for list_key in expected_list_keys:
                assert (
                    list_key in scene_set
                ), f"Expected list key '{list_key}'."
                assert isinstance(
                    scene_set[list_key], Sequence
                ), f"cfg.scene_sets - '{scene_set['name']}' '{list_key}' must be a list of strings."
            self._scene_sets[
                scene_set["name"]
            ] = cull_string_list_by_substrings(
                self.sim.metadata_mediator.get_scene_handles(),
                scene_set["included_substrings"],
                scene_set["excluded_substrings"],
            )

        # object sets
        for object_set in self.cfg.object_sets:
            assert "name" in object_set
            assert (
                object_set["name"] not in self._obj_sets
            ), f"cfg.object_sets - Duplicate name ('{object_set['name']}') detected."
            for list_key in expected_list_keys:
                assert (
                    list_key in object_set
                ), f"Expected list key '{list_key}'."
                assert isinstance(
                    object_set[list_key], Sequence
                ), f"cfg.object_sets - '{object_set['name']}' '{list_key}' must be a list of strings."
            self._obj_sets[
                object_set["name"]
            ] = cull_string_list_by_substrings(
                self.sim.get_object_template_manager().get_template_handles(),
                object_set["included_substrings"],
                object_set["excluded_substrings"],
            )

        # receptacle sets
        expected_list_keys = [
            "included_object_substrings",
            "excluded_object_substrings",
            "included_receptacle_substrings",
            "excluded_receptacle_substrings",
        ]
        for receptacle_set in self.cfg.receptacle_sets:
            assert "name" in receptacle_set
            assert (
                receptacle_set["name"] not in self._receptacle_sets
            ), f"cfg.receptacle_sets - Duplicate name ('{receptacle_set['name']}') detected."
            for list_key in expected_list_keys:
                assert (
                    list_key in receptacle_set
                ), f"Expected list key '{list_key}'."
                assert isinstance(
                    receptacle_set[list_key], Sequence
                ), f"cfg.receptacle_sets - '{receptacle_set['name']}' '{list_key}' must be a list of strings."

            self._receptacle_sets[receptacle_set["name"]] = ReceptacleSet(
                **receptacle_set
            )

    def _get_obj_samplers(self) -> None:
        """
        Extracts object sampler parameters from the yaml config file and constructs the sampler objects.
        """
        self._obj_samplers: Dict[str, samplers.ObjectSampler] = {}

        for obj_sampler_info in self.cfg.object_samplers:
            assert "name" in obj_sampler_info
            assert "type" in obj_sampler_info
            assert "params" in obj_sampler_info
            assert (
                obj_sampler_info["name"] not in self._obj_samplers
            ), f"Duplicate object sampler name '{obj_sampler_info['name']}' in config."
            if obj_sampler_info["type"] == "uniform":
                assert "object_sets" in obj_sampler_info["params"]
                assert "receptacle_sets" in obj_sampler_info["params"]
                assert "num_samples" in obj_sampler_info["params"]
                assert "orientation_sampling" in obj_sampler_info["params"]
                # merge and flatten object and receptacle sets
                object_handles = [
                    x
                    for y in obj_sampler_info["params"]["object_sets"]
                    for x in self._obj_sets[y]
                ]
                object_handles = sorted(set(object_handles))
                if len(object_handles) == 0:
                    raise ValueError(
                        f"Found no object handles for {obj_sampler_info}"
                    )
                self._obj_samplers[
                    obj_sampler_info["name"]
                ] = samplers.ObjectSampler(
                    object_set=object_handles,
                    allowed_recep_set_names=obj_sampler_info["params"][
                        "receptacle_sets"
                    ],
                    num_objects=(
                        obj_sampler_info["params"]["num_samples"][0],
                        obj_sampler_info["params"]["num_samples"][1],
                    ),
                    orientation_sample=obj_sampler_info["params"][
                        "orientation_sampling"
                    ],
                    sample_region_ratio=get_sample_region_ratios(
                        obj_sampler_info
                    ),
                    nav_to_min_distance=obj_sampler_info["params"].get(
                        "nav_to_min_distance", -1.0
                    ),
                    recep_set_sample_probs=obj_sampler_info["params"].get(
                        "sample_probs", None
                    ),
                    constrain_to_largest_nav_island=obj_sampler_info[
                        "params"
                    ].get("constrain_to_largest_nav_island", False),
                )
            else:
                logger.info(
                    f"Requested object sampler '{obj_sampler_info['type']}' is not implemented."
                )
                raise (NotImplementedError)

    def _get_object_target_samplers(self) -> None:
        """
        Initialize target samplers. Expects self.episode_data to be populated by object samples.
        """

        self._target_samplers: Dict[str, samplers.ObjectTargetSampler] = {}
        for target_sampler_info in self.cfg.object_target_samplers:
            assert "name" in target_sampler_info
            assert "type" in target_sampler_info
            assert "params" in target_sampler_info
            assert (
                target_sampler_info["name"] not in self._target_samplers
            ), f"Duplicate target sampler name '{target_sampler_info['name']}' in config."
            if target_sampler_info["type"] == "uniform":
                # merge and flatten receptacle sets

                self._target_samplers[
                    target_sampler_info["name"]
                ] = samplers.ObjectTargetSampler(
                    # Add object set later
                    object_instance_set=[],
                    allowed_recep_set_names=target_sampler_info["params"][
                        "receptacle_sets"
                    ],
                    num_objects=(
                        target_sampler_info["params"]["num_samples"][0],
                        target_sampler_info["params"]["num_samples"][1],
                    ),
                    orientation_sample=target_sampler_info["params"][
                        "orientation_sampling"
                    ],
                    sample_region_ratio=get_sample_region_ratios(
                        target_sampler_info
                    ),
                    nav_to_min_distance=target_sampler_info["params"].get(
                        "nav_to_min_distance", -1.0
                    ),
                    recep_set_sample_probs=target_sampler_info["params"].get(
                        "sample_probs", None
                    ),
                    constrain_to_largest_nav_island=target_sampler_info[
                        "params"
                    ].get("constrain_to_largest_nav_island", False),
                )
            else:
                logger.info(
                    f"Requested target sampler '{target_sampler_info['type']}' is not implemented."
                )
                raise (NotImplementedError)

    def _get_scene_sampler(self, num_episodes: int) -> None:
        """
        Initialize the scene sampler.

        :param num_episodes: The nubmer of episodes the sampler will be used for. Must be accurate for BalancedSceneSampler.
        """
        self._scene_sampler: Optional[samplers.SceneSampler] = None
        if self.cfg.scene_sampler.type == "single":
            self._scene_sampler = samplers.SingleSceneSampler(
                self.cfg.scene_sampler.params.scene
            )
        elif self.cfg.scene_sampler.type in ["subset", "scene_balanced"]:
            unified_scene_set: List[str] = []
            # concatenate all requested scene sets
            for set_name in self.cfg.scene_sampler.params.scene_sets:
                if (
                    self._limit_scene_set is not None
                    and set_name != self._limit_scene_set
                ):
                    continue
                assert (
                    set_name in self._scene_sets
                ), f"'subset' SceneSampler requested scene_set name, '{set_name}', not found."
                unified_scene_set += self._scene_sets[set_name]

            # cull duplicates
            unified_scene_set = sorted(set(unified_scene_set))
            if self.cfg.scene_sampler.type == "subset":
                self._scene_sampler = samplers.MultiSceneSampler(
                    unified_scene_set
                )
            elif self.cfg.scene_sampler.type == "scene_balanced":
                self._scene_sampler = samplers.BalancedSceneSampler(
                    unified_scene_set, num_episodes
                )
        else:
            logger.error(
                f"Requested scene sampler '{self.cfg.scene_sampler.type}' is not implemented."
            )
            raise (NotImplementedError)

    def _get_ao_state_samplers(self) -> None:
        """
        Initialize and cache all ArticulatedObject state samplers from configuration.
        """
        self._ao_state_samplers: Dict[
            str, samplers.ArticulatedObjectStateSampler
        ] = {}
        for ao_info in self.cfg.ao_state_samplers:
            assert "name" in ao_info
            assert "type" in ao_info
            assert "params" in ao_info
            assert (
                ao_info["name"] not in self._ao_state_samplers
            ), f"Duplicate AO state sampler name {ao_info['name']} in config."

            if ao_info["type"] == "uniform":
                self._ao_state_samplers[
                    ao_info["name"]
                ] = samplers.ArticulatedObjectStateSampler(
                    ao_info["params"][0],
                    ao_info["params"][1],
                    (ao_info["params"][2], ao_info["params"][3]),
                )
            elif ao_info["type"] == "categorical":
                self._ao_state_samplers[
                    ao_info["name"]
                ] = samplers.ArtObjCatStateSampler(
                    ao_info["params"][0],
                    ao_info["params"][1],
                    (ao_info["params"][2], ao_info["params"][3]),
                )
            elif ao_info["type"] == "composite":
                composite_ao_sampler_params: Dict[
                    str, Dict[str, Tuple[float, float, bool]]
                ] = {}
                for entry in ao_info["params"]:
                    ao_handle = entry["ao_handle"]
                    should_sample_all_joints = entry.get(
                        "should_sample_all_joints", False
                    )
                    link_sample_params = entry["joint_states"]
                    assert (
                        ao_handle not in composite_ao_sampler_params
                    ), f"Duplicate handle '{ao_handle}' in composite AO sampler config."
                    composite_ao_sampler_params[ao_handle] = {}
                    for link_params in link_sample_params:
                        link_name = link_params[0]
                        assert (
                            link_name
                            not in composite_ao_sampler_params[ao_handle]
                        ), f"Duplicate link name '{link_name}' for handle '{ao_handle} in composite AO sampler config."
                        composite_ao_sampler_params[ao_handle][link_name] = (
                            link_params[1],
                            link_params[2],
                            should_sample_all_joints,
                        )
                self._ao_state_samplers[
                    ao_info["name"]
                ] = samplers.CompositeArticulatedObjectStateSampler(
                    composite_ao_sampler_params,
                    ao_info.get("apply_prob", None),
                )
            else:
                logger.error(
                    f"Requested AO state sampler type '{ao_info['type']}' not implemented."
                )
                raise (NotImplementedError)

    def _reset_samplers(self) -> None:
        """
        Reset any sampler internal state related to a specific scene or episode.
        """
        self.ep_sampled_objects = []
        for sampler in self._obj_samplers.values():
            sampler.reset()

    def generate_scene(self) -> str:
        """
        Sample a new scene and re-initialize the Simulator.
        Return the generated scene's handle.
        """
        cur_scene_name = self._scene_sampler.sample()
        logger.info(f"Initializing scene {cur_scene_name}")
        self.initialize_sim(cur_scene_name, self.cfg.dataset_path)

        return cur_scene_name

    def visualize_scene_receptacles(self) -> None:
        """
        Generate a debug line representation for each receptacle in the scene, aim the camera at it and record 1 observation.
        """
        logger.info("visualize_scene_receptacles processing")
        receptacles = find_receptacles(self.sim)
        for receptacle in receptacles:
            logger.info("receptacle processing")
            receptacle.debug_draw(self.sim)
            # Receptacle does not have a position cached relative to the object it is attached to, so sample a position from it instead
            sampled_look_target = receptacle.sample_uniform_global(
                self.sim, 1.0
            )
            self.dbv.look_at(sampled_look_target)
            self.dbv.debug_obs.append(self.dbv.get_observation())

    def generate_episodes(
        self, num_episodes: int = 1, verbose: bool = False
    ) -> List[RearrangeEpisode]:
        """
        Generate a fixed number of episodes.
        """
        generated_episodes: List[RearrangeEpisode] = []
        failed_episodes = 0
        if verbose:
            pbar = tqdm(total=num_episodes)
        while len(generated_episodes) < num_episodes:
            try:
                self._scene_sampler.set_cur_episode(len(generated_episodes))
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
        if verbose:
            pbar.close()

        logger.info(
            f"Generated {num_episodes} episodes in {num_episodes+failed_episodes} tries."
        )

        return generated_episodes

    def generate_single_episode(self) -> Optional[RearrangeEpisode]:
        """
        Generate a single episode, sampling the scene.
        """

        # Reset the number of allowed objects per receptacle.
        recep_tracker = ReceptacleTracker(
            dict(self.cfg.max_objects_per_receptacle),
            self._receptacle_sets,
        )

        self._reset_samplers()
        self.episode_data: Dict[str, Dict[str, Any]] = {
            "sampled_objects": {},  # object sampler name -> sampled object instances
            "sampled_targets": {},  # target sampler name -> (object, target state)
        }

        ep_scene_handle = self.generate_scene()
        scene_base_dir = osp.dirname(osp.dirname(ep_scene_handle))

        recep_tracker.init_scene_filters(
            mm=self.sim.metadata_mediator, scene_handle=ep_scene_handle
        )

        scene_name = ep_scene_handle.split(".")[0]
        navmesh_path = osp.join(
            scene_base_dir, "navmeshes", scene_name + ".navmesh"
        )

        # Load navmesh
        regenerate_navmesh = self.cfg.regenerate_new_mesh
        if not regenerate_navmesh and not self.sim.pathfinder.load_nav_mesh(
            navmesh_path
        ):
            # if loading fails, regenerate instead
            regenerate_navmesh = True
            logger.error(
                f"Failed to load navmesh '{navmesh_path}', regenerating instead."
            )
        if regenerate_navmesh:
            navmesh_settings = NavMeshSettings()
            navmesh_settings.set_defaults()
            navmesh_settings.agent_radius = self.cfg.agent_radius
            navmesh_settings.agent_height = self.cfg.agent_height
            navmesh_settings.include_static_objects = True
            navmesh_settings.agent_max_climb = self.cfg.agent_max_climb
            navmesh_settings.agent_max_slope = self.cfg.agent_max_slope
            self.sim.recompute_navmesh(
                self.sim.pathfinder,
                navmesh_settings,
            )

        # prepare target samplers
        self._get_object_target_samplers()
        target_numbers: Dict[str, int] = {
            k: sampler.target_objects_number
            for k, sampler in self._target_samplers.items()
        }
        # prepare mapping of target samplers to their source object samplers
        targ_sampler_name_to_obj_sampler_names: Dict[str, List[str]] = {}
        for targ_sampler_cfg in self.cfg.object_target_samplers:
            sampler_name = targ_sampler_cfg["name"]
            targ_sampler_name_to_obj_sampler_names[
                sampler_name
            ] = targ_sampler_cfg["params"]["object_samplers"]

        # get the largest indoor island to constrain target navigability check
        largest_indoor_island_id = get_largest_island_index(
            self.sim.pathfinder, self.sim, allow_outdoor=False
        )

        # sample and allocate receptacles to contain the target objects
        target_receptacles = defaultdict(list)
        all_target_receptacles = []
        for sampler_name, num_targets in target_numbers.items():
            new_target_receptacles: List[Receptacle] = []
            failed_samplers: Dict[str, bool] = defaultdict(bool)
            while len(new_target_receptacles) < num_targets:
                assert len(failed_samplers.keys()) < len(
                    targ_sampler_name_to_obj_sampler_names[sampler_name]
                ), f"All target samplers failed to find a match for '{sampler_name}'."
                obj_sampler_name = random.choice(
                    targ_sampler_name_to_obj_sampler_names[sampler_name]
                )

                sampler = self._obj_samplers[obj_sampler_name]
                new_receptacle = None
                try:
                    new_receptacle = sampler.sample_receptacle(
                        self.sim, recep_tracker
                    )
                except AssertionError:
                    # No receptacle instances found matching this sampler's requirements, likely ran out of allocations and a different sampler should be tried
                    failed_samplers[obj_sampler_name]
                    continue

                if recep_tracker.allocate_one_placement(new_receptacle):
                    # used up new_receptacle, need to recompute the sampler's receptacle_candidates
                    sampler.receptacle_candidates = None
                # optionally constrain to the largest indoor island
                nav_island = -1
                if sampler._constrain_to_largest_nav_island:
                    nav_island = largest_indoor_island_id
                new_receptacle = get_navigable_receptacles(
                    self.sim, [new_receptacle], nav_island
                )  # type: ignore
                if len(new_receptacle) != 0:  # type: ignore
                    new_target_receptacles.append(new_receptacle[0])  # type: ignore

            target_receptacles[obj_sampler_name].extend(new_target_receptacles)
            all_target_receptacles.extend(new_target_receptacles)

        # sample and allocate receptacles to contain the goal states for target objects
        goal_receptacles = {}
        all_goal_receptacles = []
        for sampler, (sampler_name, num_targets) in zip(
            self._target_samplers.values(), target_numbers.items()
        ):
            new_goal_receptacles = []  # type: ignore
            num_iterations = 0
            max_iterations = num_targets * 100
            while (
                len(new_goal_receptacles) < num_targets
                and num_iterations < max_iterations
            ):
                num_iterations += 1
                new_receptacle = sampler.sample_receptacle(
                    self.sim,
                    recep_tracker,
                )
                if isinstance(new_receptacle, OnTopOfReceptacle):
                    new_receptacle.set_episode_data(self.episode_data)
                if recep_tracker.allocate_one_placement(new_receptacle):
                    # used up new_receptacle, need to recompute the sampler's receptacle_candidates
                    sampler.receptacle_candidates = None
                # optionally constrain to the largest indoor island
                nav_island = -1
                if sampler._constrain_to_largest_nav_island:
                    nav_island = largest_indoor_island_id
                new_receptacle = get_navigable_receptacles(
                    self.sim, [new_receptacle], nav_island
                )  # type: ignore
                if len(new_receptacle) != 0:  # type: ignore
                    new_goal_receptacles.append(new_receptacle[0])  # type: ignore
            assert (
                len(new_goal_receptacles) == num_targets
            ), "Unable to sample goal Receptacles for all requested targets."
            goal_receptacles[sampler_name] = new_goal_receptacles
            all_goal_receptacles.extend(new_goal_receptacles)

        # Goal and target containing receptacles are allowed 1 extra maximum object for each goal/target if a limit was defined
        for recep in [*all_goal_receptacles, *all_target_receptacles]:
            recep_tracker.inc_count(recep.unique_name)

        # sample AO states for objects in the scene
        # ao_instance_handle -> [ (link_ix, state), ... ]
        ao_states: Dict[str, Dict[int, float]] = {}
        for _sampler_name, ao_state_sampler in self._ao_state_samplers.items():
            sampler_states = ao_state_sampler.sample(
                self.sim,
                [*all_target_receptacles, *all_goal_receptacles],
            )
            if sampler_states is None:
                return None
            for sampled_instance, link_states in sampler_states.items():
                if sampled_instance.handle not in ao_states:
                    ao_states[sampled_instance.handle] = {}
                for link_ix, joint_state in link_states.items():
                    ao_states[sampled_instance.handle][link_ix] = joint_state

        # visualize after setting AO states to correctly see scene state
        if self._render_debug_obs:
            self.visualize_scene_receptacles()
            self.dbv.make_debug_video(prefix="receptacles_")

        # track a list of target objects to be used for settle culling later
        target_object_names: List[str] = []
        # sample object placements
        self.object_to_containing_receptacle: Dict[str, Receptacle] = {}
        for sampler_name, obj_sampler in self._obj_samplers.items():
            object_sample_data = obj_sampler.sample(
                self.sim,
                recep_tracker,
                target_receptacles[sampler_name],
                snap_down=True,
                dbv=(self.dbv if self._render_debug_obs else None),
            )
            if len(object_sample_data) == 0:
                return None
            new_objects, receptacles = zip(*object_sample_data)
            # collect names of all newly placed target objects
            target_object_names.extend(
                [
                    obj.handle
                    for obj in new_objects[
                        : len(target_receptacles[sampler_name])
                    ]
                ]
            )
            for obj, rec in zip(new_objects, receptacles):
                self.object_to_containing_receptacle[obj.handle] = rec
            if sampler_name not in self.episode_data["sampled_objects"]:
                self.episode_data["sampled_objects"][
                    sampler_name
                ] = new_objects
            else:
                # handle duplicate sampler names
                self.episode_data["sampled_objects"][
                    sampler_name
                ] += new_objects
            self.ep_sampled_objects += new_objects
            logger.info(
                f"Sampler {sampler_name} generated {len(new_objects)} new object placements."
            )
            # debug visualization showing each newly added object
            if self._render_debug_obs:
                logger.info(
                    f"Generating debug images for {len(new_objects)} objects..."
                )
                for new_object in new_objects:
                    self.dbv.look_at(new_object.translation)
                    self.dbv.debug_obs.append(self.dbv.get_observation())
                logger.info(
                    f"... done generating the debug images for {len(new_objects)} objects."
                )

        # simulate the world for a few seconds to validate the placements
        if self.cfg.enable_check_obj_stability and not self.settle_sim(
            target_object_names
        ):
            logger.warning(
                "Aborting episode generation due to unstable state."
            )
            return None

        for sampler, target_sampler_info in zip(
            self._target_samplers.values(), self.cfg.object_target_samplers
        ):
            sampler.object_instance_set = [
                x
                for y in target_sampler_info["params"]["object_samplers"]
                for x in self.episode_data["sampled_objects"][y]
            ]
            sampler.object_set = [
                x.creation_attributes.handle
                for x in sampler.object_instance_set
            ]

        target_refs: Dict[str, str] = {}

        # sample goal positions for target objects after all other clutter is placed and validated
        handle_to_obj = {obj.handle: obj for obj in self.ep_sampled_objects}
        for sampler_name, target_sampler in self._target_samplers.items():
            obj_sampler_name = targ_sampler_name_to_obj_sampler_names[
                sampler_name
            ][0]
            new_target_objects = target_sampler.sample(
                self.sim,
                recep_tracker,
                snap_down=True,
                dbv=self.dbv,
                target_receptacles=target_receptacles[obj_sampler_name],
                goal_receptacles=goal_receptacles[sampler_name],
                object_to_containing_receptacle=self.object_to_containing_receptacle,
            )
            if new_target_objects is None:
                return None
            for target_handle, (
                new_target_obj,
                _,
            ) in new_target_objects.items():
                match_obj = handle_to_obj[target_handle]

                dist = np.linalg.norm(
                    match_obj.translation - new_target_obj.translation
                )
                if dist < self.cfg.min_dist_from_start_to_goal:
                    return None

                # Add check for checking if the robot can navigate from start to goal
                # given the navmesh of the robot
                if self.cfg.check_navigable:
                    is_navigable = path_is_navigable_given_robot(
                        sim=self.sim,
                        start_pos=match_obj.translation,
                        goal_pos=new_target_obj.translation,
                        robot_navmesh_offsets=self.cfg.navmesh_offset,
                        collision_rate_threshold=self.cfg.max_collision_rate_for_navigable,
                        selected_island=largest_indoor_island_id,
                        angle_threshold=self.cfg.angle_threshold,
                        angular_speed=self.cfg.angular_velocity,
                        distance_threshold=self.cfg.distance_threshold,
                        linear_speed=self.cfg.linear_velocity,
                        dbv=self.dbv,
                        render_debug_video=False,
                    )
                    if not is_navigable:
                        logger.info(
                            f"Collision rate greater than {self.cfg.max_collision_rate_for_navigable}, discarding episode."
                        )
                        return None

            # cache transforms and add visualizations
            for i, (instance_handle, value) in enumerate(
                new_target_objects.items()
            ):
                target_object, target_receptacle = value
                target_receptacles[obj_sampler_name][i] = target_receptacle
                assert (
                    instance_handle not in self.episode_data["sampled_targets"]
                ), f"Duplicate target for instance '{instance_handle}'."
                rom = self.sim.get_rigid_object_manager()
                target_transform = target_object.transformation
                self.episode_data["sampled_targets"][
                    instance_handle
                ] = np.array(target_transform)
                target_refs[
                    instance_handle
                ] = f"{sampler_name}|{len(target_refs)}"
                rom.remove_object_by_handle(target_object.handle)

        # collect final object states and serialize the episode
        # TODO: creating shortened names should be automated and embedded in the objects to be done in a uniform way
        sampled_rigid_object_states = []
        for sampled_obj in self.ep_sampled_objects:
            creation_attrib = sampled_obj.creation_attributes
            file_handle = creation_attrib.handle.split(
                creation_attrib.file_directory
            )[-1].split("/")[-1]
            sampled_rigid_object_states.append(
                (
                    file_handle,
                    np.array(sampled_obj.transformation),
                )
            )

        self.num_ep_generated += 1

        def extract_recep_info(recep):
            return (recep.parent_object_handle, recep.parent_link)

        save_target_receps = [
            extract_recep_info(x) for x in all_target_receptacles
        ]
        save_goal_receps = [
            extract_recep_info(x) for x in all_goal_receptacles
        ]

        name_to_receptacle = {
            k: v.unique_name
            for k, v in self.object_to_containing_receptacle.items()
        }

        return RearrangeEpisode(
            scene_dataset_config=self.cfg.dataset_path,
            additional_obj_config_paths=self.cfg.additional_object_paths,
            episode_id=str(self.num_ep_generated - 1),
            start_position=[0, 0, 0],
            start_rotation=[
                0,
                0,
                0,
                1,
            ],
            scene_id=ep_scene_handle,
            ao_states=ao_states,
            rigid_objs=sampled_rigid_object_states,
            targets=self.episode_data["sampled_targets"],
            target_receptacles=save_target_receps,
            goal_receptacles=save_goal_receps,
            markers=self.cfg.markers,
            name_to_receptacle=name_to_receptacle,
            info={"object_labels": target_refs},
        )

    def initialize_sim(self, scene_name: str, dataset_path: str) -> None:
        """
        Initialize a new Simulator object with a selected scene and dataset.
        """
        # Setup a camera coincident with the agent body node.
        # For debugging visualizations place the default agent where you want the camera with local -Z oriented toward the point of focus.
        camera_resolution = [540, 720]
        sensors = {
            "rgb": {
                "sensor_type": habitat_sim.SensorType.COLOR,
                "resolution": camera_resolution,
                "position": [0.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0],
            }
        }

        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_dataset_config_file = dataset_path
        backend_cfg.scene_id = scene_name
        backend_cfg.enable_physics = True
        backend_cfg.gpu_device_id = self.cfg.gpu_device_id
        if not self._render_debug_obs:
            # don't bother loading textures if not intending to visualize the generation process
            backend_cfg.create_renderer = False

        sensor_specs = []
        for sensor_uuid, sensor_params in sensors.items():
            # sensor_spec = habitat_sim.EquirectangularSensorSpec()
            sensor_spec = habitat_sim.CameraSensorSpec()
            sensor_spec.uuid = sensor_uuid
            sensor_spec.sensor_type = sensor_params["sensor_type"]
            sensor_spec.resolution = sensor_params["resolution"]
            sensor_spec.position = sensor_params["position"]
            sensor_spec.orientation = sensor_params["orientation"]
            sensor_spec.sensor_subtype = (
                habitat_sim.SensorSubType.EQUIRECTANGULAR
            )
            sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor_specs.append(sensor_spec)

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs

        hab_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        if self.sim is None:
            self.sim = habitat_sim.Simulator(hab_cfg)

            object_attr_mgr = self.sim.get_object_template_manager()
            for object_path in self.cfg.additional_object_paths:
                object_attr_mgr.load_configs(osp.abspath(object_path))
        else:
            if self.sim.config.sim_cfg.scene_id != scene_name:
                self.sim.close(destroy=True)
            if self.sim.config.sim_cfg.scene_id == scene_name:
                # we need to force a reset, so reload the NONE scene
                # TODO: we should fix this to provide an appropriate reset method
                proxy_backend_cfg = habitat_sim.SimulatorConfiguration()
                proxy_backend_cfg.scene_id = "NONE"
                proxy_backend_cfg.gpu_device_id = self.cfg.gpu_device_id
                proxy_hab_cfg = habitat_sim.Configuration(
                    proxy_backend_cfg, [agent_cfg]
                )
                self.sim.reconfigure(proxy_hab_cfg)
            self.sim.reconfigure(hab_cfg)

        # setup the debug camera state to the center of the scene bounding box
        scene_bb = (
            self.sim.get_active_scene_graph().get_root_node().cumulative_bb
        )
        self.sim.agents[0].scene_node.translation = scene_bb.center()

        # initialize the debug visualizer
        output_path = (
            "rearrange_ep_gen_output/"
            if self.dbv is None
            else self.dbv.output_path
        )
        self.dbv = DebugVisualizer(self.sim, output_path=output_path)

    def settle_sim(
        self,
        target_object_names: List[str],
        duration: float = 5.0,
        make_video: bool = True,
    ) -> bool:
        """
        Run dynamics for a few seconds to check for stability of newly placed objects and optionally produce a video.
        Returns whether or not the simulation was stable.
        """
        if len(self.ep_sampled_objects) == 0:
            return True

        settle_start_time = time.time()
        logger.info("Running placement stability analysis...")

        scene_bb = (
            self.sim.get_active_scene_graph().get_root_node().cumulative_bb
        )
        new_obj_centroid = mn.Vector3()
        spawn_positions = {}
        for new_object in self.ep_sampled_objects:
            spawn_positions[new_object.handle] = new_object.translation
            new_obj_centroid += new_object.translation
        new_obj_centroid /= len(self.ep_sampled_objects)
        settle_db_obs: List[Any] = []
        if self._render_debug_obs:
            settle_db_obs.append(
                self.dbv.get_observation(
                    look_at=new_obj_centroid, look_from=scene_bb.center()
                )
            )

        while self.sim.get_world_time() < duration:
            self.sim.step_world(1.0 / 30.0)
            if self._render_debug_obs:
                settle_db_obs.append(self.dbv.get_observation())

        logger.info(
            f"   ...done with placement stability analysis in {time.time()-settle_start_time} seconds."
        )
        # check stability of placements
        logger.info(
            "Computing placement stability report:\n----------------------------------------"
        )
        max_settle_displacement = 0
        error_eps = 0.1
        unstable_placements: List[str] = []  # list of unstable object handles
        for new_object in self.ep_sampled_objects:
            error = (
                spawn_positions[new_object.handle] - new_object.translation
            ).length()
            max_settle_displacement = max(max_settle_displacement, error)
            if error > error_eps:
                unstable_placements.append(new_object.handle)
                logger.info(
                    f"    Object '{new_object.handle}' unstable. Moved {error} units from placement."
                )
                if self._render_debug_obs:
                    # NOTE: if debugging visualization is on, any unstable objects will have a peek image prefixed "unstable_" generated to aid debugging
                    self.dbv.peek(
                        new_object,
                        peek_all_axis=True,
                        debug_lines=[
                            (
                                [
                                    spawn_positions[new_object.handle],
                                    new_object.translation,
                                ],
                                mn.Color4.red(),
                            )
                        ],
                    ).save(self.dbv.output_path, prefix="unstable_")
        logger.info(
            f" : unstable={len(unstable_placements)}|{len(self.ep_sampled_objects)} ({len(unstable_placements)/len(self.ep_sampled_objects)*100}%) : {unstable_placements}."
        )
        logger.info(
            f" : Maximum displacement from settling = {max_settle_displacement}"
        )
        # TODO: maybe draw/display trajectory tubes for the displacements?

        if self._render_debug_obs and make_video:
            self.dbv.make_debug_video(
                prefix="settle_", fps=30, obs_cache=settle_db_obs
            )

        # collect detailed receptacle stability report log
        detailed_receptacle_stability_report = (
            "  Detailed receptacle stability analysis:"
        )

        # compute number of unstable objects for each receptacle
        rec_num_obj_vs_unstable: Dict[Receptacle, Dict[str, int]] = {}
        for obj_name, rec in self.object_to_containing_receptacle.items():
            if rec not in rec_num_obj_vs_unstable:
                rec_num_obj_vs_unstable[rec] = {
                    "num_objects": 0,
                    "num_unstable_objects": 0,
                }
            rec_num_obj_vs_unstable[rec]["num_objects"] += 1
            if obj_name in unstable_placements:
                rec_num_obj_vs_unstable[rec]["num_unstable_objects"] += 1
        for rec, obj_in_rec in rec_num_obj_vs_unstable.items():
            rec_unique_name = rec.unique_name if rec is not None else "floor"
            detailed_receptacle_stability_report += f"\n      receptacle '{rec_unique_name}': ({obj_in_rec['num_unstable_objects']}/{obj_in_rec['num_objects']}) (unstable/total) objects."

        success = len(unstable_placements) == 0

        # count unstable target objects, these can't be salvaged
        unstable_target_objects = [
            obj_name
            for obj_name in unstable_placements
            if obj_name in target_object_names
        ]
        logger.info(
            f"{len(unstable_target_objects)} target objects are unstable."
        )

        # optionally salvage the episode by removing unstable objects
        if (
            self.cfg.correct_unstable_results
            and not success
            and len(unstable_target_objects) == 0
        ):
            detailed_receptacle_stability_report += (
                "\n  attempting to correct unstable placements..."
            )
            for sampler_name, objects in self.episode_data[
                "sampled_objects"
            ].items():
                obj_names = [obj.handle for obj in objects]
                sampler = self._obj_samplers[sampler_name]
                unstable_subset = [
                    obj_name
                    for obj_name in unstable_placements
                    if obj_name in obj_names
                ]

                # check that we have freedom to reject some objects
                num_required_objects = sampler.num_objects[0]
                num_stable_objects = len(objects) - len(unstable_subset)
                if num_stable_objects >= num_required_objects:
                    # remove the unstable objects from datastructures
                    self.episode_data["sampled_objects"][sampler_name] = [
                        obj
                        for obj in self.episode_data["sampled_objects"][
                            sampler_name
                        ]
                        if obj.handle not in unstable_subset
                    ]
                    self.ep_sampled_objects = [
                        obj
                        for obj in self.ep_sampled_objects
                        if obj.handle not in unstable_subset
                    ]
                else:
                    detailed_receptacle_stability_report += f"\n  ... could not remove all unstable placements without violating minimum object sampler requirements for {sampler_name}"
                    detailed_receptacle_stability_report += (
                        "\n----------------------------------------"
                    )
                    logger.info(detailed_receptacle_stability_report)
                    return False
            detailed_receptacle_stability_report += f"\n  ... corrected unstable placements successfully. Final object count = {len(self.ep_sampled_objects)}"
            # we removed all unstable placements
            success = True

        detailed_receptacle_stability_report += (
            "\n----------------------------------------"
        )
        logger.info(detailed_receptacle_stability_report)

        # generate debug images of all final object placements
        if self._render_debug_obs and success:
            for obj in self.ep_sampled_objects:
                self.dbv.peek(obj, peek_all_axis=True).save(
                    self.dbv.output_path
                )

        # return success or failure
        return success
