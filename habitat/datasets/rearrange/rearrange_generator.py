#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
from typing import Any, Dict, List, Optional, Tuple

import magnum as mn
import numpy as np
from tqdm import tqdm
from yacs.config import CfgNode as CN

import habitat.datasets.rearrange.samplers as samplers
import habitat.sims.habitat_simulator.sim_utilities as sutils
import habitat_sim
from habitat.core.logging import logger
from habitat.datasets.rearrange.rearrange_dataset import (
    RearrangeDatasetV0,
    RearrangeEpisode,
)
from habitat.datasets.rearrange.receptacle import (
    find_receptacles,
    get_all_scenedataset_receptacles,
)
from habitat.sims.habitat_simulator.debug_visualizer import DebugVisualizer
from habitat.utils.common import cull_string_list_by_substrings


class RearrangeEpisodeGenerator:
    """Generator class encapsulating logic for procedurally sampling individual episodes for general rearrangement tasks.

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

    def __init__(self, cfg: CN, debug_visualization: bool = False) -> None:
        """
        Initialize the generator object for a particular configuration.
        Loads yaml, sets up samplers and debug visualization settings.
        """
        # load and cache the config
        self.cfg = cfg
        self.start_cfg = self.cfg.clone()

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
        self._get_resource_sets()
        self._get_scene_sampler()
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
        self._receptacle_sets: Dict[
            str, Tuple[List[str], List[str], List[str], List[str]]
        ] = {}

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
                assert (
                    type(scene_set[list_key]) is list
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
                assert (
                    type(object_set[list_key]) is list
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
                assert (
                    type(receptacle_set[list_key]) is list
                ), f"cfg.receptacle_sets - '{receptacle_set['name']}' '{list_key}' must be a list of strings."

            # NOTE: we can't finalize this list until sampling time when objects are instanced and receptacle metadata is scraped from the scene
            self._receptacle_sets[receptacle_set["name"]] = (
                receptacle_set["included_object_substrings"],
                receptacle_set["excluded_object_substrings"],
                receptacle_set["included_receptacle_substrings"],
                receptacle_set["excluded_receptacle_substrings"],
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
                object_handles = list(set(object_handles))
                receptacle_info = [
                    self._receptacle_sets[y]
                    for y in obj_sampler_info["params"]["receptacle_sets"]
                ]

                self._obj_samplers[
                    obj_sampler_info["name"]
                ] = samplers.ObjectSampler(
                    object_handles,
                    receptacle_info,
                    (
                        obj_sampler_info["params"]["num_samples"][0],
                        obj_sampler_info["params"]["num_samples"][1],
                    ),
                    obj_sampler_info["params"]["orientation_sampling"],
                    obj_sampler_info["params"].get("sample_region_ratio", 1.0),
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
                # merge and flatten object and receptacle sets
                object_instances = [
                    x
                    for y in target_sampler_info["params"]["object_samplers"]
                    for x in self.episode_data["sampled_objects"][y]
                ]
                receptacle_info = [
                    self._receptacle_sets[y]
                    for y in target_sampler_info["params"]["receptacle_sets"]
                ]

                self._target_samplers[
                    target_sampler_info["name"]
                ] = samplers.ObjectTargetSampler(
                    object_instances,
                    receptacle_info,
                    (
                        target_sampler_info["params"]["num_samples"][0],
                        target_sampler_info["params"]["num_samples"][1],
                    ),
                    target_sampler_info["params"]["orientation_sampling"],
                )
            else:
                logger.info(
                    f"Requested target sampler '{target_sampler_info['type']}' is not implemented."
                )
                raise (NotImplementedError)

    def _get_scene_sampler(self) -> None:
        """
        Initialize the scene sampler.
        """
        self._scene_sampler: Optional[samplers.SceneSampler] = None
        if self.cfg.scene_sampler.type == "single":
            self._scene_sampler = samplers.SingleSceneSampler(
                self.cfg.scene_sampler.params.scene
            )
        elif self.cfg.scene_sampler.type == "subset":
            unified_scene_set: List[str] = []
            # concatenate all requested scene sets
            for set_name in self.cfg.scene_sampler.params.scene_sets:
                assert (
                    set_name in self._scene_sets
                ), f"'subset' SceneSampler requested scene_set name, '{set_name}', not found."
                unified_scene_set += self._scene_sets[set_name]

            # cull duplicates
            unified_scene_set = list(set(unified_scene_set))
            self._scene_sampler = samplers.MultiSceneSampler(unified_scene_set)
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
            elif ao_info["type"] == "composite":
                composite_ao_sampler_params: Dict[
                    str, Dict[str, Tuple[float, float]]
                ] = {}
                for entry in ao_info["params"]:
                    ao_handle = entry["ao_handle"]
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
                        )
                self._ao_state_samplers[
                    ao_info["name"]
                ] = samplers.CompositeArticulatedObjectStateSampler(
                    composite_ao_sampler_params
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
        self._scene_sampler.reset()
        for sampler in self._obj_samplers.values():
            sampler.reset()

    def generate_scene(self) -> str:
        """
        Sample a new scene and re-initialize the Simulator.
        Return the generated scene's handle.
        """
        cur_scene_name = self._scene_sampler.sample()
        self.initialize_sim(cur_scene_name, self.cfg.dataset_path)

        return cur_scene_name

    def visualize_scene_receptacles(self) -> None:
        """
        Generate a wireframe bounding box for each receptacle in the scene, aim the camera at it and record 1 observation.
        """
        logger.info("visualize_scene_receptacles processing")
        receptacles = find_receptacles(self.sim)
        for receptacle in receptacles:
            logger.info("receptacle processing")
            viz_objects = receptacle.add_receptacle_visualization(self.sim)

            # sample points in the receptacles to display
            # for sample in range(25):
            #     sample_point = receptacle.sample_uniform_global(self.sim, 1.0)
            #     sutils.add_viz_sphere(self.sim, 0.025, sample_point)

            if viz_objects:
                # point the camera at the 1st viz_object for the Receptacle
                self.vdb.look_at(
                    viz_objects[0].root_scene_node.absolute_translation
                )
                self.vdb.get_observation()
            else:
                logger.warning(
                    f"visualize_scene_receptacles: no visualization object generated for Receptacle '{receptacle.name}'."
                )

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
            new_episode = self.generate_single_episode()
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

        self._reset_samplers()
        self.episode_data: Dict[str, Dict[str, Any]] = {
            "sampled_objects": {},  # object sampler name -> sampled object instances
            "sampled_targets": {},  # target sampler name -> (object, target state)
        }

        ep_scene_handle = self.generate_scene()

        # sample AO states for objects in the scene
        # ao_instance_handle -> [ (link_ix, state), ... ]
        ao_states: Dict[str, Dict[int, float]] = {}
        for sampler_name, ao_state_sampler in self._ao_state_samplers.items():
            sampler_states = ao_state_sampler.sample(self.sim)
            assert (
                sampler_states is not None
            ), f"AO sampler '{sampler_name}' failed"
            for sampled_instance, link_states in sampler_states.items():
                if sampled_instance.handle not in ao_states:
                    ao_states[sampled_instance.handle] = {}
                for link_ix, joint_state in link_states.items():
                    ao_states[sampled_instance.handle][link_ix] = joint_state

        # visualize after setting AO states to correctly see scene state
        if self._render_debug_obs:
            self.visualize_scene_receptacles()
            self.vdb.make_debug_video(prefix="receptacles_")

        # sample object placements
        for sampler_name, obj_sampler in self._obj_samplers.items():
            new_objects = obj_sampler.sample(
                self.sim,
                snap_down=True,
                vdb=(self.vdb if self._render_debug_obs else None),
            )
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
                for new_object in new_objects:
                    self.vdb.look_at(new_object.translation)
                    self.vdb.get_observation()

        # simulate the world for a few seconds to validate the placements
        if not self.settle_sim():
            logger.warning(
                "Aborting episode generation due to unstable state."
            )
            return None

        # generate the target samplers
        self._get_object_target_samplers()

        target_refs = {}

        # sample targets
        for target_idx, (sampler_name, target_sampler) in enumerate(
            self._target_samplers.items()
        ):
            new_target_objects = target_sampler.sample(
                self.sim, snap_down=True, vdb=self.vdb
            )
            # cache transforms and add visualizations
            for instance_handle, target_object in new_target_objects.items():
                assert (
                    instance_handle not in self.episode_data["sampled_targets"]
                ), f"Duplicate target for instance '{instance_handle}'."
                rom = self.sim.get_rigid_object_manager()
                target_bb_size = (
                    target_object.root_scene_node.cumulative_bb.size()
                )
                target_transform = target_object.transformation
                self.episode_data["sampled_targets"][
                    instance_handle
                ] = np.array(target_transform)
                target_refs[instance_handle] = f"{sampler_name}|{target_idx}"
                rom.remove_object_by_handle(target_object.handle)
                if self._render_debug_obs:
                    sutils.add_transformed_wire_box(
                        self.sim,
                        size=target_bb_size / 2.0,
                        transform=target_transform,
                    )
                    self.vdb.look_at(target_transform.translation)
                    self.vdb.debug_line_render.set_line_width(2.0)
                    self.vdb.debug_line_render.draw_transformed_line(
                        target_transform.translation,
                        rom.get_object_by_handle(instance_handle).translation,
                        mn.Color4(1.0, 0.0, 0.0, 1.0),
                        mn.Color4(1.0, 0.0, 0.0, 1.0),
                    )
                    self.vdb.get_observation()

        # collect final object states and serialize the episode
        # TODO: creating shortened names should be automated and embedded in the objects to be done in a uniform way
        sampled_rigid_object_states = [
            (
                x.creation_attributes.handle.split(
                    x.creation_attributes.file_directory
                )[-1].split("/")[-1],
                np.array(x.transformation),
            )
            for x in self.ep_sampled_objects
        ]
        # sampled_rigid_object_states = [
        #     (x.creation_attributes.handle, np.array(x.transformation))
        #     for x in self.ep_sampled_objects
        # ]

        self.num_ep_generated += 1
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
            markers=self.cfg.markers,
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
                "position": [0, 0, 0],
                "orientation": [0, 0, 0.0],
            }
        }

        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_dataset_config_file = dataset_path
        backend_cfg.scene_id = scene_name
        backend_cfg.enable_physics = True
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
            if self.sim.config.sim_cfg.scene_id == scene_name:
                # we need to force a reset, so change the internal config scene name
                # TODO: we should fix this to provide an appropriate reset method
                assert (
                    self.sim.config.sim_cfg.scene_id != "NONE"
                ), "Should never generate episodes in an empty scene. Mistake?"
                self.sim.config.sim_cfg.scene_id = "NONE"
            self.sim.reconfigure(hab_cfg)

        # setup the debug camera state to the center of the scene bounding box
        scene_bb = (
            self.sim.get_active_scene_graph().get_root_node().cumulative_bb
        )
        self.sim.agents[0].scene_node.translation = scene_bb.center()

        # initialize the debug visualizer
        self.vdb = DebugVisualizer(
            self.sim, output_path="rearrange_ep_gen_output/"
        )

    def settle_sim(
        self, duration: float = 5.0, make_video: bool = True
    ) -> bool:
        """
        Run dynamics for a few seconds to check for stability of newly placed objects and optionally produce a video.
        Returns whether or not the simulation was stable.
        """
        if len(self.ep_sampled_objects) == 0:
            return True
        # assert len(self.ep_sampled_objects) > 0

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
            self.vdb.get_observation(
                look_at=new_obj_centroid,
                look_from=scene_bb.center(),
                obs_cache=settle_db_obs,
            )

        while self.sim.get_world_time() < duration:
            self.sim.step_world(1.0 / 30.0)
            if self._render_debug_obs:
                self.vdb.get_observation(obs_cache=settle_db_obs)

        # check stability of placements
        logger.info("Computing placement stability report:")
        max_settle_displacement = 0
        error_eps = 0.1
        unstable_placements = []
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
        logger.info(
            f" : unstable={len(unstable_placements)}|{len(self.ep_sampled_objects)} ({len(unstable_placements)/len(self.ep_sampled_objects)*100}%) : {unstable_placements}."
        )
        logger.info(
            f" : Maximum displacement from settling = {max_settle_displacement}"
        )
        # TODO: maybe draw/display trajectory tubes for the displacements?

        if self._render_debug_obs and make_video:
            self.vdb.make_debug_video(
                prefix="settle_", fps=30, obs_cache=settle_db_obs
            )

        # return success or failure
        return len(unstable_placements) == 0


# =======================================
# Episode Configuration
# ======================================


def get_config_defaults() -> CN:
    """
    Populates and resturns a default config for a RearrangeEpisode.
    """
    _C = CN()

    # ----- import/initialization parameters ------
    # the scene dataset from which scenes and objects are sampled
    _C.dataset_path = "data/replica_cad/replicaCAD.scene_dataset_config.json"
    # any additional object assets to load before defining object sets
    _C.additional_object_paths = ["data/objects/ycb/"]

    # ----- resource set definitions ------
    # Define the sets of scenes, objects, and receptacles which can be sampled from.
    # The SceneDataset will be searched for resources of each type with handles containing ANY "included" substrings and NO "excluded" substrings.

    # Define sets of scene instance handles which can be sampled from for initialization:
    _C.scene_sets = [
        {
            "name": "any",
            "included_substrings": [""],
            "excluded_substrings": [],
            # NOTE: The "comment" key is intended for notes and descriptions and not consumed by the generator.
            "comment": "The empty substring acts like a wildcard, selecting all scenes.",
        },
    ]

    # Define the sets of object handles which can be sampled from for placement and target sampling:
    # NOTE: Each set must have a unique name.
    _C.object_sets = [
        {
            "name": "any",
            "included_substrings": [""],
            "excluded_substrings": [],
            # NOTE: The "comment" key is intended for notes and descriptions and not consumed by the generator.
            "comment": "The empty substring acts like a wildcard, selecting all objects.",
        },
    ]

    # Define the sets of receptacles which can be sampled from for placing objects and targets:
    # The SceneDataset will be searched for objects containing receptacle metadata.
    # Receptacle name substrings are used to further constrain sets.
    # NOTE: Each set must have a unique name.
    _C.receptacle_sets = [
        {
            "name": "any",
            "included_object_substrings": [""],
            "excluded_object_substrings": [],
            "included_receptacle_substrings": [""],
            "excluded_receptacle_substrings": [],
            # NOTE: The "comment" key is intended for notes and descriptions and not consumed by the generator.
            "comment": "The empty substrings act like wildcards, selecting all receptacles for all objects.",
        },
    ]

    # ----- sampler definitions ------
    # Define the scene sampling configuration
    # NOTE: There must be exactly one scene sampler!
    # "type": str ("single" or "subset")
    # "params": {
    #   "scene_sets": [str] (if type "subset")
    #   "scene": str (if type "single")
    #  },
    # NOTE: "single" scene sampler asserts that only a single scene contains the "scene" name substring
    # NOTE: "subset" scene sampler allows sampling from multiple scene sets by name
    # TODO: This default is a bit ugly, but we must use ConfigNodes and define all options to directly nest dicts with yacs|yaml...
    _C.scene_sampler = CN()
    _C.scene_sampler.type = "single"
    _C.scene_sampler.params = CN()
    _C.scene_sampler.params.scene = "v3_sc1_staging_00"
    _C.scene_sampler.params.scene_sets = []
    _C.scene_sampler.comment = ""

    # Define the object sampling configuration
    _C.object_samplers = [
        # {"name":str, "type:str", "params":{})
        # - uniform sampler params: {"object_sets":[str], "receptacle_sets":[str], "num_samples":[min, max], "orientation_sampling":str)
        # NOTE: "orientation_sampling" options: "none", "up", "all"
        # TODO: convert some special examples to yaml:
        # (
        #     "fridge_middle",
        #     "uniform",
        #     (["any"], ["fridge_middle"], 1, 30, "up"),
        # ),
        # Composite object sampling (e.g. apple in bowl)
        #  - parameterized by object and receptacle sets, but inclusive of listed samplers BEFORE the composite sampler
        # Example: sample a basket placement on a table and then place apples in the basket
        # ("basket_sampling", "uniform", (["basket"], ["table"], 1, 1, "up")),
        # (
        #     "in_basket_sampling",
        #     "uniform",
        #     (["apple"], ["basket"], 1, 2, "any"),
        # ),
        # {
        #     "name": "any_one",
        #     "type": "uniform",
        #     "params": {
        #         "object_sets": ["any"],
        #         "receptacle_sets": ["any"],
        #         "num_samples": [1, 1],
        #         "orientation_sampling": "up",
        #     },
        #     "comment": "Sample any one object from any receptacle.",
        # }
    ]

    # Define the desired object target sampling (i.e., where should an existing object be moved to)
    _C.object_target_samplers = [
        # {"name":str, "type:str", "params":{})
        # - uniform target sampler params:
        # {"object_samplers":[str], "receptacle_sets":[str], "num_samples":[min, max], "orientation_sampling":str)
        # NOTE: random instances are chosen from the specified, previously excecuted object sampler up to the maximum number specified in params.
        # NOTE: previous samplers referenced must have: combined minimum samples >= minimum requested targets
        # {
        #     "name": "any_one_target",
        #     "type": "uniform",
        #     "params": {
        #         "object_samplers": ["any_one"],
        #         "receptacle_sets": ["any"],
        #         "num_samples": [1, 1],
        #         "orientation_sampling": "up",
        #     },
        #     "comment": "Sample a target for the object instanced by the 'any_one' object sampler from any receptacle.",
        # }
    ]

    # define ArticulatedObject(AO) joint state sampling (when a scene is initialized, all samplers are run for all matching AOs)
    _C.ao_state_samplers = [
        # TODO: the cupboard asset needs to be modified to remove self-collisions or have collision geometry not intersecting the wall.
        # TODO: does not support spherical joints (3 dof joints)
        # - uniform continuous range for a single joint. params: ("ao_handle", "link name", min, max)
        # Example:
        #     {"name": "open_fridge_top_door",
        #     "type": "uniform",
        #     "params": ["fridge", "top_door", 1.5, 1.5]}
        # - "composite" type sampler (rejection sampling of composite configuration)
        # params: [{"ao_handle":str, "joint_states":[[link name, min max], ]}, ]
    ]

    # ----- marker definitions ------
    # A marker defines a point in the local space of a rigid object or articulated link which can be registered to instances in a scene and tracked
    # Format for each marker is a dict containing:
    # "name": str
    # "type": str ("articulated_object" or "rigid_object")
    # "params": {
    #   "object": str
    #   "link": str (if "articulated_object")
    #   "offset": vec3 []
    #  }
    _C.markers = []

    return _C.clone()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # necessary arguments
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Relative path to RearrangeEpisode generator config.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Relative path to output generated RearrangeEpisodeDataset.",
    )

    # mutually exclusive run and investigate options
    arg_function_group = parser.add_mutually_exclusive_group()
    arg_function_group.add_argument(
        "--list",
        action="store_true",
        help="List available datasource from the configured SceneDataset to console. Use this to quickly investigate available handles for referencing scenes, rigid and articulated objects, and object instances.",
    )
    arg_function_group.add_argument(
        "--run",
        action="store_true",
        help="Run the episode generator and serialize the results.",
    )

    # optional arguments
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Render debug frames and save images/videos during episode generation.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Display progress bar",
    )
    parser.add_argument(
        "--db-output",
        type=str,
        default="rearrange_ep_gen_output/",
        help="Relative path to output debug frames and videos.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="The number of episodes to generate.",
    )

    args, _ = parser.parse_known_args()

    # merge the configuration from file with the default
    cfg = get_config_defaults()
    logger.info(f"\n\nOriginal Config:\n{cfg}")
    if args.config is not None:
        assert osp.exists(
            args.config
        ), f"Provided config, '{args.config}', does not exist."
        cfg.merge_from_file(args.config)

    logger.info(f"\n\nModified Config:\n{cfg}\n\n")

    dataset = RearrangeDatasetV0()
    with RearrangeEpisodeGenerator(
        cfg=cfg, debug_visualization=args.debug
    ) as ep_gen:
        if not osp.isdir(args.db_output):
            os.makedirs(args.db_output)
        ep_gen.vdb.output_path = osp.abspath(args.db_output)

        # Simulator has been initialized and SceneDataset is populated
        if args.list:
            # NOTE: you can retrieve a string CSV rep of the full SceneDataset with ep_gen.sim.metadata_mediator.dataset_report()
            mm = ep_gen.sim.metadata_mediator
            receptacles = get_all_scenedataset_receptacles(ep_gen.sim)
            list_sep = "\n    "
            logger.info("==================================")
            logger.info("Listing SceneDataset Summary")
            logger.info("==================================")
            logger.info(f" SceneDataset: {mm.active_dataset}\n")
            logger.info("--------")
            logger.info(" Scenes:")
            logger.info(
                "--------\n    " + (list_sep.join(mm.get_scene_handles()))
            )
            logger.info("---------------")
            logger.info(" Rigid Objects:")
            logger.info(
                "---------------\n    "
                + (
                    list_sep.join(
                        mm.object_template_manager.get_template_handles()
                    )
                ),
            )
            logger.info("---------------------")
            logger.info(" Articulated Objects:")
            logger.info(
                "---------------------\n    " + (list_sep.join(mm.urdf_paths))
            )

            logger.info("-------------------------")
            logger.info("Stage Global Receptacles:")
            logger.info("-------------------------")
            for handle, r_list in receptacles["stage"].items():
                logger.info(f"  - {handle}\n    " + (list_sep.join(r_list)))

            logger.info("-------------------------")
            logger.info("Rigid Object Receptacles:")
            logger.info("-------------------------")
            for handle, r_list in receptacles["rigid"].items():
                logger.info(f"  - {handle}\n    " + (list_sep.join(r_list)))
            logger.info("-------------------------------")
            logger.info("Articulated Object receptacles:")
            logger.info("-------------------------------")
            for handle, r_list in receptacles["articulated"].items():
                logger.info(f"  - {handle}\n    " + (list_sep.join(r_list)))

            logger.info("==================================")
            logger.info("Done listing SceneDataset summary")
            logger.info("==================================")
        elif args.run:
            import time

            start_time = time.time()
            dataset.episodes += ep_gen.generate_episodes(
                args.num_episodes, args.verbose
            )
            output_path = args.out
            if output_path is None:
                # default
                output_path = "rearrange_ep_dataset.json.gz"
            elif osp.isdir(output_path) or output_path.endswith("/"):
                # append a default filename
                output_path = (
                    osp.abspath(output_path) + "/rearrange_ep_dataset.json.gz"
                )
            else:
                # filename
                if not output_path.endswith(".json.gz"):
                    output_path += ".json.gz"

            if (
                not osp.exists(osp.dirname(output_path))
                and len(osp.dirname(output_path)) > 0
            ):
                os.makedirs(osp.dirname(output_path))
            # serialize the dataset
            import gzip

            with gzip.open(output_path, "wt") as f:
                f.write(dataset.to_json())

            logger.info(
                "=============================================================="
            )
            logger.info(
                f"RearrangeEpisodeGenerator generated {args.num_episodes} episodes in {time.time()-start_time} seconds."
            )
            logger.info(
                f"RearrangeDatasetV0 saved to '{osp.abspath(output_path)}'"
            )
            logger.info(
                "=============================================================="
            )
