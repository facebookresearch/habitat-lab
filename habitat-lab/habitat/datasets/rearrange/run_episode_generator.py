#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List

import numpy as np
from omegaconf import OmegaConf

from habitat.core.logging import logger
from habitat.datasets.rearrange.rearrange_dataset import RearrangeDatasetV0
from habitat.datasets.rearrange.rearrange_generator import (
    RearrangeEpisodeGenerator,
)
from habitat.datasets.rearrange.samplers.receptacle import (
    get_all_scenedataset_receptacles,
)

if TYPE_CHECKING:
    from habitat.config import DictConfig


@dataclass
class SceneSamplerParamsConfig:
    scene: str = "v3_sc1_staging_00"
    scene_sets: List[Any] = field(default_factory=list)


@dataclass
class SceneSamplerConfig:
    type: str = "single"
    params: SceneSamplerParamsConfig = SceneSamplerParamsConfig()
    comment: str = ""


@dataclass
class RearrangeEpisodeGeneratorConfig:
    # The minimum distance from the target object starting position to its goal
    min_dist_from_start_to_goal: float = 0.5
    gpu_device_id: int = 0
    # ----- import/initialization parameters ------
    # the scene dataset from which scenes and objects are sampled
    dataset_path: str = "data/replica_cad/replicaCAD.scene_dataset_config.json"
    # any additional object assets to load before defining object sets
    additional_object_paths: List[str] = field(
        default_factory=lambda: ["data/objects/ycb/"]
    )
    # optionally correct unstable states by removing extra unstable objects (within minimum samples limitations)
    correct_unstable_results: bool = True
    # ----- resource set definitions ------
    # Define the sets of scenes, objects, and receptacles which can be sampled from.
    # The SceneDataset will be searched for resources of each type with handles containing ANY "included" substrings and NO "excluded" substrings.

    # Define sets of scene instance handles which can be sampled from for initialization:
    scene_sets: List[Any] = field(
        default_factory=lambda: [
            {
                "name": "any",
                "included_substrings": [""],
                "excluded_substrings": [],
                # NOTE: The "comment" key is intended for notes and descriptions and not consumed by the generator.
                "comment": "The empty substring acts like a wildcard, selecting all scenes.",
            },
        ]
    )
    # Define the sets of object handles which can be sampled from for placement and target sampling:
    # NOTE: Each set must have a unique name.
    object_sets: List[Any] = field(
        default_factory=lambda: [
            {
                "name": "any",
                "included_substrings": [""],
                "excluded_substrings": [],
                # NOTE: The "comment" key is intended for notes and descriptions and not consumed by the generator.
                "comment": "The empty substring acts like a wildcard, selecting all objects.",
            },
        ]
    )

    # Define the sets of receptacles which can be sampled from for placing objects and targets:
    # The SceneDataset will be searched for objects containing receptacle metadata.
    # Receptacle name substrings are used to further constrain sets.
    # NOTE: Each set must have a unique name.
    receptacle_sets: List[Any] = field(
        default_factory=lambda: [
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
    )
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
    # NOTE: "scene_balanced" scene sampler splits desired episodes evenly amongst scenes in the set and generates all episodes for each scene consecutively.
    # TODO: This default is a bit ugly, but we must use ConfigNodes and define all options to directly nest dicts with yacs|yaml...
    scene_sampler: SceneSamplerConfig = SceneSamplerConfig()

    # Specify name of receptacle and maximum # of placemenets in
    # receptacle. To allow only two objects in the chair, specify:
    # - ["receptacle_aabb_Chr1_Top1_frl_apartment_chair_01", 2]
    max_objects_per_receptacle: List[Any] = field(default_factory=list)

    # Define the object sampling configuration
    object_samplers: List[Any] = field(default_factory=list)
    # {"name":str, "type:str", "params":{})
    # - uniform sampler params: {"object_sets":[str], "receptacle_sets":[str], "num_samples":[min, max], "orientation_sampling":str, "nav_to_min_distance":float, "constrain_to_largest_nav_island":bool)
    # NOTE: "orientation_sampling" options: "none", "up", "all"
    # NOTE: (optional) "constrain_to_largest_nav_island" (default False): if True, valid placements must snap to the largest navmesh island
    # NOTE: (optional) "nav_to_min_distance" (default -1): if not -1, valid placements must snap to the navmesh with horizontal distance less than this value
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

    # Define the desired object target sampling (i.e., where should an existing object be moved to)
    object_target_samplers: List[Any] = field(default_factory=list)
    # {"name":str, "type:str", "params":{})
    # - uniform target sampler params:
    # {"object_samplers":[str], "receptacle_sets":[str], "num_samples":[min, max], "orientation_sampling":str)
    # NOTE: random instances are chosen from the specified, previously excecuted object sampler up to the maximum number specified in params.
    # NOTE: previous samplers referenced must have: combined minimum samples >= minimum requested targets
    # NOTE: "orientation_sampling" options: "none", "up", "all"
    # NOTE: (optional) "constrain_to_largest_nav_island" (default False): if True, valid placements must snap to the largest navmesh island
    # NOTE: (optional) "nav_to_min_distance" (default -1): if not -1, valid placements must snap to the navmesh with horizontal distance less than this value
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

    # define ArticulatedObject(AO) joint state sampling (when a scene is initialized, all samplers are run for all matching AOs)
    ao_state_samplers: List[Any] = field(default_factory=list)
    # TODO: the cupboard asset needs to be modified to remove self-collisions or have collision geometry not intersecting the wall.
    # TODO: does not support spherical joints (3 dof joints)
    # - uniform continuous range for a single joint. params: ("ao_handle", "link name", min, max)
    # Example:
    #     {"name": "open_fridge_top_door",
    #     "type": "uniform",
    #     "params": ["fridge", "top_door", 1.5, 1.5]}
    # - "composite" type sampler (rejection sampling of composite configuration)
    # params: [{"ao_handle":str, "joint_states":[[link name, min max], ], "should_sample_all_joints:bool"}, ]
    # If should_sample_all_joints is True (defaults to False) then all joints of an AO will be sampled and not just the one the target is in.
    # for example, should_sample_all_joints should be true for the fridge since the joints (the door) angle need to be sampled when the object
    # is inside. But for kitchen drawers, this should be false since the joints (all drawers) should not be sampled when the object is on the
    # countertop (only need to sample for the drawer the object is in)

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
    markers: List[Any] = field(default_factory=list)

    # If we want to re-generate the nav mesh or not
    regenerate_new_mesh: bool = True
    # The radius of the agent in meters
    agent_radius: float = 0.25
    # The height of the agent in meters
    agent_height: float = 0.61
    # The max climb of the agent
    agent_max_climb: float = 0.02
    # The maximum slope that is considered walkable in degrees
    agent_max_slope: float = 5.0
    # If we want to check the navigability of the robot
    check_navigable: bool = False
    # The navmesh setting of the robot
    navmesh_offset: List[Any] = field(default_factory=list)
    # The angle threshold of the robot
    angle_threshold: float = 0.1
    # The angualr velocity of the robot
    angular_velocity: float = 10
    # The distance threshold of the robot
    distance_threshold: float = 0.2
    # The linear velocity of the robot
    linear_velocity: float = 10.0
    # The collision rate for navigation
    max_collision_rate_for_navigable: float = 0.5
    # If we want to check the stability of object placement
    enable_check_obj_stability: bool = True


def get_config_defaults() -> "DictConfig":
    """
    Populates and resturns a default config for a RearrangeEpisode.
    """
    return OmegaConf.create(RearrangeEpisodeGeneratorConfig())  # type: ignore[call-overload]


def print_metadata_mediator(ep_gen):
    mm = ep_gen.sim.metadata_mediator
    receptacles = get_all_scenedataset_receptacles(ep_gen.sim)
    logger.info("==================================")
    logger.info("Listing SceneDataset Summary")
    logger.info("==================================")
    logger.info(f" SceneDataset: {mm.active_dataset}\n")
    logger.info("--------")
    logger.info(" Scenes:")
    logger.info("--------\n    ")
    logger.info("\n     ".join(mm.get_scene_handles()))
    logger.info("---------------")
    logger.info(" Rigid Objects:")
    logger.info("---------------\n    ")
    logger.info(
        "\n     ".join(mm.object_template_manager.get_template_handles()),
    )
    logger.info("---------------------")
    logger.info(" Articulated Objects:")
    logger.info("---------------------\n    ")
    logger.info("\n     ".join(mm.urdf_paths))

    logger.info("-------------------------")
    logger.info("Stage Global Receptacles:")
    logger.info("-------------------------")
    for handle, r_list in receptacles["stage"].items():
        logger.info(f"  - {handle}\n    ")
        logger.info("\n     ".join(r_list))

    logger.info("-------------------------")
    logger.info("Rigid Object Receptacles:")
    logger.info("-------------------------")
    for handle, r_list in receptacles["rigid"].items():
        logger.info(f"  - {handle}\n    ")
        logger.info("\n     ".join(r_list))
    logger.info("-------------------------------")
    logger.info("Articulated Object receptacles:")
    logger.info("-------------------------------")
    for handle, r_list in receptacles["articulated"].items():
        logger.info(f"  - {handle}\n    ")
        logger.info("\n     ".join(r_list))

    logger.info("==================================")
    logger.info("Done listing SceneDataset summary")
    logger.info("==================================")


def get_arg_parser():
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
        "--limit-scene-set",
        type=str,
        default=None,
        help="Limit to one of the scene set samplers. Used to differentiate scenes from training and eval.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="The number of episodes to generate.",
    )
    parser.add_argument("--seed", type=int)
    return parser


if __name__ == "__main__":
    parser = get_arg_parser()
    args, _ = parser.parse_known_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # merge the configuration from file with the default
    cfg = get_config_defaults()
    logger.info(f"\n\nOriginal Config:\n{cfg}")
    if args.config is not None:
        assert osp.exists(
            args.config
        ), f"Provided config, '{args.config}', does not exist."
        override_config = OmegaConf.load(args.config)
        cfg = OmegaConf.merge(cfg, override_config)  # type: ignore[assignment]

    logger.info(f"\n\nModified Config:\n{cfg}\n\n")

    dataset = RearrangeDatasetV0()
    with RearrangeEpisodeGenerator(
        cfg=cfg,
        debug_visualization=args.debug,
        limit_scene_set=args.limit_scene_set,
        num_episodes=args.num_episodes,
    ) as ep_gen:
        if not osp.isdir(args.db_output):
            os.makedirs(args.db_output)
        ep_gen.dbv.output_path = osp.abspath(args.db_output)

        # Simulator has been initialized and SceneDataset is populated
        if args.list:
            # NOTE: you can retrieve a string CSV rep of the full SceneDataset with ep_gen.sim.metadata_mediator.dataset_report()
            print_metadata_mediator(ep_gen)
        else:
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
