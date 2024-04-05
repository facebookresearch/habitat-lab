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
from habitat.datasets.rearrange.run_episode_generator import (
    SceneSamplerParamsConfig,
    SceneSamplerConfig,
    RearrangeEpisodeGeneratorConfig
)
import habitat_sim
from habitat.datasets.rearrange.navmesh_utils import (
    get_largest_island_index,
    path_is_navigable_given_robot,
)
from IPython import embed

def get_config_defaults() -> "DictConfig":
    """
    Populates and resturns a default config for a RearrangeEpisode.
    """
    return OmegaConf.create(RearrangeEpisodeGeneratorConfig())  # type: ignore[call-overload]


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


def get_island_sampled_point(
        pathfinder: habitat_sim.nav.PathFinder,
        sim: habitat_sim.Simulator,
        allow_outdoor: bool = True,
    ):
    print("---------------get island sampled point----------")
    largest_island, second_island = None, None
    assert pathfinder.is_loaded, "PathFinder is not loaded."
    island_areas = [
        (island_ix, pathfinder.island_area(island_index=island_ix))
        for island_ix in range(pathfinder.num_islands)
    ]
    island_areas.sort(reverse=True, key=lambda x: x[1])
    if not allow_outdoor:
        # classify indoor vs outdoor
        island_outdoor_classifications = [
            is_outdoor(pathfinder, sim, island_info[0])
            for island_info in island_areas
        ]
        if False not in island_outdoor_classifications:
            print("No indoor islands")
            largest_island, second_island = -1, -1
        indoor_islands = island_areas[island_outdoor_classifications.index(False)]
        if(len(indoor_islands) >= 2):
            largest_island, second_island =  indoor_islands[0], indoor_islands[1]
        else:
            largest_island, second_island = -1, -1
    print("Island_areas:", island_areas)
    largest_island, second_island = island_areas[0][0], island_areas[1][0]
    start_navigable = pathfinder.get_random_navigable_point(
        islandIndex=largest_island
        )
    goal_navigable = pathfinder.get_random_navigable_point(
        islandIndex=second_island
        )
    print("----start_navigable: ", start_navigable, " goal_navigable: ", goal_navigable, "----------")


def generate_fn():
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
        ep_gen.vdb.output_path = osp.abspath(args.db_output)

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
                    #KL
                    # cur_file_name = cfg.scene_sets[0].included_substrings[0]
                    # output_path += f"_{cur_file_name}"
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


if __name__ == "__main__":
    generate_fn()
    # cur_sim = habitat_sim.Simulator
    # pathfinder = habitat_sim.nav.PathFinder()
    # pathfinder.load
    # _,_ = get_island_sampled_point(, cur_sim, allow_outdoor=False)

