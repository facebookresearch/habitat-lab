#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gc
import itertools
import json
import os
import os.path as osp
import time
from glob import glob

import pytest
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

import habitat
import habitat.datasets.rearrange.run_episode_generator as rr_gen
import habitat.tasks.rearrange.rearrange_sim
import habitat.tasks.rearrange.rearrange_task
import habitat.utils.env_utils
from habitat.config.default import _HABITAT_CFG_DIR, get_config
from habitat.core.embodied_task import Episode
from habitat.core.environments import get_env_class
from habitat.core.logging import logger
from habitat.datasets.rearrange.rearrange_dataset import RearrangeDatasetV0
from habitat.tasks.rearrange.multi_task.composite_task import CompositeTask
from habitat_baselines.config.default import get_config as baselines_get_config
from habitat_baselines.rl.ddppo.ddp_utils import find_free_port
from habitat_baselines.run import run_exp

CFG_TEST = "benchmark/rearrange/pick.yaml"
GEN_TEST_CFG = (
    "habitat-lab/habitat/datasets/rearrange/configs/test_config.yaml"
)
EPISODES_LIMIT = 6


def check_json_serialization(dataset: RearrangeDatasetV0):
    start_time = time.time()
    json_str = dataset.to_json()
    logger.info(
        "JSON conversion finished. {} sec".format((time.time() - start_time))
    )
    decoded_dataset = RearrangeDatasetV0()
    decoded_dataset.from_json(json_str)
    decoded_dataset.config = dataset.config
    assert len(decoded_dataset.episodes) == len(dataset.episodes)
    episode = decoded_dataset.episodes[0]
    assert isinstance(episode, Episode)

    # The strings won't match exactly as dictionaries don't have an order for the keys
    # Thus we need to parse the json strings and compare the serialized forms
    assert json.loads(decoded_dataset.to_json()) == json.loads(
        json_str
    ), "JSON dataset encoding/decoding isn't consistent"


def test_rearrange_dataset():
    dataset_config = get_config(CFG_TEST).habitat.dataset
    if not RearrangeDatasetV0.check_config_paths_exist(dataset_config):
        pytest.skip(
            "Please download ReplicaCAD RearrangeDataset Dataset to data folder."
        )

    dataset = habitat.make_dataset(
        id_dataset=dataset_config.type, config=dataset_config
    )
    assert dataset
    dataset.episodes = dataset.episodes[0:EPISODES_LIMIT]
    check_json_serialization(dataset)


@pytest.mark.parametrize(
    "test_cfg_path",
    list(
        glob(
            "habitat-baselines/habitat_baselines/config/rearrange/**/*.yaml",
            recursive=True,
        ),
    ),
)
def test_rearrange_baseline_envs(test_cfg_path):
    """
    Test the Habitat Baseline environments
    """
    config = baselines_get_config(test_cfg_path)
    with habitat.config.read_write(config):
        config.habitat.gym.obs_keys = None
        config.habitat.gym.desired_goal_keys = []

    env_class = get_env_class(config.habitat.env_task)

    env = habitat.utils.env_utils.make_env_fn(
        env_class=env_class, config=config
    )

    with env:
        for _ in range(10):
            env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                _, _, done, _ = env.step(  # type:ignore[assignment]
                    action=action
                )


@pytest.mark.parametrize(
    "test_cfg_path",
    list(
        glob("habitat-lab/habitat/config/benchmark/rearrange/*"),
    ),
)
def test_rearrange_tasks(test_cfg_path):
    """
    Test the underlying Habitat Tasks
    """
    if not osp.isfile(test_cfg_path):
        return

    config = get_config(test_cfg_path)
    if (
        config.habitat.dataset.data_path
        == "data/ep_datasets/bench_scene.json.gz"
    ):
        pytest.skip(
            "This config is only useful for examples and does not have the generated dataset"
        )

    with habitat.Env(config=config) as env:
        for _ in range(5):
            env.reset()


@pytest.mark.parametrize(
    "test_cfg_path",
    list(
        glob("habitat-lab/habitat/config/benchmark/rearrange/*"),
    ),
)
def test_composite_tasks(test_cfg_path):
    """
    Test for the Habitat composite tasks.
    """
    if not osp.isfile(test_cfg_path):
        return

    config = get_config(
        test_cfg_path, ["habitat.simulator.concur_render=False"]
    )
    if "task_spec" not in config.habitat.task:
        return

    if (
        config.habitat.dataset.data_path
        == "data/ep_datasets/bench_scene.json.gz"
    ):
        pytest.skip(
            "This config is only useful for examples and does not have the generated dataset"
        )

    with habitat.Env(config=config) as env:
        if not isinstance(env.task, CompositeTask):
            return

        pddl_path = osp.join(
            _HABITAT_CFG_DIR,
            config.habitat.task.task_spec_base_path,
            config.habitat.task.task_spec + ".yaml",
        )
        with open(pddl_path, "r") as f:
            domain = yaml.safe_load(f)
        if "solution" not in domain:
            return
        n_stages = len(domain["solution"])

        for task_idx in range(n_stages):
            env.reset()
            env.task.jump_to_node(task_idx, env.current_episode)
            env.step(env.action_space.sample())
            env.reset()


# NOTE: set 'debug_visualization' = True to produce videos showing receptacles and final simulation state
@pytest.mark.parametrize("debug_visualization", [False])
@pytest.mark.parametrize("num_episodes", [2])
@pytest.mark.parametrize("config", [GEN_TEST_CFG])
def test_rearrange_episode_generator(
    debug_visualization, num_episodes, config
):
    cfg = rr_gen.get_config_defaults()
    override_config = OmegaConf.load(config)
    cfg = OmegaConf.merge(cfg, override_config)
    assert isinstance(cfg, DictConfig)
    dataset = RearrangeDatasetV0()
    with rr_gen.RearrangeEpisodeGenerator(
        cfg=cfg, debug_visualization=debug_visualization
    ) as ep_gen:
        start_time = time.time()
        dataset.episodes += ep_gen.generate_episodes(num_episodes)

    # test serialization of freshly generated dataset
    check_json_serialization(dataset)

    logger.info(
        f"successful_ep = {len(dataset.episodes)} generated in {time.time()-start_time} seconds."
    )


@pytest.mark.parametrize(
    "test_cfg_path,mode",
    list(
        itertools.product(
            glob("habitat-baselines/habitat_baselines/config/tp_srl_test/*"),
            ["eval"],
        )
    ),
)
def test_tp_srl(test_cfg_path, mode):
    # For testing with world_size=1
    os.environ["MAIN_PORT"] = str(find_free_port())

    run_exp(
        test_cfg_path.replace(
            "habitat-baselines/habitat_baselines/config/", ""
        ),
        mode,
        ["habitat_baselines.eval.split=train"],
    )

    # Needed to destroy the trainer
    gc.collect()

    # Deinit processes group
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# NOTE: set 'debug_visualization' = True to produce videos showing receptacles

@pytest.mark.parametrize("debug_visualization", [True])
@pytest.mark.parametrize("scene_asset", ["GLAQ4DNUx5U", 
#"NBg5UqG3di3", 
#"CFVBbU9Rsyb"
])
def test_receptacles(
    debug_visualization, scene_asset
):
    import habitat_sim
    import magnum as mn
    import numpy as np
    from habitat_sim.utils.common import d3_40_colors_hex
    replica_cad_data_path = "data/replica_cad/replicaCAD.scene_dataset_config.json"
    hm3d_data_path = "data/scene_datasets/hm3d/example/hm3d_example_basis.scene_dataset_config.json"
    
    mm = habitat_sim.metadata.MetadataMediator()
    mm.active_dataset = hm3d_data_path
    print(mm.summary)
    print(mm.dataset_report())
    print(mm.get_scene_handles())

    ##########################
    # Test Mesh Receptacles
    ##########################
    # 1. Load the parameterized scene
    sim_settings = habitat_sim.utils.settings.default_sim_settings.copy()
    sim_settings["scene"] = scene_asset
    sim_settings["scene_dataset_config_file"] = hm3d_data_path
    sim_settings["sensor_height"] = 0
    cfg = habitat_sim.utils.settings.make_cfg(sim_settings)
    with habitat_sim.Simulator(cfg) as sim:

        #place the camera in the scene center looking down
        scene_bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
        look_down = mn.Quaternion.rotation(
                        mn.Deg(-90), mn.Vector3.x_axis()
                    )
        max_dim = max(scene_bb.size_x(), scene_bb.size_z())
        cam_pos = scene_bb.center()
        cam_pos[1] += 0.52*max_dim + scene_bb.size_y()/2.0
        sim.agents[0].scene_node.translation = cam_pos
        sim.agents[0].scene_node.rotation = look_down
        
    # 2. Compute a navmesh
        if not sim.pathfinder.is_loaded:
            # compute a navmesh on the ground plane
            navmesh_settings = habitat_sim.NavMeshSettings()
            navmesh_settings.set_defaults()
            sim.recompute_navmesh(sim.pathfinder, navmesh_settings, True)
    
    # 3. Create receptacles from navmesh data
    #   a) global receptacles
        receptacles = []
        #get navmesh data per-island, convert to lists, create Receptacles
        for isl_ix in range(sim.pathfinder.num_islands):
            island_verts = sim.pathfinder.build_navmesh_vertices(isl_ix)
            island_ixs = sim.pathfinder.build_navmesh_vertex_indices(isl_ix)
            mesh_receptacle = habitat.datasets.rearrange.samplers.receptacle.TriangleMeshReceptacle(
                name=str(isl_ix),
                mesh_data = (island_verts, island_ixs)
            )
            receptacles.append(mesh_receptacle)
    #   TODO: b) load navmesh from .navmesh test files
    #       -local rigid and articulated pre-computed

    # 4. render receptacle debug (vs. navmesh vis)
        observations = []
        if debug_visualization:
            sim.navmesh_visualization = True
            observations.append(sim.get_sensor_observations())
            sim.navmesh_visualization = False
            for isl_ix,mesh_rec in enumerate(receptacles):
                isl_color = mn.Color4.from_srgb(int(d3_40_colors_hex[isl_ix], base=16))
                print(f"isl_color = {isl_color}")
                mesh_rec.debug_draw(sim, color=isl_color)
            observations.append(sim.get_sensor_observations())


    # 5. sample from receptacles
        stat_samples_per_unit_area = 500
        render_samples_per_unit_area = 50

        rec_samples = []
        for isl_ix,mesh_rec in enumerate(receptacles):
            rec_samples.append([])
            num_samples = max(1, int(mesh_rec.total_area*stat_samples_per_unit_area))
            print(f"isl {isl_ix} num samples = {num_samples}")
            for samp_ix in range(num_samples):
                sample = mesh_rec.sample_uniform_global(sim,sample_region_scale=1.0)
                #print(f"    - {sample}")
                rec_samples[-1].append(sample)

        if debug_visualization:
            dblr = sim.get_debug_line_render()
            #draw the samples
            for isl_ix,samples in enumerate(rec_samples):
                isl_color = mn.Color4.from_srgb(int(d3_40_colors_hex[isl_ix], base=16))
                num_samples = max(1, int(mesh_rec.total_area*render_samples_per_unit_area))
                for sample_ix in range(num_samples):
                    dblr.draw_circle(samples[sample_ix], 0.05, isl_color)
                observations.append(sim.get_sensor_observations())

    # 6. test sampling is correct (percent in each triangle equivalent to area weight)
        

    #show observations
    if debug_visualization:
        from habitat_sim.utils import viz_utils as vut
        for obs in observations:
            vut.observation_to_image(obs["color_sensor"], "color").show()

    logger.info(
        f"done"
    )
