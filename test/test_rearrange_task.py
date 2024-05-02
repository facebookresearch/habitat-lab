#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os.path as osp
import random
import time
from glob import glob

import magnum as mn
import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

import habitat
import habitat.datasets.rearrange.run_episode_generator as rr_gen
import habitat.datasets.rearrange.samplers.receptacle as hab_receptacle
import habitat.sims.habitat_simulator.sim_utilities as sutils
import habitat.tasks.rearrange.rearrange_sim
import habitat.tasks.rearrange.rearrange_task
import habitat.utils.env_utils
import habitat_sim
from habitat.config.default import get_config
from habitat.core.embodied_task import Episode
from habitat.core.environments import get_env_class
from habitat.core.logging import logger
from habitat.datasets.rearrange.rearrange_dataset import RearrangeDatasetV0
from habitat.utils.geometry_utils import is_point_in_triangle

CFG_TEST = "benchmark/rearrange/skills/pick.yaml"
GEN_TEST_CFG = (
    "habitat-lab/habitat/datasets/rearrange/configs/test_config.yaml"
)
EPISODES_LIMIT = 6
MAX_PER_TASK_TEST_STEPS = 500


def check_binary_serialization(dataset: RearrangeDatasetV0):
    start_time = time.time()
    bin_dict = dataset.to_binary()
    logger.info(
        "Binary conversion finished. {} sec".format((time.time() - start_time))
    )
    decoded_dataset = RearrangeDatasetV0()
    decoded_dataset.from_binary(bin_dict)
    decoded_dataset.config = dataset.config
    assert len(decoded_dataset.episodes) == len(dataset.episodes)
    episode = decoded_dataset.episodes[0]
    assert isinstance(episode, Episode)
    dataset_decoded_bin = decoded_dataset.to_binary()
    # check that the expected keys are present in both encoded Dicts
    expected_keys = ["all_transforms", "idx_to_name", "all_eps"]
    for key in expected_keys:
        assert key in bin_dict
        assert key in dataset_decoded_bin

    # use JSON serialization to test integrety of dataset binary serialization -> deserialization
    assert json.loads(decoded_dataset.to_json()) == json.loads(
        dataset.to_json()
    ), "JSON dataset encoding of binary decoding isn't consistent."


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
    check_binary_serialization(dataset)


def _get_test_pddl():
    """
    Helper to get a test PDDL instance.
    """

    config = get_config(
        "habitat-lab/habitat/config/benchmark/rearrange/multi_task/rearrange_easy.yaml",
        [
            # Concurrent rendering can cause problems on CI.
            "habitat.simulator.concur_render=False",
            "habitat.dataset.split=val",
        ],
    )
    env_class = get_env_class(config.habitat.env_task)
    env = habitat.utils.env_utils.make_env_fn(
        env_class=env_class, config=config
    )
    env.reset()
    return env.env.env._env.task.pddl_problem  # type: ignore


def test_pddl_actions():
    """
    Checks we can execute all PDDL actions.
    """

    pddl = _get_test_pddl()
    sim_info = pddl.sim_info

    poss_actions = pddl.get_possible_actions()
    for action in poss_actions:
        action.apply_if_true(sim_info)
    pddl._sim_info.sim.close()


def test_pddl_action_postconds():
    """
    Tests the PDDL system action post conditions have the expected outcome in
    the simulator.
    """

    pddl = _get_test_pddl()
    sim_info = pddl.sim_info

    # Check that the predicates are registering that the robot is not holding
    # anything at the start.
    true_preds = pddl.get_true_predicates()
    assert any(x.name == "not_holding" for x in true_preds)

    poss_actions = pddl.get_possible_actions()
    ac_strs = [x.compact_str for x in poss_actions]

    # Make sure we can't do an action if the precondition is not satisfied.
    place_ac = poss_actions[
        ac_strs.index("place(goal0|0,TARGET_goal0|0,robot_0)")
    ]
    assert not place_ac.apply_if_true(sim_info)

    # Make sure we can do an action that does have satisifed preconditions.
    nav_ac = poss_actions[ac_strs.index("nav(goal0|0,robot_0)")]
    assert nav_ac.apply_if_true(sim_info)

    # Now test preconditions hold for a longer chain of actions
    pick_ac = poss_actions[ac_strs.index("pick(goal0|0,robot_0)")]
    nav_to_goal_ac = poss_actions[ac_strs.index("nav(TARGET_goal0|0,robot_0)")]
    assert pick_ac.apply_if_true(sim_info)
    assert nav_to_goal_ac.apply_if_true(sim_info)
    assert place_ac.apply_if_true(sim_info)

    # Check the object registers at the goal now.
    true_preds = pddl.get_true_predicates()
    assert any(
        x.compact_str == "object_at(goal0|0,TARGET_goal0|0)"
        for x in true_preds
    )
    sim_info.sim.close()


TEST_CFG_PATHS = list(
    glob(
        "habitat-lab/habitat/config/benchmark/rearrange/**/*.yaml",
        recursive=True,
    ),
)
NO_TEST_CONFIGS = [
    "habitat-lab/habitat/config/benchmark/rearrange/hab3_bench/single_agent_bench.yaml",
    "habitat-lab/habitat/config/benchmark/rearrange/hab3_bench/multi_agent_bench.yaml",
    "habitat-lab/habitat/config/benchmark/rearrange/hab3_bench/spot_spot_oracle.yaml",
    "habitat-lab/habitat/config/benchmark/rearrange/hab3_bench/spot_oracle.yaml",
    "habitat-lab/habitat/config/benchmark/rearrange/hab3_bench/humanoid_oracle.yaml",
    "habitat-lab/habitat/config/benchmark/rearrange/hab3_bench/spot_humanoid_oracle.yaml",
]
for config_file_name in NO_TEST_CONFIGS:
    TEST_CFG_PATHS.remove(config_file_name)


@pytest.mark.parametrize("test_cfg_path", TEST_CFG_PATHS)
def test_rearrange_tasks(test_cfg_path):
    """
    Iterates through all the Habitat Rearrangement tasks.
    """

    config = get_config(
        test_cfg_path,
        [
            # Concurrent rendering can cause problems on CI.
            "habitat.simulator.concur_render=False",
            "habitat.dataset.split=val",
        ],
    )

    for agent_config in config.habitat.simulator.agents.values():
        if (
            agent_config.articulated_agent_type == "KinematicHumanoid"
            and not osp.exists(agent_config.motion_data_path)
        ):
            pytest.skip(
                "This test should only be run if we have the motion data files."
            )

    if (
        config.habitat.dataset.data_path
        == "data/ep_datasets/bench_scene.json.gz"
    ):
        pytest.skip(
            "This config is only useful for examples and does not have the generated dataset"
        )
    env_class = get_env_class(config.habitat.env_task)

    env = habitat.utils.env_utils.make_env_fn(
        env_class=env_class, config=config
    )

    for _ in range(2):
        env.reset()
        for _ in range(MAX_PER_TASK_TEST_STEPS):
            action = env.action_space.sample()
            _, _, done, _ = env.step(  # type:ignore[assignment]
                action=action
            )
            if done:
                break

    env.close()


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
    check_binary_serialization(dataset)

    logger.info(
        f"successful_ep = {len(dataset.episodes)} generated in {time.time()-start_time} seconds."
    )


@pytest.mark.skipif(
    not osp.exists("data/test_assets/"),
    reason="This test requires habitat-sim test assets.",
)
def test_receptacle_parsing():
    # 1. Load the parameterized scene
    sim_settings = habitat_sim.utils.settings.default_sim_settings.copy()
    sim_settings[
        "scene"
    ] = "data/test_assets/scenes/simple_room.stage_config.json"
    sim_settings["sensor_height"] = 0
    sim_settings["enable_physics"] = True
    cfg = habitat_sim.utils.settings.make_cfg(sim_settings)
    with habitat_sim.Simulator(cfg) as sim:
        # load test assets
        sim.metadata_mediator.object_template_manager.load_configs(
            "data/test_assets/objects/chair.object_config.json"
        )
        # TODO: add an AO w/ receptacles also

        # test quick receptacle listing:
        list_receptacles = hab_receptacle.get_all_scenedataset_receptacles(sim)
        print(f"list_receptacles = {list_receptacles}")
        # receptacles from stage configs:
        assert (
            "receptacle_aabb_simpleroom_test"
            in list_receptacles["stage"][
                "data/test_assets/scenes/simple_room.stage_config.json"
            ]
        )
        assert (
            "receptacle_mesh_simpleroom_test"
            in list_receptacles["stage"][
                "data/test_assets/scenes/simple_room.stage_config.json"
            ]
        )
        # receptacles from rigid object configs:
        assert (
            "receptacle_aabb_chair_test"
            in list_receptacles["rigid"][
                "data/test_assets/objects/chair.object_config.json"
            ]
        )
        assert (
            "receptacle_mesh_chair_test"
            in list_receptacles["rigid"][
                "data/test_assets/objects/chair.object_config.json"
            ]
        )
        # TODO: receptacles from articulated object configs:
        # assert "" in list_receptacles["articulated"]

        # add the chair to the scene
        chair_template_handle = (
            sim.metadata_mediator.object_template_manager.get_template_handles(
                "chair"
            )[0]
        )
        chair_obj = (
            sim.get_rigid_object_manager().add_object_by_template_handle(
                chair_template_handle
            )
        )

        def randomize_obj_state():
            chair_obj.translation = np.random.random(3)
            chair_obj.rotation = habitat_sim.utils.common.random_quaternion()
            # TODO: also randomize AO state here

        # parse the metadata into Receptacle objects
        test_receptacles = hab_receptacle.find_receptacles(sim)

        # test receptacle filtering in find
        random_receptacle = random.choice(test_receptacles)
        exclude_unique_name = random_receptacle.unique_name
        test_filtered_receptacles = hab_receptacle.find_receptacles(
            sim, exclude_filter_strings=[exclude_unique_name]
        )
        assert len(test_filtered_receptacles) == len(test_receptacles) - 1
        assert (
            len(
                [
                    rec
                    for rec in test_filtered_receptacles
                    if rec.unique_name == random_receptacle.unique_name
                ]
            )
            == 0
        )

        # test the Receptacle instances
        num_test_samples = 10
        for receptacle in test_receptacles:
            # check for contents and correct type parsing
            if receptacle.name == "receptacle_aabb_chair_test":
                assert type(receptacle) is hab_receptacle.AABBReceptacle
            elif receptacle.name == "receptacle_mesh_chair_test.0000":
                assert (
                    type(receptacle) is hab_receptacle.TriangleMeshReceptacle
                )
            elif receptacle.name == "receptacle_aabb_simpleroom_test":
                assert type(receptacle) is hab_receptacle.AABBReceptacle
            elif receptacle.name == "receptacle_mesh_simpleroom_test.0000":
                assert (
                    type(receptacle) is hab_receptacle.TriangleMeshReceptacle
                )
            else:
                # TODO: add AO receptacles
                raise AssertionError(
                    f"Unknown Receptacle '{receptacle.name}' detected. Update unit test golden values if this is expected."
                )

            for _six in range(num_test_samples):
                randomize_obj_state()
                # check that parenting and global transforms are as expected:
                parent_object = None
                expected_global_transform = mn.Matrix4.identity_init()
                global_transform = receptacle.get_global_transform(sim)
                if receptacle.parent_object_handle is not None:
                    parent_object = None
                    if receptacle.parent_link is not None:
                        # articulated object
                        assert receptacle.is_parent_object_articulated
                        parent_object = sim.get_articulated_object_manager().get_object_by_handle(
                            receptacle.parent_object_handle
                        )
                        expected_global_transform = (
                            parent_object.get_link_scene_node(
                                receptacle.parent_link
                            ).absolute_transformation()
                        )
                    else:
                        # rigid object
                        assert not receptacle.is_parent_object_articulated
                        parent_object = sim.get_rigid_object_manager().get_object_by_handle(
                            receptacle.parent_object_handle
                        )
                        # NOTE: we use absolute transformation from the 2nd visual node (scaling node) and root of all render assets to correctly account for any COM shifting, re-orienting, or scaling which has been applied.
                        expected_global_transform = (
                            parent_object.visual_scene_nodes[
                                1
                            ].absolute_transformation()
                        )
                    assert parent_object is not None
                    assert np.allclose(
                        global_transform, expected_global_transform, atol=1e-06
                    )
                else:
                    # this is a stage Receptacle (global transform)
                    if type(receptacle) is not hab_receptacle.AABBReceptacle:
                        assert np.allclose(
                            global_transform,
                            expected_global_transform,
                            atol=1e-06,
                        )
                    else:
                        # NOTE: global AABB Receptacles have special handling here which is not explicitly tested. See AABBReceptacle.get_global_transform()
                        expected_global_transform = global_transform

                for _six2 in range(num_test_samples):
                    sample_point = receptacle.sample_uniform_global(
                        sim, sample_region_scale=1.0
                    )
                    expected_local_sample_point = (
                        expected_global_transform.inverted().transform_point(
                            sample_point
                        )
                    )
                    if type(receptacle) is hab_receptacle.AABBReceptacle:
                        # check that the world->local sample point is contained in the local AABB
                        assert receptacle.bounds.contains(
                            expected_local_sample_point
                        )
                    elif (
                        type(receptacle)
                        is hab_receptacle.TriangleMeshReceptacle
                    ):
                        # check that the local point is within a mesh triangle
                        in_mesh = False
                        for f_ix in range(
                            int(len(receptacle.mesh_data.indices) / 3)
                        ):
                            verts = receptacle.get_face_verts(f_ix)
                            if is_point_in_triangle(
                                expected_local_sample_point,
                                verts[0],
                                verts[1],
                                verts[2],
                            ):
                                in_mesh = True
                                break
                        assert (
                            in_mesh
                        ), "The point must belong to a triangle of the local mesh to be valid."


@pytest.mark.skipif(
    not osp.exists("data/replica_cad/replicaCAD.scene_dataset_config.json"),
    reason="This test requires replica cad SceneDataset.",
)
@pytest.mark.skipif(
    not habitat_sim.bindings.built_with_bullet,
    reason="Bullet physics used for validation.",
)
def test_any_object_receptacle():
    sim_settings = habitat_sim.utils.settings.default_sim_settings.copy()
    sim_settings["scene"] = "apt_0"
    sim_settings[
        "scene_dataset_config_file"
    ] = "data/replica_cad/replicaCAD.scene_dataset_config.json"
    cfg = habitat_sim.utils.settings.make_cfg(sim_settings)
    with habitat_sim.Simulator(cfg) as sim:
        # get an object which does not have a Receptacle defined
        stool_object = sutils.get_obj_from_handle(
            sim, "frl_apartment_stool_02_:0000"
        )
        # get an ArticulatedObject and an ArticulatedLink to try
        counter_ao = sutils.get_obj_from_handle(sim, "kitchen_counter_:0000")
        drawer_link_id = 6
        drawer_obj_id = 125

        stool_receptacle = hab_receptacle.AnyObjectReceptacle(
            "stool_rec", stool_object.handle
        )
        counter_receptacle = hab_receptacle.AnyObjectReceptacle(
            "counter_rec", counter_ao.handle
        )
        drawer_receptacle = hab_receptacle.AnyObjectReceptacle(
            "drawer_rec", counter_ao.handle, parent_link=drawer_link_id
        )

        down = mn.Vector3(0, -1, 0)

        def validate_rec_with_raycast_samples(rec, parent_obj_id):
            samples = [rec.sample_uniform_global(sim, 1.0) for _ in range(100)]
            # NOTE: we bump the ray origin up to account for collision object inflation around the bounding volume.
            sample_raycast_results = [
                sim.cast_ray(
                    habitat_sim.geo.Ray(sample + mn.Vector3(0, 0.2, 0), down)
                )
                for sample in samples
            ]
            samples_over_obj = len(
                [
                    True
                    for raycast_result in sample_raycast_results
                    if (
                        raycast_result.has_hits()
                        and parent_obj_id
                        in [hit.object_id for hit in raycast_result.hits]
                    )
                ]
            )
            assert (
                samples_over_obj > 75
            ), "75 percent of samples should be above the object in question for this scene with high statistical likelihood."

        for rec, parent_obj_id in [
            (stool_receptacle, stool_object.object_id),
            (counter_receptacle, counter_ao.object_id),
            (drawer_receptacle, drawer_obj_id),
        ]:
            validate_rec_with_raycast_samples(rec, parent_obj_id)
            # test that samples in a scaled Receptacle region align with the global bounding box center
            center_sample = rec.sample_uniform_global(sim, 0.01)
            global_keypoints = sutils.get_global_keypoints_from_object_id(
                sim, parent_obj_id
            )
            sample_to_center = mn.Vector3(global_keypoints[0] - center_sample)
            assert mn.math.dot(sample_to_center.normalized(), down) >= 0.99
            # check that the global keypoints all fit inside the rec computed bounding box
            rec_global_bb = rec._get_global_bb(sim).padded(mn.Vector3(0.001))
            for keypoint in global_keypoints:
                assert rec_global_bb.contains(keypoint)
