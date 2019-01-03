import numpy as np

import teas
import teas.datasets.eqa.mp3d_eqa_dataset as mp3d_dataset
from teas.config.experiments.esp_nav import esp_nav_cfg
from teas.core.logging import logger
from teas.datasets import make_dataset

CLOSE_STEP_THRESHOLD = 0.028

IS_GENERATING_VIDEO = False

# List of episodes each from unique house
TEST_EPISODE_SET = [0, 1, 2, 17, 164, 173, 250, 272, 456, 695, 698, 782,
                    966, 970, 1160, 1272, 1295, 1296, 1376, 1384, 1633,
                    1836, 1841, 1967, 2175, 2396, 2575,
                    2717]

RGB_EPISODE_MEANS = [130.27629452458137, 118.33419659326944,
                     120.8483347525963, 129.8308741124471, 141.64853506874445]

EPISODES_LIMIT = 2


def get_minos_for_esp_eqa_config():
    _esp_eqa_c = esp_nav_cfg()
    _esp_eqa_c.task_name = 'EQA-v0'
    _esp_eqa_c.dataset = mp3d_dataset.get_default_mp3d_v1_config()
    _esp_eqa_c.dataset.split = "test"
    _esp_eqa_c.scene = "data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb"
    _esp_eqa_c.resolution = (512, 512)
    _esp_eqa_c.hfov = '45'
    _esp_eqa_c.vfov = '45'
    _esp_eqa_c.sensor_position = [0, 1.09, 0]
    _esp_eqa_c.forward_step_size = 0.1  # in metres
    _esp_eqa_c.turn_angle = 9  # in degrees
    _esp_eqa_c.simulator = 'EspSimulator-v0'

    # Agent configuration
    agent_c = _esp_eqa_c.agents[0]
    agent_c.height = 1.5
    agent_c.radius = 0.1
    agent_c.mass = 32.0
    agent_c.linear_acceleration = 10.0
    agent_c.angular_acceleration = 5 * 3.14
    agent_c.linear_friction = 1.0
    agent_c.angular_friction = 1.0
    agent_c.coefficient_of_restitution = 0.15707963267

    return _esp_eqa_c


def test_mp3d_eqa_dataset():
    dataset_config = mp3d_dataset.get_default_mp3d_v1_config()
    if not mp3d_dataset.Matterport3dDatasetV1.check_config_paths_exist(
            dataset_config):
        logger.info("Test skipped as dataset files are missing.")
        return
    dataset = mp3d_dataset.Matterport3dDatasetV1(dataset_config)
    assert dataset
    assert len(
        dataset.episodes) == mp3d_dataset.EQA_MP3D_V1_TEST_EPISODE_COUNT, \
        "Test split episode number mismatch"


def test_mp3d_eqa_esp():
    eqa_config = get_minos_for_esp_eqa_config()

    if not mp3d_dataset.Matterport3dDatasetV1.check_config_paths_exist(
            eqa_config.dataset):
        logger.info("Test skipped as dataset files are missing.")
        return

    dataset = make_dataset(eqa_config.dataset.name, config=eqa_config.dataset)
    env = teas.TeasEnv(config=eqa_config)
    env.episodes = dataset.episodes[:EPISODES_LIMIT]

    assert env
    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        assert env.action_space.contains(action)
        obs, rew, done, info = env.step(action)
        if not done:
            assert 'rgb' in obs, "RGB image is missing in observation."
            assert obs['rgb'].shape[:2] == eqa_config.resolution, \
                "Observation resolution doesn't correspond to config."

    env.close()


def test_mp3d_eqa_esp_correspondence():
    eqa_config = get_minos_for_esp_eqa_config()

    if not mp3d_dataset.Matterport3dDatasetV1.check_config_paths_exist(
            eqa_config.dataset):
        logger.info("Test skipped as dataset files are missing.")
        return

    dataset = make_dataset(eqa_config.dataset.name, config=eqa_config.dataset)
    env = teas.TeasEnv(config=eqa_config, dataset=dataset)
    env.episodes = dataset.get_episodes(TEST_EPISODE_SET)[:EPISODES_LIMIT]

    if IS_GENERATING_VIDEO:
        from teas.internal.visualize import gen_video

    ep_i = 0
    cycles_n = 2
    while cycles_n > 0:
        env.reset()
        episode = env.current_episode
        assert len(
            episode.goals) == 1, "Episode has no goals or more than one."
        assert len(
            episode.shortest_paths) == 1, \
            "Episode has no shortest paths or more than one."
        # TODO (maksymets) get rid of private member call with better agent
        # state interface
        start_state = env._simulator.agent_state()
        assert np.allclose(
            start_state.position,
            episode.start_position), \
            "Agent's start position diverges from the shortest path's one."

        rgb_frames = []
        depth = []
        labels = []
        rgb_mean = 0

        for step_id, point in enumerate(episode.shortest_paths[0]):
            cur_state = env._simulator.agent_state()

            logger.info(
                'diff position: {} diff rotation: {} '
                'cur_state.position: {} shortest_path.position: {} '
                'cur_state.rotation: {} shortest_path.rotation: {} action: {}'
                ''.format(cur_state.position - point.position,
                          cur_state.rotation - point.rotation,
                          cur_state.position, point.position,
                          cur_state.rotation, point.rotation, point.action
                          )
            )

            assert np.allclose(
                [cur_state.position[0], cur_state.position[2]],
                [point.position[0], point.position[2]],
                atol=CLOSE_STEP_THRESHOLD * (
                        step_id + 1)), \
                "Agent's path diverges from the shortest path."

            obs, rew, done, info = env.step(point.action)

            if not done:
                rgb_mean += obs['rgb'][:, :, :3].mean()
                if IS_GENERATING_VIDEO:
                    # Cut RGB channels from RGBA and fill empty frames of
                    # relevant resolution, collect frames
                    rgb_frames.append(obs['rgb'][:, :, :3])
                    # Fill frames with zeros using same resolution as rgb frame
                    depth.append(
                        np.zeros(obs['rgb'].shape[:2], dtype=np.uint8))
                    labels.append(
                        np.zeros(obs['rgb'].shape[:2], dtype=np.uint8))

        if ep_i < len(RGB_EPISODE_MEANS):
            assert np.isclose(
                RGB_EPISODE_MEANS[ep_i],
                rgb_mean / len(episode.shortest_paths[0])), \
                "RGB output doesn't match the ground truth."

        ep_i = (ep_i + 1) % EPISODES_LIMIT
        if ep_i == 0:
            cycles_n -= 1

        if IS_GENERATING_VIDEO and cycles_n == 0:
            gen_video.make_video(episode, rgb_frames, depth, labels)

    env.close()
