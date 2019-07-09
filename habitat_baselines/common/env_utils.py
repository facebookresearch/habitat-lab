import random

import numpy as np

import habitat
from habitat import Config, SimulatorActions, make_dataset


class NavRLEnv(habitat.RLEnv):
    def __init__(self, config_env, config_baseline, dataset):
        self._config_env = config_env.TASK
        self._config_baseline = config_baseline
        self._previous_target_distance = None
        self._previous_action = None
        self._episode_distance_covered = None
        super().__init__(config_env, dataset)

    def reset(self):
        self._previous_action = None

        observations = super().reset()

        self._previous_target_distance = self.habitat_env.current_episode.info[
            "geodesic_distance"
        ]
        return observations

    def step(self, action):
        self._previous_action = action
        return super().step(action)

    def get_reward_range(self):
        return (
            self._config_baseline.BASELINE.RL.SLACK_REWARD - 1.0,
            self._config_baseline.BASELINE.RL.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = self._config_baseline.BASELINE.RL.SLACK_REWARD

        current_target_distance = self._distance_target()
        reward += self._previous_target_distance - current_target_distance
        self._previous_target_distance = current_target_distance

        if self._episode_success():
            reward += self._config_baseline.BASELINE.RL.SUCCESS_REWARD

        return reward

    def _distance_target(self):
        current_position = self._env.sim.get_agent_state().position.tolist()
        target_position = self._env.current_episode.goals[0].position
        distance = self._env.sim.geodesic_distance(
            current_position, target_position
        )
        return distance

    def _episode_success(self):
        if (
            self._previous_action == SimulatorActions.STOP
            and self._distance_target() < self._config_env.SUCCESS_DISTANCE
        ):
            return True
        return False

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


def make_env_fn(config, env_class, rank):
    dataset = make_dataset(config.DATASET.TYPE, config=config.DATASET)
    config.defrost()
    config.SIMULATOR.SCENE = dataset.episodes[0].scene_id
    config.freeze()
    env = env_class(config_env=config, config_baseline=config, dataset=dataset)
    env.seed(rank)
    return env


def construct_envs(config: Config, env_class):

    random.seed(config.SEED)
    baseline_cfg = config.BASELINE.RL.PPO
    env_configs = []
    env_classes = [env_class for _ in range(baseline_cfg.num_processes)]
    dataset = make_dataset(config.DATASET.TYPE)
    scenes = dataset.get_scenes_to_load(config.DATASET)

    if len(scenes) > 0:
        random.shuffle(scenes)

        assert len(scenes) >= baseline_cfg.num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )
        scene_split_size = int(
            np.floor(len(scenes) / baseline_cfg.num_processes)
        )

    scene_splits = [[] for _ in range(baseline_cfg.num_processes)]
    for j, s in enumerate(scenes):
        scene_splits[j % len(scene_splits)].append(s)

    assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(baseline_cfg.num_processes):

        config_env = config.clone()
        config_env.defrost()
        if len(scenes) > 0:
            config_env.DATASET.CONTENT_SCENES = scene_splits[i]

        config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = (
            baseline_cfg.sim_gpu_id
        )

        agent_sensors = baseline_cfg.sensors.strip().split(",")
        for sensor in agent_sensors:
            assert sensor in ["RGB_SENSOR", "DEPTH_SENSOR"]
        config_env.SIMULATOR.AGENT_0.SENSORS = agent_sensors
        config_env.freeze()
        env_configs.append(config_env)

    envs = habitat.VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(
                zip(
                    env_configs, env_classes, range(baseline_cfg.num_processes)
                )
            )
        ),
    )

    return envs
