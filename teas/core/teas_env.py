import time
from typing import Type, List, Tuple, Any, Optional

import gym
from gym.spaces.dict_space import Dict as SpaceDict

import teas
from teas.core.dataset import Dataset, Episode
from teas.core.embodied_task import EmbodiedTask
from teas.core.simulator import Observation
from teas.simulators import make_simulator


class TeasEnv(gym.Env):

    def __init__(self, config: Any, dataset: Optional[Dataset] = None) -> None:
        self._config: Any = config
        self._dataset: Optional[Dataset] = dataset
        self._episodes: List[Type[Episode]] = self._dataset.episodes if \
            self._dataset else []
        self._current_episode_index: Optional[int] = None
        self._simulator = make_simulator(id_simulator=self._config.simulator,
                                         config=self._config)
        self._task: EmbodiedTask = teas.make_task(config.task_name,
                                                  config=self._config,
                                                  simulator=self._simulator,
                                                  dataset=dataset)
        self.observation_space = SpaceDict({
            **self._simulator.sensor_suite.observation_spaces.spaces,
            **self._task.sensor_suite.observation_spaces.spaces
        })
        self.action_space = self._simulator.action_space
        self._max_episode_seconds = getattr(
            self._config, "max_episode_seconds", None)
        self._max_episode_steps = getattr(
            self._config, "max_episode_steps", None)
        self._elapsed_steps = 0
        self._episode_started_at: Optional[float] = None
        self._episode_over = False

    @property
    def current_episode(self) -> Type[Episode]:
        assert self._current_episode_index is not None and \
               self._current_episode_index < len(self._episodes)
        return self._episodes[self._current_episode_index]

    @property
    def episodes(self) -> List[Type[Episode]]:
        return self._episodes

    @episodes.setter
    def episodes(self, episodes: List[Type[Episode]]):
        assert len(
            episodes) > 0, "Environment doesn't accept empty episodes list."
        self._episodes = episodes

    @property
    def _elapsed_seconds(self) -> float:
        assert self._episode_started_at, \
            "Elapsed seconds requested before episode was started."
        return time.time() - self._episode_started_at

    def _past_limit(self) -> bool:
        if self._max_episode_steps is not None and self._max_episode_steps <= \
                self._elapsed_steps:
            return True
        elif self._max_episode_seconds is not None and \
                self._max_episode_seconds <= self._elapsed_seconds:
            return True
        return False

    def reset(self) -> Tuple[Observation, Observation, bool, None]:
        self._episode_started_at = time.time()
        self._elapsed_steps = 0
        self._episode_over = False

        # Switch to next episode in a loop
        if len(self.episodes) > 0:
            if self._current_episode_index is None:
                self._current_episode_index = 0
            else:
                self._current_episode_index = \
                    (self._current_episode_index + 1) % len(self._episodes)
            self.reconfigure(self._config)

        # TODO (maksymets) make Task responsible for check if episode is done.
        observations, done = self._simulator.reset()
        observations.update(self._task.sensor_suite.get_observations(
            observations=observations, episode=self.current_episode))
        info = None
        reward = observations["reward"]
        return observations, reward, done, info

    def step(self, action):
        assert self._episode_started_at is not None, "Cannot call step " \
                                                     "before calling reset"
        assert self._episode_over is False, "Episode done, call reset " \
                                            "before calling step"

        observations, done = self._simulator.step(action)
        observations.update(
            self._task.sensor_suite.get_observations(
                observations=observations,
                episode=self.current_episode))

        self._elapsed_steps += 1

        if self._past_limit():
            done = True
            self._episode_over = True

        info = None
        reward = observations["reward"]
        return observations, reward, done, info

    def seed(self, seed: int = None) -> None:
        self._simulator.seed(seed)

    def reconfigure(self, config) -> None:
        # TODO (maksymets) switch to self._config.simulator when it will
        #  be separated
        self._config = self._task.overwrite_sim_config(self._config,
                                                       self.current_episode)
        self._simulator.reconfigure(config)

    def geodesic_distance(self, position_a, position_b) -> float:
        return self._simulator.geodesic_distance(position_a, position_b)

    def semantic_annotations(self):
        return self._simulator.semantic_annotations()

    def render(self, mode='human', close=False) -> None:
        self._simulator.render(mode, close)

    def close(self) -> None:
        self._simulator.close()

# TODO(akadian): Currently I haven't used the AgentState abstract class for
# TeasEnv. Discuss if this is needed.
# class AgentState:
#     r"""Represents the physical state of an agent.
#     """
#
#     def __init__(self):
#         self.position = np.ndarray([0, 0, 0])
#         self.orientation = np.ndarray([0, 0, 0])
#         self.velocity = np.ndarray([0, 0, 0])
#         self.angular_velocity = np.ndarray([0, 0, 0])
#         self.force = np.ndarray([0, 0, 0])
#         self.torque = np.ndarray([0, 0, 0])


# TODO(akadian): Currently I haven't used the AgentConfiguration abstract
# class for TeasEnv. Discuss if this is needed.
# class AgentConfiguration:
#     r"""Represents a configuration for an embodied Agent.
#     """
#
#     def __init__(self):
#         self.height = 1.0
#         self.radius = 0.1
#         self.mass = 32.0
#         self.linear_acceleration = 20.0
#         self.angular_acceleration = 4 * 3.14
#         self.linear_friction = 0.5
#         self.angular_friction = 1.0
#         self.coefficient_of_restitution = 0.0
#
#         self.sensor_specs = [SensorSpec()]
#         self.action_space = ActionSpace()
#         # defines agent embodiment
#         self.body_type = \
#                           'cylinder(0.1,1.0)|obj(./file.obj)|urdf(
#                           ./file.urdf)'


# TODO(akadian): Currently I haven't used the Agent abstract class for
# TeasEnv. Discuss if this is needed.
# class Agent:
#     r"""Represents an agent that can act within an environment.
#     Args:
#         cfg(AgentConfiguration) - Creates Agent with given
# :py:class:`AgentConfiguration`
#     """
#
#     def __init__(self, cfg):
#         self._body = None  # Initialize agent embodiment given
# AgentConfiguration
#         self._sensors = SensorSuite()  # Initialize sensors given
# cfg.sensor_specs
#         self._action_space = cfg.action_space
#         self._state = AgentState()  # Initialize state
#
#     @property
#     def state(self):
#         r"""The agent's current state.
#         Returns:
#             state(AgentState) - the current :py:class:`AgentState`
#         """
#         return self._state
#
#     def set_state(self, state):
#         r"""Set this Agent's state to be given state
#         """
#         self._state = state
#
#     def act(self, action_spec):
#         r"""Take an action specified by action_specification
#         """
#         action = self._action_space.get_action(action_spec)
#         # invoke action, changing internal agent state
#         return action
#
#     @property
#     def sensors(self):
#         r"""The agent's current :py:class:`SensorSuite`
#         Returns:
#             sensors(SensorSuite) - the current sensors
#         """
#         return self._sensors
#
#     def initialize_sensors(self):
#         r"""Initialize this Agent's SensorSuite
#         """
#         pass
