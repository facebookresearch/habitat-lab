import time

import gym

from teas.core.simulator import Observation
from teas.simulators import make_simulator


class TeasEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, config):
        self._simulator = make_simulator(config.simulator, config=config)
        self.observation_space = \
            self._simulator.sensor_suite.observation_spaces
        self.action_space = self._simulator.action_space
        self._max_episode_seconds = None
        if hasattr(config, 'max_episode_seconds'):
            self._max_episode_seconds = config.max_episode_seconds
        self._max_episode_steps = None
        if hasattr(config, 'max_episode_steps'):
            self._max_episode_steps = config.max_episode_steps
        self._elapsed_steps = 0
        self._episode_started_at = None
        self._episode_over = False

    @property
    def _elapsed_seconds(self) -> float:
        return time.time() - self._episode_started_at

    def _past_limit(self) -> bool:
        if self._max_episode_steps is not None and self._max_episode_steps <= \
                self._elapsed_steps:
            return True
        elif self._max_episode_seconds is not None and \
                self._max_episode_seconds <= self._elapsed_seconds:
            return True
        return False

    def reset(self) -> Observation:
        self._episode_started_at = time.time()
        self._elapsed_steps = 0
        self._episode_over = False
        return self._simulator.reset()

    def step(self, action):
        assert self._episode_started_at is not None, "Cannot call step " \
                                                     "before calling reset"
        assert self._episode_over is False, "Episode done, call reset " \
                                            "before calling step"

        observation, reward, done, info = self._simulator.step(action)
        self._elapsed_steps += 1

        if self._past_limit():
            done = True
            self._episode_over = True

        return observation, reward, done, info

    def seed(self, seed=None) -> None:
        self._simulator.seed(seed)

    def reconfigure(self, *config) -> None:
        self._simulator.reconfigure(*config)

    def geodesic_distance(self, position_a, position_b) -> float:
        return self._simulator.geodesic_distance(position_a, position_b)

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
