import gym

from teas.simulators import make_simulator


class TeasEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, config):
        self._simulator = make_simulator(config.simulator, config=config)
        self.observation_space = self._simulator.sensor_suite.observation_spaces
        self.action_space = self._simulator.action_space
    
    def step(self, action):
        # TODO(akadian): Ensure that reset is called before step
        return self._simulator.step(action)
    
    def reset(self):
        return self._simulator.reset()
    
    def render(self, mode='human', close=False):
        self._simulator.render(mode, close)
    
    def close(self):
        self._simulator.close()
    
    def seed(self, seed=None):
        self._simulator.seed(seed)
    
    def reconfigure(self, *config):
        self._simulator.reconfigure(*config)


# TODO(akadian): Consider subclass TeasEnv instead of gym.Env for EnvWrapper
# TODO(akadian): Refactor the below code in case we get rid of TeasEnv
# TODO(akadian): Reduce redundancy (possibly using getattribute)

class EnvWrapper(gym.Env):
    def __init__(self, env):
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.metadata = self.env.metadata
    
    def step(self, action):
        return self.env.step(action)
    
    def reset(self):
        return self.env.reset()
    
    def render(self, mode='human', **kwargs):
        return self.env.render(mode, **kwargs)
    
    def close(self):
        return self.env.close()
    
    def seed(self, seed=None):
        return self.env.seed(seed)
    
    def reconfigure(self, *config):
        return self.env.reconfigure(*config)
    
    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.env)
    
    def __repr__(self):
        return str(self)
    
    @property
    def unwrapped(self):
        return self.env.unwrapped

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
#         self.body_type = 'cylinder(0.1,1.0)|obj(./file.obj)|urdf(./file.urdf)'


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
