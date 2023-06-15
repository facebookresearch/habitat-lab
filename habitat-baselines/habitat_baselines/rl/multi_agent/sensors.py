import numpy as np
from gym import spaces

from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.rearrange.utils import UsesArticulatedAgentInterface


@registry.register_sensor
class OtherAgentGps(UsesArticulatedAgentInterface, Sensor):
    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return "other_agent_gps"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(shape=(2,), low=0, high=1, dtype=np.float32)

    def get_observation(self, observations, episode, *args, **kwargs):
        other_agent_id = (self.agent_id + 1) % 2
        my_pos = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.base_pos
        other_pos = self._sim.get_agent_data(
            other_agent_id
        ).articulated_agent.base_pos
        return np.array(my_pos - other_pos)[[0, 2]]


@registry.register_sensor
class MultiAgentGlobalPredicatesSensor(UsesArticulatedAgentInterface, Sensor):
    def __init__(self, sim, config, *args, task, **kwargs):
        self._task = task
        self._sim = sim
        self._predicates_list = None
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return "multi_agent_all_predicates"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    @property
    def predicates_list(self):
        if self._predicates_list is None:
            self._predicates_list = (
                self._task.pddl_problem.get_possible_predicates()
            )
        return self._predicates_list

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(len(self.predicates_list),), low=0, high=1, dtype=np.float32
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        sim_info = self._task.pddl_problem.sim_info
        truth_values = [p.is_true(sim_info) for p in self.predicates_list]
        return np.array(truth_values, dtype=np.float32)


@registry.register_sensor
class ShouldReplanSensor(Sensor):
    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._x_len = config.x_len
        self._y_len = config.y_len

        self._should_check = self._x_len > -1.0 and self._y_len > -1.0
        if not self._should_check:
            assert self._x_len <= -1.0 or self._y_len <= -1.0
        self._agent_idx = config.agent_idx

        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return f"agent_{self._agent_idx}_should_replan"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(shape=(1,), low=0, high=1, dtype=np.float32)

    def get_observation(self, observations, episode, *args, **kwargs):
        if not self._should_check:
            return np.array([0.0], dtype=np.float32)

        other_agent_id = (self._agent_idx + 1) % 2
        my_T = self._sim.get_agent_data(
            self._agent_idx
        ).articulated_agent.base_transformation

        other_pos = self._sim.get_agent_data(
            other_agent_id
        ).articulated_agent.base_pos
        rel_pos = my_T.inverted().transform_point(other_pos)

        # z coordinate is the height.
        dist = ((rel_pos[0] ** 2) / (self._x_len**2)) + (
            (rel_pos[1] ** 2) / (self._y_len**2)
        )
        return np.array([dist < 1], dtype=np.float32)
