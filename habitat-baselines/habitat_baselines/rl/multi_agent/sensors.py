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
