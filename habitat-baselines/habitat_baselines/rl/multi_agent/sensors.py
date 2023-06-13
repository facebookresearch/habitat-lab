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

        self._should_check = (
            self._x_len is not None and self._y_len is not None
        )
        if not self._should_check:
            assert self._x_len is None and self._y_len is None
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


@registry.register_sensor
class ActionHistorySensor(UsesArticulatedAgentInterface, Sensor):
    cls_uuid: str = "action_history"

    def __init__(self, sim, config, *args, task, **kwargs):
        self._task = task
        self._sim = sim
        self._pddl_action = None
        self._cur_write_idx = 0
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return ActionHistorySensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    @property
    def pddl_action(self):
        if self._pddl_action is None:
            self._pddl_action = self._task.actions[
                f"agent_{self.agent_id}_pddl_apply_action"
            ]
        return self._pddl_action

    def _get_observation_space(self, *args, config, **kwargs):
        self._action_ordering = self._task.pddl_problem.get_ordered_actions()
        entities_list = self._task.pddl_problem.get_ordered_entities_list()

        self._action_map = {}
        self._action_offsets = {}
        self._n_actions = 0
        for action in self._action_ordering:
            param = action._params[0]
            self._action_map[action.name] = [
                entity
                for entity in entities_list
                if entity.expr_type.is_subtype_of(param.expr_type)
            ]
            self._action_offsets[action.name] = self._n_actions
            self._n_actions += len(self._action_map[action.name])
        self._dat = np.zeros(
            (self.config.window_size, self._n_actions), dtype=np.float32
        )

        return spaces.Box(
            shape=(self.config.window_size * self._n_actions,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, *args, **kwargs):
        self._cur_write_idx = self._cur_write_idx % self.config.window_size
        if self._task._cur_episode_step == 0 or self._cur_write_idx == 0:
            self._cur_write_idx = 0
            self._dat *= 0.0

        ac = self.pddl_action._prev_action
        if ac is not None:
            if not self.pddl_action.was_prev_action_invalid:
                use_name = ac.name
                set_idx = self._action_offsets[use_name]
                param_value = ac.param_values[0]
                entities = self._action_map[use_name]
                set_idx += entities.index(param_value)
                self._dat[self._cur_write_idx, set_idx] = 1.0
            self._cur_write_idx += 1

        return self._dat.reshape(-1)
