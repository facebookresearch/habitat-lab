import numpy as np
from gym import spaces

from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.rearrange.multi_task.pddl_sensors import PddlSubgoalReward
from habitat.tasks.rearrange.utils import (
    UsesArticulatedAgentInterface,
    coll_name_matches,
)


@registry.register_measure
class DidAgentsCollide(Measure):
    """
    Detects if the 2 agents in the scene are colliding with each other at the
    current step. Only supports 2 agent setups.
    """

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "did_collide"

    def reset_metric(self, *args, **kwargs):
        self.update_metric(
            *args,
            **kwargs,
        )

    def update_metric(self, *args, task, **kwargs):
        sim = task._sim
        sim.perform_discrete_collision_detection()
        contact_points = sim.get_physics_contact_points()
        found_contact = False

        agent_ids = [
            articulated_agent.sim_obj.object_id
            for articulated_agent in sim.agents_mgr.articulated_agents_iter
        ]
        if len(agent_ids) != 2:
            raise ValueError(
                f"Sensor only supports 2 agents. Got {agent_ids=}"
            )

        for cp in contact_points:
            if coll_name_matches(cp, agent_ids[0]) and coll_name_matches(
                cp, agent_ids[1]
            ):
                found_contact = True

        self._metric = found_contact


@registry.register_measure
class NumAgentsCollide(Measure):
    """
    Cumulative number of steps in the episode the agents are in collision.
    """

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "num_agents_collide"

    def reset_metric(self, *args, task, **kwargs):
        task.measurements.check_measure_dependencies(
            self._get_uuid(), [DidAgentsCollide._get_uuid()]
        )
        self._metric = 0
        self.update_metric(
            *args,
            task=task,
            **kwargs,
        )

    def update_metric(self, *args, task, **kwargs):
        did_collide = task.measurements.measures[
            DidAgentsCollide._get_uuid()
        ].get_metric()
        self._metric += int(did_collide)


@registry.register_sensor
class OtherAgentGps(UsesArticulatedAgentInterface, Sensor):
    """
    Returns the GPS coordinates of the other agent.
    """

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
        assert (
            self.agent_id < 2
        ), f"OtherAgentGps only supports 2 agents, got {self.agent_id=}"
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
    """
    Returns the predicates ONLY for the agent this sensor is configured for.
    This is different from `GlobalPredicatesSensor` which returns the
    predicates for all agents.
    """

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
class AreAgentsWithinThreshold(Sensor):
    """
    Returns if the agents are close to each other and about to collide, thus
    the agents should replan.
    """

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
        return f"agent_{self._agent_idx}_agents_within_threshold"

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


@registry.register_measure
class RearrangeCooperateReward(PddlSubgoalReward):
    """
    `PddlSubgoalReward` adapted for 2 agent setups to penalize and
    potentially end the episode on agent collisions.
    """

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "rearrange_cooperate_reward"

    def __init__(self, *args, config, **kwargs):
        super().__init__(*args, config=config, **kwargs)
        self._end_on_collide = config.end_on_collide
        self._collide_penalty = config.collide_penalty

    def reset_metric(self, *args, task, **kwargs):
        task.measurements.check_measure_dependencies(
            self._get_uuid(), [DidAgentsCollide._get_uuid()]
        )
        super().reset_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        super().update_metric(*args, task=task, **kwargs)
        did_collide = task.measurements.measures[
            DidAgentsCollide._get_uuid()
        ].get_metric()

        if did_collide and self._end_on_collide:
            task.should_end = True
            self._metric -= self._collide_penalty
