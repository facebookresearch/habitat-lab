from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.tasks.rearrange.multi_task.composite_sensors import (
    CompositeSubgoalReward,
)
from habitat.tasks.rearrange.utils import coll_name_matches


@registry.register_measure
class DidAgentsCollide(Measure):
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
            raise ValueError("Sensor only supports 2 agents")

        for cp in contact_points:
            if coll_name_matches(cp, agent_ids[0]) and coll_name_matches(
                cp, agent_ids[1]
            ):
                found_contact = True

        self._metric = found_contact


@registry.register_measure
class CooperateSubgoalReward(CompositeSubgoalReward):
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "cooperate_subgoal_reward"

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
