import numpy as np
from gym import spaces

from habitat.core.embodied_task import SimulatorTaskAction
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.rearrange.grip_actions import RobotAction
from habitat.tasks.rearrange.multi_task.pddl_domain import PddlDomain
from habitat.tasks.rearrange.utils import rearrange_collision, rearrange_logger


@registry.register_task_action
class PddlApplyAction(RobotAction):
    def __init__(self, *args, task, **kwargs):
        super().__init__(*args, **kwargs)
        self._task = task

    @property
    def action_space(self):
        n_objects = len(self._task.task_def["objects"])

        action_n_args = sum(
            [
                len(action["parameters"])
                for action in self._task.domain.domain_def["actions"]
            ]
        )

        return spaces.Dict(
            {
                self._action_arg_prefix
                + "pddl_action": spaces.Box(
                    shape=(action_n_args,), low=-1, high=1, dtype=np.float32
                )
            }
        )

    def step(self, *args, is_last_action, **kwargs):
        apply_pddl_action = kwargs[self._action_arg_prefix + "pddl_action"]
        poss_objs = self._task.task_def["objects"]
        cur_i = 0
        for action in self._task.domain.domain_def["actions"]:

            n_params = len(action["parameters"])
            action_part = apply_pddl_action[cur_i : cur_i + n_params][:]
            if sum(action_part) > 0:
                action_name = action["name"]
                # Take action
                # Convert 1 indexed to 0 indexed.
                real_action_idxs = [a - 1 for a in action_part]
                for a in real_action_idxs:
                    if a < 0.0:
                        raise ValueError(
                            f"Got invalid action value < 0 in {action_part} with name {action_name} and arg length {n_params}"
                        )
                rearrange_logger.debug(f"Got action part {real_action_idxs}")
                obj_args = [poss_objs[int(i)] for i in real_action_idxs]
                rearrange_logger.debug(
                    f"Got action {action['name']} with obj args {obj_args}"
                )

                pddl_action = self._task.domain.actions[action_name].copy_new()
                pddl_action.bind(obj_args)
                rearrange_logger.debug(f"Applying action {pddl_action}")
                pddl_action.apply(self._task.domain._name_to_id, self._sim)
            cur_i += n_params
        if is_last_action:
            return self._sim.step(HabitatSimActions.ARM_ACTION)
        else:
            return {}
