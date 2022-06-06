import numpy as np
from gym import spaces

from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.rearrange.grip_actions import RobotAction
from habitat.tasks.rearrange.utils import rearrange_logger


@registry.register_task_action
class PddlApplyAction(RobotAction):
    def __init__(self, *args, task, **kwargs):
        super().__init__(*args, **kwargs)
        self._task = task
        self._entities_list = None
        self._action_ordering = None

    @property
    def action_space(self):
        if self._entities_list is None:
            self._entities_list = list(
                self._task.pddl_problem.all_entities.values()
            )
            self._action_ordering = list(
                self._task.pddl_problem.actions.values()
            )

        action_n_args = sum(
            [action.n_args for action in self._action_ordering]
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
        cur_i = 0
        for action in self._action_ordering:
            action_part = apply_pddl_action[cur_i : cur_i + action.n_args][:]
            if sum(action_part) > 0:
                # Take action
                # Convert 1 indexed to 0 indexed.
                real_action_idxs = [a - 1 for a in action_part]
                for a in real_action_idxs:
                    if a < 0.0:
                        raise ValueError(
                            f"Got invalid action value < 0 in {action_part} with action {action}"
                        )
                rearrange_logger.debug(f"Got action part {real_action_idxs}")

                param_values = [
                    self._entities_list[i] for i in real_action_idxs
                ]

                rearrange_logger.debug(
                    f"Got action {action} with obj args {args}"
                )

                apply_action = action.clone()
                apply_action.set_param_values(param_values)
                self._task.pddl_problem.apply_action(apply_action)
            cur_i += action.n_args
        if is_last_action:
            return self._sim.step(HabitatSimActions.ARM_ACTION)
        else:
            return {}
