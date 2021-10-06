import copy
import os.path as osp
from collections import defaultdict

import numpy as np
import yacs.config
import yaml

import habitat
from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.rearrange.multi_task.rearrange_pddl import (
    Action,
    Predicate,
    SetState,
)
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.tasks.rearrange.utils import CacheHelper, rearrange_collision


@registry.register_task(name="RearrangeCompositeTask-v0")
class RearrangeCompositeTaskV0(RearrangeTask):
    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(config=config, *args, dataset=dataset, **kwargs)
        full_task_path = osp.join(
            self._config.TASK_SPEC_BASE_PATH, self._config.TASK_SPEC + ".yaml"
        )
        with open(full_task_path, "r") as f:
            task_def = yaml.safe_load(f)
        self.task_def = task_def

        start_d = task_def["start"]
        self.start_state = SetState(start_d["state"])

        with open(self.task_def["domain"], "r") as f:
            domain_def = yaml.safe_load(f)

        self.name_to_id = self.domain.get_name_to_id_mapping()

        self.load_solution(task_def["solution"])

        self.cur_node = -1
        self.inf_cur_node = 0
        self.cur_task = None
        self.call_count = defaultdict(lambda: 0)
        self.cached_tasks = {}
        if self._config.SINGLE_NODE_EVAL >= 0:
            self.cur_node = self._config.SINGLE_NODE_EVAL

        use_goal_precond = self.tcfg["GOAL_PRECOND"]
        if use_goal_precond == "":
            use_goal_precond = task_def["goal"]
        else:
            if use_goal_precond.startswith("'") and use_goal_precond.endswith(
                "'"
            ):
                use_goal_precond = use_goal_precond[1:-1]

            use_goal_precond = use_goal_precond.replace("/", ",")
            use_goal_precond = {"precondition": use_goal_precond.split(".")}

        self._load_goal_preconds(use_goal_precond)
        self._load_start_preconds(start_d)
        self._load_stage_preconds(task_def.get("stage_goals", {}))

    def _load_stage_preconds(self, stage_goals):
        self.stage_goals = {}
        for k, preconds in stage_goals.items():
            self.stage_goals[k] = self._parse_precond_list(preconds)

    def _parse_precond_list(self, d):
        preds = []
        for pred_s in d:
            pred = copy.deepcopy(self.predicate_lookup(pred_s))
            _, effect_arg = parse_func(pred_s)
            effect_arg = effect_arg.split(",")
            if effect_arg[0] == "":
                effect_arg = []
            pred.bind(effect_arg)
            preds.append(pred)
        return preds

    def _load_goal_preconds(self, goal_d):
        self.goal_state = self._parse_precond_list(goal_d["precondition"])

    def _load_start_preconds(self, start_d):
        self.cur_state = self._parse_precond_list(start_d["precondition"])

    def query(self, pred_s):
        pred = self.predicate_lookup(pred_s)
        _, search_args = parse_func(pred_s)
        search_args = search_args.split(",")
        for pred in self.cur_state:
            if pred.name != pred.name:
                continue
            if pred.set_args is None:
                raise ValueError("unbound predicate in the current state")
            if len(pred.set_args) != len(search_args):
                raise ValueError("Predicate has wrong # of args")
            all_match = True
            for k1, k2 in zip(pred.set_args, search_args):
                if k2 == "*":
                    continue
                if k1 != k2:
                    all_match = False
                    break
            if all_match:
                return pred
        return None

    def load_solution(self, solution_d):
        self.solution = []
        for i, action in enumerate(solution_d):
            if (
                self.tcfg.LIMIT_TASK_NODE != -1
                and i > self.tcfg.LIMIT_TASK_NODE
            ):
                break
            name, args = parse_func(action)
            args = args.split(",")
            ac_instance = copy.deepcopy(self.actions[name])

            ac_instance.bind(
                args, self.task_def.get("add_args", {}).get(i, {})
            )
            self.solution.append(ac_instance)

    def _jump_to_node(self, node_idx, is_full_task=False):
        # We don't want to reset to this node if we are in full task mode.
        if not is_full_task:
            self.cur_node = node_idx
        sim = self._env._sim
        node_name = self.solution[node_idx].name

        # print('JUMP %i: %s' % (node_idx, str(node_name)))
        for i in range(node_idx):
            self.solution[i].apply(self.name_to_id, sim)

        self.call_count[node_name] += 1
        if node_idx in self.cached_tasks:
            self.cur_task = self.cached_tasks[node_idx]
            self.cur_task.reset()
        else:
            task = self.solution[node_idx].init_task(self)
            self.cached_tasks[node_idx] = task
            self.cur_task = task
        self._set_force_limit()

    def _set_force_limit(self):
        if self.cur_task is not None:
            self.use_max_accum_force = self.cur_task.tcfg.MAX_ACCUM_FORCE
        is_subtask = self.tcfg.EVAL_NODE >= 0
        if not is_subtask:
            if self.tcfg.MAX_ACCUM_FORCE != -1.0:
                self.use_max_accum_force = (
                    len(self.solution) * self.tcfg.MAX_ACCUM_FORCE
                )
            else:
                self.use_max_accum_force = -1.0
            if self.tcfg.LIMIT_TASK_NODE != -1:
                self._env._max_episode_steps = (
                    self.tcfg.LIMIT_TASK_LEN_SCALING
                    * (self.tcfg.LIMIT_TASK_NODE + 1)
                )

            self.cur_task = None
        else:
            self._env._max_episode_steps = 400
            # TODO: This is a bit hacky, I trained with no max force for fridge /
            # tasks, but I am evaluating with a limit.
            if self.use_max_accum_force == -1:
                self.use_max_accum_force = (
                    self.tcfg.SUBTASK_NO_SPEC_MAX_ACCUM_FORCE
                )

    def increment_node(self):
        if self.cur_node + 1 >= self.get_num_nodes():
            self.cur_node = 0
        else:
            self.cur_node = self.cur_node + 1

        self._jump_to_node(self.cur_node)

    def decrement_node(self):
        if self.cur_node - 1 <= 0:
            self.cur_node = self.get_num_nodes() - 1
        else:
            self.cur_node = self.cur_node - 1

        self._jump_to_node(self.cur_node)

    def get_num_nodes(self):
        return len(self.solution)

    def _get_next_inf_sol(self):
        # Never give reward from these nodes, skip to the next node instead.
        REWARD_SKIP_NODES = ["move_obj"]
        # Returns False if there is no next subtask in the solution
        if self.inf_cur_node >= len(self.solution):
            return False
        while self.solution[self.inf_cur_node].name in REWARD_SKIP_NODES:
            self.inf_cur_node += 1
            if self.inf_cur_node >= len(self.solution):
                return False

        if self.inf_cur_node in self.cached_tasks:
            self.inf_cur_task = self.cached_tasks[self.inf_cur_node]
            self.inf_cur_task.reset(super_reset=False)
        else:
            task = self.solution[self.inf_cur_node].init_task(
                self, should_reset=False
            )
            self.cached_tasks[self.inf_cur_node] = task
            self.inf_cur_task = task

        self.use_ignore_hold_violate = (
            self.inf_cur_task.use_ignore_hold_violate
        )

        return True

    def reset(self, episode: Episode):
        obs = super().reset(episode)

        sim = self._env._sim
        self.stage_succ = []
        self.start_state.set_state(self.name_to_id, sim)

        if self.tcfg.DEBUG_SKIP_TO_NODE != -1:
            self._jump_to_node(self.tcfg.DEBUG_SKIP_TO_NODE, is_full_task=True)

        if self.cur_node >= 0:
            self._jump_to_node(self.cur_node)

        self._set_force_limit()

        self.inf_cur_node = 0
        self._get_next_inf_sol()

        return self.get_task_obs()

    def step(self, action_name, action_args):
        sim = self._env._sim
        obs, reward, done, info = super().step(action_name, action_args)

        if self.cur_task is not None:
            if isinstance(self.cur_task, BaseHabEnv):
                self.cur_task.prev_obs = obs
                self.cur_task.last_action = action_args
                is_succ = self.cur_task._my_episode_success()
            else:
                is_succ = False
            if is_succ:
                done = True
            info["ep_success"] = is_succ
            info["node_idx"] = self.cur_node
        else:
            # Use data from which subtask we think we are at.
            self.inf_cur_task.add_force = self.add_force
            self.inf_cur_task.prev_obs = obs
            is_succ = self.inf_cur_task._my_episode_success()
            reward = self.inf_cur_task._my_get_reward(obs)
            if is_succ:
                prev_inf_cur_node = self.inf_cur_node
                self.inf_cur_node += 1
                if not self._get_next_inf_sol():
                    self.inf_cur_node = prev_inf_cur_node
            if self._my_episode_success():
                reward += self.rlcfg.SUCCESS_REWARD
            info["node_idx"] = self.inf_cur_node
            for i in range(len(self.solution)):
                info[f"reached_{i}"] = self.inf_cur_node >= i
            self._update_info_stage_succ(info)

        return obs, reward, done, info

    def _my_get_reward(self, obs):
        # reward is defined in the step function if we are training on the full
        # task.
        return 0.0

    def _is_pred_list_sat(self, preds):
        sim = self._env._sim
        for pred in reversed(preds):
            if not pred.set_state.is_satisfied(
                self.name_to_id,
                sim,
                self.rlcfg.OBJ_SUCC_THRESH,
                self.rlcfg.ART_SUCC_THRESH,
            ):
                return False
        return True

    def _update_info_stage_succ(self, info):
        for k, preds in self.stage_goals.items():
            succ_k = f"ep_{k}_success"
            if k in self.stage_succ:
                info[succ_k] = 1.0
            else:
                if self._is_pred_list_sat(preds):
                    info[succ_k] = 1.0
                    self.stage_succ.append(k)
                else:
                    info[succ_k] = 0.0

    def _my_episode_success(self):
        if self.cur_task is not None:
            # Don't check success when we are evaluating a subtask.
            return False
        return self._is_pred_list_sat(self.goal_state)
