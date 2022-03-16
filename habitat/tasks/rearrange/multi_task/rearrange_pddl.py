#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import magnum as mn
import numpy as np

from habitat import Config
from habitat.core.dataset import Episode
from habitat.datasets.rearrange.rearrange_dataset import RearrangeDatasetV0
from habitat.tasks.rearrange.multi_task.dynamic_task_utils import (
    load_task_object,
)
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.rearrange_task import RearrangeTask


def parse_func(x: str) -> Tuple[str, List[str]]:
    """
    Parses out the components of a function string.
    """
    name = x.split("(")[0]
    args = x.split("(")[1].split(")")[0]
    return name, args


def search_for_id(k: str, name_to_id: Dict[str, int]) -> str:
    """
    Checks if an object exists in the name to ID conversion. This automatically
    checks for ART prefixes as well.
    """
    if isinstance(k, str):
        if k not in name_to_id and "ART_" + k in name_to_id:
            return name_to_id["ART_" + k]
        else:
            if k not in name_to_id:
                raise ValueError(f"Cannot find {k} in {name_to_id}")
            return name_to_id[k]
    return k


class Predicate:
    def __init__(self, load_config: Dict[str, Any]):
        self.name = load_config["name"]
        if "state" in load_config:
            self.set_state = SetState(load_config["state"])
        else:
            self.set_state = None
        self.args = load_config.get("args", [])
        self.set_args = None

    def get_n_args(self) -> int:
        return len(self.args)

    def is_equal(self, b) -> bool:
        """
        Is the same definition as another predicate.
        """
        return (
            (b.name == self.name)
            and (self.args == b.args)
            and (self.set_args == b.set_args)
        )

    def bind(self, set_args: List[str]) -> None:
        """
        Instantiate the predicate with particular entities.
        """
        if len(self.args) != len(set_args):
            raise ValueError()

        self.set_args = set_args

        if self.set_state is not None:
            self.set_state.bind(self.args, set_args)

    def __str__(self):
        return f"<Predicate: {self.name} [{self.args}] [{self.set_args}]>"

    def __repr__(self):
        return str(self)


class Action:
    def __init__(
        self,
        load_config: Dict[str, Any],
        config: Config,
        dataset: RearrangeDatasetV0,
        name_to_id: Dict[str, int],
        env: RearrangeTask,
        predicate_lookup_fn: Callable[[str], Optional[Predicate]],
    ):
        """
        :param predicate_lookup_fn: A function that takes as input a predicate
            identifier and returns a predicate if one was found.
        """
        self.name = load_config["name"]
        self.parameters = load_config["parameters"]
        self.name_to_id = name_to_id
        self.task = load_config["task"]
        self.task_def = load_config["task_def"]

        self.load_task_fn = partial(
            self._load_task, load_config, config, dataset, name_to_id
        )
        self.precond = []
        self.precond_strs = []

        for precond_str in load_config["precondition"]:
            precond = copy.deepcopy(predicate_lookup_fn(precond_str))
            self.precond.append(precond)
            self.precond_strs.append(parse_func(precond_str)[1])

        self.postcond = []
        self.postcond_args = []
        for effect_s in load_config["postcondition"]:
            _, effect_arg = parse_func(effect_s)
            self.postcond_args.append(effect_arg)
            postcond = predicate_lookup_fn(effect_s)
            if postcond is None:
                raise ValueError(f"Could not find postcond for {effect_s}")

            self.postcond.append(postcond)
        self.is_bound = False

    def __repr__(self):
        return f"<Action: {self.name}, paras: {self.parameters}, preconds: {self.precond}, effects: {self.postcond}>"

    def bind(
        self, args: List[str], add_args: Optional[Dict[str, str]] = None
    ) -> None:
        """
        :param args: Args passed to the sub-task linked to this action. Must
            match the ordering in `self.parameters`
        :param add_args: Additional optional kwargs passed to the task.
        """

        if add_args is None:
            add_args = {}
        assert not self.is_bound
        self.add_args = add_args
        self.orig_applied_func_args = args[:]
        self.applied_func_args = args[:]
        if self.applied_func_args[0] == "":
            self.applied_func_args = []
        for i, k in enumerate(self.applied_func_args):
            self.applied_func_args[i] = search_for_id(k, self.name_to_id)

        if len(args) != len(self.parameters):
            raise ValueError()

        def convert_arg_str(effect_arg):
            for place_arg, set_arg in zip(self.parameters, args):
                effect_arg = effect_arg.replace(place_arg, set_arg)
            effect_arg = effect_arg.split(",")
            if effect_arg[0] == "":
                effect_arg = []
            return effect_arg

        for i in range(len(self.postcond_args)):
            self.postcond[i].bind(convert_arg_str(self.postcond_args[i]))

        for i in range(len(self.precond_strs)):
            self.precond[i].bind(convert_arg_str(self.precond_strs[i]))

        self.is_bound = True

    def _load_task(
        self,
        load_config: Dict[str, Any],
        config: Config,
        dataset: RearrangeDatasetV0,
        name_to_id: Dict[str, int],
        env: RearrangeTask,
        episode: Episode,
        should_reset: bool = True,
    ) -> RearrangeTask:
        if "task" not in load_config:
            return None
        func_kwargs = {
            k: v for k, v in zip(self.parameters, self.applied_func_args)
        }
        task_kwargs = {
            **func_kwargs,
            **self.add_args,
            **{
                "orig_applied_args": {
                    k: v
                    for k, v in zip(
                        self.parameters, self.orig_applied_func_args
                    )
                }
            },
        }
        return load_task_object(
            load_config["task"],
            load_config["task_def"],
            config,
            env,
            dataset,
            should_reset,
            task_kwargs,
            episode,
        )

    def init_task(
        self, env: RearrangeTask, episode: Episode, should_reset=True
    ) -> RearrangeTask:
        return self.load_task_fn(env, episode, should_reset=should_reset)

    def are_preconditions_true(self, preds: List[Predicate]) -> bool:
        """
        :param preds: List of currently True predicates.
        """

        def has_match(pred):
            return any([other_pred.is_equal(pred) for other_pred in preds])

        return all(has_match(precond_pred) for precond_pred in self.precond)

    def apply_postconditions(self, preds: List[Predicate]) -> List[Predicate]:
        """
        :param preds: Set of all True predicates.
        :returns: Set of all true predicates after applying post conditions.
        """
        new_preds = copy.deepcopy(preds)
        for pred in self.postcond:
            if (
                pred.name.startswith("open")
                or pred.name.startswith("closed")
                or pred.name.startswith("not")
            ):
                base_name = "_".join(pred.name.split("_")[1:])
            else:
                base_name = None
            found = False
            for i, other_pred in enumerate(preds):
                other_base_name = "_".join(other_pred.name.split("_")[1:])
                if pred.name == other_pred.name:
                    # Override
                    new_preds[i] = pred
                    found = True
                    break
                if base_name is not None and other_base_name == base_name:
                    new_preds[i] = pred
                    found = True
                    break
                if (
                    base_name == other_pred.name
                    or other_base_name == pred.name
                ):
                    new_preds[i] = pred
                    found = True
                    break
            if not found:
                new_preds.append(pred)
        return new_preds

    def apply(self, name_to_id: Dict[str, int], sim: RearrangeSim) -> None:
        """
        Applies the effects of all the post conditions.
        """
        for postcond in self.postcond:
            self._apply_effect(postcond, name_to_id, sim)

    def _apply_effect(self, postcond, name_to_id, sim):
        set_state = postcond.set_state
        set_state.set_state(name_to_id, sim)


class RoboState:
    """
    Specifies the configuration of the robot. Only used as a data structure. Not used to set the simulator state.
    """

    def __init__(self, load_config):
        self.holding = load_config.get("holding", None)
        self.pos = load_config.get("pos", None)

    def bind(self, arg_k, arg_v):
        for k, v in zip(arg_k, arg_v):
            if self.holding is not None:
                self.holding = self.holding.replace(k, v)
            if self.pos is not None:
                self.pos = self.pos.replace(k, v)


class SetState:
    """
    A partially specified state of the simulator. First this object needs to be
    bound to a specific set of arguments specifying scene entities
    (`self.bind`). After, you can query this object to get if the specified
    scene state is satifisfied and set everything specified.
    """

    def __init__(self, load_config: Dict[str, Any]):
        self.art_states = load_config.get("art_states", {})
        self.obj_states = load_config.get("obj_states", {})
        self.robo_state = RoboState(load_config.get("robo", {}))
        self.load_config = load_config

    def bind(self, arg_k: List[str], arg_v: List[str]) -> None:
        """
        Defines a state in the environment grounded in scene entities.
        :param arg_k: The names of the environment parameters to set.
        :param arg_v: The values of the environment parameters to set.
        """

        def list_replace(l, k, v):
            new_l = {}
            for l_k, l_v in l.items():
                if isinstance(l_k, str):
                    l_k = l_k.replace(k, v)
                if isinstance(l_v, str):
                    l_v = l_v.replace(k, v)
                new_l[l_k] = l_v
            return new_l

        for k, v in zip(arg_k, arg_v):
            self.art_states = list_replace(self.art_states, k, v)
            self.obj_states = list_replace(self.obj_states, k, v)
            if "catch_ids" in self.load_config:
                self.load_config["catch_ids"] = self.load_config[
                    "catch_ids"
                ].replace(k, v)

        self.robo_state.bind(arg_k, arg_v)

    def _is_id_rigid_object(self, id_str: str) -> bool:
        """
        Used to check if an identifier can be used to look up the object ID in the scene_ojbs_id list of the simulator.
        """
        return not id_str.startswith("ART_")

    def is_satisfied(
        self,
        name_to_id: Dict[str, int],
        sim: RearrangeSim,
        obj_thresh: float,
        art_thresh: float,
    ) -> bool:
        """
        Returns True if the grounded state is present in the current simulator state.
        """

        rom = sim.get_rigid_object_manager()
        for obj_name, target in self.obj_states.items():
            if not self._is_id_rigid_object(obj_name):
                # Invalid predicate
                return False

            obj_idx = name_to_id[obj_name]
            abs_obj_id = sim.scene_obj_ids[obj_idx]
            cur_pos = rom.get_object_by_id(
                abs_obj_id
            ).transformation.translation

            if not self._is_id_rigid_object(target):
                # Invalid predicate
                return False

            targ_idx = name_to_id[target]
            idxs, pos_targs = sim.get_targets()
            targ_pos = pos_targs[list(idxs).index(targ_idx)]

            dist = np.linalg.norm(cur_pos - targ_pos)
            if dist >= obj_thresh:
                return False

        for art_obj_id, set_art in self.art_states.items():
            abs_id = sim.art_obj_ids[art_obj_id]
            prev_art_pos = sim.get_articulated_object_positions(abs_id)
            if isinstance(set_art, str):
                art_sampler = eval(set_art)
                did_sat = art_sampler.is_satisfied(prev_art_pos)
            else:
                art_dist = np.linalg.norm(
                    np.array(prev_art_pos) - np.array(set_art)
                )
                did_sat = art_dist < art_thresh

            if not did_sat:
                return False
        if (
            self.robo_state.holding != "NONE"
            and self.robo_state.holding is not None
            and self._is_id_rigid_object(self.robo_state.holding)
        ):
            # Robot must be holding right object.
            if self.robo_state.holding not in name_to_id:
                raise ValueError(
                    f"Cannot find {self.robo_state.holding} in {name_to_id}"
                )
            obj_idx = name_to_id[self.robo_state.holding]
            abs_obj_id = sim.scene_obj_ids[obj_idx]
            if sim.grasp_mgr.snap_idx != abs_obj_id:
                return False
        elif (
            self.robo_state.holding == "NONE"
            and sim.grasp_mgr.snap_idx != None
        ):
            # Robot must be holding nothing
            return False

        return True

    def set_state(self, name_to_id: Dict[str, int], sim: RearrangeSim) -> None:
        """
        Set this state in the simulator. Warning, this steps the simulator.
        """
        for obj_name, target in self.obj_states.items():
            obj_idx = name_to_id[obj_name]
            abs_obj_id = sim.scene_obj_ids[obj_idx]

            if target in name_to_id:
                targ_idx = name_to_id[target]
                _, pos_targs = sim.get_targets()
                targ_pos = pos_targs[targ_idx]
                set_T = mn.Matrix4.translation(targ_pos)
            else:
                raise ValueError("Not supported")

            # Get the object id corresponding to this name
            sim.reset_obj_T(abs_obj_id, set_T)
            obj_name = sim.ep_info["static_objs"][obj_idx][0]

        for art_obj_id, set_art in self.art_states.items():
            art_obj_id = search_for_id(art_obj_id, name_to_id)
            art_obj = sim.art_objs[art_obj_id]

            art_obj.clear_joint_states()
            art_obj.joint_positions = set_art

            sim.internal_step(-1)

        # Set the snapped object information
        if (
            self.robo_state.holding == "NONE"
            and sim.snapped_obj_id is not None
        ):
            sim.desnap_object(force=True)
        elif self.robo_state.holding is not None:
            rel_obj_idx = name_to_id[self.robo_state.holding]
            sim.internal_step(-1)
            sim.full_snap(rel_obj_idx)
            sim.internal_step(-1)

        # Set the robot starting position
        if self.robo_state.pos == "rnd":
            sim.set_robot_base_to_random_point()
