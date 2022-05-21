#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy
from collections import defaultdict
from enum import Enum
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    DefaultDict,
    Dict,
    List,
    Optional,
    Tuple,
)

import magnum as mn
import numpy as np

from habitat import Config
from habitat.core.dataset import Episode
from habitat.tasks.rearrange.multi_task.task_creator_utils import (
    create_task_object,
)
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.tasks.rearrange.utils import rearrange_logger

if TYPE_CHECKING:
    from habitat.datasets.rearrange.rearrange_dataset import RearrangeDatasetV0


def parse_func(x: str) -> Tuple[str, List[str]]:
    """
    Parses out the components of a function string.
    :returns: First element is the name of the function, second argument are the function arguments.
    """
    try:
        name = x.split("(")[0]
        args = x.split("(")[1].split(")")[0]
    except IndexError as e:
        raise ValueError(f"Cannot parse '{x}'") from e

    return name, args


class RearrangeObjectTypes(Enum):
    ARTICULATED_OBJECT = "articulated"
    ARTICULATED_LINK = "articulated_link"
    RIGID_OBJECT = "rigid"
    GOAL_POSITION = "goal"


def search_for_id(
    k: str, name_to_id: Dict[str, Any]
) -> Tuple[Any, RearrangeObjectTypes]:
    """
    Checks if an object exists in the name to ID conversion. This automatically
    checks for ART prefixes as well.
    """
    ret_id = k
    if isinstance(k, str):
        if k not in name_to_id and "ART_" + k in name_to_id:
            k = "ART_" + k
        elif k not in name_to_id and "MARKER_" + k in name_to_id:
            k = "MARKER_" + k

        if k not in name_to_id:
            raise ValueError(f"Cannot find {k} in {name_to_id}")
        ret_id = name_to_id[k]

    if k.startswith("TARGET_"):
        ret_type = RearrangeObjectTypes.GOAL_POSITION
    elif k.startswith("MARKER_"):
        ret_type = RearrangeObjectTypes.ARTICULATED_LINK
    elif k.startswith("ART_"):
        ret_type = RearrangeObjectTypes.ARTICULATED_OBJECT
    else:
        ret_type = RearrangeObjectTypes.RIGID_OBJECT
    return ret_id, ret_type


class Predicate:
    def __init__(self, load_config: Dict[str, Any]):
        self.name = load_config["name"]
        if "state" in load_config:
            self.set_state = PddlSetState(load_config["state"])
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


class PddlAction:
    """
    :property task: The name of the associated task class.
    :property name: The unique identifier for this action.
    """

    def __init__(
        self,
        load_config: Dict[str, Any],
        config: Config,
        dataset: "RearrangeDatasetV0",
        name_to_id: Dict[str, Any],
        predicate_lookup_fn: Callable[[str], Optional[Predicate]],
    ):
        """
        :param predicate_lookup_fn: A function that takes as input a predicate
            identifier and returns a predicate if one was found.
        """
        self._orig_load_config = load_config
        self._orig_config = config
        self._orig_dataset = dataset
        self._orig_pred_lookup_fn = predicate_lookup_fn

        self.name = load_config["name"]
        self.parameters = load_config["parameters"]
        self.name_to_id = name_to_id
        self.task = load_config["task"]
        self.task_def = load_config["task_def"]
        self._config_task_args = load_config.get("config_args", {})
        self._add_task_args = load_config.get("add_task_args", {})
        self._arg_specs: Dict[str, SetStateArgSpec] = {}
        for param_name, arg_spec in load_config.get("arg_specs", {}).items():
            if param_name not in self.parameters:
                raise ValueError(
                    f"Could not find {param_name} in {self.parameters}"
                )
            self._arg_specs[param_name] = SetStateArgSpec(**arg_spec)

        self.load_task_fn = partial(
            self._load_task, load_config, config, dataset, name_to_id
        )

        self.precond = []
        self.precond_strs = []
        self._precond_arg_to_action_arg: Dict[str, str] = {}
        self._action_arg_to_precond_arg: Dict[str, str] = {}

        for precond_str in load_config["precondition"]:
            precond = copy.deepcopy(predicate_lookup_fn(precond_str))
            self.precond.append(precond)
            parsed_precond_args = parse_func(precond_str)[1]
            self.precond_strs.append(parsed_precond_args)
            sep_precond_args = self._convert_arg_str(parsed_precond_args, [])
            assert len(precond.args) == len(sep_precond_args)
            for precond_arg_param_name, precond_action_arg_name in zip(
                precond.args, sep_precond_args
            ):
                self._precond_arg_to_action_arg[
                    precond_arg_param_name
                ] = precond_action_arg_name
                self._action_arg_to_precond_arg[
                    precond_action_arg_name
                ] = precond_arg_param_name

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

    def copy_new(self) -> PddlAction:
        return PddlAction(
            self._orig_load_config,
            self._orig_config,
            self._orig_dataset,
            self.name_to_id,
            self._orig_pred_lookup_fn,
        )

    def _convert_arg_str(self, effect_arg: str, args: List[str]) -> List[str]:
        """
        Substitutes the `args` into the `effect_arg` string. Works when `args`
        is not the same size as `self.parameters` (meaning there are unbound
        arguments).
        """
        # Substitute in format strings for the key words we want to replace
        for i, param_name in enumerate(self.parameters[: len(args)]):
            effect_arg = effect_arg.replace(param_name, "{" + str(i) + "}")

        effect_arg = effect_arg.format(*args)
        effect_arg = effect_arg.split(",")
        if effect_arg[0] == "":
            effect_arg = []
        return effect_arg

    @property
    def task_args(self):
        if not self.is_bound:
            raise ValueError(
                "Trying to fetch task arguments when task is not yet bound"
            )
        return {
            **{k: v for k, v in zip(self.parameters, self.applied_func_args)},
            **self._add_task_args,
            **{
                f"orig_{k}": v
                for k, v in zip(self.parameters, self.orig_applied_func_args)
            },
        }

    @property
    def config_task_args(self):
        return self._config_task_args

    def __repr__(self):
        if self.is_bound:
            return f"<PddlAction: {self.name}, paras: {self.parameters} -> {self.applied_func_args}, preconds: {self.precond}, effects: {self.postcond}>"
        else:
            return f"<PddlAction: {self.name}, paras: {self.parameters}, preconds: {self.precond}, effects: {self.postcond}>"

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
            self.applied_func_args[i], _ = search_for_id(k, self.name_to_id)

        if len(args) != len(self.parameters):
            raise ValueError(
                f"The number of arguments {args} does not match the parameters {self.parameters}"
            )

        for i in range(len(self.postcond_args)):
            self.postcond[i].bind(
                self._convert_arg_str(self.postcond_args[i], args)
            )

        for i in range(len(self.precond_strs)):
            self.precond[i].bind(
                self._convert_arg_str(self.precond_strs[i], args)
            )

        self.is_bound = True

    def _load_task(
        self,
        load_config: Dict[str, Any],
        config: Config,
        dataset: "RearrangeDatasetV0",
        name_to_id: Dict[str, Any],
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
            "task_name": load_config["task"],
            **func_kwargs,
            **self.add_args,
            **self._add_task_args,
            **{
                "orig_applied_args": {
                    k: v
                    for k, v in zip(
                        self.parameters, self.orig_applied_func_args
                    )
                }
            },
        }
        rearrange_logger.debug(
            f"Loading task {load_config['task']} with definition {load_config['task_def']}"
        )
        return create_task_object(
            load_config["task"],
            load_config["task_def"],
            config,
            env,
            dataset,
            should_reset,
            task_kwargs,
            episode,
            self.config_task_args,
        )

    def init_task(
        self, env: RearrangeTask, episode: Episode, should_reset=True
    ) -> RearrangeTask:
        return self.load_task_fn(env, episode, should_reset=should_reset)

    def _get_consistent_preds(
        self,
        all_matches,
        name_to_id,
        cur_pred_list=None,
        already_bound=None,
        cur_i=0,
    ):
        if cur_pred_list is None:
            cur_pred_list = []
        if already_bound is None:
            already_bound = {}

        if len(cur_pred_list) == len(self.precond):
            return [cur_pred_list], [already_bound]

        cur_pred_cp: Predicate = copy.deepcopy(self.precond[cur_i])
        all_consistent_preds = []
        all_bound_args = []
        for match in all_matches[cur_i]:
            args_match = True
            # Does this predicate conflict with any already set arguments?
            for i, arg in enumerate(cur_pred_cp.args):
                if (
                    arg in already_bound
                    and already_bound[arg] != match.set_args[i]
                ):
                    args_match = False
                    break
            if not args_match:
                continue
            new_set_args = {
                k: v for k, v in zip(cur_pred_cp.args, match.set_args)
            }

            # Do these predicates work with this action?
            all_arg_spec_match = True
            for param_name, assign_name in new_set_args.items():
                action_param_name = self._precond_arg_to_action_arg[param_name]
                assign_name, assign_type = search_for_id(
                    assign_name, name_to_id
                )
                if (
                    action_param_name in self._arg_specs
                    and not self._arg_specs[
                        action_param_name
                    ].argument_matches(assign_name, assign_type)
                ):
                    all_arg_spec_match = False
                    break
            if not all_arg_spec_match:
                continue

            pred_result, bound_args = self._get_consistent_preds(
                all_matches,
                name_to_id,
                [*cur_pred_list, match],
                {**already_bound, **new_set_args},
                cur_i + 1,
            )
            all_consistent_preds.extend(pred_result)
            all_bound_args.extend(bound_args)
        return all_consistent_preds, all_bound_args

    def get_possible_actions(
        self, preds: List[Predicate], name_to_id: Dict[str, Any]
    ) -> List[PddlAction]:
        """
        Returns grounded actions that are possible in the current predicate state.
        :param preds: List of currently True predicates.
        :returns: List of bound actions that can currently be applied.
        """
        all_matches = []
        for precond in self.precond:
            all_matches.append(
                [
                    other_pred
                    for other_pred in preds
                    if other_pred.name == precond.name
                ]
            )

        consistent_preds, all_bound_args = self._get_consistent_preds(
            all_matches,
            name_to_id,
        )
        rearrange_logger.debug(f"Got consistent preds {consistent_preds}")
        consistent_actions = []

        for bound_args in all_bound_args:
            # Extract out the set arguments from consistent_preds
            all_set_args = [[]]

            for action_param_name in self.parameters:
                if action_param_name in self._action_arg_to_precond_arg:
                    # Assign the predicate
                    precond_arg = self._action_arg_to_precond_arg[
                        action_param_name
                    ]
                    for i in range(len(all_set_args)):
                        all_set_args[i].append(bound_args[precond_arg])
                else:
                    # Assign all possible values to to the empty action
                    # parameter.
                    ok_entities = []
                    for entity_name in name_to_id.keys():
                        entity_type = search_for_id(entity_name, name_to_id)[1]
                        if (
                            action_param_name in self._arg_specs
                            and self._arg_specs[
                                action_param_name
                            ].argument_matches(entity_name, entity_type)
                        ):
                            ok_entities.append(entity_name)
                    for entity in ok_entities:
                        for i in range(len(all_set_args)):
                            all_set_args[i] = [*all_set_args[i], entity]
            if len(all_set_args[0]) != len(self.parameters):
                continue
            for set_args in all_set_args:
                action = self.copy_new()
                action.bind(set_args)
                consistent_actions.append(action)
        return consistent_actions

    def calculate_postconditions(
        self, preds: List[Predicate]
    ) -> List[Predicate]:
        """
        Applies the post-conditons of the action to the list of currently true predicates.

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

    def apply(self, name_to_id: Dict[str, Any], sim: RearrangeSim) -> None:
        """
        Applies the effects of all the post conditions.
        """
        for postcond in self.postcond:
            self._apply_effect(postcond, name_to_id, sim)

    def _apply_effect(self, postcond, name_to_id, sim):
        set_state = postcond.set_state
        set_state.set_state(name_to_id, sim)


class PddlRobotState:
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

    def is_satisfied(self, name_to_id: Dict[str, Any], sim) -> bool:
        """
        Returns if the desired robot state is currently true in the simulator state.
        """
        if self.holding != "NONE" and self.holding is not None:
            # Robot must be holding desired object.
            match_name, match_type = search_for_id(self.holding, name_to_id)

            if match_type != RearrangeObjectTypes.RIGID_OBJECT:
                # We can only hold rigid objects.
                return False

            if self.holding not in name_to_id:
                raise ValueError(f"Cannot find {self.holding} in {name_to_id}")
            obj_idx = name_to_id[self.holding]
            if isinstance(obj_idx, str):
                raise ValueError(
                    f"Current holding object {obj_idx} is not a scene object index"
                )

            abs_obj_id = sim.scene_obj_ids[obj_idx]
            if sim.grasp_mgr.snap_idx != abs_obj_id:
                return False
        elif self.holding == "NONE" and sim.grasp_mgr.snap_idx != None:
            # For predicate to be true, robot must be holding nothing
            return False
        return True


class SetStateArgSpec:
    """
    The input types for a PDDL state. Used to limit what entities predicates can take as input.
    """

    def __init__(
        self, name_match: str = "", type_match: Optional[List[str]] = None
    ):
        self.name_match: str = name_match
        self.type_match: Optional[List[RearrangeObjectTypes]] = None
        if type_match is not None:
            self.type_match = [RearrangeObjectTypes(x) for x in type_match]

    def __repr__(self):
        return f"SetStateArgSpec {id(self)}: name_match={self.name_match}, type_match={self.type_match}"

    def argument_matches(
        self, arg_name: Any, arg_type: RearrangeObjectTypes
    ) -> bool:
        if self.name_match != "":
            if not isinstance(arg_name, str):
                return False
            if not arg_name.startswith(self.name_match):
                return False
        if self.type_match is not None and arg_type not in self.type_match:
            return False
        return True


class PddlSetState:
    """
    A partially specified state of the simulator. First this object needs to be
    bound to a specific set of arguments specifying scene entities
    (`self.bind`). After, you can query this object to get if the specified
    scene state is satifisfied and set everything specified.
    """

    def __init__(self, load_config: Dict[str, Any]):
        self.art_states = load_config.get("art_states", {})
        self.obj_states = load_config.get("obj_states", {})

        self.check_for_art_link_match: DefaultDict[str, bool] = defaultdict(
            lambda: False
        )
        self.check_for_art_link_match.update(
            load_config.get("check_for_art_link_match", {})
        )

        self.robo_state = PddlRobotState(load_config.get("robo", {}))
        self.load_config = load_config

        self.arg_spec: Optional[SetStateArgSpec] = None
        if "arg_spec" in load_config:
            self.arg_spec = SetStateArgSpec(**load_config["arg_spec"])

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
        self._set_args = arg_v

    def _is_id_rigid_object(self, id_str: str) -> bool:
        """
        Used to check if an identifier can be used to look up the object ID in the scene_ojbs_id list of the simulator.
        """
        return not (id_str.startswith("ART_") or id_str.startswith("MARKER_"))

    def _is_object_inside(self, obj_name, target, name_to_id, sim):
        obj_name, obj_type = search_for_id(obj_name, name_to_id)
        if obj_type == RearrangeObjectTypes.GOAL_POSITION:
            use_receps = sim.ep_info["goal_receptacles"]
        elif obj_type == RearrangeObjectTypes.RIGID_OBJECT:
            use_receps = sim.ep_info["target_receptacles"]
            obj_name = list(sim.get_targets()[0]).index(int(obj_name))
        else:
            return False
        obj_idx = int(obj_name)

        target_name, target_type = search_for_id(target, name_to_id)
        if target_type != RearrangeObjectTypes.ARTICULATED_LINK:
            return False
        check_marker = sim.get_marker(target_name)

        if obj_idx >= len(use_receps):
            rearrange_logger.debug(
                f"Could not find object {obj_name} in {use_receps}"
            )
            return False

        recep_name, recep_link_id = use_receps[obj_idx]
        if self.check_for_art_link_match[target_name] and (
            recep_link_id != check_marker.link_id
        ):
            return False
        if recep_name != check_marker.ao_parent.handle:
            return False
        return True

    def is_satisfied(
        self,
        name_to_id: Dict[str, Any],
        sim: RearrangeSim,
        obj_thresh: float,
        art_thresh: float,
    ) -> bool:
        """
        Returns True if the grounded state is present in the current simulator state.
        Also returns False if the arguments are incompatible. For example if input argument is supposed to be a cabinet, but the passed argument is a rigid object name.
        """
        if self.arg_spec is not None:
            for arg_name in self._set_args:
                match_name, match_type = search_for_id(arg_name, name_to_id)
                if not self.arg_spec.argument_matches(match_name, match_type):
                    return False

        rom = sim.get_rigid_object_manager()
        for obj_name, target in self.obj_states.items():
            if self._is_id_rigid_object(
                obj_name
            ) and not self._is_id_rigid_object(target):
                # object is rigid and target is receptacle, we are checking if
                # an object is inside of a receptacle.
                if self._is_object_inside(obj_name, target, name_to_id, sim):
                    continue
                else:
                    return False

            if not self._is_id_rigid_object(obj_name):
                # Invalid predicate
                return False

            obj_idx = name_to_id[obj_name]
            abs_obj_id = sim.scene_obj_ids[obj_idx]
            cur_pos = rom.get_object_by_id(
                abs_obj_id
            ).transformation.translation

            targ_idx = name_to_id[target]
            idxs, pos_targs = sim.get_targets()
            targ_pos = pos_targs[list(idxs).index(targ_idx)]

            dist = np.linalg.norm(cur_pos - targ_pos)
            if dist >= obj_thresh:
                return False

        for art_name, set_art in self.art_states.items():
            match_name, match_type = search_for_id(art_name, name_to_id)
            if match_type == RearrangeObjectTypes.ARTICULATED_OBJECT:
                art_obj = sim.art_objs[match_name]
                prev_art_pos = art_obj.joint_positions
            elif match_type == RearrangeObjectTypes.ARTICULATED_LINK:
                marker = sim.get_marker(match_name)
                prev_art_pos = marker.get_targ_js()
            else:
                # This is not a compatible argument type to the function
                return False

            if isinstance(set_art, str):
                art_sampler = eval(set_art)
                if not isinstance(art_sampler, ArtSampler):
                    raise ValueError(
                        f"Set art state is not an ArtSampler: {set_art}"
                    )
                did_sat = art_sampler.is_satisfied(prev_art_pos)
            else:
                prev_art_pos = np.array(prev_art_pos)
                set_art = np.array(set_art)
                if prev_art_pos.shape != set_art.shape:
                    # This type of receptacle is not a compatible input
                    return False
                art_dist = np.linalg.norm(prev_art_pos - set_art)
                did_sat = art_dist < art_thresh

            if not did_sat:
                return False

        if not self.robo_state.is_satisfied(name_to_id, sim):
            return False

        return True

    def set_state(self, name_to_id: Dict[str, Any], sim: RearrangeSim) -> None:
        """
        Set this state in the simulator. Warning, this steps the simulator.
        """
        for obj_name, target in self.obj_states.items():
            obj_idx = name_to_id[obj_name]
            abs_obj_id = sim.scene_obj_ids[obj_idx]

            if target in name_to_id:
                targ_idx = name_to_id[target]
                all_targ_idxs, pos_targs = sim.get_targets()
                targ_pos = pos_targs[list(all_targ_idxs).index(targ_idx)]
                set_T = mn.Matrix4.translation(targ_pos)
            else:
                raise ValueError("Not supported")

            # Get the object id corresponding to this name
            rom = sim.get_rigid_object_manager()
            set_obj = rom.get_object_by_id(abs_obj_id)
            set_obj.transformation = set_T

        for art_name, set_art in self.art_states.items():
            match_name, match_type = search_for_id(art_name, name_to_id)
            if match_type == RearrangeObjectTypes.ARTICULATED_OBJECT:
                art_obj = sim.art_objs[match_name]

                art_obj.clear_joint_states()

                art_obj.joint_positions = set_art
            elif match_type == RearrangeObjectTypes.ARTICULATED_LINK:
                marker = sim.get_marker(match_name)
                if isinstance(set_art, str):
                    art_sampler = eval(set_art)
                    if not isinstance(art_sampler, ArtSampler):
                        raise ValueError(
                            f"Set art state is not an ArtSampler: {set_art}"
                        )
                    marker.set_targ_js(art_sampler.sample())
                else:
                    marker.set_targ_js(set_art)
            else:
                raise ValueError(
                    f"Unrecognized type {match_type} and name {match_name} from {art_name}"
                )

            sim.internal_step(-1)

        # Set the snapped object information
        if self.robo_state.holding == "NONE" and sim.grasp_mgr.is_grasped:
            sim.grasp_mgr.desnap(True)
        elif self.robo_state.holding is not None:
            # Swap objects to the desired object.
            rel_obj_idx = name_to_id[self.robo_state.holding]
            sim.grasp_mgr.desnap(True)
            sim.internal_step(-1)
            sim.grasp_mgr.snap_to_obj(sim.scene_obj_ids[rel_obj_idx])
            sim.internal_step(-1)

        # Set the robot starting position
        if self.robo_state.pos == "rnd":
            sim.set_robot_base_to_random_point()


class ArtSampler:
    def __init__(self, value, cmp):
        self.value = value
        self.cmp = cmp

    def is_satisfied(self, cur_value):
        if self.cmp == "greater":
            return cur_value > self.value
        elif self.cmp == "less":
            return cur_value < self.value
        else:
            raise ValueError(f"Unrecognized cmp {self.cmp}")

    def sample(self):
        return self.value
