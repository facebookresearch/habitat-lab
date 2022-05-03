#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import yaml

from habitat import Config
from habitat.tasks.rearrange.multi_task.rearrange_pddl import (
    PddlAction,
    Predicate,
    RearrangeObjectTypes,
)
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim

if TYPE_CHECKING:
    from habitat.datasets.rearrange.rearrange_dataset import RearrangeDatasetV0


@dataclass(frozen=True)
class EntityToActionsMapping:
    match_entity_type: Optional[List[RearrangeObjectTypes]]
    match_id_str: str
    matching_skills: List[str]


class PddlDomain:
    """
    Manages the information from the PDDL domain and task definition.
    """

    def __init__(
        self,
        load_file_path: str,
        dataset: "RearrangeDatasetV0",
        cur_task_config: Config,
        sim: RearrangeSim,
    ):
        with open(load_file_path, "r") as f:
            self.domain_def = yaml.safe_load(f)
        self._create_action_to_entity_mapping()

        self._sim = sim

        self.predicates: List[Predicate] = []
        for pred_d in self.domain_def["predicates"]:
            pred = Predicate(pred_d)
            self.predicates.append(pred)

        self._config = cur_task_config

        self.dataset = dataset
        self.actions = {}
        self.reset()

    @property
    def action_names(self):
        return self._action_names

    def _create_action_to_entity_mapping(self):
        self._match_groups = []
        for _, group_cfg in self.domain_def[
            "action_to_entity_mapping"
        ].items():

            if len(group_cfg["match_entity_type"]) != 0:
                use_type = [
                    RearrangeObjectTypes(x)
                    for x in group_cfg["match_entity_type"]
                ]
            else:
                use_type = None

            self._match_groups.append(
                EntityToActionsMapping(
                    match_entity_type=use_type,
                    match_id_str=group_cfg["match_id_str"],
                    matching_skills=group_cfg["matching_skills"],
                )
            )

    def reset(self):
        self._name_to_id = self.get_name_id_conversions(self.domain_def)

        for action_d in self.domain_def["actions"]:
            action = PddlAction(
                action_d,
                self._config,
                self.dataset,
                self._name_to_id,
                self.predicate_lookup,
            )
            self.actions[action.name] = action
        self._action_names = list(self.actions.keys())

    def get_task_match_for_name(self, task_name: str) -> PddlAction:
        return self.actions[task_name]

    def predicate_lookup(self, pred_key: str) -> Optional[Predicate]:
        """
        Return a predicate that matches a name. Returns `None` if no predicate is found.
        """
        pred_name, pred_args = pred_key.split("(")
        pred_args = pred_args.split(")")[0].split(",")
        if pred_args[0] == "":
            pred_args = []
        # We take the first match
        for pred in self.predicates:
            if pred.name != pred_name:
                continue

            if len(pred_args) != len(pred.args):
                continue
            return copy.deepcopy(pred)
        return None

    def is_pred_true(self, bound_pred: Predicate) -> bool:
        return bound_pred.set_state.is_satisfied(
            self._name_to_id,
            self._sim,
            self._config.OBJ_SUCC_THRESH,
            self._config.ART_SUCC_THRESH,
        )

    def is_pred_true_args(self, pred: Predicate, input_args):
        if pred.set_state is not None:
            bound_pred = copy.deepcopy(pred)
            bound_pred.bind(input_args)
            return self.is_pred_true(bound_pred), bound_pred

        return False

    def get_true_predicates(self):
        all_entities = self.get_all_entities()
        true_preds = []
        for pred in self.predicates:
            for entity_input in itertools.combinations(
                all_entities, pred.get_n_args()
            ):
                is_pred_true, bound_pred = self.is_pred_true_args(
                    pred, entity_input
                )

                if is_pred_true:
                    true_preds.append(bound_pred)
        return true_preds

    def get_all_entities(self) -> List[str]:
        return list(self._name_to_id.keys())

    def get_name_to_id_mapping(self) -> Dict[str, Any]:
        return self._name_to_id

    def get_name_id_conversions(self, domain_def) -> Dict[str, Any]:
        """
        Returns a map of constant scene identifiers, such as `kitchen_counter_targets|0`, to the scene ID. If the scene identifier starts with "TARGET_" it is the goal position and refers to an index in the targets array. If it begins with "ART_" it is an articulated object and refers to an index in `self._sim.art_objs`.
        """
        name_to_id = {}

        id_to_name = {}
        for k, i in self._sim.ref_handle_to_rigid_obj_id.items():
            id_to_name[i] = k
            name_to_id[k] = i

        for targ_idx in self._sim.get_targets()[0]:
            # The object this is the target for.
            ref_id = id_to_name[targ_idx]
            name_to_id[f"TARGET_{ref_id}"] = targ_idx

        for i, art_obj in enumerate(self._sim.art_objs):
            name_to_id["ART_" + art_obj.handle] = i

        for k in self._sim.get_all_markers():
            name_to_id["MARKER_" + k] = k

        return name_to_id

    def get_matching_skills(
        self, entity_type: RearrangeObjectTypes, entity_id: str
    ) -> List[str]:
        """
        Gets the skills that have argument types compatible with an entity.
        :entity_type: One of the inputs types to the action must match this type.
        :entity_id: One of the input names to the action must match this name.
        """
        matching_skills = None
        for match_group in self._match_groups:
            if (
                match_group.match_entity_type is not None
                and entity_type not in match_group.match_entity_type
            ):
                continue
            if not entity_id.startswith(match_group.match_id_str):
                continue

            if matching_skills is not None:
                raise ValueError(
                    f"Multiple matching skills for {entity_type}, {entity_id}"
                )
            matching_skills = match_group.matching_skills
        return matching_skills
