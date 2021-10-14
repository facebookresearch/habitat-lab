import copy
import itertools

import yaml

from habitat.tasks.rearrange.multi_task.rearrange_pddl import Action, Predicate


class PddlDomain:
    def __init__(self, load_file, dataset, cur_task_config, sim):
        with open(load_file, "r") as f:
            domain_def = yaml.safe_load(f)

        self.sim = sim
        self._name_to_id = self.get_name_id_conversions(domain_def)

        self.predicates = []
        for pred_d in domain_def["predicates"]:
            pred = Predicate(pred_d)
            self.predicates.append(pred)
        self.types = domain_def["types"]

        self._config = cur_task_config

        self.actions = {}
        for action_d in domain_def["actions"]:
            action = Action(
                action_d,
                cur_task_config,
                dataset,
                self._name_to_id,
                self,
                self.predicate_lookup,
            )
            self.actions[action.name] = action

    def get_task_match_for_name(self, task_name_cls):
        matches = []
        for action in self.actions.values():
            if action.task == task_name_cls:
                matches.append(action)
        if len(matches) != 1:
            raise ValueError("Invalid or too many matches for task name")
        return matches[0]

    def predicate_lookup(self, pred_key):
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
            is_match = True
            for q_arg, k_arg in zip(pred_args, pred.args):
                if k_arg in self.types and q_arg not in self.types:
                    is_match = False
                    break
            if is_match:
                return pred
        return None

    def is_pred_true(self, bound_pred):
        return bound_pred.set_state.is_satisfied(
            self._name_to_id,
            self.sim,
            self._config.OBJ_SUCC_THRESH,
            self._config.ART_SUCC_THRESH,
        )

    def is_pred_true_args(self, pred, input_args):
        if pred.set_state is not None:
            bound_pred = copy.deepcopy(pred)
            bound_pred.bind(input_args)
            return self.is_pred_true(bound_pred)

        return False

    def get_true_predicates(self):
        all_entities = self.get_all_entities()
        true_preds = []
        for pred in self.predicates:
            for entity_input in itertools.combinations(
                all_entities, pred.get_n_args()
            ):
                if self.is_pred_true_args(pred, entity_input):
                    true_preds.append(pred)
        return true_preds

    def get_all_entities(self):
        return list(self._name_to_id.keys())

    def get_name_to_id_mapping(self):
        return self._name_to_id

    def get_name_id_conversions(self, domain_def):
        name_to_id = {}

        id_to_name = {}
        for k, i in self.sim.ref_handle_to_rigid_obj_id.items():
            id_to_name[i] = k
            name_to_id[k] = i

        for targ_idx in self.sim.get_targets()[0]:
            # The object this is the target for.
            # abs_rigid_idx = self.sim.scene_obj_ids[targ_idx]
            ref_id = id_to_name[targ_idx]
            name_to_id[f"TARGET_{ref_id}"] = targ_idx

        for i, art_obj in enumerate(self.sim.art_objs):
            name_to_id["ART_" + art_obj.handle] = i

        # for name, marker_name in domain_def["markers"].items():
        #    name_to_id[name] = marker_name
        return name_to_id
