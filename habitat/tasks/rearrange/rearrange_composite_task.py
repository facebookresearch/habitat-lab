from habitat_baselines.common.baseline_registry import baseline_registry
import yaml
from collections import defaultdict
import yacs.config
from functools import partial
from habitat_baselines.common.environments import NavRLEnv
import copy
import habitat
import time


import os.path as osp

import numpy as np

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.tasks.rearrange.utils import CacheHelper, rearrange_collision

BASE_TASK_SPECS = './orp/full_tasks/'
TASK_CONFIGS_DIR = './orp/configs/task'
BASE_SCENE_DIR = './orp/scenes/'

DOMAIN_FILE = 'domain'


def load_task_object(task, task_def, cur_config, cur_env, cur_dataset,
                     should_super_reset, task_kwargs):
    prev_base = NavRLEnv.__bases__[0]
    NavRLEnv.__bases__ = (DummyRLEnv,)
    task_cls = baseline_registry.get_env(task)

    yacs.config._VALID_TYPES.add(type(cur_env._env))
    task_config_name = task_def

    config = copy.copy(cur_config)
    config.defrost()
    if task_config_name is not None:
        task_config = habitat.get_config(osp.join(TASK_CONFIGS_DIR,
                                                  task_config_name + ".yaml"))
        config.merge_from_other_cfg(task_config)
    config.TASK_CONFIG.tmp_env = cur_env._env
    config.freeze()
    task = task_cls(config, cur_dataset)
    config.defrost()
    del config.TASK_CONFIG['tmp_env']
    config.freeze()
    yacs.config._VALID_TYPES.remove(type(cur_env._env))

    task.set_args(**task_kwargs)
    task.set_sim_reset(False)

    # THIS COULD SET THE SIMULATOR STATE
    task.reset(super_reset=should_super_reset)
    NavRLEnv.__bases__ = (prev_base,)
    return task

class DummyRLEnv(object):
    def __init__(self, config, dataset=None):
        self.config = config
        self._env = config.tmp_env
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.number_of_episodes = self._env.number_of_episodes
        self.reward_range = self.get_reward_range()

    def set_args(self, **kwargs):
        pass

    def reset(self):
        sim = self._env._sim
        sim._try_acquire_context()
        prev_sim_obs = sim._sim.get_sensor_observations()
        obs = sim._sensor_suite.get_observations(prev_sim_obs)
        return obs


def parse_func(x):
    name = x.split('(')[0]
    args = x.split('(')[1].split(')')[0]
    return name, args

def search_for_id(k, name_to_id):
    if isinstance(k, str):
        if k not in name_to_id and 'ART_'+k in name_to_id:
            return name_to_id['ART_' + k]
        else:
            return name_to_id[k]
    return k


class Action:
    def __init__(self, load_d, config, dataset, name_to_id, env,
                 predicate_lookup_fn):
        self.name = load_d['name']
        self.parameters = load_d['parameters']
        self.name_to_id = name_to_id

        self.load_task_fn = partial(self._load_task, load_d, config, dataset, name_to_id)
        self.precond = []
        self.precond_strs = []

        for precond_str in load_d['precondition']:
            precond = copy.deepcopy(predicate_lookup_fn(precond_str))
            self.precond.append(precond)
            self.precond_strs.append(parse_func(precond_str)[1])

        self.itpreconds = []
        self.itprecond_strs = []
        for itprecond in load_d.get('if_then_preconds', []):
            self.itprecond_strs.append([parse_func(itprecond[0])[1],
                                        parse_func(itprecond[1])[1]])
            self.itpreconds.append([
                copy.deepcopy(predicate_lookup_fn(itprecond[0])),
                copy.deepcopy(predicate_lookup_fn(itprecond[1]))
            ])

        self.effect = []
        self.effect_args = []
        for effect_s in load_d['effect']:
            _, effect_arg = parse_func(effect_s)
            self.effect_args.append(effect_arg)
            effect = predicate_lookup_fn(effect_s)
            if effect is None:
                raise ValueError(f"Could not find effect for {effect_s}")

            self.effect.append(effect)
        self.is_bound = False

    def __repr__(self):
        return f"<Action: {self.name}, paras: {self.parameters}, preconds: {self.precond}, effects: {self.effect}>"

    def bind(self, args, add_args={}):
        """
        - args list[str]
        """
        assert not self.is_bound
        self.add_args = add_args
        action_args = {}
        self.orig_applied_func_args = args[:]
        self.applied_func_args = args[:]
        if self.applied_func_args[0] == '':
            self.applied_func_args = []
        for i, k in enumerate(self.applied_func_args):
            self.applied_func_args[i] = search_for_id(k, self.name_to_id)

        if len(args) != len(self.parameters):
            raise ValueError()

        def convert_arg_str(effect_arg):
            for place_arg, set_arg in zip(self.parameters, args):
                effect_arg = effect_arg \
                    .replace(place_arg, set_arg) \
 \
            effect_arg = effect_arg.split(',')
            if effect_arg[0] == '':
                effect_arg = []
            return effect_arg

        for i in range(len(self.effect_args)):
            self.effect[i].bind(convert_arg_str(self.effect_args[i]))

        for i in range(len(self.precond_strs)):
            self.precond[i].bind(convert_arg_str(self.precond_strs[i]))

        for i in range(len(self.itprecond_strs)):
            self.itpreconds[i][0].bind(convert_arg_str(self.itprecond_strs[i][0]))
            self.itpreconds[i][1].bind(convert_arg_str(self.itprecond_strs[i][1]))

        self.is_bound=True


    def _load_task(self, load_d, config, dataset, name_to_id, env,
                   should_reset=True):
        if 'task' not in load_d:
            return None
        func_kwargs = {k:v for k,v in zip(self.parameters, self.applied_func_args)}
        task_kwargs = {**func_kwargs, **self.add_args,
                       **{'orig_applied_args': {k: v for k,v in zip(self.parameters, self.orig_applied_func_args)}}
                       }
        return load_task_object(load_d['task'], load_d['task_def'], config,
                                env, dataset, should_reset, task_kwargs)

    def init_task(self, env, should_reset=True):
        return self.load_task_fn(env, should_reset=should_reset)

    def can_apply(self, preds):
        def has_match(pred):
            return any([other_pred.is_equal(pred) for other_pred in preds])
        for precond_pred in self.precond:
            if not has_match(precond_pred):
                return False
        for pred1, pred2 in self.itpreconds:
            # X -> Y is same as not X or Y
            if not (not has_match(pred1) or has_match(pred2)):
                return False
        return True

    def update_predicates(self, preds):
        new_preds = copy.deepcopy(preds)
        for pred in self.effect:
            if pred.name.startswith('open') or pred.name.startswith('closed') or pred.name.startswith('not') :
                base_name = '_'.join(pred.name.split('_')[1:])
            else:
                base_name = None
            found = False
            for i, other_pred in enumerate(preds):
                other_base_name = '_'.join(other_pred.name.split('_')[1:])
                if pred.name == other_pred.name:
                    # Override
                    new_preds[i] = pred
                    found = True
                    break
                if base_name is not None and other_base_name == base_name:
                    new_preds[i] = pred
                    found = True
                    break
                if base_name == other_pred.name or other_base_name == pred.name:
                    new_preds[i] = pred
                    found = True
                    break
            if not found:
                new_preds.append(pred)
        return new_preds

    def apply(self, name_to_id, sim):
        for effect in self.effect:
            self._apply_effect(effect, name_to_id, sim)

    def _apply_effect(self, effect, name_to_id, sim):
        set_state = effect.set_state
        set_state.set_state(name_to_id, sim)



class Predicate:
    def __init__(self, load_d):
        self.name = load_d['name']
        if 'state' in load_d:
            self.set_state = SetState(load_d['state'])
        else:
            self.set_state = None
        self.args = load_d.get('args', [])
        self.set_args = None

    def is_equal(self, b):
        return (b.name == self.name) and (self.args == b.args) and (self.set_args == b.set_args)

    def bind(self, set_args):
        if len(self.args) != len(set_args):
            raise ValueError()

        self.set_args = set_args

        if self.set_state is not None:
            self.set_state.bind(self.args, set_args)

    def __str__(self):
        return f"<Predicate: {self.name} [{self.args}] [{self.set_args}]>"

    def __repr__(self):
        return str(self)


class RoboState:
    def __init__(self, load_d):
        self.holding = load_d.get('holding', None)
        self.pos = load_d.get('pos', None)

    def bind(self, arg_k, arg_v):
        for k, v in zip(arg_k, arg_v):
            if self.holding is not None:
                self.holding = self.holding.replace(k, v)
            if self.pos is not None:
                self.pos = self.pos.replace(k,v)


class SetState:
    def __init__(self, load_d):
        self.art_states = load_d.get('art_states', {})
        self.obj_states = load_d.get('obj_states', {})
        self.robo_state = RoboState(load_d.get('robo', {}))
        self.load_d = load_d

    def bind(self, arg_k, arg_v):
        def list_replace(l, k, v):
            new_l = {}
            for l_k, l_v in l.items():
                if isinstance(l_k, str):
                    l_k = l_k.replace(k,v)
                if isinstance(l_v, str):
                    l_v = l_v.replace(k,v)
                new_l[l_k] = l_v
            return new_l

        for k, v in zip(arg_k, arg_v):
            self.art_states = list_replace(self.art_states, k, v)
            self.obj_states = list_replace(self.obj_states, k, v)
            if 'catch_ids' in self.load_d:
                self.load_d['catch_ids'] = self.load_d['catch_ids'].replace(k,v)

        self.robo_state.bind(arg_k, arg_v)

    def is_satisfied(self, name_to_id, sim, obj_thresh, art_thresh):
        for obj_name, target in self.obj_states.items():
            obj_idx = name_to_id[obj_name]
            abs_obj_id = sim.scene_obj_ids[obj_idx]
            cur_pos = sim._sim.get_translation(abs_obj_id)

            targ_idx = name_to_id[target]
            _, pos_targs = sim.get_targets()
            targ_pos = pos_targs[targ_idx]

            dist = np.linalg.norm(cur_pos - targ_pos)
            if not dist < obj_thresh:
                return False

        for art_obj_id, set_art in self.art_states.items():
            abs_id = sim.art_obj_ids[art_obj_id]
            prev_art_pos = sim._sim.get_articulated_object_positions(abs_id)
            if isinstance(set_art, str):
                art_sampler = eval(set_art)
                did_sat = art_sampler.is_satisfied(prev_art_pos)
            else:
                art_dist = np.linalg.norm(np.array(prev_art_pos) - np.array(set_art))
                did_sat = art_dist < art_thresh

            if not did_sat:
                return False
        if self.robo_state.holding != 'NONE' and self.robo_state.holding is not None:
            # Robot must be holding right object.
            obj_idx = name_to_id[self.robo_state.holding]
            abs_obj_id = sim.scene_obj_ids[obj_idx]
            if sim.snapped_obj_id != abs_obj_id:
                return False
        elif self.robo_state.holding == 'NONE':
            # Robot must be holding nothing
            if sim.snapped_obj_id != None:
                return False

        return True

    def set_state(self, name_to_id, sim):
        for obj_name, target in self.obj_states.items():
            obj_idx = name_to_id[obj_name]
            abs_obj_id = sim.scene_obj_ids[obj_idx]

            T = sim._sim.get_transformation(abs_obj_id)
            if target in name_to_id:
                targ_idx = name_to_id[target]
                _, pos_targs = sim.get_targets()
                targ_pos = pos_targs[targ_idx]
                set_T = mn.Matrix4.translation(targ_pos)
            else:
                raise ValueError('Not supported')

            # Get the object id corresponding to this name
            sim.reset_obj_T(abs_obj_id, set_T)
            obj_name = sim.ep_info['static_objs'][obj_idx][0]
            if obj_name in SET_ROTATIONS:
                sim.set_rotation(SET_ROTATIONS[obj_name], abs_obj_id)
            # Freeze the object at that position.
            # This causes some issues with later picking. I don't think there
            # is any need for this anymore?
            #make_render_only(abs_obj_id, sim)

        for art_obj_id, set_art in self.art_states.items():
            art_obj_id = search_for_id(art_obj_id, name_to_id)

            abs_id = sim.art_obj_ids[art_obj_id]
            prev_sleep = sim.get_articulated_object_sleep(abs_id)
            sim.set_articulated_object_sleep(abs_id, False)

            if isinstance(set_art, str):
                prev_art_pos = sim._sim.get_articulated_object_positions(abs_id)
                # This is some sampling object we need to instantiate
                art_sampler = eval(set_art)
                set_art = art_sampler.sample(prev_art_pos, abs_id)
                art_sampler.catch(self.load_d.get('catch_ids', None), name_to_id, sim)

            sim.reset_art_obj_pos(abs_id, set_art)

            sim.internal_step(-1)
            sim.set_articulated_object_sleep(abs_id, prev_sleep)

        robo_state = self.robo_state
        #if robo_state.pos is not None:
        #    if robo_state.pos == 'rnd':
        #        start_pos = sim.pathfinder.get_random_navigable_point()
        #    else:
        #        start_pos = robo_state.pos
        #    if isinstance(start_pos, str):
        #        name = start_pos
        #        if name in name_to_id:
        #            obj_id = name_to_id[name]
        #            if isinstance(obj_id, str):
        #                # marker id.
        #                pos = sim.get_marker_nav_pos(obj_id)
        #            else:
        #                abs_id = sim.scene_obj_ids[obj_id]
        #                pos = sim.get_translation(abs_id)
        #        else:
        #            # This must be an articulated object
        #            art_id = name_to_id['ART_' + name]
        #            abs_id = sim.art_obj_ids[art_id]
        #            pos = sim.get_articulated_object_root_state(abs_id).translation
        #        start_pos = sim.get_nav_pos(pos)

        #    sim.set_robot_pos(start_pos)
        if robo_state.holding == 'NONE' and sim.snapped_obj_id is not None:
            sim.desnap_object(force=True)
        elif robo_state.holding is not None:
            rel_obj_idx = name_to_id[robo_state.holding]
            sim.internal_step(-1)
            sim.full_snap(rel_obj_idx)
            sim.internal_step(-1)




@registry.register_task(name="RearrangeCompositeTask-v0")
class RearrangeCompositeTaskV0(RearrangeTask):
    def __init__(self, config, dataset=None):
        super().__init__(config, dataset)
        full_task_path = osp.join(BASE_TASK_SPECS, self.tcfg.TASK_SPEC + '.yaml')
        with open(full_task_path, 'r') as f:
            task_def = yaml.safe_load(f)
        self.task_def = task_def

        start_d = task_def['start']
        self.start_state = SetState(start_d['state'])

        scene_layout = start_d['scene_name']
        full_scene_path = osp.join(BASE_SCENE_DIR, scene_layout + '.yaml')
        with open(full_scene_path, 'r') as f:
            scene_def = yaml.safe_load(f)

        domain_path = osp.join(BASE_TASK_SPECS, DOMAIN_FILE + '.yaml')
        with open(domain_path, 'r') as f:
            domain_def = yaml.safe_load(f)

        self.name_to_id = self.get_name_id_conversions(scene_def, domain_def)

        self.predicates = []
        for pred_d in domain_def['predicates']:
            pred = Predicate(pred_d)
            self.predicates.append(pred)
        self.types = domain_def['types']

        self.actions = {}
        for action_d in domain_def['actions']:
            action = Action(action_d, config, dataset, self.name_to_id, self,
                            self.predicate_lookup)
            self.actions[action.name] = action
        self.load_solution(task_def['solution'])
        self.last_change = time.time()
        self.cur_node = -1
        self.inf_cur_node = 0
        self.cur_task = None
        self.call_count = defaultdict(lambda: 0)
        self.cached_tasks = {}
        assert isinstance(self.tcfg.EVAL_NODE, int)
        if self.tcfg.EVAL_NODE >= 0:
            self.cur_node = self.tcfg.EVAL_NODE

        use_goal_precond = self.tcfg['GOAL_PRECOND']
        if use_goal_precond == '':
            use_goal_precond = task_def['goal']
        else:
            if use_goal_precond.startswith("'") and use_goal_precond.endswith("'"):
                use_goal_precond = use_goal_precond[1:-1]

            use_goal_precond = use_goal_precond.replace('/', ',')
            use_goal_precond = {
                'precondition': use_goal_precond.split('.')
            }

        self._load_goal_preconds(use_goal_precond)
        self._load_start_preconds(start_d)
        self._load_stage_preconds(task_def.get('stage_goals', {}))

    def _load_stage_preconds(self, stage_goals):
        self.stage_goals = {}
        for k, preconds in stage_goals.items():
            self.stage_goals[k] = self._parse_precond_list(preconds)

    def _parse_precond_list(self, d):
        preds = []
        for pred_s in d:
            pred = copy.deepcopy(self.predicate_lookup(pred_s))
            _, effect_arg = parse_func(pred_s)
            effect_arg = effect_arg.split(',')
            if effect_arg[0] == '':
                effect_arg = []
            pred.bind(effect_arg)
            preds.append(pred)
        return preds

    def _load_goal_preconds(self, goal_d):
        self.goal_state = self._parse_precond_list(goal_d['precondition'])

    def _load_start_preconds(self, start_d):
        self.cur_state = self._parse_precond_list(start_d['precondition'])

    def query(self, pred_s):
        pred = self.predicate_lookup(pred_s)
        _, search_args = parse_func(pred_s)
        search_args = search_args.split(',')
        for pred in self.cur_state:
            if pred.name != pred.name:
                continue
            if pred.set_args is None:
                raise ValueError('unbound predicate in the current state')
            if len(pred.set_args) != len(search_args):
                raise ValueError('Predicate has wrong # of args')
            all_match = True
            for k1, k2 in zip(pred.set_args, search_args):
                if k2 == '*':
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
            if self.tcfg.LIMIT_TASK_NODE != -1 and i > self.tcfg.LIMIT_TASK_NODE:
                break
            name, args = parse_func(action)
            args = args.split(',')
            ac_instance = copy.deepcopy(self.actions[name])

            ac_instance.bind(args, self.task_def.get('add_args', {}).get(i, {}))
            self.solution.append(ac_instance)

    def predicate_lookup(self, pred_key):
        pred_name, pred_args = pred_key.split('(')
        pred_args = pred_args.split(')')[0].split(',')
        if pred_args[0] == '':
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

    def get_name_id_conversions(self, scene_def, domain_def):
        name_to_id = {}
        counts = defaultdict(lambda: 0)
        for i, x in enumerate(scene_def['obj_inits']):
            if not isinstance(x, str):
                obj_name = x[0]
            else:
                # This must be a clutter
                continue
            real_obj_name = f"{obj_name}|{counts[obj_name]}"
            name_to_id[real_obj_name] = i
            counts[obj_name] += 1

        for i, x in enumerate(scene_def['target_gens']):
            obj_name = x[0]
            name_to_id['TARGET_' + obj_name] = i

        for name, idx in domain_def['art_objs'].items():
            name_to_id['ART_' + name] = idx

        for name, marker_name in domain_def['markers'].items():
            name_to_id[name] = marker_name
        return name_to_id

    def _jump_to_node(self, node_idx, is_full_task=False):
        self.last_change = time.time()

        # We don't want to reset to this node if we are in full task mode.
        if not is_full_task:
            self.cur_node = node_idx
        sim = self._env._sim
        node_name = self.solution[node_idx].name

        #print('JUMP %i: %s' % (node_idx, str(node_name)))
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
                self.use_max_accum_force = len(self.solution) * self.tcfg.MAX_ACCUM_FORCE
            else:
                self.use_max_accum_force = -1.0
            if self.tcfg.LIMIT_TASK_NODE != -1:
                self._env._max_episode_steps = self.tcfg.LIMIT_TASK_LEN_SCALING*(self.tcfg.LIMIT_TASK_NODE+1)

            self.cur_task = None
        else:
            self._env._max_episode_steps = 400
            #TODO: This is a bit hacky, I trained with no max force for fridge /
            # tasks, but I am evaluating with a limit.
            if self.use_max_accum_force == -1:
                self.use_max_accum_force = self.tcfg.SUBTASK_NO_SPEC_MAX_ACCUM_FORCE

    def increment_node(self):
        if time.time() - self.last_change <= 2:
            print('Too fast not allowing')
        print('incrementing node')
        if self.cur_node+1 >= self.get_num_nodes():
            self.cur_node = 0
        else:
            self.cur_node = self.cur_node+1
        # Visualizing nav is not exciting.
        if self.solution[self.cur_node].name in ['nav', 'move_obj']:
            self.increment_node()
            return
        self.reset()

    def decrement_node(self):
        if time.time() - self.last_change <= 2:
            print('Too fast not allowing')
        if self.cur_node - 1 <= 0:
            self.cur_node = self.get_num_nodes() - 1
        else:
            self.cur_node = self.cur_node - 1
        print('decrementing node now at ', self.cur_node)
        # Visualizing nav is not exciting.
        if self.solution[self.cur_node].name in ['nav', 'move_obj']:
            self.decrement_node()
            return
        self._jump_to_node(self.cur_node)
        #self.reset()

    def get_num_nodes(self):
        return len(self.solution)

    def _get_next_inf_sol(self):
        # Never give reward from these nodes, skip to the next node instead.
        REWARD_SKIP_NODES = ['move_obj']
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
            task = self.solution[self.inf_cur_node].init_task(self,
                                                              should_reset=False)
            self.cached_tasks[self.inf_cur_node] = task
            self.inf_cur_task = task

        self.use_ignore_hold_violate = self.inf_cur_task.use_ignore_hold_violate

        return True

    def reset(self):
        sim = self._env._sim
        obs = super().reset()
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
            info['ep_success'] = is_succ
            info['node_idx'] = self.cur_node
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
            info['node_idx'] = self.inf_cur_node
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
            if not pred.set_state.is_satisfied(self.name_to_id, sim,
                                               self.rlcfg.OBJ_SUCC_THRESH, self.rlcfg.ART_SUCC_THRESH):
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


if __name__ == '__main__':
    import sys
    sys.path.insert(0, "./")
    from orp.env import get_env
    from orp_env_adapter import OrpInterface
    import argparse
    import orp.tasks.rearrang_env
    import orp.tasks.rearrang_pick_env
    import orp.tasks.rearrang_place_env
    import orp.tasks.shut_env
    import orp.tasks.open_env
    import orp.tasks.push_env
    import orp.tasks.reach_env

    #TEST_NAME = 'stock_fridge'
    TEST_NAME = 'set_table'

    parser = argparse.ArgumentParser()
    OrpInterface(None, False).get_add_args(parser)
    args = parser.parse_args(['--hab-scene-name', TEST_NAME,
                              '--hab-agent-config', 'all', '--hab-env-config', 'full_task',
                              '--hab-set', 'TASK_CONFIG.TASK_SPEC='+TEST_NAME])

    env_interface = OrpInterface(args, False)
    env = env_interface.create_from_id("")
    env = env_interface.env_trans_fn(env, set_eval=True)

    rl_env = env.env.env

    env.reset()
    for i in range(rl_env.get_num_nodes()):
        rl_env.increment_node()
        for _ in range(10):
            obs, reward, done, info = env.step(env.action_space.sample())

    #for stage in stages:
    #    print('Jumping to', stage)
    #    rl_env._jump_to_node(stage)
    #    for _ in range(10):
    #        obs, reward, done, info = env.step(env.action_space.sample())


