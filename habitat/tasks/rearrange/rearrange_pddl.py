def parse_func(x):
    name = x.split("(")[0]
    args = x.split("(")[1].split(")")[0]
    return name, args


def search_for_id(k, name_to_id):
    if isinstance(k, str):
        if k not in name_to_id and "ART_" + k in name_to_id:
            return name_to_id["ART_" + k]
        else:
            return name_to_id[k]
    return k


def load_task_object(
    task,
    task_def,
    cur_config,
    cur_env,
    cur_dataset,
    should_super_reset,
    task_kwargs,
):
    prev_base = NavRLEnv.__bases__[0]
    NavRLEnv.__bases__ = (DummyRLEnv,)
    task_cls = baseline_registry.get_env(task)

    yacs.config._VALID_TYPES.add(type(cur_env._env))
    task_config_name = task_def

    config = copy.copy(cur_config)
    config.defrost()
    if task_config_name is not None:
        task_config = habitat.get_config(
            osp.join(TASK_CONFIGS_DIR, task_config_name + ".yaml")
        )
        config.merge_from_other_cfg(task_config)
    config.TASK_CONFIG.tmp_env = cur_env._env
    config.freeze()
    task = task_cls(config, cur_dataset)
    config.defrost()
    del config.TASK_CONFIG["tmp_env"]
    config.freeze()
    yacs.config._VALID_TYPES.remove(type(cur_env._env))

    task.set_args(**task_kwargs)
    task.set_sim_reset(False)

    # THIS COULD SET THE SIMULATOR STATE
    task.reset(super_reset=should_super_reset)
    NavRLEnv.__bases__ = (prev_base,)
    return task


class Action:
    def __init__(
        self, load_d, config, dataset, name_to_id, env, predicate_lookup_fn
    ):
        self.name = load_d["name"]
        self.parameters = load_d["parameters"]
        self.name_to_id = name_to_id

        self.load_task_fn = partial(
            self._load_task, load_d, config, dataset, name_to_id
        )
        self.precond = []
        self.precond_strs = []

        for precond_str in load_d["precondition"]:
            precond = copy.deepcopy(predicate_lookup_fn(precond_str))
            self.precond.append(precond)
            self.precond_strs.append(parse_func(precond_str)[1])

        self.itpreconds = []
        self.itprecond_strs = []
        for itprecond in load_d.get("if_then_preconds", []):
            self.itprecond_strs.append(
                [parse_func(itprecond[0])[1], parse_func(itprecond[1])[1]]
            )
            self.itpreconds.append(
                [
                    copy.deepcopy(predicate_lookup_fn(itprecond[0])),
                    copy.deepcopy(predicate_lookup_fn(itprecond[1])),
                ]
            )

        self.effect = []
        self.effect_args = []
        for effect_s in load_d["effect"]:
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

        for i in range(len(self.effect_args)):
            self.effect[i].bind(convert_arg_str(self.effect_args[i]))

        for i in range(len(self.precond_strs)):
            self.precond[i].bind(convert_arg_str(self.precond_strs[i]))

        for i in range(len(self.itprecond_strs)):
            self.itpreconds[i][0].bind(
                convert_arg_str(self.itprecond_strs[i][0])
            )
            self.itpreconds[i][1].bind(
                convert_arg_str(self.itprecond_strs[i][1])
            )

        self.is_bound = True

    def _load_task(
        self, load_d, config, dataset, name_to_id, env, should_reset=True
    ):
        if "task" not in load_d:
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
            load_d["task"],
            load_d["task_def"],
            config,
            env,
            dataset,
            should_reset,
            task_kwargs,
        )

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

    def apply(self, name_to_id, sim):
        for effect in self.effect:
            self._apply_effect(effect, name_to_id, sim)

    def _apply_effect(self, effect, name_to_id, sim):
        set_state = effect.set_state
        set_state.set_state(name_to_id, sim)


class Predicate:
    def __init__(self, load_d):
        self.name = load_d["name"]
        if "state" in load_d:
            self.set_state = SetState(load_d["state"])
        else:
            self.set_state = None
        self.args = load_d.get("args", [])
        self.set_args = None

    def is_equal(self, b):
        return (
            (b.name == self.name)
            and (self.args == b.args)
            and (self.set_args == b.set_args)
        )

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
        self.holding = load_d.get("holding", None)
        self.pos = load_d.get("pos", None)

    def bind(self, arg_k, arg_v):
        for k, v in zip(arg_k, arg_v):
            if self.holding is not None:
                self.holding = self.holding.replace(k, v)
            if self.pos is not None:
                self.pos = self.pos.replace(k, v)


class SetState:
    def __init__(self, load_d):
        self.art_states = load_d.get("art_states", {})
        self.obj_states = load_d.get("obj_states", {})
        self.robo_state = RoboState(load_d.get("robo", {}))
        self.load_d = load_d

    def bind(self, arg_k, arg_v):
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
            if "catch_ids" in self.load_d:
                self.load_d["catch_ids"] = self.load_d["catch_ids"].replace(
                    k, v
                )

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
                art_dist = np.linalg.norm(
                    np.array(prev_art_pos) - np.array(set_art)
                )
                did_sat = art_dist < art_thresh

            if not did_sat:
                return False
        if (
            self.robo_state.holding != "NONE"
            and self.robo_state.holding is not None
        ):
            # Robot must be holding right object.
            obj_idx = name_to_id[self.robo_state.holding]
            abs_obj_id = sim.scene_obj_ids[obj_idx]
            if sim.snapped_obj_id != abs_obj_id:
                return False
        elif self.robo_state.holding == "NONE":
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
                raise ValueError("Not supported")

            # Get the object id corresponding to this name
            sim.reset_obj_T(abs_obj_id, set_T)
            obj_name = sim.ep_info["static_objs"][obj_idx][0]
            if obj_name in SET_ROTATIONS:
                sim.set_rotation(SET_ROTATIONS[obj_name], abs_obj_id)
            # Freeze the object at that position.
            # This causes some issues with later picking. I don't think there
            # is any need for this anymore?
            # make_render_only(abs_obj_id, sim)

        for art_obj_id, set_art in self.art_states.items():
            art_obj_id = search_for_id(art_obj_id, name_to_id)

            abs_id = sim.art_obj_ids[art_obj_id]
            prev_sleep = sim.get_articulated_object_sleep(abs_id)
            sim.set_articulated_object_sleep(abs_id, False)

            if isinstance(set_art, str):
                prev_art_pos = sim._sim.get_articulated_object_positions(
                    abs_id
                )
                # This is some sampling object we need to instantiate
                art_sampler = eval(set_art)
                set_art = art_sampler.sample(prev_art_pos, abs_id)
                art_sampler.catch(
                    self.load_d.get("catch_ids", None), name_to_id, sim
                )

            sim.reset_art_obj_pos(abs_id, set_art)

            sim.internal_step(-1)
            sim.set_articulated_object_sleep(abs_id, prev_sleep)

        robo_state = self.robo_state
        # if robo_state.pos is not None:
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
        if robo_state.holding == "NONE" and sim.snapped_obj_id is not None:
            sim.desnap_object(force=True)
        elif robo_state.holding is not None:
            rel_obj_idx = name_to_id[robo_state.holding]
            sim.internal_step(-1)
            sim.full_snap(rel_obj_idx)
            sim.internal_step(-1)
