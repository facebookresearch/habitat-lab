import gzip
import json
import os
import os.path as osp
import pickle
import shutil
from collections import defaultdict
from functools import partial
from typing import List

import attr
import magnum as mn
import numpy as np
import rlf.rl.utils as rutils
from orp.obj_loaders import add_obj, init_art_objs
from orp.samplers import *
from orp.scenes.zatun_samplers import *
from orp.utils import (
    dist_2d,
    euler_to_quat,
    get_aabb,
    get_nav_mesh_settings_from_height,
    inter_any_bb,
    make_render_only,
)
from rlf.exp_mgr.viz_utils import save_mp4
from tqdm import tqdm
from yacs.config import CfgNode as CN

import habitat_sim
from habitat_sim.nav import NavMeshSettings
from habitat_sim.physics import MotionType

###########
# Magic constants
# Max dist from nearest navigable point to target object in 2D plane
MAX_OBJ_DIST = 1.2
###########


def mag_mat_to_list(trans):
    return [list(x) for x in list(trans)]


def set_state(static_T, obj_ids, sim):
    for T, i in zip(static_T, obj_ids):
        sim.set_transformation(T, i)
        sim.set_linear_velocity(mn.Vector3(0, 0, 0), i)
        sim.set_angular_velocity(mn.Vector3(0, 0, 0), i)


def is_stable(sim, watch_ids, is_move_legal_dict, num_sec_sim=1):
    dt = 1 / 30
    num_steps = int((1 / dt) * num_sec_sim)

    static_T = [sim.get_transformation(i) for i in watch_ids]
    ret = True

    obs = []
    for _ in range(num_steps):
        sim.step_world(dt)
        img = sim.get_sensor_observations()["rgb"]
        img = np.flip(img, axis=1)
        obs.append(img)
        is_legal = [
            is_move_legal_dict[other_id](pos=sim.get_translation(other_id))
            for other_id in watch_ids
        ]
        if not all(is_legal):
            # Recover from a bad state.
            set_state(static_T, watch_ids, sim)
            ret = False
            break

    for obj_id in watch_ids:
        sim.set_linear_velocity(mn.Vector3(0, 0, 0), obj_id)
        sim.set_angular_velocity(mn.Vector3(0, 0, 0), obj_id)

    return ret, obs


def get_sampled_obj(
    sim,
    pos_gen,
    pos,
    obj_id,
    existing_obj_ids,
    is_move_legal_dict,
    restrict_bbs,
    should_stabilize,
    is_target,
):
    """
    - is_move_legal_dict: Mapping from obj_idx to lambda which says if position
      is valid.
    """
    timeout_tries = 100
    found = False

    # Keep trying until we get a non-overlapping placement.
    for i in range(timeout_tries):
        new_pos = pos_gen.sample(pos, obj_id)
        if new_pos == "void":
            return new_pos
        sim.set_translation(new_pos, obj_id)
        bb = get_aabb(obj_id, sim, transformed=True)

        if inter_any_bb(bb, restrict_bbs):
            continue
        if is_target:
            closest_nav = sim.pathfinder.snap_point(new_pos)
            dist = dist_2d(closest_nav, new_pos)
            if dist > MAX_OBJ_DIST:
                continue
        found = True
        break

    if not found:
        print(
            "E: Could not get a valid position for %i with %s"
            % (obj_id, str(pos_gen))
        )
        return None

    # also provide proper vertical offset
    if pos_gen.should_add_offset():
        new_pos[1] += (bb.size()[1] / 2.0) + 0.01
    return new_pos


def remove_objs(sim, obj_ids):
    for obj_id in obj_ids:
        sim.remove_object(obj_id)


@attr.s(auto_attribs=True, slots=True)
class ObjDat:
    pos: List[float]
    fname: str
    rot: mn.Quaternion
    obj_type: int

    def to_output_spec(self):
        trans = mn.Matrix4.from_(self.rot.to_matrix(), mn.Vector3(self.pos))
        trans = mag_mat_to_list(trans)
        return [self.fname, [trans, self.obj_type]]


def place_articulated_objs(obj_dats, sim, obj_ids=[]):
    art_obj_ids = []
    for i, obj_dat in enumerate(obj_dats):
        if obj_dat.obj_type == -2:
            motion_type = MotionType.DYNAMIC
            fixed_base = False
        else:
            motion_type = MotionType(obj_dat.obj_type)
            fixed_base = True

        if len(obj_ids) == 0:
            obj_id = sim.add_articulated_object_from_urdf(
                obj_dat.fname, fixed_base
            )
        else:
            obj_id = obj_ids[i]

        T = mn.Matrix4.from_(obj_dat.rot.to_matrix(), mn.Vector3(obj_dat.pos))

        sim.set_articulated_object_root_state(obj_id, T)
        sim.set_articulated_object_sleep(obj_id, True)
        sim.set_articulated_object_motion_type(obj_id, motion_type)
        art_obj_ids.append(obj_id)
    if len(obj_ids) != 0:
        return obj_ids
    return art_obj_ids


def place_static_objs(
    obj_dats,
    sim,
    on_pre_place_fn=None,
    on_post_place_fn=None,
    obj_ids=[],
    rotations={},
):
    """
    - rotations: (str -> mn.Quaternion) mapping from the name of the object to
      its rotation.
    """
    static_obj_ids = []
    for i, obj_dat in enumerate(obj_dats):
        if len(obj_ids) == 0:
            obj_id = add_obj(obj_dat.fname, sim)
        else:
            obj_id = obj_ids[i]

        if obj_dat.fname in rotations:
            sim.set_rotation(rotations[obj_dat.fname], obj_id)
        else:
            sim.set_rotation(obj_dat.rot, obj_id)

        pos = obj_dat.pos
        if on_pre_place_fn is not None:
            pos = on_pre_place_fn(
                i, obj_dat.fname, pos, obj_id, static_obj_ids, sim
            )
        if pos is None:
            # Deallocate all objects
            remove_objs(sim, static_obj_ids)
            # Failed to place the object. We need to start all over.
            return None
        use_motion_type = MotionType(obj_dat.obj_type)
        set_void = False
        if pos == "void":
            set_void = True
            # DO NOT LOAD THIS OBJECT
            sim.remove_object(obj_id)
            continue

        sim.set_translation(mn.Vector3(*pos), obj_id)
        if set_void:
            make_render_only(obj_id, sim)
        else:
            sim.set_object_motion_type(use_motion_type, obj_id)
        sim.set_linear_velocity(mn.Vector3(0, 0, 0), obj_id)
        sim.set_angular_velocity(mn.Vector3(0, 0, 0), obj_id)
        if on_post_place_fn is not None:
            on_post_place_fn(i, obj_dat.fname, pos, obj_id, sim)
        static_obj_ids.append(obj_id)
    if len(obj_ids) != 0:
        return obj_ids
    return static_obj_ids


class OrpEpisodeGen(object):
    def __init__(self, cfg, reset_interval, ftype):
        self.cfg = cfg
        self.start_cfg = self.cfg.clone()
        self.reset_cache()
        self.reset_interval = reset_interval
        self.ftype = ftype

    def get_matching_idxs(self, objs):
        noise_idxs = []
        for obj, _ in objs:
            name, idx = obj.split("|")
            rel_idx = int(idx)

            poss_idxs = [
                i for i, o in enumerate(self.cfg.obj_inits) if o[0] == name
            ]
            noise_idx = poss_idxs[rel_idx]
            noise_idxs.append(noise_idx)
        return noise_idxs

    def get_obj_inits(self, obj_list):
        """
        Extracts all the information about static objects from the yaml file.
        The format is always:
            - obj_name
            - pos
            - rot
            - obj_type
        """
        # Object domain randomization
        def get_obj_name(obj_name):
            # A file name can get appended to the end of every object.
            parts = obj_name.split("/")
            rest = "/".join(parts[:-1])
            final_obj_name = parts[-1]
            if final_obj_name in OBJ_SAMPLERS:
                choice = OBJ_SAMPLERS[final_obj_name].sample(self.ftype)
                return "/".join([rest, choice])
            return obj_name

        for obj_dat in obj_list:
            name, pos = obj_dat[:2]
            use_obj_name = get_obj_name(name)
            rot = mn.Quaternion.identity_init()
            obj_type = int(MotionType.DYNAMIC)
            if len(obj_dat) > 2:
                rot = obj_dat[2]
                if isinstance(rot, str):
                    try:
                        rot = eval(rot)
                    except Exception as e:
                        print(f"Parse error for {rot}")
                        print(e)
                        rot = mn.Quaternion.identity_init()
                elif isinstance(rot, list):
                    rot = mn.Quaternion(mn.Vector3(rot[:3]), rot[3])
                elif not isinstance(rot, mn.Quaternion):
                    raise ValueError("invalid data type for rotation")
            if len(obj_dat) > 3:
                obj_type = obj_dat[3]
            yield ObjDat(
                pos=pos, fname=use_obj_name, rot=rot, obj_type=obj_type
            )

    def place_targets(self, use_obj_inits, sim, target_ids, added_objs):
        # Compute the target bounding boxes
        target_idxs = self.get_matching_idxs(self.cfg.target_gens)
        target_bbs = []
        target_spec = []
        i = 0
        set_targ_ids = target_ids[:]
        for target_idx, (_, pos_gen) in zip(target_idxs, self.cfg.target_gens):
            obj_dat = use_obj_inits[target_idx]

            if len(target_ids) == 0:
                target_obj_id = add_obj(obj_dat.fname, sim)
                set_targ_ids.append(target_obj_id)
            else:
                target_obj_id = target_ids[i]

            # Not checking for any constraints on the target objects.
            if added_objs is not None:
                start_idx = added_objs[target_idx]
            else:
                start_idx = None
            new_pos = pos_gen.sample(obj_dat.pos, target_obj_id, start_idx)

            # make_render_only(target_obj_id, sim)
            if obj_dat.fname in SET_ROTATIONS:
                sim.set_rotation(SET_ROTATIONS[obj_dat.fname], target_obj_id)
            sim.set_translation(new_pos, target_obj_id)
            target_bbs.append(get_aabb(target_obj_id, sim, transformed=True))
        if len(set_targ_ids) != 0:
            target_ids = set_targ_ids

        return target_ids

    def init_ep(self, sim, rough_place=False):
        target_spec = []
        obj_ids = []
        target_ids = []
        start_sampled = {}

        # use the articulated objects from the previous episode if possible.
        new_obj_names = [x[0] for x in self.cfg.art_objs]
        if self.prev_art_names == new_obj_names:
            use_obj_ids = self.prev_art_objs
        else:
            for art_id in self.prev_art_objs:
                sim.remove_articulated_object(art_id)
            use_obj_ids = []

        art_objs = list(self.get_obj_inits(self.cfg.art_objs))
        art_obj_ids = place_articulated_objs(art_objs, sim, use_obj_ids)
        self.prev_art_names = new_obj_names
        self.prev_art_objs = art_obj_ids
        art_spec = [x.to_output_spec() for x in art_objs]

        init_art_objs(
            [
                (art_obj_ids[i], art_state)
                for i, art_state in self.cfg.art_states
            ],
            sim,
        )

        use_obj_inits = list(self.get_obj_inits(self.cfg.obj_inits))

        noise_idxs = self.get_matching_idxs(self.cfg.noise_gens)
        targ_idxs = self.get_matching_idxs(self.cfg.target_gens)

        def on_pre_place(i, name, pos, obj_id, existing_obj_ids, sim):
            existing_obj_bbs = [
                get_aabb(i, sim, transformed=True) for i in existing_obj_ids
            ]
            if i in noise_idxs:
                pos_gen = self.get_noise_for_idx(i)
                is_move_legal_dict = {
                    other_obj_id: partial(
                        self.get_noise_for_idx(j).is_legal
                        if j in noise_idxs
                        else def_is_legal,
                        start_pos=other_pos,
                    )
                    for j, (other_pos, other_obj_id) in start_sampled.items()
                }
                pos = get_sampled_obj(
                    sim,
                    pos_gen,
                    pos,
                    obj_id,
                    existing_obj_ids,
                    is_move_legal_dict,
                    existing_obj_bbs,
                    pos_gen.should_stabilize(),
                    i in targ_idxs,
                )

            return pos

        obj_id_to_names = {}

        def on_post_place(i, name, pos, obj_id, sim):
            start_sampled[i] = (pos, obj_id)
            obj_id_to_names[obj_id] = name

        object_spec = None
        timeout = 50

        found = False
        added_objs = []

        # TO force place, useful for debugging
        rough_place = True
        for timeout_i in range(timeout):
            added_objs = place_static_objs(
                use_obj_inits,
                sim,
                on_pre_place,
                on_post_place,
                obj_ids=added_objs,
                rotations=SET_ROTATIONS,
            )
            target_ids = self.place_targets(
                use_obj_inits, sim, target_ids, added_objs
            )

            if added_objs is None:
                added_objs = []
                continue

            # Final stability check
            is_move_legal_dict = {
                other_obj_id: partial(
                    self.get_noise_for_idx(j).is_legal
                    if j in noise_idxs
                    else def_is_legal,
                    start_pos=other_pos,
                )
                for j, (other_pos, other_obj_id) in start_sampled.items()
            }

            stable, obs = is_stable(
                sim,
                added_objs,
                is_move_legal_dict,
                num_sec_sim=self.cfg.sim_time,
            )
            if rough_place:
                # stop on the first placement.
                found = True
                break
            if stable:
                # save_mp4(obs, './data/vids/episode_gen_debug/',
                #        'good_sim_result_%i' % np.random.randint(20),
                #        fps=30, no_frame_drop=True)

                found = True
                break
            # else:
            #    save_mp4(obs, './data/vids/episode_gen_debug/',
            #            'sim_result_%i' % i,
            #            fps=30, no_frame_drop=True)

        if not found:
            return None, None, None

        obj_ids.extend(added_objs)
        obj_ids.extend(target_ids)

        object_spec = []
        for i, static_obj_id in enumerate(added_objs):
            trans = sim.get_transformation(static_obj_id)
            object_spec.append(
                (
                    obj_id_to_names[static_obj_id],
                    (mag_mat_to_list(trans), use_obj_inits[i].obj_type),
                )
            )

        targ_spec = []
        correspond_target_idxs = self.get_matching_idxs(self.cfg.target_gens)
        for (correspond_target_idx, targ_obj_id) in zip(
            correspond_target_idxs, target_ids
        ):
            trans = sim.get_transformation(targ_obj_id)
            target_spec.append((correspond_target_idx, mag_mat_to_list(trans)))

        remove_objs(sim, obj_ids)

        return art_spec, object_spec, target_spec

    def get_noise_for_idx(self, idx):
        # Gets the noise generator for the index into the objects array.
        # NOT object id.
        noise_idxs = self.get_matching_idxs(self.cfg.noise_gens)
        return self.cfg.noise_gens[noise_idxs.index(idx)][1]

    def load_set_objs(self, obj_fname, targ_ids, goal_ids):
        """
        Modify the task definition to include any preplaced figure objects.
        Note this is not actually placing anything in the scene, it is just
        modifying the definition.
        """
        if obj_fname == "":
            return
        with open(obj_fname, "rb") as f:
            data = pickle.load(f)
            fig_objs = data["objs"]

        assert len(targ_ids) == len(goal_ids)
        for targ_id, goal_id in zip(targ_ids, goal_ids):
            obj_name = fig_objs[targ_id][0]
            self.cfg.target_gens.append(
                (obj_name + "|0", ReplaceSampler(fig_objs[goal_id][1]))
            )

        for goal_id in sorted(goal_ids, reverse=True):
            del fig_objs[goal_id]
        robot_trans = mn.Matrix4(*data["robot_start"])
        x = robot_trans.transposed().translation
        self.cfg.start_pos = [x[0], x[2]]

        for i in range(len(fig_objs)):
            fig_objs[i][1] = list(fig_objs[i][1].astype(float))
            fig_objs[i] = tuple(fig_objs[i])

        self.cfg.obj_inits.extend(fig_objs)

    def reset_cache(self):
        self.prev_art_objs = []
        self.prev_art_names = []

    def _get_ep_dict(
        self,
        episode_id,
        scene_config_path,
        art_spec,
        object_spec,
        target_spec,
        sim,
        scene_name,
    ):
        return {
            "episode_id": episode_id,
            "scene_id": scene_name,
            "scene_config_path": scene_config_path,
            "start_position": self.cfg.start_pos,
            "start_rotation": 180,
            "allowed_region": self.cfg.allowed_region,
            "fixed_base": self.cfg.fixed_base,
            "art_objs": art_spec,
            "static_objs": object_spec,
            "art_states": self.cfg.art_states,
            "targets": target_spec,
            "nav_mesh_path": self.cfg.nav_mesh_path,
            "markers": self.cfg.markers,
            "force_spawn_pos": self.cfg.force_spawn_pos,
        }

    def _on_reset(self, sim):
        pass

    def reset_samplers(self, sim):
        for _, noise_gen in [*self.cfg.target_gens, *self.cfg.noise_gens]:
            noise_gen.reset_balance()
            noise_gen.set_sim(sim)

    def get_scene_sampler(self):
        if osp.exists(self.cfg.scene_name):
            return SingleSceneSampler(self.cfg.scene_name)
        scene_sampler = eval(self.cfg.scene_name)
        return scene_sampler

    def fill_scene_details(self, mdat, sim, cur_scene_name):
        self.cfg = self.start_cfg.clone()
        could_get_navmesh = self._load_navmesh(
            self.cfg.nav_mesh_path, sim, cur_scene_name
        )
        if not could_get_navmesh:
            return None
        self.cfg = eval_cfg(self.cfg, mdat, sim)

        return self.cfg

    def _load_navmesh(self, desired_path, sim, cur_scene_name):
        navmesh_settings = get_nav_mesh_settings_from_height(1.5)
        sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
        return True

    def generate_scene(self, scene_sampler, sim):
        cur_scene_name, metadata = scene_sampler.sample()
        if sim is not None:
            sim.close(destroy=True)
            del sim
        self.reset_cache()
        sim = get_sim(self, cur_scene_name)
        result = self.fill_scene_details(metadata, sim, cur_scene_name)
        if result is None:
            return cur_scene_name, False, sim

        self._on_reset(sim)
        print("Reset the simulator")
        print("Using scene name ", cur_scene_name)
        return cur_scene_name, True, sim

    def gen_episodes(self, num_eps, scene_config_path):
        episodes = []
        scene_sampler = self.get_scene_sampler()
        if scene_sampler.num_scenes() >= num_eps:
            switch_interval = scene_sampler.num_scenes()
        elif scene_sampler.num_scenes() != 0:
            switch_interval = num_eps // scene_sampler.num_scenes()
        else:
            switch_interval = 1
        if self.reset_interval is not None:
            reset_interval = self.reset_interval
        else:
            reset_interval = switch_interval

        sim = None
        cur_scene_name = None
        episode_id = 0
        problem_scenes = []

        with tqdm(total=num_eps) as pbar:
            while episode_id < num_eps:
                if (episode_id) % reset_interval == 0:
                    was_success = False
                    timeout = 10
                    scene_try_i = 0
                    while not was_success:
                        scene_try_i += 1
                        if scene_try_i > timeout:
                            raise ValueError("Problem processing scenes")
                        cur_scene_name, was_success, sim = self.generate_scene(
                            scene_sampler, sim
                        )
                        if not was_success:
                            print(cur_scene_name, "was a problem")
                            problem_scenes.append("+" + cur_scene_name)

                if (episode_id + 1) % 10 == 0:
                    print(f"[{scene_config_path}]")

                self.reset_samplers(sim)

                object_spec = None
                timeout = 3

                for _ in range(timeout):
                    art_spec, object_spec, target_spec = self.init_ep(sim)
                    if object_spec is not None:
                        break
                if object_spec is None:
                    print(cur_scene_name + " could be problematic")
                    problem_scenes.append(cur_scene_name)
                    continue

                if not sim.get_existing_object_ids() != 0:
                    raise ValueError("Did not clear all objects!")
                episodes.append(
                    self._get_ep_dict(
                        episode_id,
                        scene_config_path,
                        art_spec,
                        object_spec,
                        target_spec,
                        sim,
                        cur_scene_name,
                    )
                )
                episode_id += 1
                pbar.update(1)

        with open(
            "data/debug_scene_%i.txt" % np.random.randint(100), "w"
        ) as f:
            f.writelines(problem_scenes)
        return episodes


def make_test_cfg(task_gen, scene_name):
    camera_resolution = [540, 720]
    start_p = task_gen.cfg.start_pos
    if start_p is None:
        start_p = [0, 0]
    sensors = {
        "rgb": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": camera_resolution,
            # You will have to manually tweak the position to look where you
            # want for debugging.
            # "position": [-0.6, 1.5, start_p[1]],
            # "orientation": [0, np.pi/2, np.pi/6],
            "position": [2.4, 1.5, 2.5],
            "orientation": [0, np.pi, 0.0],
            # "position": [2.4, 1.5, 2.5],
            # "orientation": [0, np.pi/2, np.pi/6],
        }
    }
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_name
    backend_cfg.enable_physics = True

    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        sensor_spec = habitat_sim.CameraSensorSpec()
        sensor_spec.uuid = sensor_uuid
        sensor_spec.sensor_type = sensor_params["sensor_type"]
        sensor_spec.resolution = sensor_params["resolution"]
        sensor_spec.position = sensor_params["position"]
        sensor_spec.orientation = sensor_params["orientation"]
        sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(sensor_spec)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


def get_config_defaults():
    _C = CN()
    _C.art_objs = []
    _C.art_states = []
    _C.obj_inits = []
    _C.noise_inits = []
    _C.target_gens = []
    _C.noise_def = []
    _C.fixed_base = True
    _C.allowed_region = []
    _C.start_pos = [0, 0]
    _C.urdf_base = ""
    _C.obj_base = ""
    _C.nav_mesh_path = None
    _C.noise_gens = []
    _C.markers = []
    _C.scene_name = "./orp/start_data/frl_apartment_stage_pvizplan_full.glb"
    _C.sim_time = 5
    _C.force_spawn_pos = None
    return _C.clone()


def eval_cfg(cfg, mdat, sim):
    dyn_surfaces = get_dyn_samplers(mdat)
    cfg.defrost()
    for name, ndef in cfg.noise_def:
        exec("%s = %s" % (name, ndef))

    for i in range(len(cfg.target_gens)):
        cfg.target_gens[i] = (
            cfg.target_gens[i][0],
            eval(cfg.target_gens[i][1]),
        )

    for i in range(len(cfg.art_objs)):
        if isinstance(cfg.art_objs[i][1], str):
            if cfg.art_objs[i][1].startswith("rnd|"):
                set_height = cfg.art_objs[i][1].split("|")[1]
                rnd_start = sim.pathfinder.get_random_navigable_point()
                rnd_start[1] = set_height
                cfg.art_objs[i][1] = rnd_start
            else:
                cfg.art_objs[i][1] = eval(cfg.art_objs[i][1])
            if len(cfg.art_objs[i]) > 2:
                # Rotation
                cfg.art_objs[i][2] = eval(cfg.art_objs[i][2])

    for i in range(len(cfg.noise_gens)):
        cfg.noise_gens[i] = (cfg.noise_gens[i][0], eval(cfg.noise_gens[i][1]))

    def is_clutter_gen(x):
        if isinstance(x, list):
            return False
        return x in CLUTTER_GENS

    clutter_gens = [x for x in cfg.obj_inits if is_clutter_gen(x)]
    cfg.obj_inits = [x for x in cfg.obj_inits if not is_clutter_gen(x)]

    name_counts = defaultdict(lambda: 0)
    for x in cfg.obj_inits:
        name_counts[x[0]] += 1

    # Convert clutter gens to individual gens
    for clutter_name in clutter_gens:
        obj_init, noise_gen = CLUTTER_GENS[clutter_name](
            name_counts, mdat, dyn_surfaces
        )
        cfg.obj_inits.extend(obj_init)
        cfg.noise_gens.extend(noise_gen)

    cfg.art_objs = [
        [osp.join(cfg.urdf_base, x[0]), *x[1:]] for x in cfg.art_objs
    ]
    cfg.obj_inits = [
        [osp.join(cfg.obj_base, x[0]), *x[1:]] for x in cfg.obj_inits
    ]
    cfg.noise_gens = [
        (osp.join(cfg.obj_base, a), b) for a, b in cfg.noise_gens
    ]
    cfg.target_gens = [
        (osp.join(cfg.obj_base, a), b) for a, b in cfg.target_gens
    ]
    cfg.freeze()

    return cfg


def get_sim(task_gen, scene_name):
    sim = habitat_sim.Simulator(make_test_cfg(task_gen, scene_name))
    obj_attr_mgr = sim.get_object_template_manager()
    obj_attr_mgr.load_configs("data/objects")
    return sim


def def_get_episode_gen(cfg, reset_interval, ftype):
    return OrpEpisodeGen(cfg, reset_interval, ftype)


def create_episode_dataset(
    scene_config_paths,
    all_num_eps,
    out_name,
    ftype,
    obj_fname="",
    targ_ids=[],
    goal_ids=[],
    reset_interval=None,
    get_episode_gen=def_get_episode_gen,
):
    ep_folder = osp.join("data/episode_data")
    episodes = []
    use_out_name = "%s_%s" % (ftype, out_name)
    cache_path = osp.join(rutils.CACHE_PATH, use_out_name)
    if osp.exists(cache_path):
        shutil.rmtree(cache_path)

    for num_eps, scene_config_path in zip(all_num_eps, scene_config_paths):
        cfg = get_config_defaults()
        cfg.merge_from_file(scene_config_path)

        task_gen = get_episode_gen(cfg, reset_interval, ftype)

        task_gen.load_set_objs(obj_fname, targ_ids, goal_ids)
        episodes.extend(task_gen.gen_episodes(num_eps, scene_config_path))
        if not osp.exists(ep_folder):
            os.makedirs(ep_folder)
    episodes = {"episodes": episodes}

    ep_file = osp.join(ep_folder, use_out_name + ".json.gz")

    with gzip.open(ep_file, "wt") as f:
        json.dump(episodes, f)

    print("Saved to ", ep_file)
    return ep_file
