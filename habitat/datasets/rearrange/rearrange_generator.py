import os.path as osp
import pickle
from collections import defaultdict
from functools import partial
from typing import Generator

import attr
import magnum as mn
import numpy as np
import rlf.rl.utils as rutils
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
from habitat.datasets.obj_placer import (
    get_sampled_obj,
    is_stable,
    place_articulated_objs,
    place_static_objs,
)
from habitat.datasets.rearrange.rearrange_dataset import RearrangeEpisode
from habitat.datasets.rearrange.samplers import did_object_fall
from habitat.tasks.rearrange.obj_loaders import add_obj, init_art_objs
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


class OrpEpisodeGen(object):
    def __init__(self, cfg, reset_interval, ftype):
        self.cfg = cfg
        self.start_cfg = self.cfg.clone()
        self.ftype = ftype

        # Get the scene sampler information
        self._scene_sampler = self.get_scene_sampler()
        if self._scene_sampler.num_scenes() >= num_eps:
            self._switch_interval = self._scene_sampler.num_scenes()
        elif self._scene_sampler.num_scenes() != 0:
            self._switch_interval = num_eps // self._scene_sampler.num_scenes()
        else:
            self._switch_interval = 1

        # Get the object sampler information
        self._obj_type_samplers = self._get_obj_samplers()

    def _get_obj_samplers(self):
        obj_samplers = {}
        for obj_sampler_name, info in self.cfg.BASE_CFG.OBJ_SAMPLERS.items():
            obj_samplers[obj_sampler_name] = ObjTypeSampler(
                info.included, info.holdout
            )
        return obj_samplers

    def _get_obj_name(self, obj_name):
        # A file name can get appended to the end of every object.
        parts = obj_name.split("/")
        rest = "/".join(parts[:-1])
        final_obj_name = parts[-1]
        if final_obj_name in self._obj_type_samplers:
            choice = self._obj_type_samplers[final_obj_name].sample(self.ftype)
            return "/".join([rest, choice])
        return obj_name

    def _get_matching_idxs(self, objs):
        noise_idxs = []
        for obj, _ in objs:
            name, idx = obj.split("|")
            rel_idx = int(idx)

            poss_idxs = [
                i for i, o in enumerate(self.cfg.obj_gens) if o[0] == name
            ]
            noise_idx = poss_idxs[rel_idx]
            noise_idxs.append(noise_idx)
        return noise_idxs

    def get_obj_inits(self, obj_list):
        """
        Extracts all the information about static objects from the yaml file.
        The format is always:
            - obj_name
            - position_generator
            - rot
            - obj_type
        """
        for obj_dat in obj_list:
            name, pos = obj_dat[:2]
            pos_generator = None
            use_obj_name = self._get_obj_name(name)
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
                pos_generator=pos_generator,
                fname=use_obj_name,
                rot=rot,
                obj_type=obj_type,
            )

    def place_targets(self, use_obj_inits, sim, target_ids, added_objs):
        # Compute the target bounding boxes
        target_idxs = self._get_matching_idxs(self.cfg.target_gens)
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

            sim.set_translation(new_pos, target_obj_id)
            target_bbs.append(get_aabb(target_obj_id, sim, transformed=True))
        if len(set_targ_ids) != 0:
            target_ids = set_targ_ids

        return target_ids

    def _get_place_pos(self, i, name, obj_id, existing_obj_ids, sim):
        existing_obj_bbs = [
            get_aabb(i, sim, transformed=True) for i in existing_obj_ids
        ]
        # Resample the position
        pos_gen = self.get_noise_for_idx(i)
        is_move_legal_dict = {
            other_obj_id: partial(
                self.get_noise_for_idx(j).is_legal,
                start_pos=other_pos,
            )
            for j, (other_pos, other_obj_id) in start_sampled.items()
        }
        return get_sampled_obj(
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

    def _on_post_place(self, i, name, pos, obj_id, sim):
        start_sampled[i] = (pos, obj_id)
        self._obj_id_to_names[obj_id] = name

    def _setup_episode(self, sim):
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

        self._use_obj_inits = list(self.get_obj_inits(self.cfg.obj_inits))
        self._noise_idxs = self._get_matching_idxs(self.cfg.noise_gens)
        self._targ_idxs = self._get_matching_idxs(self.cfg.target_gens)
        self._obj_id_to_names = {}

        object_spec = None
        timeout = 50

        found = False
        added_objs = []

        for timeout_i in range(timeout):
            added_objs = place_static_objs(
                self._use_obj_inits,
                sim,
                self._on_pre_place,
                self._on_post_place,
                obj_ids=added_objs,
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
                    if j in self._noise_idxs
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
            if stable:
                found = True
                break

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
        correspond_target_idxs = self._get_matching_idxs(self.cfg.target_gens)
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
        return self.cfg.noise_gens[idx][1]

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

    def _on_reset(self, sim):
        pass

    def _reset_samplers(self, sim):
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
        sim = get_sim(self, cur_scene_name)
        result = self.fill_scene_details(metadata, sim, cur_scene_name)
        if result is None:
            return cur_scene_name, False, sim

        self._on_reset(sim)
        print("Reset the simulator")
        print("Using scene name ", cur_scene_name)
        return cur_scene_name, True, sim

    def gen_episode(self):
        sim = None

        if self._num_episodes % self._switch_interval == 0:
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
                    return None

        self._reset_samplers(sim)

        object_spec = None
        timeout = 3

        for _ in range(timeout):
            art_spec, object_spec, target_spec = self._setup_episode(sim)
            if object_spec is not None:
                break
        if object_spec is None:
            return None

        if not sim.get_existing_object_ids() != 0:
            raise ValueError("Did not clear all objects!")
        self._num_episodes += 1

        return RearrangeEpisode(
            art_objs=art_spec,
            static_objs=object_spec,
            targets=target_spec,
            scene_confg_path=scene_config_path,
            alowed_region=self.cfg.allowed_region,
            markers=self.cfg.markers,
        )


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

    hab_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(hab_cfg)
    obj_attr_mgr = sim.get_object_template_manager()
    obj_attr_mgr.load_configs("data/objects")
    return sim


# def create_episode_dataset(scene_config_paths, all_num_eps, out_name, ftype,
#        obj_fname="", targ_ids=[], goal_ids=[], reset_interval=None,
#        get_episode_gen=def_get_episode_gen):


def generate_rearrange_episode(
    sim: "HabitatSim",
    num_episodes: int,
    scene_cfg_path: str,
) -> Generator[RearrangeEpisode, None, None]:

    cfg = get_config_defaults()
    cfg.merge_from_file(ecene_cfg_path)

    ep_gen = OrpEpisodeGen(cfg, ftype)

    task_gen.load_set_objs(obj_fname, targ_ids, goal_ids)
    task_gen.gen_episodes(num_eps, scene_config_path)
