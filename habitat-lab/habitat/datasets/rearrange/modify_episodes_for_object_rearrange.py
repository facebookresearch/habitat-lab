import argparse
import ctypes
import gzip
import json
import os
import os.path as osp
import pickle
import re
import sys
from typing import Any, Dict, List, Set

flags = sys.getdlopenflags()
sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)

import magnum as mn
import numpy as np
import pandas as pd
from tqdm import tqdm

import habitat_sim
from habitat.core.utils import DatasetFloatJSONEncoder
from habitat.datasets.rearrange.navmesh_utils import (
    compute_navmesh_island_classifications,
)
from habitat.datasets.rearrange.samplers.receptacle import find_receptacles
from habitat.datasets.rearrange.viewpoints import (
    generate_viewpoints,
    populate_semantic_graph,
)
from habitat_sim.agent.agent import ActionSpec
from habitat_sim.agent.controls import ActuationSpec


def get_rec_category(rec_id, rec_category_mapping=None):
    """
    Retrieves the receptacle category from id
    """
    rec_id = rec_id.rstrip("_")
    if rec_category_mapping is None:
        # for replicaCAD objects
        rec_id = rec_id.rstrip("_").replace("frl_apartment_", "")
        return re.sub(r"_[0-9]+", "", rec_id)
    elif rec_id in rec_category_mapping:
        return rec_category_mapping[rec_id]
    else:
        rec_category = rec_id
        remove_strings = [
            "receptacle",
            "kitchen",
            "living",
            "dining",
            "hall",
            "office",
            "bath",
            "closet",
            "laundry",
            "coffee",
            "garage",
            "bd",
            "deck",
            "half",
            "upper",
            "side",
            "top",
            "left",
            "right",
            "corner",
            "midle",
            "middle",
            "round",
            "patio",
            "lower",
            "aabb",
            "mesh",
        ]
        for r in remove_strings:
            rec_category = rec_category.replace(r, "")
        rec_category = re.sub("_+", "_", rec_category).lstrip("_").rstrip("_")
        rec_category = re.sub("[.0-9]", "", rec_category)
        print(
            f"Warning: {rec_id} not found in receptacle category mapping. Using {rec_category} as the category."
        )
        return rec_category


def get_obj_category(obj_name, obj_category_mapping=None):
    """
    Retrieves the object category from id
    """
    if obj_category_mapping is None:
        # for YCB objects
        return obj_name[4:]
    elif obj_name in obj_category_mapping:
        return obj_category_mapping[obj_name]
    else:
        raise ValueError(f"Cannot find matching category for {obj_name}")


def get_obj_rec_cat_in_eps(
    episode, obj_category_mapping=None, rec_category_mapping=None
):
    """
    Returns the category names for the object to be picked up, start and goal receptacles
    """
    obj_cat = get_obj_category(
        list(episode["info"]["object_labels"].keys())[0].split(":")[0][:-1],
        obj_category_mapping=obj_category_mapping,
    )
    start_rec_cat = get_rec_category(
        episode["target_receptacles"][0][0].split(":")[0],
        rec_category_mapping=rec_category_mapping,
    )
    goal_rec_cat = get_rec_category(
        episode["goal_receptacles"][0][0].split(":")[0],
        rec_category_mapping=rec_category_mapping,
    )
    return obj_cat, start_rec_cat, goal_rec_cat


def read_obj_category_mapping(filename, keep_only_recs=False):
    """
    Returns a mapping from object name to category.
    Setting keep_only_recs to True keeps only receptacles.
    """
    df = pd.read_csv(filename)
    name_key = "id" if "id" in df else "name"
    category_key = (
        "main_category" if "main_category" in df else "clean_category"
    )

    if keep_only_recs:
        df = df[df["hasReceptacles"] == True]

    df["category"] = (
        df[category_key]
        .fillna("")
        .apply(lambda x: x.replace(" ", "_").split(".")[0])
    )
    return dict(zip(df[name_key], df["category"]))


def get_cats_list(obj_category_mapping=None, rec_category_mapping=None):
    """
    Extracts the lists of object and receptacle categories from episodes and returns name to id mappings
    """
    cat_to_id_mappings = []
    for mapping in [obj_category_mapping, rec_category_mapping]:
        cats = sorted(set(mapping.values()))
        cat_to_id_mappings.append({cat: i for i, cat in enumerate(cats)})
    return cat_to_id_mappings


def collect_receptacle_goals(sim, receptacles, rec_category_mapping=None):
    recep_goals = []
    rom = sim.get_rigid_object_manager()
    aom = sim.get_articulated_object_manager()

    for receptacle in receptacles:
        if receptacle.parent_object_handle is None:
            object_id = -1
        elif aom.get_library_has_handle(receptacle.parent_object_handle):
            object_id = aom.get_object_id_by_handle(
                receptacle.parent_object_handle
            )
        elif rom.get_library_has_handle(receptacle.parent_object_handle):
            object_id = rom.get_object_id_by_handle(
                receptacle.parent_object_handle
            )
        else:
            object_id = -1
        pos = receptacle.get_surface_center(sim)
        rec_name = receptacle.parent_object_handle
        # if the handle name is None, try extracting category from receptacle name
        rec_name = (
            receptacle.name if rec_name is None else rec_name.split(":")[0]
        )
        recep_goals.append(
            {
                "position": [pos.x, pos.y, pos.z],
                "object_name": receptacle.name,
                "object_id": str(object_id),
                "object_category": get_rec_category(
                    rec_name,
                    rec_category_mapping=rec_category_mapping,
                ),
                "view_points": [],
            }
        )

    return recep_goals


def initialize_sim(
    sim,
    existing_rigid_objects: Set,
    scene_name: str,
    dataset_path: str,
    additional_obj_config_paths: List[str],
    debug_viz: bool = False,
) -> habitat_sim.Simulator:
    """
    Initialize a new Simulator object with a selected scene and dataset.
    """
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_dataset_config_file = dataset_path
    backend_cfg.scene_id = scene_name
    backend_cfg.create_renderer = True
    backend_cfg.enable_physics = True

    if debug_viz:
        sensors = {
            "semantic": {
                "sensor_type": habitat_sim.SensorType.SEMANTIC,
                "resolution": [512, 512],
                "position": [0, 0, 0],
                "orientation": [0, 0, 0.0],
            },
            "color": {
                "sensor_type": habitat_sim.SensorType.COLOR,
                "resolution": [512, 512],
                "position": [0, 0, 0],
                "orientation": [0, 0, 0.0],
            },
        }
    else:
        sensors = {
            "semantic": {
                "sensor_type": habitat_sim.SensorType.SEMANTIC,
                "resolution": [256, 256],
                "position": [0, 0, 0],
                "orientation": [0, 0, 0.0],
            },
        }
    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        sensor_spec = habitat_sim.CameraSensorSpec()
        sensor_spec.uuid = sensor_uuid
        sensor_spec.sensor_type = sensor_params["sensor_type"]
        sensor_spec.resolution = sensor_params["resolution"]
        sensor_spec.position = sensor_params["position"]
        sensor_spec.orientation = sensor_params["orientation"]
        sensor_spec.sensor_subtype = habitat_sim.SensorSubType.EQUIRECTANGULAR
        sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(sensor_spec)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "look_up": ActionSpec("look_up", ActuationSpec(amount=10.0)),
        "look_down": ActionSpec("look_down", ActuationSpec(amount=10.0)),
    }

    hab_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    if sim is None:
        sim = habitat_sim.Simulator(hab_cfg)
        object_attr_mgr = sim.get_object_template_manager()
        for object_path in additional_obj_config_paths:
            object_attr_mgr.load_configs(osp.abspath(object_path))
    else:
        if sim.config.sim_cfg.scene_id == scene_name:
            rom = sim.get_rigid_object_manager()
            for obj in rom.get_object_handles():
                if obj not in existing_rigid_objects:
                    rom.remove_object_by_handle(obj)
        else:
            sim.reconfigure(hab_cfg)

    return sim


def get_matching_recep_handle(receptacle_name, handles):
    return [handle for handle in handles if handle in receptacle_name][0]


def get_candidate_starts(
    objects,
    category,
    start_rec_cat,
    obj_idx_to_name,
    name_to_receptacle,
    obj_category_mapping=None,
    rec_category_mapping=None,
    rec_to_parent_obj=None,
):
    obj_goals_hard = []
    obj_goals_easy = []
    for i, (obj, pos) in enumerate(objects):
        if (
            get_obj_category(
                obj.split(".")[0], obj_category_mapping=obj_category_mapping
            )
            == category
        ):
            obj_goal = {
                "position": np.array(pos)[:3, 3].tolist(),
                "object_name": obj,
                "object_id": str(i),
                "object_category": category,
                "view_points": [],
            }
            obj_goals_hard.append(obj_goal)
            # use parent object name if it exists
            rec_name = rec_to_parent_obj[
                name_to_receptacle[obj_idx_to_name[i]]
            ]
            # otherwise use receptacle name
            if rec_name is None:
                rec_name = name_to_receptacle[obj_idx_to_name[i]]
            recep_cat = get_rec_category(
                rec_name.split(":")[0],
                rec_category_mapping=rec_category_mapping,
            )
            if recep_cat == start_rec_cat:
                obj_goals_easy.append(obj_goal)
    return obj_goals_easy, obj_goals_hard


def get_candidate_receptacles(recep_goals, goal_recep_category):
    return [
        g for g in recep_goals if g["object_category"] == goal_recep_category
    ]


def load_receptacle_cache(scene, cache_dir, cache):
    if scene in cache:
        return cache

    scene_name = osp.basename(scene).split(".")[0]
    receptacle_cache_path = osp.join(cache_dir, scene_name + ".pkl")
    assert osp.exists(
        receptacle_cache_path
    ), f"Receptacle viewpoints cache file missing: {receptacle_cache_path}"
    with open(receptacle_cache_path, "rb") as f:
        cache[scene] = pickle.load(f)
    return cache


def load_objects(sim, objects, additional_obj_config_paths):
    rom = sim.get_rigid_object_manager()
    obj_idx_to_name = {}
    for i, (obj_handle, transform) in enumerate(objects):
        template = None
        for obj_path in additional_obj_config_paths:
            template = osp.abspath(osp.join(obj_path, obj_handle))
            if osp.isfile(template):
                break
        assert (
            template is not None
        ), f"Could not find config file for object {obj_handle}"
        ro = rom.add_object_by_template_handle(template)

        # The saved matrices need to be flipped when reloading.
        ro.transformation = mn.Matrix4(
            [[transform[j][i] for j in range(4)] for i in range(4)]
        )
        ro.angular_velocity = mn.Vector3.zero_init()
        ro.linear_velocity = mn.Vector3.zero_init()
        obj_idx_to_name[i] = ro.handle
    return obj_idx_to_name


def add_viewpoints(
    sim,
    episode,
    obj_idx_to_name,
    rec_viewpoints,
    rec_to_parent_obj,
    debug_viz=False,
):
    rom = sim.get_rigid_object_manager()

    for obj in episode["candidate_objects"]:
        obj_handle = obj_idx_to_name[int(obj["object_id"])]
        sim_obj = rom.get_object_by_handle(obj_handle)
        obj["view_points"] = generate_viewpoints(
            sim, sim_obj, debug_viz=debug_viz
        )
        if len(obj["view_points"]) == 0:
            print("Need at least 1 view point for object")
            return False

    for rec in episode["candidate_goal_receps"]:
        rec_handle = rec_to_parent_obj[rec["object_name"]]
        rec["view_points"] = rec_viewpoints[rec_handle]

    return True


def load_navmesh(sim, scene):
    scene_base_dir = osp.dirname(osp.dirname(scene))
    scene_name = osp.basename(scene).split(".")[0]
    navmesh_path = osp.join(
        scene_base_dir, "navmeshes", scene_name + ".navmesh"
    )
    if osp.exists(navmesh_path):
        sim.pathfinder.load_nav_mesh(navmesh_path)
    else:
        raise RuntimeError(
            f"No navmesh found for scene {scene}, please generate one."
        )
    compute_navmesh_island_classifications(sim)


def add_cat_fields_to_episodes(
    episodes_file,
    obj_to_id,
    rec_to_id,
    rec_cache_dir,
    num_episodes,
    obj_category_mapping=None,
    rec_category_mapping=None,
    enable_add_viewpoints=False,
    debug_viz=False,
):
    """
    Adds category fields to episodes
    """
    episodes = json.load(gzip.open(episodes_file))
    episodes["obj_category_to_obj_category_id"] = obj_to_id
    episodes["recep_category_to_recep_category_id"] = rec_to_id
    receptacle_cache: Dict[str, Any] = {}

    sim = None
    sim = initialize_sim(
        sim,
        set(),
        episodes["episodes"][0]["scene_id"],
        episodes["episodes"][0]["scene_dataset_config"],
        episodes["episodes"][0]["additional_obj_config_paths"],
        debug_viz,
    )

    rom = sim.get_rigid_object_manager()
    existing_rigid_objects = set(rom.get_object_handles())

    new_episodes = {k: v for k, v in episodes.items()}
    new_episodes["episodes"] = []

    if num_episodes == -1:
        num_episodes = len(episodes["episodes"])

    with tqdm(total=num_episodes) as pbar:
        for episode in episodes["episodes"]:
            scene_id = episode["scene_id"]
            sim = initialize_sim(
                sim,
                existing_rigid_objects,
                scene_id,
                episode["scene_dataset_config"],
                episode["additional_obj_config_paths"],
                debug_viz,
            )
            load_navmesh(sim, scene_id)

            rom = sim.get_rigid_object_manager()
            existing_rigid_objects = set(rom.get_object_handles())

            rec = find_receptacles(sim)
            rec_viewpoints, _ = load_receptacle_cache(
                scene_id, rec_cache_dir, receptacle_cache
            )[scene_id]
            rec = [r for r in rec if r.parent_object_handle in rec_viewpoints]
            rec_to_parent_obj = {r.name: r.parent_object_handle for r in rec}
            obj_idx_to_name = load_objects(
                sim,
                episode["rigid_objs"],
                episode["additional_obj_config_paths"],
            )
            populate_semantic_graph(sim)
            all_rec_goals = collect_receptacle_goals(
                sim, rec, rec_category_mapping=rec_category_mapping
            )
            obj_cat, start_rec_cat, goal_rec_cat = get_obj_rec_cat_in_eps(
                episode,
                obj_category_mapping=obj_category_mapping,
                rec_category_mapping=rec_category_mapping,
            )

            if start_rec_cat == goal_rec_cat:
                continue

            episode["object_category"] = obj_cat
            episode["start_recep_category"] = start_rec_cat
            name_to_receptacle = {
                k: v.split("|", 1)[1]
                for k, v in episode["name_to_receptacle"].items()
            }
            try:
                episode["goal_recep_category"] = goal_rec_cat
                (
                    episode["candidate_objects"],
                    episode["candidate_objects_hard"],
                ) = get_candidate_starts(
                    episode["rigid_objs"],
                    obj_cat,
                    start_rec_cat,
                    obj_idx_to_name,
                    name_to_receptacle,
                    obj_category_mapping=obj_category_mapping,
                    rec_category_mapping=rec_category_mapping,
                    rec_to_parent_obj=rec_to_parent_obj,
                )
            except KeyError:
                print("Skipping episode", episode["episode_id"])
                continue
            episode["candidate_start_receps"] = get_candidate_receptacles(
                all_rec_goals, start_rec_cat
            )
            episode["candidate_goal_receps"] = get_candidate_receptacles(
                all_rec_goals, goal_rec_cat
            )
            if enable_add_viewpoints and not add_viewpoints(
                sim,
                episode,
                obj_idx_to_name,
                rec_viewpoints,
                rec_to_parent_obj,
                debug_viz,
            ):
                print("Skipping episode", episode["episode_id"])
                continue
            assert (
                len(episode["candidate_objects"]) > 0
                and len(episode["candidate_start_receps"]) > 0
                and len(episode["candidate_goal_receps"]) > 0
            )
            new_episodes["episodes"].append(episode)
            pbar.update()
            if len(new_episodes["episodes"]) >= num_episodes:
                break
    return new_episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_data_dir",
        type=str,
        default="data/datasets/replica_cad/rearrange/v1/",
    )
    parser.add_argument(
        "--source_episodes_tag", type=str, default="rearrange_easy"
    )
    parser.add_argument(
        "--target_data_dir",
        type=str,
        default="data/datasets/replica_cad/rearrange/v1/",
    )
    parser.add_argument(
        "--target_episodes_tag", type=str, default="categorical_rearrange_easy"
    )
    parser.add_argument(
        "--rec_cache_dir",
        type=str,
        default="data/cache/receptacle_viewpoints/fphab",
        help="Path to cache where receptacle viewpoints were saved during first stage of episode generation",
    )
    parser.add_argument("--obj_category_mapping_file", type=str, default=None)
    parser.add_argument("--rec_category_mapping_file", type=str, default=None)
    parser.add_argument(
        "--num_episodes", type=int, default=-1
    )  # -1 uses all episodes
    parser.add_argument("--add_viewpoints", action="store_true")
    parser.add_argument("--debug_viz", action="store_true")

    args = parser.parse_args()

    source_data_dir = args.source_data_dir
    source_episodes_tag = args.source_episodes_tag
    target_data_dir = args.target_data_dir
    target_episodes_tag = args.target_episodes_tag
    rec_cache_dir = args.rec_cache_dir
    num_episodes = args.num_episodes

    obj_category_mapping = None
    if args.obj_category_mapping_file is not None:
        obj_category_mapping = read_obj_category_mapping(
            args.obj_category_mapping_file
        )

    rec_category_mapping = None
    if args.rec_category_mapping_file is not None:
        rec_category_mapping = read_obj_category_mapping(
            args.rec_category_mapping_file, keep_only_recs=True
        )

    obj_to_id, rec_to_id = get_cats_list(
        obj_category_mapping=obj_category_mapping,
        rec_category_mapping=rec_category_mapping,
    )
    print(f"Number of object categories: {len(obj_to_id)}")
    print(f"Number of receptacle categories: {len(rec_to_id)}")
    print(obj_to_id, rec_to_id)
    # Add category fields and save episodes
    for split in os.listdir(source_data_dir):
        episodes_file = osp.join(
            source_data_dir, split, f"{source_episodes_tag}.json.gz"
        )
        if not osp.exists(episodes_file):
            continue
        episodes = add_cat_fields_to_episodes(
            episodes_file,
            obj_to_id,
            rec_to_id,
            rec_cache_dir,
            num_episodes,
            obj_category_mapping=obj_category_mapping,
            rec_category_mapping=rec_category_mapping,
            enable_add_viewpoints=args.add_viewpoints,
            debug_viz=args.debug_viz,
        )

        print(f"Number of episodes in {split}: {len(episodes['episodes'])}")
        episodes_json = DatasetFloatJSONEncoder().encode(episodes)
        os.makedirs(osp.join(target_data_dir, split), exist_ok=True)
        target_episodes_file = osp.join(
            target_data_dir, split, f"{target_episodes_tag}.json.gz"
        )
        with gzip.open(target_episodes_file, "wt") as f:
            f.write(episodes_json)
    print("All episodes written")
