import argparse
import gzip
import json
import os
import os.path as osp
import re

import numpy as np
import pandas as pd

from habitat.core.utils import DatasetFloatJSONEncoder


def get_rec_category(rec_id, rec_category_mapping=None):
    """
    Retrieves the receptacle category from id
    """
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
            "left",
            "right",
            "corner",
            "midle",
            "middle",
            "lower",
        ]
        for r in remove_strings:
            rec_category = rec_category.replace(r, "")
        rec_category = re.sub("_+", "_", rec_category).lstrip("_").rstrip("_")
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


def read_obj_category_mapping(filename):
    df = pd.read_csv(filename)
    df["category"] = df["clean_category"].apply(lambda x: x.replace(" ", "_"))
    return dict(zip(df["name"], df["category"]))


def get_cats_list(
    train_episodes_file, obj_category_mapping=None, rec_category_mapping=None
):
    """
    Extracts the lists of object and receptacle categories from episodes and returns name to id mappings
    """
    episodes = json.load(gzip.open(train_episodes_file))
    obj_cats, rec_cats = set(), set()
    o = list()
    for episode in episodes["episodes"]:
        obj_cat, start_rec_cat, goal_rec_cat = get_obj_rec_cat_in_eps(
            episode,
            obj_category_mapping=obj_category_mapping,
            rec_category_mapping=rec_category_mapping,
        )
        obj_cats.add(obj_cat)
        rec_cats.add(start_rec_cat)
        rec_cats.add(goal_rec_cat)
        o.append(obj_cat)
    obj_cats = sorted(obj_cats)
    rec_cats = sorted(rec_cats)
    obj_to_id_mapping = {cat: i for i, cat in enumerate(obj_cats)}
    rec_to_id_mapping = {cat: i for i, cat in enumerate(rec_cats)}
    return obj_to_id_mapping, rec_to_id_mapping


def collect_receptacle_positions(episode):
    with open(episode["scene_id"], "r") as f:
        scene_data = json.load(f)
    # For ReplicaCAD: positions of receptacle object models. TODO: Replace these with receptacle sample volumes, need to load scene in sim for this.
    recep_positions = {}
    for recep_type in ["object_instances", "articulated_object_instances"]:
        if recep_type not in scene_data:
            continue
        for recep_data in scene_data[recep_type]:
            recep_positions[recep_data["template_name"]] = recep_data[
                "translation"
            ]

    # Receptacle positions are taken from sampling volumes for Floorplanner scenes
    with open(episode["scene_dataset_config"], "r") as f:
        scene_config_file = json.load(f)
    stage_config_folder = scene_config_file["stages"]["paths"][".json"][0]
    if not osp.isdir(stage_config_folder):
        stage_config_folder = osp.dirname(stage_config_folder)

    stage_config_file = osp.join(
        osp.dirname(episode["scene_dataset_config"]),
        stage_config_folder,
        scene_data["stage_instance"]["template_name"] + ".stage_config.json",
    )

    if osp.exists(stage_config_file):
        with open(stage_config_file, "r") as f:
            stage_configs = json.load(f)
        if "user_defined" in stage_configs:
            for receptacle, recep_data in stage_configs[
                "user_defined"
            ].items():
                recep_positions[receptacle] = recep_data["position"]

    return recep_positions


def get_matching_recep_handle(receptacle_name, handles):
    return [handle for handle in handles if handle in receptacle_name][0]


def get_candidate_starts(
    objects,
    category,
    start_rec_cat,
    name_to_receptacle,
    obj_rearrange_easy=True,
    obj_category_mapping=None,
):
    obj_goals = []
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
                "object_id": i,
                "object_category": category,
                "view_points": [],
            }
            if not obj_rearrange_easy:
                obj_goals.append(obj_goal)
            else:
                # TODO: Needed for the nav before pick in the easy version of the task. Check if start receptacle matches before appending. This information is missing in GeoGoal rearrange episodes. Need to load scene in sim for this.
                obj_goals.append(obj_goal)
    return obj_goals


def get_candidate_receptacles(
    rec_positions, goal_recep_category, rec_category_mapping=None
):
    goals = []
    for recep, position in rec_positions.items():
        recep_category = get_rec_category(
            recep, rec_category_mapping=rec_category_mapping
        )
        if recep_category == goal_recep_category:
            goal = {
                "position": position,
                "object_name": recep,
                "object_id": -1,
                "object_category": recep_category,
                "view_points": [],
            }
            goals.append(goal)
    return goals


def add_cat_fields_to_episodes(
    episodes_file,
    obj_to_id,
    rec_to_id,
    obj_rearrange_easy=True,
    obj_category_mapping=None,
    rec_category_mapping=None,
):
    """
    Adds category fields to episodes
    """
    episodes = json.load(gzip.open(episodes_file))
    episodes["obj_category_to_obj_category_id"] = obj_to_id
    episodes["recep_category_to_recep_category_id"] = rec_to_id
    for episode in episodes["episodes"]:
        rec_positions = collect_receptacle_positions(episode)
        obj_cat, start_rec_cat, goal_rec_cat = get_obj_rec_cat_in_eps(
            episode,
            obj_category_mapping=obj_category_mapping,
            rec_category_mapping=rec_category_mapping,
        )
        episode["object_category"] = obj_cat
        episode["start_recep_category"] = start_rec_cat
        episode["goal_recep_category"] = goal_rec_cat
        episode["candidate_objects"] = get_candidate_starts(
            episode["rigid_objs"],
            obj_cat,
            start_rec_cat,
            episode["name_to_receptacle"],
            obj_rearrange_easy=obj_rearrange_easy,
            obj_category_mapping=obj_category_mapping,
        )
        episode["candidate_start_receps"] = get_candidate_receptacles(
            rec_positions,
            start_rec_cat,
            rec_category_mapping=rec_category_mapping,
        )
        episode["candidate_goal_receps"] = get_candidate_receptacles(
            rec_positions,
            goal_rec_cat,
            rec_category_mapping=rec_category_mapping,
        )

    return episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/datasets/replica_cad/rearrange/v1/",
    )
    parser.add_argument(
        "--source_episodes_tag", type=str, default="rearrange_easy"
    )
    parser.add_argument(
        "--target_episodes_tag", type=str, default="categorical_rearrange_easy"
    )
    parser.add_argument("--obj_category_mapping_file", type=str, default=None)
    parser.add_argument("--rec_category_mapping_file", type=str, default=None)
    parser.add_argument("--obj_rearrange_easy", type=bool, default=True)

    args = parser.parse_args()

    data_dir = args.data_dir
    source_episodes_tag = args.source_episodes_tag
    target_episodes_tag = args.target_episodes_tag

    obj_category_mapping = None
    if args.obj_category_mapping_file is not None:
        obj_category_mapping = read_obj_category_mapping(
            args.obj_category_mapping_file
        )

    rec_category_mapping = None
    if args.rec_category_mapping_file is not None:
        rec_category_mapping = read_obj_category_mapping(
            args.rec_category_mapping_file
        )

    # Retrieve object and receptacle categories in train episodes
    train_episode_file = osp.join(
        data_dir, "train", f"{source_episodes_tag}.json.gz"
    )
    obj_to_id, rec_to_id = get_cats_list(
        train_episode_file,
        obj_category_mapping=obj_category_mapping,
        rec_category_mapping=rec_category_mapping,
    )
    print(f"Number of object categories: {len(obj_to_id)}")
    print(f"Number of receptacle categories: {len(rec_to_id)}")
    print(obj_to_id, rec_to_id)
    # Add category fields and save episodes
    for split in os.listdir(data_dir):
        episodes_file = osp.join(
            data_dir, split, f"{source_episodes_tag}.json.gz"
        )
        if not osp.exists(episodes_file):
            continue
        episodes = add_cat_fields_to_episodes(
            episodes_file,
            obj_to_id,
            rec_to_id,
            obj_rearrange_easy=args.obj_rearrange_easy,
            obj_category_mapping=obj_category_mapping,
            rec_category_mapping=rec_category_mapping,
        )
        episodes_json = DatasetFloatJSONEncoder().encode(episodes)
        target_episodes_file = osp.join(
            data_dir, split, f"{target_episodes_tag}.json.gz"
        )
        with gzip.open(target_episodes_file, "wt") as f:
            f.write(episodes_json)
    print("All episodes written")
