# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from typing import Any, Dict, List, Tuple

import magnum as mn

# import habitat.sims.habitat_simulator.sim_utilities as sutils
from habitat.datasets.rearrange.rearrange_dataset import (
    RearrangeDatasetV0,
    RearrangeEpisode,
)
from habitat.sims.habitat_simulator.debug_visualizer import DebugVisualizer
from habitat_sim import Simulator
from habitat_sim.metadata import MetadataMediator
from habitat_sim.physics import JointType, ManagedRigidObject
from habitat_sim.utils.settings import default_sim_settings, make_cfg


def load_grasp_pose_json(grasp_json_filepath: str):
    """
    Load a grasp pose JSON file and return the map of object id to grasp poses.
    The JSON file should contain a dict mapping object names to a list of grasp dicts containing:
    - "rotation": a 3x3 matrix representing the wrist rotation
    - "translation": a 3D vector representing the wrist translation
    - "fingers": the finger joint degrees of freedom (DOF) for the hand model
    """

    with open(grasp_json_filepath, "r") as f:
        grasp_pose_data = json.load(f)

    obj_to_grasps: Dict[
        str, List[Tuple[mn.Quaternion, mn.Vector3, List[float]]]
    ] = {}
    for object_name, grap_poses in grasp_pose_data.items():
        obj_to_grasps[object_name] = []
        for grasp in grap_poses:
            # Convert the grasp dict to a tuple of (rotation, translation, fingers)
            rotation = mn.Quaternion.from_matrix(
                mn.Matrix3(
                    [
                        [grasp["rotation"][i][j] for j in range(3)]
                        for i in range(3)
                    ]
                )
            )
            translation = mn.Vector3(grasp["translation"])
            finger_joints = grasp["fingers"]
            obj_to_grasps[object_name].append(
                (rotation, translation, finger_joints)
            )

    return obj_to_grasps


def load_episode_dataset(dataset_file: str):
    """
    Load a RearrangeDataset and setup local variables.
    """
    import gzip

    with gzip.open(dataset_file, "rt") as f:
        episode_dataset = RearrangeDatasetV0()
        episode_dataset.from_json(f.read())
        return episode_dataset


def load_episode_contents(sim: Simulator, episode: RearrangeEpisode):
    """
    Load the contents of a RearrangeEpisode into the simulator.
    Assumes the additional object templates are already loaded in the MetadataMediator.
    Returns the list of added ManagedObjects.
    """
    added_objects = []
    for obj_config_name, transform in episode.rigid_objs:
        try:
            # NOTE: restoring 4x4 matrix from serialized file can fail due to unnormalized rotation resulting from floating point precision issues
            ro_t = mn.Matrix4(
                [[transform[j][i] for j in range(4)] for i in range(4)]
            )
            ro_t = mn.Matrix4.from_(
                ro_t.rotation_normalized(), ro_t.translation
            )
            obj_shortname = obj_config_name.split(".object_config.json")[0]
            matching_obj_template_handles = list(
                sim.get_object_template_manager()
                .get_templates_by_handle_substring(obj_shortname)
                .keys()
            )
            ro = sim.get_rigid_object_manager().add_object_by_template_handle(
                matching_obj_template_handles[0]
            )
            if ro is not None:
                ro.transformation = ro_t
                added_objects.append(ro)
        except ValueError as e:
            print(f"Failed to add object '{obj_config_name}' with error: {e}")
            continue
    return added_objects


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--episode-dataset",
        default="data/datasets/hssd/rearrange/val/social_rearrange.json.gz",
        type=str,
        help="the RearrangeEpisode dataset file to load.",
    )
    parser.add_argument(
        "--grasps-json",
        type=str,
        help="the grasp pose JSON file to load. The file should contain a dict mapping object names to a list of grasp dicts.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/grasp_pose_validator_results/",
        type=str,
        help="directory in which to cache images and results csv.",
    )
    parser.add_argument(
        "--save-images",
        default=False,
        action="store_true",
        help="save images during validation into the output directory.",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    episode_dataset = load_episode_dataset(args.episode_dataset)
    # TODO: get this episode index another way, e.g. iteration or command line argument
    episode_index = 0
    episode = episode_dataset.episodes[  # type:ignore[attr-defined]
        episode_index
    ]
    # NOTE: split the scene_id to get the scene name separate from the file path
    scene_name = episode.scene_id.split("/")[-1].split(".")[0]

    # load the grasp pose JSON file
    obj_to_grasps = load_grasp_pose_json(args.grasps_json)

    # create an initial simulator config
    sim_settings: Dict[str, Any] = default_sim_settings
    sim_settings["scene_dataset_config_file"] = episode.scene_dataset_config
    cfg = make_cfg(sim_settings)

    # pre-initialize a MetadataMediator to iterate over scenes
    mm = MetadataMediator()
    mm.active_dataset = episode.scene_dataset_config
    # here we load any additional objects which are not in the scene dataset
    for obj_config_path in episode.additional_obj_config_paths:
        print(f"Loading additional object config: {obj_config_path}")
        mm.object_template_manager.load_configs(obj_config_path)
    cfg.metadata_mediator = mm

    # NOTE: could iterate over all episodes in the dataset here

    # Get scene id from episode metadata
    cfg.sim_cfg.scene_id = scene_name

    # create the Simulator object
    with Simulator(cfg) as sim:
        # create a dbv for easy camera customization
        dbv = DebugVisualizer(sim)

        # load the episode contents into the simulator
        episode_objects: List[ManagedRigidObject] = load_episode_contents(
            sim, episode
        )

        if len(episode_objects) == 0:
            print("No objects loaded for the episode, aborting.")
            exit(1)

        # NOTE: assuming the 1st object is the star object
        star_object = episode_objects[0]
        star_object_name = star_object.handle.split("_:")[0]

        # load the hand URDF
        # NOTE: I modified the mesh filepaths in these urdfs in order to load them
        hand_ao = sim.get_articulated_object_manager().add_articulated_object_from_urdf(
            # "data/hab_murp/allegro/allegro.urdf", maintain_link_order=True,
            "data/hab_murp/allegro/allegro_digit360_right_full_collision.urdf",
            maintain_link_order=True,  # NOTE: this is more logically likely than tree traversal order (default)
        )
        if hand_ao is None:
            print("Failed to load the Allegro URDF, aborting.")
            exit(1)

        # print the links and joint names of the hand pose by index
        print("Link Names:")
        link_names = [
            (hand_ao.get_link_name(idx), hand_ao.get_link_joint_name(idx))
            for idx in hand_ao.get_link_ids()
            if hand_ao.get_link_joint_type(idx) != JointType.Fixed
        ]
        for idx, (link_name, joint_name) in enumerate(link_names):
            print(f" {idx}: {link_name} | {joint_name}")

        # pose the hand
        if len(obj_to_grasps) > 0:
            # TODO: use multiple grasps for the object
            # load the first grasp pose for the star object
            first_grasp = obj_to_grasps[star_object_name][0]

            # pose the hand relative to the object transformation
            hand_ao.translation = star_object.transformation.transform_point(
                first_grasp[1]
            )
            hand_ao.rotation = star_object.rotation * first_grasp[0]

            # set the finger joints to the specified DOF
            # hand_ao.joint_positions = first_grasp[2]
            # NOTE: transposing thumb and fingers here
            hand_ao.joint_positions = first_grasp[2][4:] + first_grasp[2][0:4]
            print(hand_ao.joint_positions)
        else:
            # for now we will just place it near the first object
            hand_ao.translation = star_object.translation + mn.Vector3(
                0, 0.25, 0
            )
        hand_object_ids: Dict[int, int] = hand_ao.link_object_ids

        # run collision detection and check for contact between the hand and environment
        sim.perform_discrete_collision_detection()
        cps = sim.get_physics_contact_points()
        for cp in cps:
            if (
                cp.object_id_a in hand_object_ids
                or cp.object_id_b in hand_object_ids
            ):
                # TODO: add filtering logic for contacts here
                print(
                    f"Contact detected between hand and environment: {cp.object_id_a} - {cp.object_id_b}"
                )

        # set camera to look at the object
        # TODO: refine this to use the hemisphere samples, for now just use a fixed offset direction and distance for prototype
        for cam_pos in [
            mn.Vector3(1.0, 1.0, 0),
            mn.Vector3(-1.0, 1.0, 0),
            mn.Vector3(0, 1.0, 1.0),
            mn.Vector3(0, 1.0, -1.0),
        ]:
            dbv.peek(
                star_object,
                cam_local_pos=cam_pos,
                distance_from_subject=0.5,
            ).show()
