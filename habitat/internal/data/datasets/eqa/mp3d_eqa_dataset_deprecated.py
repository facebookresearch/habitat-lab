import gzip
import json
import os
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
from habitat.core.dataset import Dataset
from habitat.core.logging import logger
from habitat.tasks.eqa.eqa_task import EQAEpisode, QuestionData
from habitat.tasks.nav.nav_task import (
    ObjectGoal, NavigationGoal, ShortestPathPoint
)
from yacs.config import CfgNode

ALLOWED_QUESTION_TYPES = ["location", "color", "color_room"]

HABITAT_NAME_TO_ACTION = {
    'forwards': 0,
    'turnLeft': 1,
    'turnRight': 2,
    'stop': 3
}

MINOS_NAME_TO_ACTION = {
    'forwards': 0,
    'turnLeft': 1,
    'turnRight': 2,
    'stop': 100
}

ACTION_MAPPING = {
    MINOS_NAME_TO_ACTION['forwards']: HABITAT_NAME_TO_ACTION['forwards'],
    MINOS_NAME_TO_ACTION['turnLeft']: HABITAT_NAME_TO_ACTION['turnLeft'],
    MINOS_NAME_TO_ACTION['turnRight']: HABITAT_NAME_TO_ACTION['turnRight'],
    MINOS_NAME_TO_ACTION['stop']: HABITAT_NAME_TO_ACTION['stop']
}

EQA_MP3D_V1_VAL_EPISODE_COUNT = 1950


def get_default_mp3d_v1_config(split: str = "val"):
    config = CfgNode()
    config.name = "MP3DEQA-v1"
    config.data_path = "data/datasets/eqa_mp3d_v1/full_data.json.gz"
    config.data_split_h5_path = "data/datasets/eqa_mp3d_v1/full_{split}.h5"
    config.scenes_path = "data/scene_datasets/mp3d"
    config.split = split
    return config


# TODO(maksymets): Get rid of conversion methods from MINOS coordinate
# systems to Habitat Sim, as soon relative
# transformation for whole space will be implemented.

def minos_angle_to_sim_quaternion(angle_xy: float) -> List[float]:
    r"""
    :param angle_xy: float angle in radians for agent direction on xy plane

    :return: list with 4 entries for (x, y, z, w) elements of unit
    quaternion (versor) representing agent 3D orientation,
    ref: https://en.wikipedia.org/wiki/Versor
    :rtype: List[float]
    """
    angle_xy += np.pi / 2
    return [0, np.sin(angle_xy / 2), 0, np.cos(angle_xy / 2)]


def minos_direction_to_sim_quaternion(orientation: List[float]) -> List[float]:
    r"""
    :param orientation: list with 3 entries for (x, y, z) elements
    of unit representing agent 3D orientation,

    :return: list with 4 entries for (x, y, z, w) elements
    of unit quaternion (versor) representing
    agent 3D orientation,
    ref: https://en.wikipedia.org/wiki/Versor
    :rtype: List[float]
    """
    # TODO (maksymets): this function isn't critical, but it has inconsistency
    # with Habitat Sim orientation
    assert len(orientation) == 3
    orientation = minos_to_sim(orientation)
    return minos_angle_to_sim_quaternion(
        -1 * np.arcsin(orientation[0] / np.sqrt(
            orientation[0] ** 2 + orientation[2] ** 2)) + 3 * np.pi / 2)


def minos_to_sim(position: List[float]) -> List[float]:
    return [position[2], position[1], -position[0]]


def swap_yz(position: List[float]) -> List[float]:
    return [position[0], position[2], position[1]]


class Matterport3dDatasetV1(Dataset):
    r"""Class inherited from Dataset that loads Matterport3D
    Embodied Question Answering dataset.

    This class can then be used as follows::


    eqa_config.dataset = get_default_mp3d_v1_config()
    eqa = habitat.make_task(eqa_config.task_name, config=eqa_config)

    """
    episodes: List[EQAEpisode]

    @staticmethod
    def check_config_paths_exist(config: Any) -> bool:
        return os.path.exists(config.data_path) and os.path.exists(
            config.data_split_h5_path.format(
                split=config.split)) and os.path.exists(config.scenes_path)

    @staticmethod
    def _create_goal(episode_id: str,
                     episode_data: Dict[str, Any]) -> Optional[NavigationGoal]:
        if episode_data["bbox"][0]["type"] != "object":
            logger.info("Unknown navigational goal for episode id: {}.".format(
                episode_id))
            return None

        assert len(episode_data[
                       "bbox"]) > 0, \
            "No goals found for episode id: {}.".format(episode_id)

        goal = ObjectGoal(
            position=minos_to_sim(
                swap_yz(episode_data["target"]["centroid"])),
            room_id=episode_data["target"]["room_id"],
            object_id=episode_data["target"]["obj_id"],
            object_name=episode_data["bbox"][0]["name"]
        )

        return goal

    @staticmethod
    def _create_shortest_path(episode_id, path_points, path_actions) \
            -> List[ShortestPathPoint]:
        shortest_path = []
        for step_id, ds_point in enumerate(
                path_actions):
            point = ShortestPathPoint(
                position=minos_to_sim(path_points[step_id]["position"]),
                rotation=minos_direction_to_sim_quaternion(
                    path_points[step_id]["orientation"]),
                action=ACTION_MAPPING[int(path_actions[step_id])]
            )
            shortest_path.append(point)
            if point.action == HABITAT_NAME_TO_ACTION["stop"]:
                break

        if not len(path_points) == len(shortest_path):
            logger.info(
                "Number of positions and actions doesn't match for episode"
                " {}.".format(episode_id))
        return shortest_path

    @staticmethod
    def _create_question(episode_data) -> QuestionData:
        question = QuestionData(
            question_text=episode_data["question"],
            answer_text=episode_data["answer"],
            question_type=episode_data["type"]
        )
        return question

    @staticmethod
    def _create_episode(episode_id, episode_data, path_points, path_actions,
                        scene_id, scenes_path) -> Optional[EQAEpisode]:
        episode_id = str(episode_id)
        question = Matterport3dDatasetV1._create_question(
            episode_data["qn"])

        if question.question_type not in ALLOWED_QUESTION_TYPES:
            return None

        goal = Matterport3dDatasetV1._create_goal(episode_id,
                                                  episode_data["qn"])
        if goal:
            goals = [goal]
        shortest_paths = [
            Matterport3dDatasetV1._create_shortest_path(
                episode_id=episode_id,
                path_points=path_points,
                path_actions=path_actions)]

        house = episode_data["qn"]["house"]
        level = episode_data["qn"]["level"]
        assert scene_id == "{}.{}".format(house,
                                          level), \
            "Scene mismatch between metadata and environment index " \
            "for episode {}.".format(
                episode_id)
        scene_id = "{scenes_path}/{house}/{house}.glb".format(
            scenes_path=scenes_path,
            house=house)
        start_position = minos_to_sim(
            episode_data["start"]["position"])
        start_rotation = minos_angle_to_sim_quaternion(
            episode_data["start"]["angle"])
        start_room = episode_data["start"]["room"]

        episode = EQAEpisode(episode_id=episode_id, question=question,
                             goals=goals,
                             scene_id=scene_id, start_position=start_position,
                             start_rotation=start_rotation,
                             start_room=start_room,
                             shortest_paths=shortest_paths)
        return episode

    def __init__(self, config=None):
        self.episodes: List[EQAEpisode] = []

        if config is None:
            return

        with gzip.open(config.data_path, "rt") as f:
            dataset = json.load(f)

        with h5py.File(config.data_split_h5_path.format(split=config.split),
                       "r") as questions_h5:
            shortest_path_actions = questions_h5['action_labels']

            env_ids_key = "{}_env_idx".format(config.split)
            meta_data_key = "{}_meta_data".format(config.split)
            path_data_key = "{}_pos_queue".format(config.split)

            for episode_id, episode_data in enumerate(dataset[meta_data_key]):
                episode_data = dataset[meta_data_key][episode_id]
                inner_scene_id = dataset[env_ids_key][episode_id]
                scene_id = dataset["envs"][inner_scene_id]
                episode = self._create_episode(
                    episode_id=episode_id, episode_data=episode_data,
                    path_points=dataset[path_data_key][episode_id],
                    path_actions=shortest_path_actions[episode_id],
                    scene_id=scene_id,
                    scenes_path=config.scenes_path)

                if episode:
                    self.episodes.append(episode)

    def from_json(self, deserialized: Dict[str, Any]) -> None:
        self.__dict__.update(deserialized)
        for episode in self.episodes:
            episode = EQAEpisode(**episode)
            episode.question = QuestionData(**episode.question)
            for index, goal in enumerate(episode.goals):
                episode.goals[index] = ObjectGoal(**goal)
            for path in episode.shortest_paths:
                for p_index, point in enumerate(path):
                    path[p_index] = ShortestPathPoint(**point)
