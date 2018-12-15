import gzip
import json
import os
from typing import Any, Dict, List

import h5py
import numpy as np
from yacs.config import CfgNode

import teas.core.dataset as ds
from teas.core.logging import logger

BLACKLIST_QUESTION_TYPE = []

TEAS_NAME_TO_ACTION = {
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
    MINOS_NAME_TO_ACTION['forwards']: TEAS_NAME_TO_ACTION['forwards'],
    MINOS_NAME_TO_ACTION['turnLeft']: TEAS_NAME_TO_ACTION['turnLeft'],
    MINOS_NAME_TO_ACTION['turnRight']: TEAS_NAME_TO_ACTION['turnRight'],
    MINOS_NAME_TO_ACTION['stop']: TEAS_NAME_TO_ACTION['stop']
}

EQA_MP3D_V1_TEST_EPISODE_COUNT = 3766


def get_default_mp3d_v1_config(split: str = "test"):
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

def minos_angle_to_esp_quaternion(angle_xy: float) -> List[float]:
    r"""
    :param angle_xy: float angle in radians for agent direction on xy plane

    :return: list with 4 entries for (x, y, z, w) elements of unit
    quaternion (versor) representing agent 3D orientation,
    ref: https://en.wikipedia.org/wiki/Versor
    :rtype: List[float]
    """
    angle_xy += np.pi / 2
    return [0, np.sin(angle_xy / 2), 0, np.cos(angle_xy / 2)]


def minos_direction_to_esp_quaternion(orientation: List[float]) -> List[float]:
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
    orientation = minos_to_esp(orientation)
    return minos_angle_to_esp_quaternion(
        -1 * np.arcsin(orientation[0] / np.sqrt(
            orientation[0] ** 2 + orientation[2] ** 2)) + 3 * np.pi / 2)


def minos_to_esp(position: List[float]) -> List[float]:
    return [position[2], position[1], -position[0]]


def swap_yz(position: List[float]) -> List[float]:
    return [position[0], position[2], position[1]]


class MP3DDatasetV1(ds.Dataset):
    r"""Class inherited from Dataset that loads Matterport3D
    Embodied Question Answering dataset.

    This class can then be used as follows::


    eqa_config.dataset = get_default_mp3d_v1_config()
    eqa = teas.make_task(eqa_config.task_name, config=eqa_config)

    """
    _data: Dict[int, ds.EQAEpisode] = None

    @staticmethod
    def check_config_paths_exist(config: Any) -> bool:
        return os.path.exists(config.data_path) and os.path.exists(
            config.data_split_h5_path.format(
                split=config.split)) and os.path.exists(
            config.scenes_path)

    @staticmethod
    def _create_goal(episode_data: Dict = None) -> ds.ObjectGoal:
        goal = ds.ObjectGoal()
        if episode_data["bbox"][0]["type"] != "object":
            goal = ds.NavigationGoal()
            goal.type = ds.NavGoalType.UNKNOWN.value
            logger.info("Unknown navigational goal for episode id: {}.".format(
                episode_data.id))
        else:
            goal.type = ds.NavGoalType.OBJECT.value
        goal.position = episode_data["target"]["centroid"]
        goal.position = minos_to_esp(
            swap_yz(episode_data["target"]["centroid"]))
        goal.room_id = episode_data["target"]["room_id"]
        goal.object_id = episode_data["target"]["obj_id"]
        assert len(episode_data["bbox"]) > 0
        goal.object_name = episode_data["bbox"][0]["name"]
        return goal

    @staticmethod
    def _create_shortest_path(episode, path_points, path_actions):
        shortest_path = []
        for step_id, ds_point in enumerate(
                path_actions):
            point = ds.ShortestPathPoint()
            point.position = minos_to_esp(path_points[step_id]["position"])
            point.rotation = minos_direction_to_esp_quaternion(
                path_points[step_id]["orientation"])
            point.action = int(path_actions[step_id])
            point.action = ACTION_MAPPING[point.action]
            shortest_path.append(point)
            if point.action == TEAS_NAME_TO_ACTION["stop"]:
                break

        if not len(path_points) == len(shortest_path):
            logger.info(
                "Number of positions and actions doesn't match for episode"
                " {}.".format(episode.id))
        return shortest_path

    @staticmethod
    def _create_question(episode_data):
        question = ds.QuestionData()
        question.question_text = episode_data["question"]
        question.answer_text = episode_data["answer"]
        question.question_type = episode_data["type"]
        return question

    def __init__(self, config):
        self._data = {}

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

                episode = ds.EQAEpisode()
                episode.id = str(episode_id)
                episode.question = self._create_question(
                    episode_data["qn"])
                if episode.question in BLACKLIST_QUESTION_TYPE:
                    continue
                episode.goals = [
                    self._create_goal(episode_data["qn"])]
                episode.shortest_paths = [
                    self._create_shortest_path(
                        episode=episode,
                        path_points=dataset[path_data_key][episode_id],
                        path_actions=shortest_path_actions[episode_id])]

                house = episode_data["qn"]["house"]
                level = episode_data["qn"]["level"]
                inner_scene_id = dataset[env_ids_key][episode_id]
                episode.scene_id = dataset["envs"][inner_scene_id]
                assert episode.scene_id == "{}.{}".format(house,
                                                          level), \
                    "Scene mismatch between metadata and environment index " \
                    "for episode {}.".format(
                        episode_id)
                episode.scene_file = \
                    "{scenes_path}/{house}/{house}.glb".format(
                        scenes_path=config.scenes_path,
                        house=house)
                episode.start_position = minos_to_esp(
                    episode_data["start"]["position"])
                episode.start_rotation = minos_angle_to_esp_quaternion(
                    episode_data["start"]["angle"])
                episode.start_room = episode_data["start"]["room"]
                self._data[len(self._data)] = episode

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)
