from typing import Any, List
from enum import Enum


class NavGoalType(Enum):
    UNKNOWN = 'unknown'
    POSITION = 'position'
    OBJECT = 'object'
    ROOM = 'room'


class ShortestPathPoint:
    position: List[Any] = None
    rotation: List[Any] = None
    action: int = None


class NavigationGoal:
    r"""Base class for a goal specification hierarchy.
    """
    type: NavGoalType


class PositionGoal(NavigationGoal):
    r"""Positional goal when only position and acceptable radius provided.
    """
    position: List[float] = None
    radius: float = None


class ObjectGoal(PositionGoal):
    r"""Object goal that can be specified by object_id or position or object
    category.
    """
    object_id: str = None
    object_name: str = None
    category: str = None
    room_id: str = None
    room_name: str = None


class RoomGoal(NavigationGoal):
    r"""Room goal that can be specified by room_id or position with radius.
    """
    room_id: str = None
    room_name: str = None


class QuestionData:
    r"""Class saves data about question asked to the agent and correct answer.
    """
    question_text: str
    answer_text: str
    question_type: str

    def __init__(self, question_text: str = None, answer_text: str = None,
                 question_type: str = None) -> None:
        self.question_text = question_text
        self.answer_text = answer_text
        self.question_type = question_type


class NavigationEpisode:
    r"""Base class for episode specification that includes initial position and
    rotation of agent, scene name, goal and optional shortest paths. An
    episode is a description of one task instance for the agent.
    """
    id: str = None
    scene_id: str = None
    start_position: List[float] = None
    start_rotation: List[float] = None
    start_room: str = None
    goals: List[NavigationGoal] = None

    def __init__(self, scene_id: str = None,
                 start_position: List[float] = None,
                 start_rotation: List[float] = None, start_room: str = None,
                 goals: List[NavigationGoal] = None) -> None:
        r"""
        :param scene_id: id of scene in scene dataset
        :param start_position: numpy ndarray containing 3 entries for (x, y, z)
        :param start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
        elements of unit quaternion (versor) representing agent 3D orientation,
        ref: https://en.wikipedia.org/wiki/Versor
        :param goals: list of goals specifications
        """
        self.scene_id = scene_id
        self.start_position = start_position
        self.start_rotation = start_rotation
        self.start_room = start_room
        self.goals = goals


class EQAEpisode(NavigationEpisode):
    r"""Specification of episode that includes initial position and rotation of
    agent, goal, question specifications and optional shortest paths.
    """
    question: QuestionData = None

    def __init__(self, scene_id: str = None,
                 start_position: List[float] = None,
                 start_rotation: List[float] = None, start_room: str = None,
                 goals: List[NavigationGoal] = None,
                 question: QuestionData = None) -> None:
        r"""
        :param scene_id:
        :param start_position: numpy ndarray containing 3 entries for (x, y, z)
        :param start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
        elements of unit quaternion (versor) representing agent 3D orientation,
        ref: https://en.wikipedia.org/wiki/Versor
        :param goals: relevant goal object/room
        :param question: question related to goal object
        """
        super(self.__class__, self).__init__(scene_id=scene_id,
                                             start_position=start_position,
                                             start_rotation=start_rotation,
                                             start_room=start_room,
                                             goals=goals)
        self.question = question


class Dataset:
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
