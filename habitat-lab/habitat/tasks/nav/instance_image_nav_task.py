# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import attr
import numpy as np
from gym import Space, spaces

import habitat_sim
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import (
    RGBSensor,
    Sensor,
    SensorTypes,
    VisualObservation,
)
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.tasks.nav.object_nav_task import ObjectGoal, ObjectNavigationTask
from habitat.utils.geometry_utils import quaternion_from_coeff
from habitat_sim import bindings as hsim
from habitat_sim.agent.agent import AgentState, SixDOFPose

try:
    from habitat.datasets.image_nav.instance_image_nav_dataset import (
        InstanceImageNavDatasetV1,
    )
except ImportError:
    pass

if TYPE_CHECKING:
    from omegaconf import DictConfig


@attr.s(auto_attribs=True, kw_only=True)
class InstanceImageGoalNavEpisode(NavigationEpisode):
    """Instance ImageGoal Navigation Episode

    Args:
        object_category: Category of the object
        goal_object_id: the object ID of the instance to navigate to
        goal_image_id: the image ID of which goal image to observe
    """

    goal_object_id: str = attr.ib(default=None, validator=not_none_validator)
    goal_image_id: int = attr.ib(default=None, validator=not_none_validator)
    object_category: Optional[str] = None

    @property
    def goal_key(self) -> str:
        """The key to retrieve the instance goal"""
        sid = os.path.basename(self.scene_id)
        for x in [".glb", ".basis"]:
            sid = sid[: -len(x)] if sid.endswith(x) else sid
        return f"{sid}_{self.goal_object_id}"


@attr.s(auto_attribs=True, kw_only=True)
class InstanceImageParameters:
    position: List[float] = attr.ib(default=None, validator=not_none_validator)
    rotation: List[float] = attr.ib(default=None, validator=not_none_validator)
    hfov: Union[int, float] = attr.ib(
        default=None, validator=not_none_validator
    )
    image_dimensions: Tuple[int, int] = attr.ib(
        default=None, validator=not_none_validator
    )
    frame_coverage: Optional[float] = None
    object_coverage: Optional[float] = None


@attr.s(auto_attribs=True, kw_only=True)
class InstanceImageGoal(ObjectGoal):
    """An instance image goal is an ObjectGoal that also contains a collection
    of InstanceImageParameters.

    Args:
        image_goals: a list of camera parameters each used to generate an
        image goal.
    """

    image_goals: List[InstanceImageParameters] = attr.ib(
        default=None, validator=not_none_validator
    )
    object_surface_area: Optional[float] = None


@registry.register_sensor
class InstanceImageGoalSensor(RGBSensor):
    """A sensor for instance-based image goal specification used by the
    InstanceImageGoal Navigation task. Image goals are rendered according to
    camera parameters (resolution, HFOV, extrinsics) specified by the dataset.

    Args:
        sim: a reference to the simulator for rendering instance image goals.
        config: a config for the InstanceImageGoalSensor sensor.
        dataset: a Instance Image Goal navigation dataset that contains a
        dictionary mapping goal IDs to instance image goals.
    """

    cls_uuid: str = "instance_imagegoal"
    _current_image_goal: Optional[VisualObservation]
    _current_episode_id: Optional[str]

    def __init__(
        self,
        sim,
        config: "DictConfig",
        dataset: "InstanceImageNavDatasetV1",
        *args: Any,
        **kwargs: Any,
    ):
        from habitat.datasets.image_nav.instance_image_nav_dataset import (
            InstanceImageNavDatasetV1,
        )

        assert isinstance(
            dataset, InstanceImageNavDatasetV1
        ), "Provided dataset needs to be InstanceImageNavDatasetV1"

        self._dataset = dataset
        self._sim = sim
        super().__init__(config=config)
        self._current_episode_id = None
        self._current_image_goal = None

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        H, W = (
            next(iter(self._dataset.goals.values()))
            .image_goals[0]
            .image_dimensions
        )
        return spaces.Box(low=0, high=255, shape=(H, W, 3), dtype=np.uint8)

    def _add_sensor(
        self, img_params: InstanceImageParameters, sensor_uuid: str
    ) -> None:
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = sensor_uuid
        spec.sensor_type = habitat_sim.SensorType.COLOR
        spec.resolution = img_params.image_dimensions
        spec.hfov = img_params.hfov
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        self._sim.add_sensor(spec)

        agent = self._sim.get_agent(0)
        agent_state = agent.get_state()
        agent.set_state(
            AgentState(
                position=agent_state.position,
                rotation=agent_state.rotation,
                sensor_states={
                    **agent_state.sensor_states,
                    sensor_uuid: SixDOFPose(
                        position=np.array(img_params.position),
                        rotation=quaternion_from_coeff(img_params.rotation),
                    ),
                },
            ),
            infer_sensor_states=False,
        )

    def _remove_sensor(self, sensor_uuid: str) -> None:
        agent = self._sim.get_agent(0)
        del self._sim._sensors[sensor_uuid]
        hsim.SensorFactory.delete_subtree_sensor(agent.scene_node, sensor_uuid)
        del agent._sensors[sensor_uuid]
        agent.agent_config.sensor_specifications = [
            s
            for s in agent.agent_config.sensor_specifications
            if s.uuid != sensor_uuid
        ]

    def _get_instance_image_goal(
        self, img_params: InstanceImageParameters
    ) -> VisualObservation:
        """To render the instance image goal, a temporary HabitatSim sensor is
        created with the specified InstanceImageParameters. This sensor renders
        the image and is then removed.
        """
        sensor_uuid = f"{self.cls_uuid}_sensor"
        self._add_sensor(img_params, sensor_uuid)

        self._sim._sensors[sensor_uuid].draw_observation()
        img = self._sim._sensors[sensor_uuid].get_observation()[:, :, :3]

        self._remove_sensor(sensor_uuid)
        return img

    def get_observation(
        self,
        *args: Any,
        episode: InstanceImageGoalNavEpisode,
        **kwargs: Any,
    ) -> Optional[VisualObservation]:
        if len(episode.goals) == 0:
            logger.error(
                f"No goal specified for episode {episode.episode_id}."
            )
            return None
        if not isinstance(episode.goals[0], InstanceImageGoal):
            logger.error(
                f"First goal should be InstanceImageGoal, episode {episode.episode_id}."
            )
            return None

        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id == self._current_episode_id:
            return self._current_image_goal

        img_params = episode.goals[0].image_goals[episode.goal_image_id]
        self._current_image_goal = self._get_instance_image_goal(img_params)
        self._current_episode_id = episode_uniq_id

        return self._current_image_goal


@registry.register_sensor
class InstanceImageGoalHFOVSensor(Sensor):
    """A sensor that returns the horizontal field of view (HFOV) in degrees
    of the current episode's instance image goal.
    """

    cls_uuid: str = "instance_imagegoal_hfov"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        return spaces.Box(low=0.0, high=360.0, shape=(1,), dtype=np.float32)

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    def get_observation(
        self, *args: Any, episode: InstanceImageGoalNavEpisode, **kwargs: Any
    ) -> np.ndarray:
        if len(episode.goals) == 0:
            logger.error(
                f"No goal specified for episode {episode.episode_id}."
            )
            return None
        if not isinstance(episode.goals[0], InstanceImageGoal):
            logger.error(
                f"First goal should be InstanceImageGoal, episode {episode.episode_id}."
            )
            return None

        img_params = episode.goals[0].image_goals[episode.goal_image_id]
        return np.array([img_params.hfov], dtype=np.float32)


@registry.register_task(name="InstanceImageNav-v1")
class InstanceImageNavigationTask(ObjectNavigationTask):
    """A task for navigating to a specific object instance specified by a goal
    image. Built on top of ObjectNavigationTask. Used to explicitly state a
    type of the task in config.
    """
