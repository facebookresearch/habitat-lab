#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional, Tuple
from collections import defaultdict

import attr
import numpy as np
from gym import spaces

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import (
    EmbodiedTask,
    Measure,
    SimulatorTaskAction,
)
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    RGBSensor,
    Sensor,
    SensorTypes,
    ShortestPathPoint,
    Simulator,
)
from habitat.core.spaces import ActionSpace
from habitat.core.utils import not_none_validator, try_cv2_import
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat.utils.visualizations import fog_of_war, maps

try:
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
    from habitat_sim import RigidState
    from habitat_sim.physics import VelocityControl
except ImportError:
    pass

try:
    import magnum as mn
except ImportError:
    pass


@registry.register_simulator(name="Ant-v2-sim")
class AntV2Sim(HabitatSim):
    def __init__(self, config):
        print("Sim loaded..")
        super().__init__(config)

        agent_config = self.habitat_config
        self.first_setup = True
        self.is_render_obs = False
        self.ep_info = None
        self.prev_loaded_navmesh = None
        self.prev_scene_id = None

        # Number of physics updates per action
        self.ac_freq_ratio = agent_config.AC_FREQ_RATIO
        # The physics update time step.
        self.ctrl_freq = agent_config.CTRL_FREQ
        # Effective control speed is (ctrl_freq/ac_freq_ratio)

        self.art_objs = []
        self.start_art_states = {}
        self.cached_art_obj_ids = []
        self.scene_obj_ids = []
        self.viz_obj_ids = []
        # Used to get data from the RL environment class to sensors.
        self.track_markers = []
        self._goal_pos = None
        self.viz_ids: Dict[Any, Any] = defaultdict(lambda: None)

    def reconfigure(self, config):
            ep_info = config["ep_info"][0]
            ep_info = self._update_config(ep_info)

            config["SCENE"] = ep_info["scene_id"]
            super().reconfigure(config)

            self.ep_info = ep_info
            self.fixed_base = ep_info["fixed_base"]

            self.target_obj_ids = []

            if ep_info["scene_id"] != self.prev_scene_id:
                # Object instances are not valid between scenes.
                self.art_objs = []
                self.scene_obj_ids = []
                self.robot = None
                self.viz_ids = defaultdict(lambda: None)
                self.viz_obj_ids = []
            self.grasp_mgr.desnap(force=True)
            self.prev_scene_id = ep_info["scene_id"]

            self._try_acquire_context()

            # add ant
            self._add_objs(ep_info)
            if self.robot is None:
                self.robot = AntV2Robot(self.habitat_config.ROBOT_URDF, self)
                self.robot.reconfigure()
            self.robot.reset()

            # add floor
            cube_handle = obj_templates_mgr.get_template_handles("cube")[0]
            floor = obj_templates_mgr.get_template_by_handle(cube_handle)
            floor.scale = np.array([2.0, 0.05, 2.0])

            obj_templates_mgr.register_template(floor, "floor")
            floor_obj = rigid_obj_mgr.add_object_by_template_handle("floor")
            floor_obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC

            floor_obj.translation = np.array([2.50, -1, 0.5])
            floor_obj.motion_type = habitat_sim.physics.MotionType.STATIC
    
    def step(self, action):
        # what to do with action?

        # returns new observation after step
        sim.step_physics(1.0 / 60.0)
        obs = self._sensor_suite.get_observations(self._prev_sim_obs)
        return obs


    # Need to figure out MVP for the simulator + how to set up a camera which isn't part of the obs space.
    # Also need to figure out how to define rewards based on measurements/observations


@registry.register_sensor(name="ANT_OBSERVATION_SPACE_SENSOR")
class AntObservationalSpaceSensor(Sensor):

    cls_uuid: str = "ANT_OBSERVATION_SPACE_SENSOR"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(27,), dtype=np.float)

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        obs = self._sim.robot.observational_space
        return obs


@registry.register_measure
class XLocation(Measure):
    """The measure calculates the x component of the robot's location."""

    cls_uuid: str = "X_LOCATION"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._metric = None
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._metric = None

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        current_position = self._sim.robot.base_pos
        self._metric["x_location"] = current_position.x

@registry.register_task_action
class LegRelPosAction(SimulatorTaskAction):
    """
    The leg motor targets are offset by the delta joint values specified by the
    action
    """

    @property
    def action_space(self):
        return spaces.Box(
            shape=(self._config.LEG_JOINT_DIMENSIONALITY,),
            low=0,
            high=1,
            dtype=np.float32,
        )

    def step(self, delta_pos, should_step=True, *args, **kwargs):
        # clip from -1 to 1
        delta_pos = np.clip(delta_pos, -1, 1)
        delta_pos *= self._config.DELTA_POS_LIMIT
        # The actual joint positions
        self._sim: AntV2Sim
        self._sim.robot.leg_joint_pos = (
            delta_pos + self._sim.robot.leg_joint_pos
        )
        if should_step:
            return self._sim.step(HabitatSimActions.LEG_VEL)
        return None

@registry.register_task_action
class LegAction(SimulatorTaskAction):
    """A continuous leg control into one action space."""

    def __init__(self, *args, config, sim: AntV2Sim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        leg_controller_cls = eval(self._config.LEG_CONTROLLER)
        self.leg_ctrlr = leg_controller_cls(
            *args, config=config, sim=sim, **kwargs
        )

    def reset(self, *args, **kwargs):
        self.leg_ctrlr.reset(*args, **kwargs)

    @property
    def action_space(self):
        action_spaces = {
            "leg_action": self.leg_ctrlr.action_space,
        }
        return spaces.Dict(action_spaces)

    def step(self, leg_action, *args, **kwargs):
        self.leg_ctrlr.step(leg_action, should_step=False)
        return self._sim.step(HabitatSimActions.LEG_ACTION)


@registry.register_task(name="Ant-v2-task")
class AntV2Task(EmbodiedTask):
    def __init__(
        self, config: Config, sim: Simulator, dataset: Optional[Dataset] = None
    ) -> None:
        print("Task loaded!!!")
        super().__init__(config=config, sim=sim, dataset=dataset)

    def overwrite_sim_config(self, sim_config: Any, episode: Episode) -> Any:
        return merge_sim_episode_config(sim_config, episode)

    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        return not getattr(self, "is_stop_called", False)