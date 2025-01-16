# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional

import magnum as mn
import numpy as np

from habitat.articulated_agents.humanoids.kinematic_humanoid import (
    KinematicHumanoid,
)
from habitat.articulated_agents.mobile_manipulator import MobileManipulator

# flake8: noqa
from habitat.articulated_agents.robots import (
    FetchRobot,
    FetchRobotNoWheels,
    SpotRobot,
    StretchRobot,
)
from habitat.articulated_agents.robots.fetch_suction import FetchSuctionRobot
from habitat.tasks.rearrange.rearrange_grasp_manager import (
    RearrangeGraspManager,
)
from habitat.tasks.rearrange.utils import (
    IkHelper,
    add_perf_timing_func,
    is_pb_installed,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from habitat.config.default_structured_configs import (
        AgentConfig,
        SimulatorConfig,
    )
    from habitat_sim.simulator import Simulator


@dataclass
class ArticulatedAgentData:
    """
    Data needed to manage an agent instance.
    """

    articulated_agent: MobileManipulator
    grasp_mgrs: List[RearrangeGraspManager]
    cfg: "AgentConfig"
    start_js: np.ndarray
    is_pb_installed: bool
    _ik_helper: Optional[IkHelper] = None

    @property
    def grasp_mgr(self):
        if len(self.grasp_mgrs) == 0:
            raise Exception("Agent data has no grasp manager defined")
        return self.grasp_mgrs[0]

    @property
    def ik_helper(self):
        if not self.is_pb_installed:
            raise ImportError(
                "Need to install PyBullet to use IK (`pip install pybullet==3.0.4`)"
            )
        return self._ik_helper


class ArticulatedAgentManager:
    """
    Handles creating, updating and managing all agent instances.
    """

    def __init__(self, cfg: "SimulatorConfig", sim: "Simulator"):
        self._sim = sim
        self._all_agent_data: List[ArticulatedAgentData] = []
        self._is_pb_installed = is_pb_installed()
        self.agent_names: Dict[str, Any] = cfg.agents
        self._agent_index_to_name: Dict[int, str] = {}
        self._agent_name_to_index: Dict[str, int] = {}

        for agent_index in range(len(cfg.agents_order)):
            agent_name = cfg.agents_order[agent_index]
            self._agent_index_to_name[agent_index] = agent_name
            self._agent_name_to_index[agent_name] = agent_index

            agent_cfg = cfg.agents[agent_name]
            agent_cls = eval(agent_cfg.articulated_agent_type)
            assert issubclass(agent_cls, MobileManipulator)
            agent = agent_cls(agent_cfg, sim)
            grasp_managers = []
            for grasp_manager_id in range(agent_cfg.grasp_managers):
                grasp_mgr = RearrangeGraspManager(
                    sim, cfg, agent, grasp_manager_id
                )
                grasp_managers.append(grasp_mgr)

            if len(cfg.agents) > 1:
                # Prefix sensors if there is more than 1 agent in the scene.
                agent.params.cameras = {
                    f"{agent_name}_{k}": v
                    for k, v in agent.params.cameras.items()
                }
                for camera_prefix in agent.params.cameras:
                    for sensor_name in self._sim._sensors:
                        if sensor_name.startswith(camera_prefix):
                            agent._cameras[camera_prefix].append(sensor_name)

            if agent_cfg.joint_start_override is None:
                use_arm_init = np.array(agent.params.arm_init_params)
            else:
                use_arm_init = np.array(agent_cfg.joint_start_override)
            self._all_agent_data.append(
                ArticulatedAgentData(
                    articulated_agent=agent,
                    grasp_mgrs=grasp_managers,
                    cfg=agent_cfg,
                    start_js=use_arm_init,
                    is_pb_installed=self._is_pb_installed,
                )
            )

    def get_agent_name_from_index(self, agent_index: int) -> Optional[str]:
        """
        Get the name of the agent from its index.
        Returns `None` if the index does not exist.
        """
        return self._agent_index_to_name.get(agent_index, None)

    def get_agent_index_from_name(self, agent_name: str) -> Optional[int]:
        """
        Get the index of an agent from its name.
        Returns `None` if the name does not exist.
        """
        return self._agent_name_to_index.get(agent_name, None)

    @add_perf_timing_func()
    def on_new_scene(self) -> None:
        """
        Call on a new scene. This will destroy and re-create the robot
        simulator instances.
        """
        ao_mgr = self._sim.get_articulated_object_manager()
        for agent_data in self._all_agent_data:
            agent_data.grasp_mgr.reconfigure()
            if (
                agent_data.articulated_agent.sim_obj is not None
                and agent_data.articulated_agent.sim_obj.is_alive
            ):
                ao_mgr.remove_object_by_id(
                    agent_data.articulated_agent.sim_obj.object_id
                )

            agent_data.articulated_agent.reconfigure()

    def pre_obj_clear(self) -> None:
        """
        Must call before all objects in the scene are removed before the next
        episode. This will reset the grasp constraints and any references to
        previously existing objects.
        """

        for agent_data in self._all_agent_data:
            for grasp_manager in agent_data.grasp_mgrs:
                grasp_manager.reset()
            agent_data.grasp_mgr.reconfigure()

    @add_perf_timing_func()
    def post_obj_load_reconfigure(self):
        """
        Called at the end of the simulator reconfigure method. Used to set the starting configurations of the robots if specified in the task config.
        """
        for agent_data in self._all_agent_data:
            target_arm_init_params = (
                agent_data.start_js
                + agent_data.cfg.joint_start_noise
                * np.random.randn(len(agent_data.start_js))
            )

            # We only randomly set the location of the particular joint if that joint can be controlled
            # and given joint_that_can_control value.
            if agent_data.cfg.joint_that_can_control is not None:
                assert len(agent_data.start_js) == len(
                    agent_data.cfg.joint_that_can_control
                )
                for i in range(len(agent_data.cfg.joint_that_can_control)):
                    # We cannot control this joint
                    if agent_data.cfg.joint_that_can_control[i] == 0:
                        # The initial parameter for this joint should be the original angle
                        target_arm_init_params[i] = agent_data.start_js[i]

            agent_data.articulated_agent.params.arm_init_params = (
                target_arm_init_params
            )
            # note this does not reset the base position and rotation
            agent_data.articulated_agent.reset()

            # Consume a fixed position from SIMULATOR.agent_0 if configured
            if agent_data.cfg.is_set_start_state:
                agent_data.articulated_agent.base_pos = mn.Vector3(
                    agent_data.cfg.start_position
                )
                agent_rot = agent_data.cfg.start_rotation
                agent_data.articulated_agent.sim_obj.rotation = mn.Quaternion(
                    mn.Vector3(agent_rot[:3]), agent_rot[3]
                )

    def __getitem__(self, key: int) -> ArticulatedAgentData:
        """
        Fetches the agent data at the agent index.
        """

        return self._all_agent_data[key]

    def __len__(self) -> int:
        """
        The number of agents.
        """

        return len(self._all_agent_data)

    @property
    def articulated_agents_iter(self) -> Iterator[MobileManipulator]:
        """
        Iterator over all agent interfaces.
        """

        for agent_data in self._all_agent_data:
            yield agent_data.articulated_agent

    @property
    def grasp_iter(self) -> Iterator[RearrangeGraspManager]:
        for agent_data in self._all_agent_data:
            yield agent_data.grasp_mgr

    def first_setup(self):
        for agent_data in self._all_agent_data:
            ik_arm_urdf = agent_data.cfg.ik_arm_urdf
            if ik_arm_urdf is not None and self._is_pb_installed:
                agent_data._ik_helper = IkHelper(
                    agent_data.cfg.ik_arm_urdf,
                    agent_data.start_js,
                )

    def update_agents(self):
        """
        Update all agent instance managers.
        """

        for agent_data in self._all_agent_data:
            for grasp_mgr in agent_data.grasp_mgrs:
                grasp_mgr.update()
            agent_data.articulated_agent.update()

    def update_debug(self):
        """
        Only call when in debug mode. This renders visualization helpers for
        the agents.
        """

        for agent_data in self._all_agent_data:
            for grasp_mgr in agent_data.grasp_mgrs:
                grasp_mgr.update_debug()
