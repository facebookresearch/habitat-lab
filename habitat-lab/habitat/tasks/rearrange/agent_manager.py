from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, List, Optional

import magnum as mn
import numpy as np

from habitat.agents import AgentInterface as Agent
from habitat.agents.humanoids.amass_human import AmassHuman
from habitat.agents.robots import (
    FetchRobot,
    FetchRobotNoWheels,
    SpotRobot,
    StretchRobot,
)
from habitat.agents.robots.fetch_suction import FetchSuctionRobot
from habitat.tasks.rearrange.rearrange_grasp_manager import (
    RearrangeGraspManager,
)
from habitat.tasks.rearrange.utils import IkHelper, is_pb_installed

if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass
class AgentData:
    """
    Data needed to manage a robot instance.
    """

    agent: Agent
    grasp_mgrs: List[RearrangeGraspManager]
    cfg: "DictConfig"
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


class AgentManager:
    """
    Creates all agent instances
    """

    def __init__(self, cfg, sim):
        self._sim = sim
        self.agent_names = cfg.agents
        self._all_agent_data = []

        self._is_pb_installed = is_pb_installed()
        for agent_name in cfg.agents:
            agent_cfg = cfg.agents[agent_name]
            agent_cls = eval(agent_cfg.agent_type)
            agent = agent_cls(agent_cfg.agent_urdf, sim)

            grasp_managers = []
            for grasp_manager_id in range(agent_cfg.grasp_managers):
                graps_mgr = RearrangeGraspManager(
                    sim, cfg, agent, grasp_manager_id
                )
                grasp_managers.append(graps_mgr)
            self._all_agent_data.append(
                # TODO: start_js should be more flexible to support
                # multiple end effectors
                AgentData(
                    agent=agent,
                    grasp_mgrs=grasp_managers,
                    start_js=np.array(agent.params.arm_init_params),
                    cfg=agent_cfg,
                    is_pb_installed=self._is_pb_installed,
                )
            )

    def reconfigure(self, is_new_scene: bool):
        # TODO: Some documentation here
        ao_mgr = self._sim.get_articulated_object_manager()
        for agent_data in self._all_agent_data:
            if is_new_scene:
                agent_data.grasp_mgr.reconfigure()
                if (
                    agent_data.agent.sim_obj is not None
                    and agent_data.agent.sim_obj.is_alive
                ):
                    ao_mgr.remove_object_by_id(
                        agent_data.agent.sim_obj.object_id
                    )

                agent_data.agent.reconfigure()
            for ind in range(len(agent_data.grasp_mgrs)):
                agent_data.grasp_mgrs[ind].reset()

    def post_obj_load_reconfigure(self):
        """
        Called at the end of the simulator reconfigure method. Used to set the starting configurations of the humanoid if specified in the task config.
        """
        for agent_data in self._all_agent_data:
            agent_data.agent.params.arm_init_params = (
                agent_data.start_js
                + agent_data.cfg.joint_start_noise
                * np.random.randn(len(agent_data.start_js))
            )
            agent_data.agent.reset()

            # consume a fixed position from SIMUALTOR.agent_0 if configured
            if agent_data.cfg.is_set_start_state:
                agent_data.agent.base_pos = mn.Vector3(
                    agent_data.cfg.start_position
                )
                agent_rot = agent_data.cfg.start_rotation
                agent_data.agent.sim_obj.rotation = mn.Quaternion(
                    mn.Vector3(agent_rot[:3]), agent_rot[3]
                )

    def __getitem__(self, key: int):
        """
        Fetches the robot data at the robot index.
        """

        return self._all_agent_data[key]

    def __len__(self):
        """
        The number of robots.
        """

        return len(self._all_agent_data)

    @property
    def robots_iter(self) -> Iterator[Agent]:
        """
        Iterator over all robot interfaces.
        """

        for agent_data in self._all_agent_data:
            yield agent_data.agent

    @property
    def grasp_iter(self) -> Iterator[RearrangeGraspManager]:
        return iter(())

    def first_setup(self):
        for robot_data in self._all_agent_data:
            ik_arm_urdf = robot_data.cfg.ik_arm_urdf
            if ik_arm_urdf is not None and self._is_pb_installed:
                robot_data._ik_helper = IkHelper(
                    robot_data.cfg.ik_arm_urdf,
                    robot_data.start_js,
                )
                # breakpoint()

    def update_agents(self):
        """
        Update all agent instance managers.
        """

        for agent_data in self._all_agent_data:
            agent_data.grasp_mgr.update()
            agent_data.agent.update()

    def update_debug(self):
        """
        Only call when in debug mode. This renders visualization helpers for
        the agents.
        """

        for agent_data in self._all_agent_data:
            agent_data.grasp_mgr.update_debug()
