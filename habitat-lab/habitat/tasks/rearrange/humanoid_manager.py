
from dataclasses import dataclass

from typing import TYPE_CHECKING, Iterator, Optional

from typing import Iterator, Optional, List


import magnum as mn
import numpy as np

from habitat.humanoids.amass_human  import AmassHuman
from habitat.humanoids.human_base import Humanoid

from habitat.tasks.rearrange.rearrange_grasp_manager import (
    RearrangeGraspManager, HumanRearrangeGraspManager
)

from habitat.tasks.rearrange.utils import IkHelper, is_pb_installed


if TYPE_CHECKING:
    from omegaconf import DictConfig

@dataclass
class HumanData:
    """
    Data needed to manage a robot instance.
    """
    humanoid: Humanoid
    grasp_mgrs: List[RearrangeGraspManager]
    robot: Humanoid
    start_js: np.ndarray
    is_pb_installed: bool
    cfg: "DictConfig"
    _ik_helper: Optional[IkHelper] = None



    @property
    def grasp_mgr(self):
        if len(self.grasp_mgrs) == 0:
            raise Exception(
                "Human data has no grasp manager defined"
            )
        return self.grasp_mgrs[0]

    @property
    def ik_helper(self):
        if not self.is_pb_installed:
            raise ImportError(
                "Need to install PyBullet to use IK (`pip install pybullet==3.0.4`)"
            )
        return self._ik_helper


class HumanoidManager:
    """
    Creates all humanoid instances
    """

    def __init__(self, cfg, sim):
        self._sim = sim
        self.agent_names = cfg.agents
        self._all_human_data = []

        self._is_pb_installed = is_pb_installed()
        # breakpoint()
        for agent_name in cfg.agents:
            agent_cfg = cfg.agents[agent_name]
            agent_type = agent_cfg.agent_type
            assert agent_type == "AmassHuman", "Only AmassHuman is implemented"

            human_cls = AmassHuman
            human = human_cls(agent_cfg.agent_urdf, sim)


            grasp_mgr_left = HumanRearrangeGraspManager(sim, cfg, human, 1)
            grasp_mgr_right = HumanRearrangeGraspManager(sim, cfg, human, 0)
            self._all_human_data.append(
                HumanData(
                    humanoid=human,
                    grasp_mgrs=[grasp_mgr_left, grasp_mgr_right],
                    robot=human,
                    start_js=np.array(human.params.arm_init_params_left),
                    cfg=agent_cfg,
                    is_pb_installed=self._is_pb_installed
                )
            )

    def reconfigure(self, is_new_scene: bool):
        ao_mgr = self._sim.get_articulated_object_manager()
        for human_data in self._all_human_data:
            if is_new_scene:
                # robot_data.grasp_mgr.reconfigure()
                if (
                    human_data.humanoid.sim_obj is not None
                    and human_data.humanoid.sim_obj.is_alive
                ):
                    ao_mgr.remove_object_by_id(
                        human_data.humanoid.sim_obj.object_id
                    )

                human_data.humanoid.reconfigure()
            for ind in range(len(human_data.grasp_mgrs)):
                human_data.grasp_mgrs[ind].reset()

    def post_obj_load_reconfigure(self):
        """
        Called at the end of the simulator reconfigure method. Used to set the starting configurations of the humanoid if specified in the task config.
        """
        for human_data in self._all_human_data:
            # human_data.robot.params.arm_init_params = (
            #     human_data.start_js
            #     * np.random.randn(len(robot_data.start_js))
            # )
            human_data.humanoid.reset()

            # consume a fixed position from SIMUALTOR.agent_0 if configured
            if human_data.cfg.is_set_start_state:
                breakpoint()
                human_data.robot.translation_offset = mn.Vector3(
                    human_data.cfg.start_position
                )
                agent_rot = human_data.cfg.start_rotation
                human_data.robot.sim_obj.rotation = mn.Quaternion(
                    mn.Vector3(agent_rot[:3]), agent_rot[3]
                )

    def __getitem__(self, key: int):
        """
        Fetches the robot data at the robot index.
        """

        return self._all_human_data[key]

    def __len__(self):
        """
        The number of robots.
        """

        return len(self._all_human_data)

    @property
    def robots_iter(self) -> Iterator[Humanoid]:
        """
        Iterator over all robot interfaces.
        """

        for human_data in self._all_human_data:
            yield human_data.humanoid

    @property
    def grasp_iter(self) -> Iterator[RearrangeGraspManager]:
        return iter(())

    # def first_setup(self):
    #     pass

    def first_setup(self):
        for robot_data in self._all_human_data:
            ik_arm_urdf = robot_data.cfg.ik_arm_urdf
            if ik_arm_urdf is not None and self._is_pb_installed:
                robot_data._ik_helper = IkHelper(
                    robot_data.cfg.ik_arm_urdf,
                    robot_data.start_js,
                )
                # breakpoint()

    def update_agents(self):
        """
        Update all robot instance managers.
        """

        for human_data in self._all_human_data:
            human_data.humanoid.update()

    def update_debug(self):
        """
        Only call when in debug mode. This renders visualization helpers for
        the robots.
        """

        for robot_data in self._all_human_data:
            robot_data.grasp_mgr.update_debug()
