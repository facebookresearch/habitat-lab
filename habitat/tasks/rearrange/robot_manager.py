from dataclasses import dataclass
from typing import Any, Optional

import magnum as mn
import numpy as np
from yacs.config import CfgNode

# flake8: noqa
from habitat.robots import FetchRobot, FetchRobotNoWheels
from habitat.robots.fetch_suction import FetchSuctionRobot
from habitat.tasks.rearrange.rearrange_grasp_manager import (
    RearrangeGraspManager,
)
from habitat.tasks.rearrange.utils import IkHelper, is_pb_installed


@dataclass
class RobotData:
    robot: Any
    grasp_mgr: RearrangeGraspManager
    cfg: CfgNode
    start_js: np.ndarray
    is_pb_installed: bool
    _ik_helper: Optional[IkHelper] = None

    @property
    def ik_helper(self):
        if not self.is_pb_installed:
            raise ImportError(
                "Need to install PyBullet to use IK (`pip install pybullet==3.0.4`)"
            )
        return self._ik_helper


class RobotManager:
    def __init__(self, cfg, sim):
        self._sim = sim
        self._all_robot_data = []
        self._is_pb_installed = is_pb_installed()
        self.agent_names = cfg.AGENTS

        for agent_name in cfg.AGENTS:
            agent_cfg = cfg[agent_name]
            robot_cls = eval(agent_cfg.ROBOT_TYPE)
            robot = robot_cls(agent_cfg.ROBOT_URDF, sim)
            grasp_mgr = RearrangeGraspManager(sim, cfg, robot)

            if len(cfg.AGENTS) > 1:
                # Only prefix cameras if there is more than one agent.
                robot.params.cameras = {
                    f"{agent_name}_{k}": v
                    for k, v in robot.params.cameras.items()
                }
                for camera_prefix in robot.params.cameras:
                    for sensor_name in self._sim._sensors:
                        if sensor_name.startswith(camera_prefix):
                            robot._cameras[camera_prefix].append(sensor_name)

            self._all_robot_data.append(
                RobotData(
                    robot=robot,
                    grasp_mgr=grasp_mgr,
                    cfg=agent_cfg,
                    start_js=np.array(robot.params.arm_init_params),
                    is_pb_installed=self._is_pb_installed,
                )
            )

    def reconfigure(self, is_new_scene):
        ao_mgr = self._sim.get_articulated_object_manager()
        for robot_data in self._all_robot_data:
            if is_new_scene:
                robot_data.grasp_mgr.reconfigure()
                if (
                    robot_data.robot.sim_obj is not None
                    and robot_data.robot.sim_obj.is_alive
                ):
                    ao_mgr.remove_object_by_id(
                        robot_data.robot.sim_obj.object_id
                    )

                robot_data.robot.reconfigure()
            robot_data.grasp_mgr.reset()

    def post_obj_load_reconfigure(self):
        for robot_data in self._all_robot_data:
            robot_data.robot.arm_init_params = (
                robot_data.start_js
                + robot_data.cfg.JOINT_START_NOISE
                * np.random.randn(len(robot_data.start_js))
            )
            robot_data.robot.reset()

            # consume a fixed position from SIMUALTOR.AGENT_0 if configured
            if robot_data.cfg.IS_SET_START_STATE:
                robot_data.robot.base_pos = mn.Vector3(
                    robot_data.cfg.START_POSITION
                )
                agent_rot = robot_data.cfg.START_ROTATION
                robot_data.robot.sim_obj.rotation = mn.Quaternion(
                    mn.Vector3(agent_rot[:3]), agent_rot[3]
                )

    def __getitem__(self, key):
        return self._all_robot_data[key]

    def __len__(self):
        return len(self._all_robot_data)

    @property
    def robots_iter(self):
        for robot_data in self._all_robot_data:
            yield robot_data.robot

    @property
    def grasp_iter(self):
        for robot_data in self._all_robot_data:
            yield robot_data.grasp_mgr

    def first_setup(self):
        for robot_data in self._all_robot_data:
            ik_arm_urdf = robot_data.cfg.IK_ARM_URDF
            if ik_arm_urdf is not None and self._is_pb_installed:
                robot_data._ik_helper = IkHelper(
                    robot_data.cfg.IK_ARM_URDF,
                    robot_data.start_js,
                )

    def update_robots(self):
        for robot_data in self._all_robot_data:
            robot_data.grasp_mgr.update()
            robot_data.robot.update()

    def update_debug(self):
        for robot_data in self._all_robot_data:
            robot_data.grasp_mgr.update_debug()
