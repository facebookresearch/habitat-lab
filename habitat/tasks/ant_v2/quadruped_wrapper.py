from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from os.path import exists

import attr
import magnum as mn
import numpy as np

from habitat_sim.physics import JointMotorSettings
from habitat_sim.robots.robot_interface import RobotInterface
from habitat_sim.robots.mobile_manipulator import RobotCameraParams
from habitat_sim.simulator import Simulator

@attr.s(auto_attribs=True, slots=True)
class QuadrupedRobotParams:
    hip_joints: List[int]
    ankle_joints: List[int]

    hip_init_params: Optional[List[float]]
    ankle_init_params: Optional[List[float]]

    cameras: Dict[str, RobotCameraParams]

    hip_mtr_pos_gain: float
    hip_mtr_vel_gain: float
    hip_mtr_max_impulse: float

    ankle_mtr_pos_gain: float
    ankle_mtr_vel_gain: float
    ankle_mtr_max_impulse: float

    base_offset: mn.Vector3
    base_link_names: Set[str]

class QuadrupedRobot(RobotInterface):
    """Quadruped with hip and ankle joints"""
    def __init__(
        self,
        params: QuadrupedRobotParams,
        urdf_path: str,
        sim: Simulator,
        limit_robo_joints: bool = True,
        fixed_base: bool = True,
    ):
        super().__init__()
        self.urdf_path = urdf_path
        self.params = params

        self._sim = sim
        self._limit_robo_joints = limit_robo_joints
        self._fixed_base = fixed_base

        # set up cameras
        self._cameras = defaultdict(list)
        for camera_prefix in self.params.cameras:
            for sensor_name in self._sim._sensors:
                if sensor_name.startswith(camera_prefix):
                    self._cameras[camera_prefix].append(sensor_name)
        
        # NOTE: the follow members cache static info for improved efficiency over querying the API
        # maps joint ids to motor settings for convenience
        self.joint_motors: Dict[int, Tuple[int, JointMotorSettings]] = {}
        # maps joint ids to position index
        self.joint_pos_indices: Dict[int, int] = {}
        # maps joint ids to velocity index
        self.joint_dof_indices: Dict[int, int] = {}
        self.joint_limits: Tuple[np.ndarray, np.ndarray] = None

        # defaults for optional params
        if self.params.hip_init_params is None:
            self.params.hip_init_params = [
                0 for i in range(len(self.params.hip_joints))
            ]
        if self.params.ankle_init_params is None:
            self.params.ankle_init_params = [
                0 for i in range(len(self.params.ankle_joints))
            ]

    
    def reconfigure(self) -> None:
        """Instantiates the robot the scene. Loads the URDF, sets initial state of parameters, joints, motors, etc..."""
        ao_mgr = self._sim.get_articulated_object_manager()
        print(self.urdf_path)
        print(exists(self.urdf_path))
        self.sim_obj = ao_mgr.add_articulated_object_from_urdf(
            filepath=self.urdf_path, fixed_base=self._fixed_base
        )
        print("SIM OBJ:", self.sim_obj)
        for link_id in self.sim_obj.get_link_ids():
            self.joint_pos_indices[link_id] = self.sim_obj.get_link_joint_pos_offset(
                link_id
            )
            self.joint_dof_indices[link_id] = self.sim_obj.get_link_dof_offset(link_id)
        self.joint_limits = self.sim_obj.joint_position_limits

        # remove any default damping motors
        for motor_id in self.sim_obj.existing_joint_motor_ids:
            self.sim_obj.remove_joint_motor(motor_id)
        # re-generate all joint motors with leg gains.
        jms = JointMotorSettings(
            0,  # position_target
            self.params.hip_mtr_pos_gain,  # position_gain
            0,  # velocity_target
            self.params.hip_mtr_vel_gain,  # velocity_gain
            self.params.hip_mtr_max_impulse,  # max_impulse
        )
        self.sim_obj.create_all_motors(jms)
        self._update_motor_settings_cache()

        # set correct gains for ankles
        if self.params.ankle_joints is not None:
            jms = JointMotorSettings(
                0,  # position_target
                self.params.ankle_mtr_pos_gain,  # position_gain
                0,  # velocity_target
                self.params.ankle_mtr_vel_gain,  # velocity_gain
                self.params.ankle_mtr_max_impulse,  # max_impulse
            )
            # pylint: disable=not-an-iterable
            for i in self.params.ankle_joints:
                self.sim_obj.update_joint_motor(self.joint_motors[i][0], jms)

        # set initial states and targets
        self.hip_joint_pos = self.params.hip_init_params
        self.ankle_joint_pos = self.params.ankle_init_params

        self._update_motor_settings_cache()

    def update(self) -> None:
        """Updates the camera transformations and performs necessary checks on
        joint limits and sleep states.
        """
        agent_node = self._sim._default_agent.scene_node
        inv_T = agent_node.transformation.inverted()

        for cam_prefix, sensor_names in self._cameras.items():
            for sensor_name in sensor_names:
                sens_obj = self._sim._sensors[sensor_name]._sensor_object
                cam_info = self.params.cameras[cam_prefix]

                if cam_info.attached_link_id == -1:
                    link_trans = self.sim_obj.transformation
                else:
                    link_trans = self.sim_obj.get_link_scene_node(
                        self.params.ee_link
                    ).transformation

                cam_transform = mn.Matrix4.look_at(
                    cam_info.cam_offset_pos,
                    cam_info.cam_look_at_pos,
                    mn.Vector3(0, 1, 0),
                )
                cam_transform = link_trans @ cam_transform @ cam_info.relative_transform
                cam_transform = inv_T @ cam_transform

                sens_obj.node.transformation = cam_transform

        # Guard against out of limit joints
        # TODO: should auto clamping be enabled instead? How often should we clamp?
        if self._limit_robo_joints:
            self.sim_obj.clamp_joint_limits()

        self.sim_obj.awake = True

    def reset(self) -> None:
        """Reset the joints on the existing robot.
        NOTE: only hip and ankle joint motors (not gains) are reset by default, derived class should handle any other changes."""

        # reset the initial joint positions
        self.hip_joint_pos = self.params.hip_init_params
        self.ankle_joint_pos = self.params.ankle_init_params

        self._update_motor_settings_cache()
        self.update()

    # MOTOR CONTROLS
    
    @property
    def hip_joint_pos(self) -> np.ndarray:
        """Get the current target of the hip joints motors."""
        motor_targets = np.zeros(len(self.params.hip_init_params))
        for i, jidx in enumerate(self.params.hip_joints):
            motor_targets[i] = self._get_motor_pos(jidx)
        return motor_targets

    @hip_joint_pos.setter
    def hip_joint_pos(self, ctrl: List[float]) -> None:
        """Set the desired target of the hip joint motors."""
        if len(ctrl) != len(self.params.hip_joints):
            raise ValueError("Control dimension does not match joint dimension")

        for i, jidx in enumerate(self.params.hip_joints):
            self._set_motor_pos(jidx, ctrl[i])

    @property
    def ankle_joint_pos(self) -> np.ndarray:
        """Get the current target of the hip joints motors."""
        motor_targets = np.zeros(len(self.params.ankle_init_params))
        for i, jidx in enumerate(self.params.ankle_joints):
            motor_targets[i] = self._get_motor_pos(jidx)
        return motor_targets

    @ankle_joint_pos.setter
    def ankle_joint_pos(self, ctrl: List[float]) -> None:
        """Set the desired target of the ankle joint motors."""
        if len(ctrl) != len(self.params.ankle_joints):
            raise ValueError("Control dimension does not match joint dimension")

        for i, jidx in enumerate(self.params.ankle_joints):
            self._set_motor_pos(jidx, ctrl[i])

    @property
    def leg_joint_pos(self) -> np.ndarray:
        """Get the current target of both the hip and ankle joints motors."""
        motor_targets = np.zeros(len(self.params.hip_init_params) + len(self.params.ankle_init_params))
        for i, jidx in enumerate(self.params.hip_joints + self.params.ankle_joints) :
            motor_targets[i] = self._get_motor_pos(jidx)
        
        return motor_targets

    @leg_joint_pos.setter
    def leg_joint_pos(self, ctrl: List[float]) -> None:
        """Set the desired target of both the hip and ankle joint motors."""
        if len(ctrl) != len(self.params.hip_init_params) + len(self.params.ankle_init_params):
            raise ValueError("Control dimension does not match joint dimension")

        for i, jidx in enumerate(self.params.hip_joints + self.params.ankle_joints):
            self._set_motor_pos(jidx, ctrl[i])

    def _set_motor_pos(self, joint, ctrl):
        self.joint_motors[joint][1].position_target = ctrl
        self.sim_obj.update_joint_motor(
            self.joint_motors[joint][0], self.joint_motors[joint][1]
        )
    
    def _get_motor_pos(self, joint):
        return self.joint_motors[joint][1].position_target

    #############################################
    # BASE RELATED
    #############################################

    @property
    def base_pos(self):
        """Get the robot base ground position via configured local offset from origin."""
        return self.sim_obj.translation + self.sim_obj.transformation.transform_vector(
            self.params.base_offset
        )

    @base_pos.setter
    def base_pos(self, position):
        """Set the robot base to a desired ground position (e.g. NavMesh point) via configured local offset from origin."""
        self.sim_obj.translation = (
            position
            - self.sim_obj.transformation.transform_vector(self.params.base_offset)
        )

    @property
    def base_rot(self) -> mn.Quaternion:
        return self.sim_obj.rotation

    @base_rot.setter
    def base_rot(self, rotation_y_rad: float):
        self.sim_obj.rotation = mn.Quaternion.rotation(
            mn.Rad(rotation_y_rad), mn.Vector3(1, 0, 0)
        )

    @property
    def base_transformation(self):
        return self.sim_obj.transformation

    def is_base_link(self, link_id: int) -> bool:
        return self.sim_obj.get_link_name(link_id) in self.params.base_link_names

    @property
    def base_velocity(self):
        return self.sim_obj.root_linear_velocity
    
    @property
    def base_angular_velocity(self):
        return self.sim_obj.root_angular_velocity

    @property
    def joint_velocities(self):
        return self.sim_obj.joint_velocities

    def _update_motor_settings_cache(self):
        """Updates the JointMotorSettings cache for cheaper future updates"""
        self.joint_motors = {}
        for motor_id, joint_id in self.sim_obj.existing_joint_motor_ids.items():
            self.joint_motors[joint_id] = (
                motor_id,
                self.sim_obj.get_joint_motor_settings(motor_id),
            )






