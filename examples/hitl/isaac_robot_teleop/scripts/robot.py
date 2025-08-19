#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from typing import Any, Dict, List, Tuple, Union

import magnum as mn
import numpy as np
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation as R

import habitat_sim
from habitat_sim.gfx import DebugLineRender
from scripts.utils import debug_draw_axis, normalize_angle

# path to this example app directory
dir_path = os.path.dirname(os.path.realpath(__file__)).split("scripts")[0]
default_pose_cache_path = os.path.join(dir_path, "robot_poses.json")

from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.types import ArticulationAction
from omni.physx import get_physx_simulation_interface
from pxr import PhysxSchema, Usd, UsdPhysics  # Sdf UsdGeom

from habitat.isaac_sim import isaac_prim_utils

# NOTE: must be imported after Isaac initialization


class LinkSubset:
    """
    A class to encapsulate logic related to querying properties of a grouped subset of links.
    This class is not intended to be used for direct reference to DoFs. See ConfigurationSubset.
    """

    def __init__(
        self,
        robot: "RobotAppWrapper",
        links: Union[List[int], List[str]] = None,
    ) -> None:
        self._robot = robot
        self.link_ixs: List[int] = None
        if links is None:
            # by default collect all links
            self.link_ixs = list(
                range(len(self._robot._body_prim_paths))
            )  # type:ignore[assignment]
        else:
            if isinstance(links[0], int):
                self.link_ixs = links  # type:ignore[assignment]
            else:
                self.link_ixs = []
                model_body_paths = self._robot._body_prim_paths
                for link_name in links:
                    found = False
                    for ix, body_path in enumerate(model_body_paths):
                        if link_name in body_path:  # type:ignore[operator]
                            self.link_ixs.append(ix)
                            found = True
                            break
                    if not found:
                        raise ValueError(f"No body for link named {link_name}")


class ConfigurationSubset:
    """
    A class to encapsulate logic related to querying, setting, and caching labeled subsets of the full robot configuration.
    For example, a single hand, arm, or wheeled base is controlled by a subset of the full configuration.
    """

    def __init__(
        self,
        robot: "RobotAppWrapper",
        dofs: Union[List[int], List[str]] = None,
    ) -> None:
        self._robot: "RobotAppWrapper" = robot
        if dofs is None:
            # by default collect all 1 DoF links
            self.joint_ixs = range(self._robot._robot.num_dof)
        else:
            if isinstance(dofs[0], int):
                # dof indices
                self.joint_ixs = dofs  # type:ignore[assignment]
            else:
                # dof names
                model_dof_names = self._robot._robot.dof_names
                self.joint_ixs = [  # type:ignore[assignment]
                    model_dof_names.index(dof_name) for dof_name in dofs
                ]

    def set_pos_from_full(self, all_joint_pos: List[float]) -> None:
        """
        Set this subset configuration from a fullbody configuration input.
        """
        specific_dofs = [
            x for ix, x in enumerate(all_joint_pos) if ix in self.joint_ixs
        ]
        self._robot._robot.set_joint_positions(
            positions=np.array(specific_dofs), joint_indices=self.joint_ixs
        )

    def set_pos(self, joint_positions: List[float]) -> None:
        """
        Set this configuration subset from a precisely sized list of joint positions.
        NOTE: Input joint positions list must be the same size as this configuration subset. To set this subset from a full pose use set_pos_from_full instead.
        """
        self._robot._robot.set_joint_positions(
            positions=np.array(joint_positions), joint_indices=self.joint_ixs
        )

    def get_pos(self) -> List[float]:
        """
        Get the current configuration of the configured subset of joints.
        """
        return self._robot._robot.get_joint_positions(self.joint_ixs)

    def clear_velocities(self) -> None:
        """
        Clear all velocities associated with this configuration subset.
        """
        self._robot._robot.set_joint_velocities(
            velocities=np.zeros(len(self.joint_ixs)),
            joint_indices=self.joint_ixs,
        )

    def set_velocities(self, velocities: List[float]) -> None:
        """
        Set the instantaneous joint velocities for this configuration subset.
        """
        self._robot._robot.set_joint_velocities(
            velocities=np.array(velocities),
            joint_indices=self.joint_ixs,
        )

    def set_motor_velocities(self, velocities: List[float]) -> None:
        """
        Sets target motor velocities for this configuration subset.
        """
        action = ArticulationAction(
            joint_velocities=np.array(velocities),
            joint_indices=self.joint_ixs,
        )
        self._robot._robot.apply_action(action)

    def set_motor_pos_from_full(
        self, all_joint_pos_targets: List[float]
    ) -> None:
        """
        Set this subset configuration's joint motor targets from a fullbody joint position target input.
        """
        specific_dofs = [
            x
            for ix, x in enumerate(all_joint_pos_targets)
            if ix in self.joint_ixs
        ]
        action = ArticulationAction(
            joint_positions=np.array(specific_dofs),
            joint_indices=self.joint_ixs,
        )
        self._robot._robot.apply_action(action)

    def set_motor_pos(
        self, motor_targets: List[float], clamp: bool = True
    ) -> None:
        """
        Set this configuration subset's joint motor targets from a precisely sized list of joint position targets.
        NOTE: Input joint position targets list must be the same size as this configuration subset. To set this subset from a full pose use set_motor_pos_from_full instead.
        """
        action = ArticulationAction(
            joint_positions=np.array(motor_targets),
            joint_indices=self.joint_ixs,
        )
        self._robot._robot.apply_action(action)

    def get_motor_pos(self) -> List[float]:
        """
        Get the current motor targets for the subset.
        Assumes exactly one motor per joint.
        """
        applied_action = self._robot._robot.get_applied_action().get_dict()[
            "joint_positions"
        ]

        subset_target = [applied_action[ix] for ix in self.joint_ixs]
        return subset_target

    def set_cached_pose(
        self,
        pose_file: str = default_pose_cache_path,
        pose_name: str = "default",
        set_motor_targets: bool = False,
        set_positions: bool = False,
    ) -> None:
        """
        Loads a robot pose from a json file which could have multiple poses.
        """

        # fetch and validate the pose
        pose = self.fetch_cached_pose(pose_file, pose_name)

        if pose == None:
            return

        if len(pose) == self._robot._robot.num_dof:
            # this is a full pose subset so use the shortcut APIs
            if set_positions:
                self.set_pos_from_full(pose)
            if set_motor_targets:
                self.set_motor_pos_from_full(pose)
            return

        assert len(pose) == len(self.joint_ixs)
        # this is a partial pose subset, so set each dof from a full pose
        if set_motor_targets:
            self.set_motor_pos(pose)
        if set_positions:
            self.set_pos(pose)

    def fetch_cached_pose(
        self,
        pose_file: str = default_pose_cache_path,
        pose_name: str = "default",
    ) -> List[float]:
        """
        Reads a pose cache file and returns this ConfigurationSubset's pose vector from the cache without changing the robot's state.
        NOTE: Use this to deserialize and store pose waypoints for applications such as LERPing between cached poses.
        Returns None if the requested operation is invalid.
        """
        if not os.path.exists(pose_file):
            print(
                f"Cannot load cached pose. Configured pose file {pose_file} does not exist."
            )
            return None

        with open(pose_file, "r") as f:
            poses = json.load(f)
            if self._robot._robot_prim_path not in poses:
                print(
                    f"Cannot load cached pose. No poses cached for robot {self._robot._robot_prim_path}."
                )
                return None
            if pose_name not in poses[self._robot._robot_prim_path]:
                print(
                    f"Cannot load cached pose. No pose named {pose_name} cached for robot {self._robot._robot_prim_path}. Options are {poses[self._robot._robot_prim_path].keys()}"
                )
                return None
            pose = poses[self._robot._robot_prim_path][pose_name]
            if len(pose) == self._robot._robot.num_dof:
                # loaded a full pose so cut it down if necessary
                if len(pose) != len(self.joint_ixs):
                    pose = [pose[ix] for ix in self.joint_ixs]
            elif len(pose) == len(self.joint_ixs):
                # subset pose is correctly sized so no work
                pass
            else:
                print(
                    f"Cannot load cached pose (size {len(pose)}) as it does not match number of dofs ({self._robot._robot.num_dof}) full or {len(self.joint_ixs)} subset)"
                )
                return None
            return pose

    def cache_pose(
        self,
        pose_file: str = default_pose_cache_path,
        pose_name: str = "default",
    ) -> None:
        """
        Saves the current subset pose state in a json cache file with the given name.
        """
        # create the directory if it doesn't exist
        cache_dir = pose_file[: -len(pose_file.split("/")[-1])]
        os.makedirs(cache_dir, exist_ok=True)
        poses = {}
        if os.path.exists(pose_file):
            # load existing contents
            with open(pose_file, "r") as f:
                poses = json.load(f)
        if self._robot._robot_prim_path not in poses:
            poses[self._robot._robot_prim_path] = {}
        full_pose = self._robot._robot.get_joint_positions()
        # NOTE: Isaac returns float32 which are not directly serializable to JSON without casting
        subset_pose = [float(full_pose[ix]) for ix in self.joint_ixs]
        poses[self._robot._robot_prim_path][pose_name] = subset_pose
        print(poses)
        with open(pose_file, "w") as f:
            json.dump(
                poses,
                f,
                indent=4,
            )


# class FingerRaycastSensor:
#     """
#     A class to encapsulate the necessary logic for using raycasting between fingertips to detect an object within the "pre-grasp" volume of the hand.
#     """

#     def __init__(
#         self,
#         sim: habitat_sim.Simulator,
#         ao: habitat_sim.physics.ManagedArticulatedObject,
#         thumb_link: LinkSubset,
#         finger_links: LinkSubset,
#     ):
#         """
#         Requires: the simulator, the ArticulatedObject, a thumb link, some finger links
#         """
#         self.sim = sim
#         self.ao = ao
#         self.obj_ids = [self.ao.object_id] + list(
#             self.ao.link_object_ids.keys()
#         )
#         self.thumb = thumb_link
#         self.fingers = finger_links
#         assert len(self.thumb.link_ixs) > 0
#         assert len(self.fingers.link_ixs) > 0
#         # cache the most recent update for debug drawing and query. Keyed by (thumb,finger) link pairs.
#         self.results: Dict[Tuple[int, int], RaycastResults] = None

#     def update_sensor_raycasts(self) -> None:
#         """
#         Runs the raycasting routine to fill the RaycastResults cache.
#         """
#         self.results = {}
#         for thumb_link in self.thumb.link_ixs:
#             thumb_pos = self.ao.get_link_scene_node(thumb_link).translation
#             for finger_link in self.fingers.link_ixs:
#                 finger_pos = self.ao.get_link_scene_node(
#                     finger_link
#                 ).translation
#                 # NOTE: scaled to distance between the finger tips such that valid distances are [0,1]
#                 ray = Ray(thumb_pos, finger_pos - thumb_pos)
#                 results = self.sim.cast_ray(ray)
#                 self.results[(thumb_link, finger_link)] = results

#     def get_obj_ids_in_sensor_reading(
#         self, update_sensor: bool = True
#     ) -> Dict[int, float]:
#         """
#         Checks the raycast and returns a dict keyed by object_id of any objects detected by the sensor and mapping to the ratio of rays for that object.
#         """
#         if update_sensor:
#             self.update_sensor_raycasts()
#         if self.results is None:
#             print(
#                 "FingerRaycastSensor:get_obj_ids_in_sensor_reading - No raycast results cached, returning null sensor reading."
#             )
#             return {}
#         num_rays = len(self.thumb.link_ixs) * len(self.fingers.link_ixs)
#         detected_objs: Dict[int, float] = {}
#         for ray_result in self.results.values():
#             if ray_result.has_hits():
#                 # find any non-robot hits between the finger and thumb and record them
#                 for hit in ray_result.hits:
#                     # if a hit is greater distance than 1 it is outside the grip
#                     if hit.ray_distance > 1:
#                         break
#                     if hit.object_id not in self.obj_ids:
#                         if hit.object_id not in detected_objs:
#                             detected_objs[hit.object_id] = 1
#                         else:
#                             detected_objs[hit.object_id] += 1
#         for detected_obj, num_hits in detected_objs.items():
#             detected_objs[detected_obj] = num_hits / num_rays
#         return detected_objs

#     def get_scalar_sensor_reading(self, update_sensor: bool = True) -> float:
#         """
#         Checks the raycast and returns the ratio between [0,1] of detected hits.
#         Set `update_sensor=False` to re-interpret the previous results without recasting the rays.
#         """
#         if update_sensor:
#             self.update_sensor_raycasts()
#         if self.results is None:
#             print(
#                 "FingerRaycastSensor:get_sensor_reading - No raycast results cached, returning null sensor reading."
#             )
#             return None
#         hits = 0
#         for ray_result in self.results.values():
#             if ray_result.has_hits():
#                 # find the first non-robot hit between the finger and thumb
#                 for hit in ray_result.hits:
#                     # if a hit is greater distance than 1 it is outside the grip
#                     if hit.ray_distance > 1:
#                         break
#                     if hit.object_id not in self.obj_ids:
#                         hits += 1
#                         break
#         return hits / (len(self.thumb.link_ixs) * len(self.fingers.link_ixs))

#     def draw(self, dblr: DebugLineRender) -> None:
#         """
#         Debug draw the current state of the sensor.
#         """
#         if self.results is None:
#             return
#         for ray_result in self.results.values():
#             hit_positions = []
#             # collect the hit points
#             if ray_result.has_hits():
#                 # find the first non-robot hit between the finger and thumb
#                 for hit in ray_result.hits:
#                     # if a hit is greater distance than 1 it is outside the grip
#                     if hit.ray_distance > 1:
#                         break
#                     if hit.object_id not in self.obj_ids:
#                         hit_positions.append(hit.point)
#                         break
#             # color based on results
#             color = mn.Color4.green()
#             if len(hit_positions) > 0:
#                 color = mn.Color4.red()
#             # draw the ray
#             dblr.draw_transformed_line(
#                 ray_result.ray.origin,
#                 ray_result.ray.origin + ray_result.ray.direction,
#                 color,
#             )
#             # draw the hit points
#             for hit_point in hit_positions:
#                 # draw the navmesh circle
#                 dblr.draw_circle(
#                     hit_point,
#                     radius=0.025,
#                     color=mn.Color4.yellow(),
#                     normal=ray_result.ray.direction,
#                 )


class RobotBaseVelController:
    """
    Encapsulates logic for controlling the robot's base via linear and angular velocity commands.
    """

    def __init__(self, robot: "RobotAppWrapper"):
        self.robot = robot
        # NOTE: we assume the wheeled mobile base can only turn in place and move forward/backward
        self.target_linear_vel: float = 0
        self.target_angular_vel: float = 0

        # optional waypoint control
        self._track_waypoints: bool = False
        self._pause_track_waypoints: bool = False
        self.max_linear_speed = self.robot.robot_cfg.max_linear_speed
        self.max_angular_speed = self.robot.robot_cfg.max_angular_speed
        self._target_rotation: float = 0
        self.target_position: mn.Vector3 = mn.Vector3()

    @property
    def target_rotation(self) -> float:
        return self._target_rotation

    @target_rotation.setter
    def target_rotation(self, target_rot: float):
        self._target_rotation = normalize_angle(target_rot)

    def apply(self, dt: float):
        """
        Apply the target velocities given the timestep.
        NOTE: should be called in the robot's physics callback
        NOTE: should be called BEFORE height fix velocity controller
        """
        pos, rot = self.robot.get_root_pose()

        max_pos_error = 0.035
        max_ang_error = 0.035
        if self._track_waypoints and not self._pause_track_waypoints:
            pos_error = self.target_position - pos
            pos_error[1] = 0
            dist_to_target = pos_error.length()
            if dist_to_target > max_pos_error:
                # first move to the position waypoint
                angle_to_waypoint = self.robot.angle_to(pos_error)
                if abs(angle_to_waypoint) > max_ang_error:
                    one_step_correction = angle_to_waypoint / dt
                    self.target_angular_vel = min(
                        self.max_angular_speed,
                        max(-self.max_angular_speed, one_step_correction),
                    )
                    self.target_linear_vel = 0
                else:
                    self.target_linear_vel = min(
                        self.max_linear_speed,
                        max(-self.max_linear_speed, dist_to_target / dt),
                    )
                    self.target_angular_vel = 0
            else:
                self.target_linear_vel = 0
                # if in the correct position, rotate to desired angle
                # NOTE: target angle is always in the range [-pi, pi] by construction
                angle_error = normalize_angle(
                    self.target_rotation - self.robot.base_rot
                )
                if abs(angle_error) > max_ang_error:
                    self.target_angular_vel = min(
                        self.max_angular_speed,
                        max(-self.max_angular_speed, angle_error / dt),
                    )
                else:
                    # REACHED THE GOAL
                    self.target_angular_vel = 0

        # NOTE: we zero lateral and vertical velocities
        forward_dir = rot.transform_vector(mn.Vector3(1, 0, 0))
        forward_dir[1] = 0

        if self.target_linear_vel != 0:
            desired_linear_vel = (
                forward_dir.normalized() * self.target_linear_vel
            )
            self.robot._robot.set_linear_velocity(
                isaac_prim_utils.habitat_to_usd_position(desired_linear_vel)
            )

        if self.target_angular_vel != 0:
            desired_angular_vel = [0, 0, self.target_angular_vel]
            self.robot._robot.set_angular_velocity(desired_angular_vel)

    def reset(self):
        """
        Zeros any target velocities.
        """
        self.target_linear_vel = 0
        self.target_angular_vel = 0
        self._pause_track_waypoints = False

        # set waypoint targets to current state (stops the robot)
        if self._track_waypoints:
            self.target_rotation = self.robot.base_rot
            self.target_position, _ = self.robot.get_root_pose()

    @property
    def track_waypoints(self) -> bool:
        return self._track_waypoints

    @track_waypoints.setter
    def track_waypoints(self, track_waypoints: bool) -> None:
        """
        Set the controller to track waypoints or not.
        If setting to True from False the waypoint targets are set to the current state.
        """
        if track_waypoints == self._track_waypoints:
            return
        self._track_waypoints = track_waypoints
        self.reset()

    def debug_draw_waypoint(self, dblr: DebugLineRender) -> None:
        """
        Draw a frame at the configured waypoint.
        """
        rot = mn.Quaternion.rotation(
            mn.Rad(self.target_rotation), mn.Vector3(0, 1, 0)
        )
        tform = mn.Matrix4.from_(rot.to_matrix(), self.target_position)
        debug_draw_axis(dblr, tform)


class RobotAppWrapper:
    """
    Wrapper class for robots imported as simulated ArticulatedObjects.
    Wraps the ManagedObjectAPI.
    """

    def __init__(
        self,
        isaac_service,
        sim: habitat_sim.Simulator,
        robot_cfg: DictConfig,
        instance_id: int = 0,
    ):
        """
        Initialize the robot in a Simulator from its config object.
        """

        self.sim = sim
        self.robot_cfg = robot_cfg

        # apply this local viewpoint offset to the ao.translation to align the cursor
        self.viewpoint_offset = mn.Vector3(robot_cfg.viewpoint_offset)

        # Isaac setup
        self._isaac_service = isaac_service
        self._robot_prim_path = (
            f"/World/env_{instance_id}/{self.robot_cfg.robot_prim_path_name}"
        )

        # here we add the robot to the USD contents
        from omni.isaac.core.utils.stage import add_reference_to_stage

        add_reference_to_stage(
            usd_path=self.robot_cfg.usd, prim_path=self._robot_prim_path
        )
        self._isaac_service.usd_visualizer.on_add_reference_to_stage(
            usd_path=self.robot_cfg.usd, prim_path=self._robot_prim_path
        )

        self.robot_prim = self._isaac_service.world.stage.GetPrimAtPath(
            self._robot_prim_path
        )
        if not self.robot_prim.IsValid():
            raise ValueError(f"Prim at {self._robot_prim_path} is not valid.")

        # create joint motors from configured settings
        self.motor_params = {
            "max_joint_velocity": self.robot_cfg.max_joint_velocity,
            "angular_drive_stiffness": self.robot_cfg.angular_drive_stiffness,
            "angular_drive_damping": self.robot_cfg.angular_drive_damping,
            "angular_drive_max_force": self.robot_cfg.angular_drive_max_force,
            "linear_drive_stiffness": self.robot_cfg.linear_drive_stiffness,
        }
        self.set_motor_params(self.motor_params)

        # create rigid object API
        for prim in Usd.PrimRange(self.robot_prim):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                # UsdPhysics.RigidBodyAPI doesn't support damping but PhysxRigidBodyAPI does
                if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                    physx_api = PhysxSchema.PhysxRigidBodyAPI(prim)
                else:
                    physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)

                # todo: decide hard-coded values here
                physx_api.CreateLinearDampingAttr(
                    self.robot_cfg.angular_damping
                )
                physx_api.CreateAngularDampingAttr(
                    self.robot_cfg.linear_damping
                )

        self._name = f"{self.robot_cfg.robot_prim_path_name}_{instance_id}"
        self._robot: Robot = self._isaac_service.world.scene.add(
            Robot(prim_path=self._robot_prim_path, name=self._name)
        )
        self._robot_controller = self._robot.get_articulation_controller()
        self.approximate_arm_length = self.robot_cfg.approximate_arm_length

        self.right_arm_locked = self.robot_cfg.ik.right_arm_locked
        self.left_arm_locked = self.robot_cfg.ik.left_arm_locked

        # control waypoints for the platform
        # NOTE: should be set from navmesh
        self.target_base_height = 0.1
        # how much extra clearance to add between navmesh heights and base position heights
        # prevents the robot from getting stuck in the ground
        self.ground_to_base_offset = self.robot_cfg.ground_to_base_offset
        # TODO: configure the base controller from settings yaml
        self.base_vel_controller = RobotBaseVelController(self)
        # if true, use the velocity based base constraint control, otherwise fully dynamic
        self.do_vel_fix_base = True
        self._do_kin_fixed_base = False
        self.kin_fixed_base_state: Tuple[
            List[float], List[float], mn.Vector3
        ] = None
        # datastructure to aggregate the results of contact callbacks in a queryable format
        self.contact_state: Dict[Any, Any] = {}
        self._in_contact = False
        # NOTE: filtering contacts is relatively expensive so we'll only do so when necessary
        self._contact_sensors_active = True

    def snap_to_navmesh(self, pathfinder):
        """
        Snaps the base position to the navmesh.
        """
        pos = self.get_root_pose()[0]
        nav_pos = pathfinder.snap_point(pos)
        if not np.isnan(nav_pos).any():
            self.set_root_pose(pos=nav_pos)
            self.target_base_height = nav_pos[1] + self.ground_to_base_offset

    @property
    def do_kin_fixed_base(self) -> bool:
        return self._do_kin_fixed_base

    @do_kin_fixed_base.setter
    def do_kin_fixed_base(self, value: bool) -> None:
        """
        To set kinematic fixe base we record the current state to maintain.
        """
        if value:
            # self.kin_fixed_base_state = self.get_root_pose()
            usd_world_pose = self._robot.get_world_pose()
            _, rot = self.get_root_pose()
            glob_forward = rot.transform_vector(mn.Vector3(1.0, 0, 0))
            self.kin_fixed_base_state = (
                usd_world_pose[0],
                usd_world_pose[1],
                glob_forward,
            )
            # self.do_vel_fix_base = False
        # else:
        # self.do_vel_fix_base = True
        self._do_kin_fixed_base = value

    def set_motor_params(self, motor_params: Dict[str, Any]) -> None:
        """
        Set the motor parameters for the robot using provided settings
        """

        for prim in Usd.PrimRange(self.robot_prim):
            # joint vel max
            if prim.HasAPI(PhysxSchema.PhysxJointAPI):
                joint_api = PhysxSchema.PhysxJointAPI(prim)
                joint_api.GetMaxJointVelocityAttr().Set(
                    motor_params["max_joint_velocity"]
                )

            # drivers
            if prim.HasAPI(UsdPhysics.DriveAPI):
                # Access the existing DriveAPI
                drive_api = UsdPhysics.DriveAPI(prim, "angular")
                if drive_api:
                    # Modify drive parameters
                    drive_api.GetStiffnessAttr().Set(
                        motor_params["angular_drive_stiffness"]
                    )  # Position gain
                    drive_api.GetDampingAttr().Set(
                        motor_params["angular_drive_damping"]
                    )  # Velocity gain
                    drive_api.GetMaxForceAttr().Set(
                        motor_params["angular_drive_max_force"]
                    )  # Maximum force/torque

                drive_api = UsdPhysics.DriveAPI(prim, "linear")
                if drive_api:
                    drive_api = UsdPhysics.DriveAPI.Get(prim, "linear")
                    drive_api.GetStiffnessAttr().Set(
                        motor_params["linear_drive_stiffness"]
                    )  # Example for linear stiffness

    def get_global_view_offset(self) -> mn.Vector3:
        """
        Applies the robot's local to global transform to the view offset vector bringing it into global space.
        """
        pos, rot = self.get_root_pose()
        return rot.transform_vector(self.viewpoint_offset) + pos

    def post_init(self):
        """
        Called after the Robot object is initialized by a world reset.
        Collects pose subsets and set the initial pose.
        """
        print("DOF NAMES:")
        for ix, name in enumerate(self._robot.dof_names):
            print(f"{ix}: {name}")
        print("")

        # define configuration subsets
        self.pos_subsets = {"full": ConfigurationSubset(self)}

        # load the configuration subsets from the robot config file
        for (
            subset_cfg_name,
            subset_cfg_links,
        ) in self.robot_cfg.configuration_subsets.items():
            print(subset_cfg_name)
            self.pos_subsets[subset_cfg_name] = ConfigurationSubset(
                self, dofs=subset_cfg_links
            )
            print(self.pos_subsets[subset_cfg_name].joint_ixs)

        # left_arm
        # [12, 14, 16, 18, 20, 22, 24]
        # right_arm
        # [13, 15, 17, 19, 21, 23, 25]
        # left_hand
        # [26, 34, 42, 50, 28, 36, 44, 52, 29, 37, 45, 53, 27, 35, 43, 51]
        # right_hand
        # [30, 38, 46, 54, 32, 40, 48, 56, 33, 41, 49, 57, 31, 39, 47, 55]

        # # setup the finger raycast sensors
        # self.finger_raycast_sensors = [
        #     FingerRaycastSensor(
        #         self.sim,
        #         self.ao,
        #         self.link_subsets["left_thumb_tip"],
        #         self.link_subsets["left_finger_tips"],
        #     ),
        #     FingerRaycastSensor(
        #         self.sim,
        #         self.ao,
        #         self.link_subsets["right_thumb_tip"],
        #         self.link_subsets["right_finger_tips"],
        #     ),
        # ]

        # beware this poses the object
        self._create_rigid_prim_view()

        # cache of states updated each step to reduce transform queries
        self._body_prim_states: Tuple[
            List[mn.Vector3], List[mn.Quaternion]
        ] = None
        self._body_prim_states_dirty: bool = True
        self.update_body_prim_states()

        self._joint_relationships = self._collect_joint_relationships()
        self._joint_names_to_dof_ix = self._construct_dof_name_map()

        print("BODY NAMES")
        for ix, name in enumerate(self._body_prim_paths):
            print(f"{ix}: {name}")
        # load link subsets from the robot config file
        self.link_subsets: Dict[str, LinkSubset] = {}
        for (
            subset_cfg_name,
            subset_cfg_links,
        ) in self.robot_cfg.link_subsets.items():
            self.link_subsets[subset_cfg_name] = LinkSubset(
                self, subset_cfg_links
            )

        # set initial pose
        self.set_cached_pose(
            pose_name=self.robot_cfg.initial_pose,
            set_positions=True,
            set_motor_targets=True,
        )

        # add the physics callback which constrains the base position via velocity
        # NOTE: this function will be called internally before each isaac step
        self._isaac_service.world.add_physics_callback(
            f"{self._name}_physics_callback", callback_fn=self.physics_callback
        )
        # set the initial base targets from the initial state
        self.base_vel_controller.track_waypoints = (
            self.robot_cfg.track_waypoints
        )

        self.apply_contact_report_sensors()

    def apply_contact_report_sensors(self):
        """
        Here we are hooking the robot's links into a contact callback.
        NOTE: We also set the sensitivity of the contact report to cull small erroneous contacts.
        """

        # NOTE: hardcoded for now because iteration caused some (e.g. base) to invalidate the physx view
        self.contact_sensor_links = [
            "/World/env_0/Murp/base_link",
            "/World/env_0/Murp/torso_link",
            "/World/env_0/Murp/left_base",
            "/World/env_0/Murp/left_fr3_link0",
            "/World/env_0/Murp/left_fr3_link1",
            "/World/env_0/Murp/left_fr3_link2",
            "/World/env_0/Murp/left_fr3_link3",
            "/World/env_0/Murp/left_fr3_link4",
            "/World/env_0/Murp/left_fr3_link5",
            "/World/env_0/Murp/left_fr3_link6",
            "/World/env_0/Murp/left_fr3_link7",
            "/World/env_0/Murp/left_fr3_link8",
            "/World/env_0/Murp/base_link_lhand",
            "/World/env_0/Murp/link_l0_0",
            "/World/env_0/Murp/link_l1_0",
            "/World/env_0/Murp/link_l2_0",
            "/World/env_0/Murp/link_l3_0",
            "/World/env_0/Murp/link_l3_0_digit2_sensor_base",
            "/World/env_0/Murp/link_l3_0_tip",
            "/World/env_0/Murp/link_l12_0",
            "/World/env_0/Murp/link_l13_0",
            "/World/env_0/Murp/link_l14_0",
            "/World/env_0/Murp/link_l15_0",
            "/World/env_0/Murp/link_l15_0_digit2_sensor_base",
            "/World/env_0/Murp/link_l15_0_tip",
            "/World/env_0/Murp/link_l4_0",
            "/World/env_0/Murp/link_l5_0",
            "/World/env_0/Murp/link_l6_0",
            "/World/env_0/Murp/link_l7_0",
            "/World/env_0/Murp/link_l7_0_digit2_sensor_base",
            "/World/env_0/Murp/link_l7_0_tip",
            "/World/env_0/Murp/link_l8_0",
            "/World/env_0/Murp/link_l9_0",
            "/World/env_0/Murp/link_l10_0",
            "/World/env_0/Murp/link_l11_0",
            "/World/env_0/Murp/link_l11_0_digit2_sensor_base",
            "/World/env_0/Murp/link_l11_0_tip",
            "/World/env_0/Murp/right_base",
            "/World/env_0/Murp/right_fr3_link0",
            "/World/env_0/Murp/right_fr3_link1",
            "/World/env_0/Murp/right_fr3_link2",
            "/World/env_0/Murp/right_fr3_link3",
            "/World/env_0/Murp/right_fr3_link4",
            "/World/env_0/Murp/right_fr3_link5",
            "/World/env_0/Murp/right_fr3_link6",
            "/World/env_0/Murp/right_fr3_link7",
            "/World/env_0/Murp/right_fr3_link8",
            "/World/env_0/Murp/base_link_hand",
            "/World/env_0/Murp/link_0_0",
            "/World/env_0/Murp/link_1_0",
            "/World/env_0/Murp/link_2_0",
            "/World/env_0/Murp/link_3_0",
            "/World/env_0/Murp/link_3_0_digit2_sensor_base",
            "/World/env_0/Murp/link_3_0_tip",
            "/World/env_0/Murp/link_12_0",
            "/World/env_0/Murp/link_13_0",
            "/World/env_0/Murp/link_14_0",
            "/World/env_0/Murp/link_15_0",
            "/World/env_0/Murp/link_15_0_digit2_sensor_base",
            "/World/env_0/Murp/link_15_0_tip",
            "/World/env_0/Murp/link_4_0",
            "/World/env_0/Murp/link_5_0",
            "/World/env_0/Murp/link_6_0",
            "/World/env_0/Murp/link_7_0",
            "/World/env_0/Murp/link_7_0_digit2_sensor_base",
            "/World/env_0/Murp/link_7_0_tip",
            "/World/env_0/Murp/link_8_0",
            "/World/env_0/Murp/link_9_0",
            "/World/env_0/Murp/link_10_0",
            "/World/env_0/Murp/link_11_0",
            "/World/env_0/Murp/link_11_0_digit2_sensor_base",
            "/World/env_0/Murp/link_11_0_tip",
        ]
        for prim_path in self.contact_sensor_links:
            my_prim = self._isaac_service.world.stage.GetPrimAtPath(prim_path)
            contactReportAPI = PhysxSchema.PhysxContactReportAPI.Apply(my_prim)
            # NOTE: this should be tuned empirically for application.
            contactReportAPI.CreateThresholdAttr().Set(1.0)

        self._contact_report_sub = (
            get_physx_simulation_interface().subscribe_contact_report_events(
                self._on_contact_report_event
            )
        )

    def disable_contact_report_sensors(self):
        """
        Disables the contact sensors assuming the've been created already.
        """
        for prim_path in self.contact_sensor_links:
            my_prim = self._isaac_service.world.stage.GetPrimAtPath(prim_path)
            my_prim.RemoveAPI(PhysxSchema.PhysxContactReportAPI)

    def enable_contact_report_sensors(self):
        """
        Called to re-enable contact sensors after the've been disabled.
        """
        for prim_path in self.contact_sensor_links:
            my_prim = self._isaac_service.world.stage.GetPrimAtPath(prim_path)
            contactReportAPI = PhysxSchema.PhysxContactReportAPI.Apply(my_prim)
            contactReportAPI.CreateThresholdAttr().Set(1.0)

    def _on_contact_report_event(self, contact_headers, contact_data):
        """
        Callback function triggered within physx loop when contacts are detected.
        Populates internal datastructures for later query.
        #NOTE: this function profiled on Lambda at 0.0015sec in heavy contact (both hands)
        """
        if not self._contact_sensors_active:
            return

        from omni.physx.bindings._physx import ContactEventType
        from pxr import PhysicsSchemaTools

        # reset internal datastructures
        self._in_contact = False
        self.contact_state = {}

        # TODO: this should fill our datastructure which we will query when we need recent contact info
        for contact_header in contact_headers:
            # print("Got contact header type: " + str(contact_header.type))
            actor0 = PhysicsSchemaTools.intToSdfPath(contact_header.actor0)
            actor1 = PhysicsSchemaTools.intToSdfPath(contact_header.actor1)
            if (
                contact_header.type == ContactEventType.CONTACT_PERSIST
                or contact_header.type == ContactEventType.CONTACT_FOUND
            ) and (
                actor0 in self.contact_sensor_links
                or actor1 in self.contact_sensor_links
            ):
                self._in_contact = True

                print(
                    "Actor0: "
                    + str(
                        PhysicsSchemaTools.intToSdfPath(contact_header.actor0)
                    )
                )
                print(
                    "Actor1: "
                    + str(
                        PhysicsSchemaTools.intToSdfPath(contact_header.actor1)
                    )
                )
            continue

            # print(
            #     "Collider0: "
            #     + str(
            #         PhysicsSchemaTools.intToSdfPath(contact_header.collider0)
            #     )
            # )
            # print(
            #     "Collider1: "
            #     + str(
            #         PhysicsSchemaTools.intToSdfPath(contact_header.collider1)
            #     )
            # )
            # print("StageId: " + str(contact_header.stage_id))
            # print(
            #     "Number of contacts: " + str(contact_header.num_contact_data)
            # )

            # contact_data_offset = contact_header.contact_data_offset
            # num_contact_data = contact_header.num_contact_data

            # for index in range(
            #     contact_data_offset, contact_data_offset + num_contact_data, 1
            # ):
            #     print("Contact:")
            #     print("Contact position: " + str(contact_data[index].position))
            #     print("Contact normal: " + str(contact_data[index].normal))
            #     print("Contact impulse: " + str(contact_data[index].impulse))
            #     print(
            #         "Contact separation: "
            #         + str(contact_data[index].separation)
            #     )
            #     print(
            #         "Contact faceIndex0: "
            #         + str(contact_data[index].face_index0)
            #     )
            #     print(
            #         "Contact faceIndex1: "
            #         + str(contact_data[index].face_index1)
            #     )
            #     print(
            #         "Contact material0: "
            #         + str(
            #             PhysicsSchemaTools.intToSdfPath(
            #                 contact_data[index].material0
            #             )
            #         )
            #     )
            #     print(
            #         "Contact material1: "
            #         + str(
            #             PhysicsSchemaTools.intToSdfPath(
            #                 contact_data[index].material1
            #             )
            #         )
            #     )

    @property
    def in_contact(self) -> bool:
        """
        Boolean contact function for the full body.
        Useful for rejection sampling.
        NOTE: does not update from state without Isaac simulation step.
        """
        return self._in_contact

    def physics_callback(self, step_size: float):
        """
        A callback function which is called withing isaac's sim loop before each step
        Applies velocities to constrain the robot's base position.
        """
        base_position, base_orientation = self._robot.get_world_pose()

        # update the target base height from navmesh if possible
        # NOTE: as the robot base drifts we want to keep this updated
        if self.sim.pathfinder.is_loaded:
            hab_base_pos = mn.Vector3(
                isaac_prim_utils.usd_to_habitat_position(base_position)
            )
            if self.sim.pathfinder.is_navigable(hab_base_pos):
                self.target_base_height = (
                    self.sim.pathfinder.snap_point(hab_base_pos)[1]
                    + self.ground_to_base_offset
                )
            else:
                self.snap_to_navmesh(self.sim.pathfinder)

        self.base_vel_controller.apply(step_size)

        # NOTE: this is a velocity hack to constrain the base position. It results in non-physical behavior, but accurate base positioning
        # if self.do_kin_fixed_base:
        #    self.set_root_pose(self.kin_fixed_base_state[0], self.kin_fixed_base_state[1])
        if self.do_vel_fix_base:
            self.fix_base(step_size, base_position, base_orientation)

        # NOTE: set the state cache to dirty. Next time a query is made, update it.
        self._body_prim_states_dirty = True
        # TODO: apply robot controller drive actions here
        # self._step_count += 1

    def init_ik(self):
        """
        Initialize pymomentum and load a model.
        """
        try:
            import pymomentum.geometry as pym_geo

            self.momentum_character = pym_geo.Character.load_urdf(
                self.robot_cfg.urdf
            )
            # TODO: the above character is available for ik
        except ImportError:
            print("Could not initialize pymomentum IK library.")

    def clean(self) -> None:
        """
        Cleans up the robot. This object is expected to be deleted immediately after calling this function.
        """
        # self.sim.get_articulated_object_manager().remove_object_by_handle(
        #    self.ao.handle
        # )
        # TODO: how to remove an object "nicely" in Isaac

    def set_root_pose(
        self,
        pos: mn.Vector3 = None,
        rot: mn.Quaternion = None,
        convention: str = "hab",
    ) -> None:
        """
        Sets the robot's base position and orientation globally.
        """
        usd_rot = None
        usd_pos = None
        if rot is not None and convention == "hab":
            usd_rot = isaac_prim_utils.habitat_to_usd_rotation(
                isaac_prim_utils.magnum_quat_to_list_wxyz(rot)
            )
        if pos is not None and convention == "hab":
            usd_pos = mn.Vector3(pos)
            usd_pos[1] += self.ground_to_base_offset
            self.target_base_height = usd_pos[1]
            usd_pos = isaac_prim_utils.habitat_to_usd_position(usd_pos)

        self._robot.set_world_pose(usd_pos, usd_rot)
        self.base_vel_controller.reset()
        self._robot.set_linear_velocity(np.zeros(3))
        self._robot.set_angular_velocity(np.zeros(3))

    def get_root_pose(
        self, convention: str = "hab"
    ) -> Tuple[mn.Vector3, mn.Quaternion]:
        """
        Get the robot's base position and orientation in global space.
        """

        pos_usd, rot_usd = self._robot.get_world_pose()
        if convention == "hab":
            pos = mn.Vector3(isaac_prim_utils.usd_to_habitat_position(pos_usd))
            pos[1] -= self.ground_to_base_offset
            rot = isaac_prim_utils.rotation_wxyz_to_magnum_quat(
                isaac_prim_utils.usd_to_habitat_rotation(rot_usd)
            )
        else:
            pos = mn.Vector3(pos_usd)
            rot = isaac_prim_utils.rotation_wxyz_to_magnum_quat(rot_usd)
        return pos, rot

    def global_forward(self):
        """
        Return the global forward vector for the robot's base.
        """
        _, rot = self.get_root_pose()
        glob_forward = rot.transform_vector(mn.Vector3(1.0, 0, 0))
        return glob_forward

    def angle_to(
        self,
        dir_target: mn.Vector3,
        dir_init: mn.Vector3 = None,
        right_hand: bool = False,
    ) -> float:
        """
        Gets the angular error in yaw rotation space between a global target vector and a global initial vector.
        If not provided, the global initial vector is the robot's current forward.
        """
        _, rot = self.get_root_pose()
        glob_forward = (
            rot.transform_vector(mn.Vector3(1.0, 0, 0))
            if dir_init is None
            else dir_init
        )
        norm_dir = mn.Vector3([dir_target[0], 0, dir_target[2]]).normalized()
        up = mn.Vector3(0, 1.0, 0)
        tar_right = mn.math.cross(up, norm_dir)
        angle = float(mn.math.angle(glob_forward, norm_dir))
        det = mn.math.dot(glob_forward, tar_right)
        if det > 0:
            angle = -abs(angle)
        else:
            angle = abs(angle)
        # flip the sign if right_hand coordinate system
        right_hand_sign = 1.0 if not right_hand else -1.0
        return angle * right_hand_sign

    @property
    def base_rot(self) -> float:
        """
        Returns scalar rotation angle of the agent around the Y axis.
        Within range (-pi,pi) consistency with setter is tested. Outside that range, an equivalent but distinct rotation angle may be returned (e.g. 2pi == -2pi == 0).
        NOTE: assumes the robot is upright
        """
        return -self.angle_to(mn.Vector3(1.0, 0, 0))

    @base_rot.setter
    def base_rot(self, rotation_y_rad: float):
        """
        Set the scalar rotation angle of the agent around the Y axis.
        """
        rot = mn.Quaternion.rotation(
            mn.Rad(rotation_y_rad), mn.Vector3(0, 1, 0)
        ) * mn.Quaternion.rotation(
            mn.Rad(-mn.math.pi / 2.0), mn.Vector3(1, 0, 0)
        )
        self.set_root_pose(rot=rot)

    def get_hand_centers(self) -> List[mn.Vector3]:
        """
        Get points for the left and right hand representing the center of the palm.
        """
        hand_ixs = (
            self.link_subsets["left_hand"].link_ixs
            + self.link_subsets["right_hand"].link_ixs
        )
        hand_transforms = self.get_link_world_poses(hand_ixs)
        hand_centers = hand_transforms[0]
        return hand_centers

    def set_cached_pose(
        self,
        pose_file: str = default_pose_cache_path,
        pose_name: str = "default",
        set_motor_targets: bool = True,
        set_positions: bool = True,
    ) -> None:
        """
        Loads a robot pose from a json file which could have multiple poses.
        """
        self.pos_subsets["full"].set_cached_pose(
            pose_file, pose_name, set_motor_targets, set_positions
        )

    def cache_pose(
        self,
        pose_file: str = default_pose_cache_path,
        pose_name: str = "default",
    ) -> None:
        """
        Saves the current robot pose in a json cache file with the given name.
        """
        self.pos_subsets["full"].cache_pose(pose_file, pose_name)

    def fix_base(self, step_size: float, base_position, base_orientation):
        """
        Applies velocities to constrain the robot base state.
        """
        self.damp_robot_vel(0.9)
        self.fix_base_height_via_linear_vel_z(
            step_size, base_position, base_orientation
        )
        # if self.do_kin_fixed_base:
        #    self._robot.set_world_pose(None, self.kin_fixed_base_state[1])
        # else:
        self.fix_base_orientation_via_angular_vel(
            step_size, base_position, base_orientation
        )

    def damp_robot_vel(self, damping_factor: float):
        """
        Applies a damping reduction to the robot's base velocity.
        """
        cur_ang_vel = self._robot.get_angular_velocity()
        self._robot.set_angular_velocity(cur_ang_vel * damping_factor)
        cur_lin_vel = self._robot.get_linear_velocity()
        self._robot.set_linear_velocity(cur_lin_vel * damping_factor)

    def fix_base_orientation_via_angular_vel(
        self, step_size, base_position, base_orientation
    ):
        curr_angular_velocity = self._robot.get_angular_velocity()

        # Constants
        max_angular_velocity = 0.5  # Maximum angular velocity (rad/s)

        # wxyz to xyzw
        base_orientation_xyzw = np.array(
            [
                base_orientation[1],
                base_orientation[2],
                base_orientation[3],
                base_orientation[0],
            ]
        )
        base_rotation = R.from_quat(base_orientation_xyzw)

        # Define the local "up" axis and transform it to world space
        local_up = np.array([0, 0, 1])  # Object's local up axis (hack: -1?)
        world_up = base_rotation.apply(local_up)  # Local up in world space

        # Define the global up axis
        global_up = np.array([0, 0, 1])  # Global up direction

        # Compute the axis of rotation to align world_up to global_up
        rotation_axis = np.cross(world_up, global_up)
        rotation_axis_norm = np.linalg.norm(rotation_axis)

        # Handle special cases where no rotation is needed
        if rotation_axis_norm < 1e-6:  # Already aligned
            desired_angular_velocity = np.array(
                [0.0, 0.0, 0.0]
            )  # No correction needed
        else:
            # Normalize the rotation axis
            rotation_axis /= rotation_axis_norm

            # Compute the angle of rotation using the dot product
            rotation_angle = np.arccos(
                np.clip(np.dot(world_up, global_up), -1.0, 1.0)
            )

            # Calculate the angular velocity to correct the tilt in one step
            hack_scale = 1.0
            tilt_correction_velocity = (
                (rotation_axis * rotation_angle) / step_size
            ) * hack_scale

            # Cap the angular velocity to the maximum allowed value
            angular_velocity_magnitude = np.linalg.norm(
                tilt_correction_velocity
            )
            if angular_velocity_magnitude > max_angular_velocity:
                tilt_correction_velocity *= (
                    max_angular_velocity / angular_velocity_magnitude
                )

            desired_angular_velocity = tilt_correction_velocity

        # don't change rotation about z
        desired_angular_velocity[2] = curr_angular_velocity[2]

        if self.do_kin_fixed_base:
            # also control z angle to keep orientation
            tar_forward = self.kin_fixed_base_state[2]
            angle_to = self.angle_to(tar_forward)
            desired_angular_velocity[2] = max(
                -max_angular_velocity,
                min(max_angular_velocity, angle_to / step_size),
            )

        self._robot.set_angular_velocity(desired_angular_velocity)

    def fix_base_height_via_linear_vel_z(
        self, step_size, base_position, base_orientation
    ):
        max_linear_vel = 3.0
        if self.do_kin_fixed_base:
            vel = [0, 0, 0]
            for i in range(3):
                val_cur = base_position[i]
                val_tar = self.kin_fixed_base_state[0][i]
                pos_error = val_tar - val_cur
                desired_linear_vel = pos_error / step_size
                desired_linear_vel = max(
                    -max_linear_vel, min(max_linear_vel, desired_linear_vel)
                )
                vel[i] = desired_linear_vel

            self._robot.set_linear_velocity(vel)
            return

        curr_linear_velocity = self._robot.get_linear_velocity()

        z_target = self.target_base_height

        # Extract the vertical position and velocity
        z_current = base_position[2]

        # Compute the position error
        position_error = z_target - z_current

        desired_linear_vel_z = position_error / step_size
        desired_linear_vel_z = max(
            -max_linear_vel, min(max_linear_vel, desired_linear_vel_z)
        )

        self._robot.set_linear_velocity(
            [
                curr_linear_velocity[0],
                curr_linear_velocity[1],
                desired_linear_vel_z,
            ]
        )

    def draw_debug(self, dblr: DebugLineRender):
        """
        Draw some representation of the robot state.
        """

        # draw an axis frame at the origin
        root_pos, root_rot = self.get_root_pose()
        # tform = mn.Matrix4.from_(root_rot.to_matrix(), root_pos)
        # debug_draw_axis(dblr, tform)

        # draw a line in the forward direction of the robot
        # if forward_color is None:
        #     forward_color = mn.Color4(0.0, 1.0, 0.0, 1.0)
        # forward_dir = root_rot.transform_vector(mn.Vector3(1.0, 0, 0)) * 2.0
        # dblr.draw_transformed_line(
        #     root_pos,
        #     root_pos + forward_dir,
        #     from_color=forward_color,
        #     to_color=forward_color,
        # )

        # draw the navmesh circle
        dblr.draw_circle(
            root_pos,
            radius=self.robot_cfg.navmesh_radius,
            color=mn.Color4(0.8, 0.7, 0.9, 0.8),
            normal=mn.Vector3(0, 1, 0),
        )

        if self.base_vel_controller.track_waypoints:
            self.base_vel_controller.debug_draw_waypoint(dblr)

        # draw the finger raycast sensors
        # for finger_raycast_sensor in self.finger_raycast_sensors:
        #     finger_raycast_sensor.draw(dblr)

    def get_rigid_prim_ix(self, prim_path: str):
        """
        Get a body/link prim index from the rigidbody path name.
        """

        try:
            return self._rigid_prim_view.prim_paths.index(prim_path)
        except ValueError:
            # not in the list
            return None

    def get_joint_for_rigid_prim(self, rigid_prim_path: str) -> str:
        """
        Retrieves the joint prim that is connected to the given rigid prim in an articulation.

        Args:
            rigid_prim: The rigid prim to find the joint for (Usd.Prim).

        Returns:
            The joint prim connected to the rigid prim, or None if not found (Usd.Prim).
        """

        for joint_prim_path, targets in self._joint_relationships.items():
            # looking for the joint this is a child of
            if rigid_prim_path == targets[1]:
                return joint_prim_path

        # couldn't find a match
        return None

    def _collect_joint_relationships(self, root_prim=None):
        """
        Collects a mapping of joint relationships for later query.
        """

        if root_prim is None:
            root_prim = self._isaac_service.world.stage.GetPrimAtPath(
                self._robot_prim_path
            )
        joint_relationships: Dict[str, Tuple[str, str]] = {}
        children = root_prim.GetChildren()
        for child in children:
            if child.IsA(UsdPhysics.PrismaticJoint) or child.IsA(
                UsdPhysics.RevoluteJoint
            ):
                joint_relationships[str(child.GetPath())] = (
                    str(
                        child.GetRelationship("physics:body0").GetTargets()[0]
                    ),
                    str(
                        child.GetRelationship("physics:body1").GetTargets()[0]
                    ),
                )
            child_joint_relationships = self._collect_joint_relationships(
                child
            )
            for joint, relationship in child_joint_relationships.items():
                joint_relationships[joint] = relationship
        return joint_relationships

    def _construct_dof_name_map(self):
        """
        Maps the full joint prim paths to their internal dof indices.
        """

        joint_names_to_dof_ix: Dict[str, int] = {}
        for joint_name in self._joint_relationships.keys():
            joint_short_name = joint_name.split("/")[-1]
            joint_names_to_dof_ix[joint_name] = self._robot.dof_names.index(
                joint_short_name
            )
        return joint_names_to_dof_ix

    def _create_rigid_prim_view(self):
        """
        Construct a rigid prim view over the rigid body elements of the robot.
        """

        root_prim_path = self._robot_prim_path
        root_prim = self._isaac_service.world.stage.GetPrimAtPath(
            root_prim_path
        )

        prim_paths = []

        # lazy import
        from pxr import Usd, UsdPhysics

        prim_range = Usd.PrimRange(root_prim)
        it = iter(prim_range)
        for prim in it:
            prim_path = str(prim.GetPath())

            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                print(f"prim_path {prim_path} not RigidBody")
                continue
            else:
                pass

            # we found a rigid body, so let's ignore children
            it.PruneChildren()
            prim_paths.append(prim_path)

        assert len(prim_paths)
        print("prim_paths: ", prim_paths, len(prim_paths))

        self._body_prim_paths = prim_paths

        from omni.isaac.core.prims.rigid_prim_view import RigidPrimView

        self._rigid_prim_view = RigidPrimView(prim_paths)
        physics_sim_view = self._isaac_service.world.physics_sim_view
        assert physics_sim_view

    def update_body_prim_states(self) -> None:
        """
        Pulls the world transforms of all links and converts them into habitat conventions.
        """
        body_prim_states = self._rigid_prim_view.get_world_poses(usd=False)
        rotations = []
        positions = []
        for ix in range(len(body_prim_states[0])):
            pos = body_prim_states[0][ix]
            rot = body_prim_states[1][ix]
            # convert to hab
            pos = isaac_prim_utils.usd_to_habitat_position(pos)
            rot = isaac_prim_utils.usd_to_habitat_rotation(rot)
            # wrap for magnum
            pos = mn.Vector3(pos)
            rot = isaac_prim_utils.rotation_wxyz_to_magnum_quat(rot)
            positions.append(pos)
            rotations.append(rot)
        self._body_prim_states = positions, rotations
        self._body_prim_states_dirty = False

    def get_link_world_poses(
        self, indices: List[int] = None
    ) -> Tuple[List[mn.Vector3], List[mn.Quaternion]]:
        """
        Get the global position and orientation of all specified prims by index.
        """
        if self._body_prim_states_dirty:
            # NOTE: This costs 0.02sec so cached
            self.update_body_prim_states()
        if indices is None:
            return self._body_prim_states
        states = self._body_prim_states
        positions = []
        rotations = []
        for ix in indices:
            positions.append(states[0][ix])
            rotations.append(states[1][ix])

        return positions, rotations

    def draw_dof(
        self, dblr: DebugLineRender, link_ix: int, cam_pos: mn.Vector3
    ) -> None:
        """
        Draw a visual indication of the given dof state.
        A circle aligned with the dof axis for revolute joints.
        A line with bars representing the min and max joint limits and a bar between them representing state.
        """

        # NOTE: These are local CoMs and useless for the purpose of querying state directly.
        # NOTE: These are 3D arrays: 1st layer is sim index(always 0), 2nd is body index (always 0)
        # body_com, body_rot = self._robot_view.get_body_coms(indices=[0], body_indices=[link_ix])
        # hab_body_rot = isaac_prim_utils.rotation_wxyz_to_magnum_quat(isaac_prim_utils.usd_to_habitat_rotation(body_rot[0][0]))
        # hab_body_com = mn.Vector3(isaac_prim_utils.usd_to_habitat_position(body_com[0][0]))

        body_positions, body_rotations = self.get_link_world_poses(
            indices=[link_ix]
        )
        body_pos = body_positions[0]
        body_rot = body_rotations[0]

        tform = mn.Matrix4.from_(body_rot.to_matrix(), body_pos)

        prim_path = self._rigid_prim_view.prim_paths[link_ix]
        parent_joint = self.get_joint_for_rigid_prim(prim_path)
        if parent_joint is not None:
            # this link has a parent joint so draw the joint circle
            joint_prim = self._isaac_service.world.stage.GetPrimAtPath(
                parent_joint
            )
            if joint_prim.IsA(UsdPhysics.RevoluteJoint):
                axis = joint_prim.GetAttribute("physics:axis").Get()
                if axis == "Z":
                    axis = mn.Vector3(0, 0, 1.0)
                elif axis == "X":
                    axis = mn.Vector3(1.0, 0, 0)
                elif axis == "Y":
                    axis = mn.Vector3(0, 1.0, 0)
                else:
                    raise NotImplementedError()

                global_axis = tform.transform_vector(axis)
                dblr.draw_circle(
                    body_pos,
                    radius=0.1,
                    color=mn.Color3(0, 0.75, 0),  # green
                    normal=global_axis,
                )

    def print_joint_limits(self):
        """
        Get the joints limits of each dof.
        """
        limits = self._robot.dof_properties
        lower_limit = limits["lower"]
        upper_limit = limits["upper"]
        print(list(lower_limit))
        print(list(upper_limit))


def unit_test_robot(robot: RobotAppWrapper):
    """
    Unit tests some assumptions about the robot. If these fail then other parts of the code are likely to exhibit undefined behavior.
    """

    initial_state = robot.get_root_pose()

    def all_close(val0, val1, eps=0.001):
        """
        two values are close within epsilon error
        """
        if isinstance(val0, mn.Vector3):
            return (val0 - val1).length() < eps
        else:
            return abs(val0 - val1) < eps

    # robot local forward is X axis
    local_forward = mn.Vector3(1.0, 0, 0)
    local_up = mn.Vector3(0, 1.0, 0)
    # in default pose this corresponds to the forward x axis
    robot.set_root_pose(mn.Vector3(), mn.Quaternion())
    pos, rot = robot.get_root_pose()
    tform = mn.Matrix4.from_(rot.to_matrix(), pos)
    global_forward = tform.transform_vector(local_forward)
    assert all_close(
        global_forward, mn.Vector3(1.0, 0, 0)
    ), f"global_forward={global_forward}"
    global_up = tform.transform_vector(local_up)
    assert all_close(
        global_up, mn.Vector3(0, 1.0, 0)
    ), f"global_up={global_up}"
    assert all_close(
        robot.base_rot, 0
    ), f"robot.base_rot = {robot.base_rot}, should be 0"
    # breakpoint()
    for angle in [
        -mn.math.pi + 0.01 + (i / 10.0) * (mn.math.pi * 2) for i in range(10)
    ]:
        robot.base_rot = angle
        print(f"base rot angel {angle} -> {robot.base_rot}")
        assert all_close(robot.base_rot, angle)

    robot.set_root_pose(initial_state[0], initial_state[1])
