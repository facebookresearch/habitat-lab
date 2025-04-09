#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from typing import Dict, List, Tuple

import magnum as mn
from omegaconf import DictConfig

import habitat.sims.habitat_simulator.sim_utilities as sutils
import habitat_sim  # unfortunately we can't import this earlier
from habitat_sim.geo import Ray
from habitat_sim.gfx import DebugLineRender
from habitat_sim.physics import RaycastResults

# path to this example app directory
dir_path = os.path.dirname(os.path.realpath(__file__)).split("scripts")[0]
default_pose_cache_path = os.path.join(dir_path, "robot_poses.json")


def debug_draw_axis(
    dblr: DebugLineRender, transform: mn.Matrix4 = None, scale: float = 1.0
) -> None:
    if transform is not None:
        dblr.push_transform(transform)
    for unit_axis in range(3):
        vec = mn.Vector3()
        vec[unit_axis] = 1.0
        color = mn.Color3(0.5)
        color[unit_axis] = 1.0
        dblr.draw_transformed_line(mn.Vector3(), vec * scale, color)
    if transform is not None:
        dblr.pop_transform()


class LinkSubset:
    """
    A class to encapsulate logic related to querying properties of a grouped subset of links.
    This class is not intended to be used for direct reference to DoFs. See ConfigurationSubset.
    """

    def __init__(
        self,
        ao: habitat_sim.physics.ManagedArticulatedObject,
        links: List[int] = None,
    ) -> None:
        self.ao = ao
        if links is None:
            # by default collect all links
            self.link_ixs = ao.get_link_ids()
        else:
            self.link_ixs = links


class ConfigurationSubset:
    """
    A class to encapsulate logic related to querying, setting, and caching labeled subsets of the full robot configuration.
    For example, a single hand, arm, or wheeled base is controlled by a subset of the full configuration.
    """

    def __init__(
        self,
        ao: habitat_sim.physics.ManagedArticulatedObject,
        links: List[int] = None,
    ) -> None:
        self.ao = ao
        if links is None:
            # by default collect all 1 DoF links
            self.link_ixs = [
                link_ix
                for link_ix in ao.get_link_ids()
                if ao.get_link_num_joint_pos(link_ix) == 1
            ]
        else:
            self.link_ixs = links
        # check that each link has only one degree of freedom
        for link_ix in self.link_ixs:
            assert (
                ao.get_link_num_joint_pos(link_ix) == 1
            ), f"link ({link_ix} - {self.ao.get_link_name(link_ix)}) has {ao.get_link_num_joint_pos(link_ix)} dofs, not 1."
        self.joint_pos_ixs = [
            ao.get_link_joint_pos_offset(link_ix) for link_ix in self.link_ixs
        ]
        # each index contains a list of motor indices for motors associated with the DoF at index
        self.joint_motors: List[List[int]] = []
        self.update_motor_correspondence()

    def update_motor_correspondence(self) -> None:
        """
        Queries all active joint motors to collect the set corresponding to this configuration subset.
        """
        # first organize all motors for all links
        motor_ids_by_link: Dict[int, List[int]] = {}
        for motor_id, link_id in self.ao.existing_joint_motor_ids.items():
            if link_id not in motor_ids_by_link:
                motor_ids_by_link[link_id] = []
            motor_ids_by_link[link_id].append(motor_id)
        # then re-organize all motors for each link in this subset sequentially
        self.joint_motors = []
        for link_ix in self.link_ixs:
            self.joint_motors.append([])
            if link_ix in motor_ids_by_link:
                self.joint_motors[-1] = motor_ids_by_link[link_ix]

    def set_pos_from_full(self, all_joint_pos: List[float]) -> None:
        """
        Set this subset configuration from a fullbody configuration input.
        """
        cur_pose = self.ao.joint_positions
        for j_pos_ix in self.joint_pos_ixs:
            cur_pose[j_pos_ix] = all_joint_pos[j_pos_ix]
        self.ao.joint_positions = cur_pose

    def set_pos(self, joint_positions: List[float]) -> None:
        """
        Set this configuration subset from a precisely sized list of joint positions.
        NOTE: Input joint positions list must be the same size as this configuration subset. To set this subset from a full pose use set_pos_from_full instead.
        """
        cur_pose = self.ao.joint_positions
        for ix, j_pos_ix in enumerate(self.joint_pos_ixs):
            cur_pose[j_pos_ix] = joint_positions[ix]
        self.ao.joint_positions = cur_pose

    def get_pos(self) -> List[float]:
        """
        Get the current configuration of the configured subset of joints.
        """
        full_pos = self.ao.joint_positions
        cur_pos = [full_pos[ix] for ix in self.joint_pos_ixs]
        return cur_pos

    def set_motor_pos_from_full(
        self, all_joint_pos_targets: List[float]
    ) -> None:
        """
        Set this subset configuration's joint motor targets from a fullbody joint position target input.
        """
        for local_ix, j_pos_ix in enumerate(self.joint_pos_ixs):
            for motor_id in self.joint_motors[local_ix]:
                cur_settings = self.ao.get_joint_motor_settings(motor_id)
                cur_settings.position_target = all_joint_pos_targets[j_pos_ix]
                self.ao.update_joint_motor(motor_id, cur_settings)

    def set_motor_pos(self, motor_targets: List[float]) -> None:
        """
        Set this configuration subset's joint motor targets from a precisely sized list of joint position targets.
        NOTE: Input joint position targets list must be the same size as this configuration subset. To set this subset from a full pose use set_motor_pos_from_full instead.
        """
        for ix, pos in enumerate(motor_targets):
            for motor_id in self.joint_motors[ix]:
                cur_settings = self.ao.get_joint_motor_settings(motor_id)
                cur_settings.position_target = pos
                self.ao.update_joint_motor(motor_id, cur_settings)

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
        if not os.path.exists(pose_file):
            print(
                f"Cannot load cached pose. Configured pose file {pose_file} does not exist."
            )
            return

        with open(pose_file, "r") as f:
            poses = json.load(f)
            if self.ao.handle not in poses:
                print(
                    f"Cannot load cached pose. No poses cached for robot {self.ao.handle}."
                )
                return
            if pose_name not in poses[self.ao.handle]:
                print(
                    f"Cannot load cached pose. No pose named {pose_name} cached for robot {self.ao.handle}. Options are {poses[self.ao.handle].keys()}"
                )
                return
            pose = poses[self.ao.handle][pose_name]
            if len(pose) == len(self.ao.joint_positions):
                # loaded a full pose so cut it down if necessary
                if len(pose) != len(self.joint_pos_ixs):
                    pose = [pose[ix] for ix in self.joint_pos_ixs]
                else:
                    # this is a full pose subset so use the shortcut APIs
                    if set_positions:
                        self.ao.joint_positions = pose
                    if set_motor_targets:
                        self.ao.update_all_motor_targets(pose)
                    return
            elif len(pose) == len(self.joint_pos_ixs):
                # subset pose is correctly sized so no work
                pass
            else:
                print(
                    f"Cannot load cached pose (size {len(pose)}) as it does not match number of dofs ({len(self.ao.joint_positions)} full or {len(self.joint_pos_ixs)} subset)"
                )
                return
            if set_motor_targets:
                for ix, pos in enumerate(pose):
                    for motor_id in self.joint_motors[ix]:
                        cur_settings = self.ao.get_joint_motor_settings(
                            motor_id
                        )
                        cur_settings.position_target = pos
                        self.ao.update_joint_motor(motor_id, cur_settings)
            if set_positions:
                cur_pose = self.ao.joint_positions
                for ix in range(len(pose)):
                    cur_pose[self.joint_pos_ixs[ix]] = pose[ix]
                self.ao.joint_positions = cur_pose

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
        if self.ao.handle not in poses:
            poses[self.ao.handle] = {}
        subset_pose = [
            self.ao.joint_positions[ix] for ix in self.joint_pos_ixs
        ]
        poses[self.ao.handle][pose_name] = subset_pose
        with open(pose_file, "w") as f:
            json.dump(
                poses,
                f,
                indent=4,
            )


class FingerRaycastSensor:
    """
    A class to encapsulate the necessary logic for using raycasting between fingertips to detect an object within the "pre-grasp" volume of the hand.
    """

    def __init__(
        self,
        sim: habitat_sim.Simulator,
        ao: habitat_sim.physics.ManagedArticulatedObject,
        thumb_link: LinkSubset,
        finger_links: LinkSubset,
    ):
        """
        Requires: the simulator, the ArticulatedObject, a thumb link, some finger links
        """
        self.sim = sim
        self.ao = ao
        self.obj_ids = [self.ao.object_id] + list(
            self.ao.link_object_ids.keys()
        )
        self.thumb = thumb_link
        self.fingers = finger_links
        assert len(self.thumb.link_ixs) > 0
        assert len(self.fingers.link_ixs) > 0
        # cache the most recent update for debug drawing and query. Keyed by (thumb,finger) link pairs.
        self.results: Dict[Tuple[int, int], RaycastResults] = None

    def update_sensor_raycasts(self) -> None:
        """
        Runs the raycasting routine to fill the RaycastResults cache.
        """
        self.results = {}
        for thumb_link in self.thumb.link_ixs:
            thumb_pos = self.ao.get_link_scene_node(thumb_link).translation
            for finger_link in self.fingers.link_ixs:
                finger_pos = self.ao.get_link_scene_node(
                    finger_link
                ).translation
                # NOTE: scaled to distance between the finger tips such that valid distances are [0,1]
                ray = Ray(thumb_pos, finger_pos - thumb_pos)
                results = self.sim.cast_ray(ray)
                self.results[(thumb_link, finger_link)] = results

    def get_obj_ids_in_sensor_reading(
        self, update_sensor: bool = True
    ) -> Dict[int, float]:
        """
        Checks the raycast and returns a dict keyed by object_id of any objects detected by the sensor and mapping to the ratio of rays for that object.
        """
        if update_sensor:
            self.update_sensor_raycasts()
        if self.results is None:
            print(
                "FingerRaycastSensor:get_obj_ids_in_sensor_reading - No raycast results cached, returning null sensor reading."
            )
            return {}
        num_rays = len(self.thumb.link_ixs) * len(self.fingers.link_ixs)
        detected_objs: Dict[int, float] = {}
        for ray_result in self.results.values():
            if ray_result.has_hits():
                # find any non-robot hits between the finger and thumb and record them
                for hit in ray_result.hits:
                    # if a hit is greater distance than 1 it is outside the grip
                    if hit.ray_distance > 1:
                        break
                    if hit.object_id not in self.obj_ids:
                        if hit.object_id not in detected_objs:
                            detected_objs[hit.object_id] = 1
                        else:
                            detected_objs[hit.object_id] += 1
        for detected_obj, num_hits in detected_objs.items():
            detected_objs[detected_obj] = num_hits / num_rays
        return detected_objs

    def get_scalar_sensor_reading(self, update_sensor: bool = True) -> float:
        """
        Checks the raycast and returns the ratio between [0,1] of detected hits.
        Set `update_sensor=False` to re-interpret the previous results without recasting the rays.
        """
        if update_sensor:
            self.update_sensor_raycasts()
        if self.results is None:
            print(
                "FingerRaycastSensor:get_sensor_reading - No raycast results cached, returning null sensor reading."
            )
            return None
        hits = 0
        for ray_result in self.results.values():
            if ray_result.has_hits():
                # find the first non-robot hit between the finger and thumb
                for hit in ray_result.hits:
                    # if a hit is greater distance than 1 it is outside the grip
                    if hit.ray_distance > 1:
                        break
                    if hit.object_id not in self.obj_ids:
                        hits += 1
                        break
        return hits / (len(self.thumb.link_ixs) * len(self.fingers.link_ixs))

    def draw(self, dblr: DebugLineRender) -> None:
        """
        Debug draw the current state of the sensor.
        """
        if self.results is None:
            return
        for ray_result in self.results.values():
            hit_positions = []
            # collect the hit points
            if ray_result.has_hits():
                # find the first non-robot hit between the finger and thumb
                for hit in ray_result.hits:
                    # if a hit is greater distance than 1 it is outside the grip
                    if hit.ray_distance > 1:
                        break
                    if hit.object_id not in self.obj_ids:
                        hit_positions.append(hit.point)
                        break
            # color based on results
            color = mn.Color4.green()
            if len(hit_positions) > 0:
                color = mn.Color4.red()
            # draw the ray
            dblr.draw_transformed_line(
                ray_result.ray.origin,
                ray_result.ray.origin + ray_result.ray.direction,
                color,
            )
            # draw the hit points
            for hit_point in hit_positions:
                # draw the navmesh circle
                dblr.draw_circle(
                    hit_point,
                    radius=0.025,
                    color=mn.Color4.yellow(),
                    normal=ray_result.ray.direction,
                )


class Robot:
    """
    Wrapper class for robots imported as simulated ArticulatedObjects.
    Wraps the ManagedObjectAPI.
    """

    def __init__(self, sim: habitat_sim.Simulator, robot_cfg: DictConfig):
        """
        Initialize the robot in a Simulator from its config object.
        """

        self.sim = sim
        self.robot_cfg = robot_cfg

        # apply this local viewpoint offset to the ao.translation to align the cursor
        self.viewpoint_offset = mn.Vector3(robot_cfg.viewpoint_offset)

        # expect a "urdf" config field with the filepath
        self.ao = self.sim.get_articulated_object_manager().add_articulated_object_from_urdf(
            self.robot_cfg.urdf,
            fixed_base=self.robot_cfg.fixed_base
            if hasattr(self.robot_cfg, "fixed_base")
            else False,
            force_reload=True,
        )
        self.obj_ids = [self.ao.object_id] + list(
            self.ao.link_object_ids.keys()
        )

        # create joint motors
        self.motor_ids_to_link_ids: Dict[int, int] = {}
        self.using_joint_motors = self.robot_cfg.create_joint_motors
        if self.using_joint_motors:
            self.create_joint_motors()

        # define configuration subsets
        self.pos_subsets = {"full": ConfigurationSubset(self.ao)}

        # load the configuration subsets from the robot config file
        for (
            subset_cfg_name,
            subset_cfg_links,
        ) in self.robot_cfg.configuration_subsets.items():
            print(subset_cfg_name)
            self.pos_subsets[subset_cfg_name] = ConfigurationSubset(
                self.ao, links=subset_cfg_links
            )

        # load link subsets from the robot config file
        self.link_subsets = {}
        for (
            subset_cfg_name,
            subset_cfg_links,
        ) in self.robot_cfg.link_subsets.items():
            self.link_subsets[subset_cfg_name] = LinkSubset(
                self.ao, subset_cfg_links
            )

        # setup the finger raycast sensors
        self.finger_raycast_sensors = [
            FingerRaycastSensor(
                self.sim,
                self.ao,
                self.link_subsets["left_thumb_tip"],
                self.link_subsets["left_finger_tips"],
            ),
            FingerRaycastSensor(
                self.sim,
                self.ao,
                self.link_subsets["right_thumb_tip"],
                self.link_subsets["right_finger_tips"],
            ),
        ]

        # set initial pose
        self.set_cached_pose(
            pose_name=self.robot_cfg.initial_pose,
            set_positions=True,
            set_motor_targets=True,
        )
        # clamp to joint limits
        self.clamp_joints_to_limits()
        self.clamp_motor_targets_to_limits()
        # self.init_ik()

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

    def clear_joint_motors(self) -> None:
        """
        Removes all active joint motors.
        """
        for motor_id in self.ao.existing_joint_motor_ids:
            self.ao.remove_joint_motor(motor_id)

    def create_joint_motors(self) -> None:
        """
        Creates a full set of joint motors for the robot.
        """
        # first clear any existing joint motors in case there are damping motors or stale motors from other code.
        self.clear_joint_motors()
        self.motor_settings = habitat_sim.physics.JointMotorSettings(
            0,  # position_target
            self.robot_cfg.joint_motor_pos_gains,  # position_gain
            0,  # velocity_target
            self.robot_cfg.joint_motor_vel_gains,  # velocity_gain
            self.robot_cfg.joint_motor_max_impulse,  # max_impulse
        )

        self.motor_ids_to_link_ids = self.ao.create_all_motors(
            self.motor_settings
        )

    def clean(self) -> None:
        """
        Cleans up the robot. This object is expected to be deleted immediately after calling this function.
        """
        self.sim.get_articulated_object_manager().remove_object_by_handle(
            self.ao.handle
        )

    def place_robot(self, base_pos: mn.Vector3):
        """
        Place the robot at a given position.
        """
        y_size, center = sutils.get_obj_size_along(
            self.sim, self.ao.object_id, mn.Vector3(0, -1, 0)
        )
        offset = (self.ao.translation - center)[1] + y_size
        self.ao.translation = base_pos + mn.Vector3(0, offset, 0)

    def clamp_joints_to_limits(self) -> None:
        """
        Clamps the robot's joint positions to fall within the joint limits.
        """
        cur_joint_pos = self.ao.joint_positions
        joint_limits = self.ao.joint_position_limits
        for dof_ix in range(len(cur_joint_pos)):
            min_dof = joint_limits[0][dof_ix]
            max_dof = joint_limits[1][dof_ix]
            new_dof = min(max(cur_joint_pos[dof_ix], min_dof), max_dof)
            if cur_joint_pos[dof_ix] != new_dof:
                print(
                    f" Clamped joint position dof {dof_ix} from {cur_joint_pos[dof_ix]} to {new_dof} in range [{min_dof},{max_dof}] "
                )
            cur_joint_pos[dof_ix] = new_dof
        self.ao.joint_positions = cur_joint_pos

    def clamp_motor_targets_to_limits(self) -> None:
        """
        Clamps the robot's joint motor target positions to fall within the joint limits.
        """
        joint_limits = self.ao.joint_position_limits
        for motor_id, link_ix in self.ao.existing_joint_motor_ids.items():
            cur_settings = self.ao.get_joint_motor_settings(motor_id)
            dof_ix = self.ao.get_link_dof_offset(link_ix)
            min_dof = joint_limits[0][dof_ix]
            max_dof = joint_limits[1][dof_ix]
            new_target = min(
                max(cur_settings.position_target, min_dof), max_dof
            )
            if new_target != cur_settings.position_target:
                print(
                    f" Clamped motor target dof {dof_ix} from {cur_settings.position_target} to {new_target} in range [{min_dof},{max_dof}] "
                )
                cur_settings.position_target = new_target
                self.ao.update_joint_motor(motor_id, cur_settings)

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

    def draw_debug(self, dblr: DebugLineRender):
        """
        Draw the bounding box of the robot.
        """
        dblr.push_transform(self.ao.transformation)
        bb = self.ao.aabb
        dblr.draw_box(bb.min, bb.max, mn.Color3(1.0, 1.0, 1.0))
        dblr.pop_transform()
        debug_draw_axis(dblr, transform=self.ao.transformation)

        # draw the navmesh circle
        dblr.draw_circle(
            self.ao.translation,
            radius=self.robot_cfg.navmesh_radius,
            color=mn.Color4(0.8, 0.7, 0.9, 0.8),
            normal=mn.Vector3(0, 1, 0),
        )

        # draw the finger raycast sensors
        for finger_raycast_sensor in self.finger_raycast_sensors:
            finger_raycast_sensor.draw(dblr)

    def draw_dof(
        self, dblr: DebugLineRender, link_ix: int, cam_pos: mn.Vector3
    ) -> None:
        """
        Draw a visual indication of the given dof state.
        A circle aligned with the dof axis for revolute joints.
        A line with bars representing the min and max joint limits and a bar between them representing state.
        """
        if self.ao.get_link_num_dofs(link_ix) == 0:
            return

        link_obj_id = self.ao.link_ids_to_object_ids[link_ix]
        obj_bb, transform = sutils.get_bb_for_object_id(self.sim, link_obj_id)
        center = transform.transform_point(obj_bb.center())
        size_to_camera, center = sutils.get_obj_size_along(
            self.sim, link_obj_id, cam_pos - center
        )
        # draw_at = center + (cam_pos - center).normalized() * size_to_camera

        link_T = self.ao.get_link_scene_node(link_ix).transformation
        global_link_pos = link_T.translation - link_T.transform_vector(
            self.ao.get_link_joint_to_com(link_ix)
        )

        # joint_limits = self.ao.joint_position_limits
        # joint_positions = self.ao.joint_positions

        for _local_dof in range(self.ao.get_link_num_dofs(link_ix)):
            # this link has dofs

            # dof = self.ao.get_link_joint_pos_offset(link_ix) + local_dof
            # dof_value = joint_positions[dof]
            # min_dof = joint_limits[0][dof]
            # max_dof = joint_limits[1][dof]
            # interp_dof = (dof_value - min_dof) / (max_dof - min_dof)

            j_type = self.ao.get_link_joint_type(link_ix)
            dof_axes = self.ao.get_link_joint_axes(link_ix)
            debug_draw_axis(
                dblr,
                transform=self.ao.get_link_scene_node(link_ix).transformation,
            )
            if j_type == habitat_sim.physics.JointType.Revolute:
                # points out of the rotation plane
                dof_axis = dof_axes[0]
                dblr.draw_circle(
                    global_link_pos,
                    radius=0.1,
                    color=mn.Color3(0, 0.75, 0),  # green
                    normal=link_T.transform_vector(dof_axis),
                )
            elif j_type == habitat_sim.physics.JointType.Prismatic:
                # points along the translation axis
                dof_axis = dof_axes[1]
                # TODO
            # no other options are supported presently
