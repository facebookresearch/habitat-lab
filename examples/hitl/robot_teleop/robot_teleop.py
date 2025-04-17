#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import hydra
import magnum as mn
import numpy as np
from hydra import compose
from omegaconf import DictConfig

import habitat.sims.habitat_simulator.sim_utilities as sutils
import habitat_sim  # unfortunately we can't import this earlier
from habitat.datasets.rearrange.navmesh_utils import get_largest_island_index
from habitat_hitl._internal.networking.average_rate_tracker import (
    AverageRateTracker,
)
from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.app_states.app_state_abc import AppState
from habitat_hitl.core.hitl_main import hitl_main
from habitat_hitl.core.hydra_utils import (
    omegaconf_to_object,
    register_hydra_plugins,
)
from habitat_hitl.core.key_mapping import KeyCode, MouseButton
from habitat_hitl.core.text_drawer import TextOnScreenAlignment
from habitat_hitl.core.xr_input import HAND_LEFT, HAND_RIGHT
from habitat_hitl.environment.camera_helper import CameraHelper
from scripts.DoFeditor import DoFEditor
from scripts.ik import DifferentialInverseKinematics
from scripts.quest_reader import pos as to_ik_pose
from scripts.robot import LERP, Robot
from scripts.xr_pose_adapter import XRPose, XRPoseAdapter, XRTrajectory

# path to this example app directory
dir_path = os.path.dirname(os.path.realpath(__file__))

# TODO LIST:
# - setup pymomentum IK API

# GRASPING
# - Add a grasp manager API to constrain an object in the palm's coordinate space with successful pre-grasp and update the constraint frame when moving
# - Add a "release grasp" control to toggle default hand pose and break the constraint

# IK:
# - get IK frame aligned (Asjad)
# - test and stabilize IK (Asjad)

# COMBINE:
# - use ik to guide the wrist and try grasping objects
# - try grasping furniture
# - try "hand off" cycles

# BASE CONTROL
# hook up a constraint to lock the base to the navmesh which is controlled by a target. Should be more stable for dynamic interactions.


class HitDetails:
    """
    Data class for details about a single raycast hit.
    Could be a Dict, but this provides IDE API reference.
    """

    def __init__(self):
        self.object_id: int = None
        self.obj: Union[
            habitat_sim.physics.ManagedArticulatedObject,
            habitat_sim.physics.ManagedRigidObject,
        ] = None
        self.obj_name: str = None
        self.link_id: int = None
        self.link_name: str = None
        self.point: mn.Vector3 = None


class HitObjectInfo:
    """
    A data class with a simple API for identifying which object and/or link was hit by a raycast
    """

    def __init__(
        self,
        raycast_results: habitat_sim.physics.RaycastResults,
        sim: habitat_sim.Simulator,
    ):
        self.raycast_results = raycast_results
        self.sim = sim  # cache this for simplicity
        assert (
            raycast_results is not None
        ), "Must provide a valid RaycastResults"

    def has_hits(self) -> bool:
        """
        Are there any registered hits. If not other API calls are likely invalid or return None.
        """
        return self.raycast_results.has_hits()

    @property
    def hits(self) -> List[habitat_sim.physics.RayHitInfo]:
        """
        Get the RayHitInfo objects associated with the RaycastResults.
        """
        return self.raycast_results.hits

    @property
    def ray(self) -> habitat_sim.geo.Ray:
        """
        The cast Ray.
        """
        return self.raycast_results.ray

    def hit_stage(self, hit_ix: int = 0) -> bool:
        """
        Return whether or not the hit index provided was the STAGE.
        Defaults to first hit.
        """
        if self.has_hits and self.hits[hit_ix] == habitat_sim.stage_id:
            return True
        return False

    def hit_obj(
        self, hit_ix: int = 0
    ) -> Union[
        habitat_sim.physics.ManagedArticulatedObject,
        habitat_sim.physics.ManagedRigidObject,
    ]:
        """
        Get the ManagedObject at the specified hit index.
        Defaults to first hit object.
        """
        return sutils.get_obj_from_id(self.sim, self.hits[hit_ix].object_id)

    def hit_link(
        self,
        hit_ix: int = 0,
        hit_obj: Union[
            habitat_sim.physics.ManagedArticulatedObject,
            habitat_sim.physics.ManagedRigidObject,
        ] = None,
    ) -> Union[int, None]:
        """
        Gets the link index of the given hit index or None if not a link.
        :param hit_obj: Optionally provide the hit object if already known to avoid a lookup.
        """
        if self.has_hits():
            if hit_obj is None:
                hit_obj = self.hit_obj(hit_ix)
            hit_obj_id = self.hits[hit_ix].object_id
            if hit_obj is not None and hit_obj.object_id != hit_obj_id:
                return hit_obj.link_object_ids[hit_obj_id]
        return None

    def get_hit_details(self, hit_ix: int = 0) -> HitDetails:
        """
        Returns a Dict with the details of the first hit for convenience.
        """
        hit = self.hits[hit_ix]
        hit_details = HitDetails()
        hit_details.obj = self.hit_obj(hit_ix)
        hit_details.obj_name = (
            hit_details.obj.handle
            if hit.object_id != habitat_sim.stage_id
            else "STAGE"
        )
        hit_details.link_id = self.hit_link(hit_ix, hit_details.obj)
        hit_details.link_name = (
            None
            if not hit_details.link_id
            else hit_details.obj.get_link_name(hit_details.link_id)
        )
        hit_details.object_id = hit.object_id
        hit_details.point = self.hits[hit_ix].point
        return hit_details


class AppStateRobotTeleopViewer(AppState):
    """
    HitL application for posing and teleoperating robots in simulation.
    """

    def __init__(self, app_service: AppService):
        self._app_service = app_service
        self._sim = app_service.sim
        self._current_scene_index = 0

        # bind the app specific config values
        self._app_cfg: DictConfig = omegaconf_to_object(
            app_service.config.robot_teleop
        )

        # todo: probably don't need video-recording stuff for this app
        self._video_output_prefix = "video"

        self._camera_helper = CameraHelper(
            self._app_service.hitl_config, self._app_service.gui_input
        )
        # not supported
        assert not self._app_service.hitl_config.camera.first_person_mode

        # setup the initial camera state
        self.cursor_cast_results: HitObjectInfo = None
        self.mouse_cast_results: HitObjectInfo = None
        self._hide_gui = False

        # additional object managment
        self.added_object_ids: List[int] = []

        # set with SPACE, when true cursor is locked on robot
        self.cursor_follow_robot = True

        self.robot: Optional[Robot] = None
        # in-memory cache of hand poses for quick LERP grasping
        self.hand_grasp_poses: List[List[float]] = []
        self.dof_editor: Optional[DoFEditor] = None
        # self._quest_reader: Optional[QuestReader] = QuestReader(self._app_service)
        self.xr_origin_offset = mn.Vector3(0, 0, 0)
        self.xr_origin_rotation = mn.Quaternion()
        self.xr_pose_adapter = XRPoseAdapter()

        # API for recording and playing back XR trajectories
        self.xr_traj: XRTrajectory = XRTrajectory()

        # debug xr replay
        self.replay_xr_traj = False
        self.recording_xr_traj = False
        self.xr_replay_frame = 0

        self._ik: Optional[
            DifferentialInverseKinematics
        ] = DifferentialInverseKinematics()
        self._cursor_pos = mn.Vector3(0.0, 0.0, 0.0)
        self.navmesh_lines: Optional[
            List[Tuple[mn.Vector3, mn.Vector3]]
        ] = None

        # setup the simulator
        self.set_scene(self._current_scene_index)

        # DEBUG: if replaying from start, sync at start
        if self.replay_xr_traj:
            # set the initial sync state for testing
            self.xr_traj.load_json("test_xr_pose.json")
            self.sync_xr_local_state(self.xr_traj.get_pose(0))

        self._sps_tracker = AverageRateTracker(2.0)
        self._do_pause_physics = False
        self._app_service.users.activate_user(0)

    def next_scene(self) -> None:
        scenes = self._app_cfg.scenes
        self._current_scene_index = (self._current_scene_index + 1) % len(
            scenes
        )
        self.set_scene(self._current_scene_index)

    def set_scene(self, scene_index: int) -> None:
        if self.robot is not None:
            self.robot.clean()

        scene = self._app_cfg.scenes[scene_index]
        print(f"Loading scene: {scene}.")
        self._app_service.reconfigure_sim(self._app_cfg.scene_dataset, scene)

        self.set_aos_dynamic()
        # import and setup the robot from config
        self.import_robot()

        # set the initial state of the UI cursor to the robot viewpoint
        self._cursor_pos = self.robot.ao.transformation.transform_point(
            self.robot.viewpoint_offset
        )

        self._camera_helper.update(self._cursor_pos, 0.0)
        self.dof_editor = None
        self.navmesh_lines = None
        print(f"Loaded scene: {scene}.")

        client_message_manager = self._app_service._client_message_manager
        if client_message_manager is not None:
            client_message_manager.signal_scene_change()

            if True:  # TODO: If in VR...
                self.recompute_navmesh()
                user_pos = AppStateRobotTeleopViewer._find_navmesh_position_near_target(
                    target=self._cursor_pos,
                    distance_from_target=1.2,
                    pathfinder=self._sim.pathfinder,
                )
                if user_pos is not None:
                    client_message_manager.change_humanoid_position(user_pos)
                client_message_manager.update_navmesh_triangles(
                    self._get_navmesh_triangle_vertices()
                )

    def _get_navmesh_triangle_vertices(self):
        """Return vertices (nonindexed triangles) for triangulated NavMesh polys"""
        largest_island_index = get_largest_island_index(
            self._sim.pathfinder, self._sim, allow_outdoor=False
        )
        pts = self._sim.pathfinder.build_navmesh_vertices(largest_island_index)
        assert len(pts) > 0
        assert len(pts) % 3 == 0
        assert len(pts[0]) == 3
        navmesh_fixup_y = -0.17  # sloppy
        return [
            (
                float(point[0]),
                float(point[1]) + navmesh_fixup_y,
                float(point[2]),
            )
            for point in pts
        ]

    def set_aos_dynamic(self) -> None:
        """
        Sets all AOs to dynamic for interaction in case the scene is STATIC.
        """
        for _handle, ao in (
            self._sim.get_articulated_object_manager()
            .get_objects_by_handle_substring()
            .items()
        ):
            ao.motion_type = habitat_sim.physics.MotionType.DYNAMIC
            for link_ix in ao.get_link_ids():
                ao.set_link_friction(link_ix, 1.0)

    def recompute_navmesh(self) -> None:
        """
        Recomputes the scene navmesh using the currently loaded robot's radius.
        """
        mt_cache = {}
        # set all Aos to STATIC so they are baked into the navmesh
        for _handle, ao in (
            self._sim.get_articulated_object_manager()
            .get_objects_by_handle_substring()
            .items()
        ):
            if self.robot.ao.handle != ao.handle:
                mt_cache[ao.handle] = ao.motion_type
                ao.motion_type = habitat_sim.physics.MotionType.STATIC

        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        # get height and radius from the robot
        navmesh_settings.agent_height = (
            self.robot.robot_cfg.navmesh_height
            if self.robot is not None
            else 1.3
        )
        navmesh_settings.agent_radius = (
            self.robot.robot_cfg.navmesh_radius
            if self.robot is not None
            else 0.25
        )
        navmesh_settings.include_static_objects = True
        self._sim.recompute_navmesh(
            self._sim.pathfinder,
            navmesh_settings,
        )
        # set all Aos to back to their previous motion types
        for _handle, ao in (
            self._sim.get_articulated_object_manager()
            .get_objects_by_handle_substring()
            .items()
        ):
            if self.robot.ao.handle != ao.handle:
                ao.motion_type = mt_cache[ao.handle]

    def build_navmesh_lines(self) -> None:
        """
        Constructs a set of debug line endpoint pairs for all edges on the navmesh.
        """
        self.navmesh_lines = []
        # NOTE: This could be color coded by island or restricted to a particular island
        i = self._sim.pathfinder.build_navmesh_vertex_indices()
        v = self._sim.pathfinder.build_navmesh_vertices()
        for tri in range(int(len(i) / 3)):
            v0 = v[i[tri * 3]]
            v1 = v[i[tri * 3 + 1]]
            v2 = v[i[tri * 3 + 2]]
            self.navmesh_lines.extend([(v0, v1), (v1, v2), (v2, v0)])

    def draw_navmesh_lines(self, color: mn.Color4 = None) -> None:
        """
        Draw navmesh as a wireframe.
        """
        if self.navmesh_lines is None:
            return
        if color is None:
            color = mn.Color4(0.8, 0.8, 0.8, 0.8)
        for p0, p1 in self.navmesh_lines:
            self._app_service.gui_drawer.draw_transformed_line(p0, p1, color)

    def import_robot(self) -> None:
        """
        Imports the robot defined in the yaml config.
        """
        # hot reload the config file
        robot_cfg = compose(config_name="robot_settings")
        if robot_cfg is not None:
            # initialize the robot from config
            self.robot = Robot(self._sim, robot_cfg)
            self.recompute_navmesh()
            self.robot.place_robot(
                self._sim.pathfinder.get_random_navigable_point()
            )
            # load the grasping poses. If this line fails (i.e. integrating a new robot, comment it and record the configuration subset and poses with the same key names below)
            self.hand_grasp_poses = [
                self.robot.pos_subsets["left_hand"].fetch_cached_pose(
                    pose_name=hand_subset_key
                )
                for hand_subset_key in ["hand_open", "grasp"]
            ]
        else:
            print("No robot configured.")

    def remove_object(self, object_id: int) -> None:
        """
        Remove the specified object. Only accepts removal of objects added to the scene.
        """
        try:
            added_id = next(
                obj_id
                for obj_id in self.added_object_ids
                if object_id == obj_id
            )
            self._sim.get_rigid_object_manager().remove_object_by_id(added_id)
            self.added_object_ids.remove(added_id)
        except StopIteration:
            print(f"No object with id {object_id} to remove.")

    def add_object_at(
        self,
        obj_template_handle: str,
        translation: mn.Vector3 = None,
        rotation: mn.Quaternion = None,
    ) -> habitat_sim.physics.ManagedRigidObject:
        """
        Add the desired object and assign it the provided translation and rotation if applicable.
        Returns a ManagedObject and also registers the object_id in a global list for later cleanup.
        """
        ro = (
            self._sim.get_rigid_object_manager().add_object_by_template_handle(
                obj_template_handle
            )
        )
        if ro is not None:
            self.added_object_ids.append(ro.object_id)
            if translation is not None:
                ro.translation = translation
            if rotation is not None:
                ro.rotation = rotation
        return ro

    def _update_help_text(self) -> None:
        """
        Draw help text to the screen.
        """
        if self._hide_gui:
            return

        help_text = (
            "Controls:\n"
            + " 'WASD' to translate laterally and 'ZX' to move up|down.\n"
            + " hold 'R' and move the mouse to rotate camera and mouse wheel to zoom.\n"
            + " '0' to change scene.\n"
        )

        # show some details about hits under the cursor
        cursor_cast_results_text = "\nCursor RayCast: "
        if (
            self.cursor_cast_results is not None
            and self.cursor_cast_results.has_hits()
        ):
            hit_details = self.cursor_cast_results.get_hit_details()
            cursor_cast_results_text += (
                f"{hit_details.object_id} {hit_details.obj_name}"
            )
            if hit_details.link_id is not None:
                cursor_cast_results_text += (
                    f" > {hit_details.link_id} {hit_details.link_name}"
                )
        mouse_cast_results_text = "\nMouse RayCast: "
        if (
            self.mouse_cast_results is not None
            and self.mouse_cast_results.has_hits()
        ):
            hit_details = self.mouse_cast_results.get_hit_details()
            mouse_cast_results_text += (
                f"{hit_details.object_id} {hit_details.obj_name}"
            )
            if hit_details.link_id is not None:
                mouse_cast_results_text += (
                    f" > {hit_details.link_id} {hit_details.link_name}"
                )
                if (
                    self.robot is not None
                    and hit_details.obj_name == self.robot.ao.handle
                ):
                    link_dof = self.robot.ao.get_link_joint_pos_offset(
                        hit_details.link_id
                    )
                    mouse_cast_results_text += f"--{link_dof}"

        controls_str = (
            help_text + cursor_cast_results_text + mouse_cast_results_text
        )
        if len(controls_str) > 0:
            self._app_service.text_drawer.add_text(
                controls_str, TextOnScreenAlignment.TOP_LEFT
            )

    def _update_cursor_pos(self) -> None:
        """
        Updates the position of the camera focus point when keys are pressed.
        Equivalent to "walking" in the scene.
        """
        gui_input = self._app_service.gui_input
        y_speed = 0.1
        # NOTE: cursor elevation is always controllable
        if gui_input.get_key(KeyCode.Z):
            self._cursor_pos.y -= y_speed
        if gui_input.get_key(KeyCode.X):
            self._cursor_pos.y += y_speed

        if self.cursor_follow_robot and self.robot is not None:
            # lock the cursor to the robot
            new_cursor_position = self.robot.ao.transformation.transform_point(
                self.robot.viewpoint_offset
            )
            # only set horizontal each frame to allow y to be adjusted via UI
            self._cursor_pos[0] = new_cursor_position[0]
            self._cursor_pos[2] = new_cursor_position[2]
        else:
            # manual cursor control
            xz_forward = self._camera_helper.get_xz_forward()
            xz_right = mn.Vector3(-xz_forward.z, 0.0, xz_forward.x)
            speed = (
                self._app_cfg.camera_move_speed
                * self._camera_helper.cam_zoom_dist
            )
            if gui_input.get_key(KeyCode.W):
                self._cursor_pos += xz_forward * speed
            if gui_input.get_key(KeyCode.S):
                self._cursor_pos -= xz_forward * speed
            if gui_input.get_key(KeyCode.D):
                self._cursor_pos += xz_right * speed
            if gui_input.get_key(KeyCode.A):
                self._cursor_pos -= xz_right * speed

    def _find_navmesh_position_near_target(
        target: mn.Vector3,
        distance_from_target: float,
        pathfinder: habitat_sim.PathFinder,
    ) -> Optional[mn.Vector3]:
        r"""
        Get a position from which the robot can be seen.
        """
        # Look up for a camera position by sampling radially around the target for a navigable position.
        navigable_point = None
        sample_count = 8
        max_dist_to_obstacle = 0.0
        for i in range(sample_count):
            radial_angle = i * 2.0 * math.pi / float(sample_count)
            dist_x = math.sin(radial_angle) * distance_from_target
            dist_z = math.cos(radial_angle) * distance_from_target
            candidate = mn.Vector3(
                target.x + dist_x, target.y, target.z + dist_z
            )
            if pathfinder.is_navigable(candidate, 3.0):
                dist_to_closest_obstacle = (
                    pathfinder.distance_to_closest_obstacle(candidate, 2.0)
                )
                if dist_to_closest_obstacle > max_dist_to_obstacle:
                    max_dist_to_obstacle = dist_to_closest_obstacle
                    navigable_point = candidate

        return navigable_point

    def move_robot_on_navmesh(self) -> None:
        """
        Handles key press updates the robot on the navmesh.
        """
        # TODO: using camera move speed, but could be separated
        gui_input = self._app_service.gui_input
        speed = self._app_cfg.camera_move_speed
        if self.robot is not None:
            start = self.robot.ao.translation
            end = mn.Vector3(start)
            if gui_input.get_key(KeyCode.I):
                end = end + self.robot.ao.transformation.transform_vector(
                    mn.Vector3(speed, 0, 0)
                )
            if gui_input.get_key(KeyCode.K):
                end = end + self.robot.ao.transformation.transform_vector(
                    mn.Vector3(-speed, 0, 0)
                )
            r_speed = 0.05
            if gui_input.get_key(KeyCode.L):
                r = mn.Quaternion.rotation(
                    mn.Rad(-r_speed), mn.Vector3(0, 1, 0)
                )
                self.robot.ao.rotation = r * self.robot.ao.rotation
            if gui_input.get_key(KeyCode.J):
                r = mn.Quaternion.rotation(
                    mn.Rad(r_speed), mn.Vector3(0, 1, 0)
                )
                self.robot.ao.rotation = r * self.robot.ao.rotation

            # XR input for robot motion
            if self._app_service.remote_client_state is not None:
                xr_input = self._app_service.remote_client_state.get_xr_input(
                    0
                )
                if xr_input is not None:
                    left = xr_input.controllers[HAND_LEFT]
                    right = xr_input.controllers[HAND_RIGHT]

                    # use left thumbstick to move the robot
                    left_thumbstick = left.get_thumbstick()
                    end = end + self.robot.ao.transformation.transform_vector(
                        mn.Vector3(speed * left_thumbstick[1], 0, 0)
                    )
                    r = mn.Quaternion.rotation(
                        mn.Rad(-r_speed * left_thumbstick[0]),
                        mn.Vector3(0, 1, 0),
                    )
                    self.robot.ao.rotation = r * self.robot.ao.rotation

                    # use right thumbstick to rotate the camera
                    right_thumbstick = right.get_thumbstick()
                    scale = 0.06
                    self._camera_helper._lookat_offset_yaw += (
                        scale * right_thumbstick[0]
                    )
                    self._camera_helper._lookat_offset_pitch += (
                        scale * -right_thumbstick[1]
                    )
                    self._camera_helper._lookat_offset_pitch = np.clip(
                        self._camera_helper._lookat_offset_pitch,
                        self._camera_helper._min_lookat_offset_pitch,
                        self._camera_helper._max_lookat_offset_pitch,
                    )

            self.robot.ao.translation = self._sim.pathfinder.try_step(
                start, end
            )

            # sample new placement for robot on navmesh
            if gui_input.get_key_down(KeyCode.M):
                self.robot.place_robot(
                    self._sim.pathfinder.get_random_navigable_point()
                )

    def draw_lookat(self) -> None:
        """
        Draws the yellow circle centered in the camera view for reference.
        """
        if self._hide_gui:
            return

        # a cursor centered yellow circle with Y-up normal for reference
        self._app_service.gui_drawer.draw_circle(
            self._cursor_pos,
            radius=0.05,
            color=mn.Color3(1, 0.75, 0),  # yellow
            normal=mn.Vector3(0, 1, 0),
        )
        # a cursor centered green circle with camera normal
        self._app_service.gui_drawer.draw_circle(
            self._cursor_pos,
            radius=0.02,
            color=mn.Color3(0.8, 0.8, 0.8),  # yellow
            normal=self.cursor_cast_results.ray.direction,
        )

    def handle_keys(
        self, dt: float, post_sim_update_dict: Dict[str, Any]
    ) -> None:
        """
        This function processes the key press events.
        """

        gui_input = self._app_service.gui_input
        if gui_input.get_key_down(KeyCode.ESC):
            post_sim_update_dict["application_exit"] = True

        if gui_input.get_key_down(KeyCode.SPACE):
            self.cursor_follow_robot = not self.cursor_follow_robot
            print(f"cursor_follow_robot = {self.cursor_follow_robot}")

        if gui_input.get_key_down(KeyCode.H):
            self._hide_gui = not self._hide_gui

        if gui_input.get_key_down(KeyCode.T):
            # remove the current robot and reload from URDF
            # I.E., Hot-reload robot
            self.robot.clean()
            self.import_robot()

        # navmesh features
        if gui_input.get_key_down(KeyCode.N):
            if self.navmesh_lines is not None:
                self.navmesh_lines = None
            else:
                self.build_navmesh_lines()

        # robot pose caching and loading
        if gui_input.get_key_down(KeyCode.P) and self.robot is not None:
            # cache the current pose
            print("Save pose to cache:")
            configuration_subset_name = input(
                " Enter the configuration subset key. (Press ENTER without input for full pose.) >"
            )
            if (
                configuration_subset_name != ""
                and configuration_subset_name not in self.robot.pos_subsets
            ):
                print(
                    f" Invalid configuration subset name. Defined subsets are: {self.robot.pos_subsets.keys()}"
                )
                return
            cached_pose_name = input(
                "Enter the desired cache key name of the pose > "
            )
            if cached_pose_name == "":
                print("No cache key name entered, defaulted to 'default'.")
                cached_pose_name = "default"
            if configuration_subset_name == "":
                self.robot.cache_pose(pose_name=cached_pose_name)
            else:
                self.robot.pos_subsets[configuration_subset_name].cache_pose(
                    pose_name=cached_pose_name
                )
        if gui_input.get_key_down(KeyCode.O) and self.robot is not None:
            # self.robot.set_cached_pose()
            # load a pose from the cache
            configuration_subset_name = input(
                "Enter the configuration subset key. (Press ENTER without input for full pose.) >"
            )
            if (
                configuration_subset_name != ""
                and configuration_subset_name not in self.robot.pos_subsets
            ):
                print(
                    f" Invalid configuration subset name. Defined subsets are: {self.robot.pos_subsets.keys()}"
                )
                return
            cached_pose_name = input("Enter the cache key name of the pose > ")
            if cached_pose_name == "":
                print("No cache key name entered, defaulted to 'default'.")
                cached_pose_name = "default"
            if configuration_subset_name == "":
                self.robot.set_cached_pose(
                    pose_name=cached_pose_name,
                    set_motor_targets=self.robot.using_joint_motors,
                    set_positions=not self.robot.using_joint_motors,
                )
            else:
                self.robot.pos_subsets[
                    configuration_subset_name
                ].set_cached_pose(
                    pose_name=cached_pose_name,
                    set_motor_targets=self.robot.using_joint_motors,
                    set_positions=not self.robot.using_joint_motors,
                )

        if gui_input.get_key_down(KeyCode.Y):
            # insert an object at the mouse raycast position
            obj_shortname = "000bbe302b53fd2904e7ae92e1516a18f29de02d"
            obj_template_handle = list(
                self._sim.get_object_template_manager()
                .get_templates_by_handle_substring(obj_shortname)
                .keys()
            )[0]
            new_obj = self.add_object_at(obj_template_handle)
            obj_size_down = sutils.get_obj_size_along(
                self._sim, new_obj.object_id, mn.Vector3(0, -1, 0)
            )
            placement_position = self.mouse_cast_results.hits[
                0
            ].point + mn.Vector3(0, obj_size_down[0], 0)
            new_obj.translation = placement_position

        if gui_input.get_key_down(KeyCode.U):
            self.remove_object(self.mouse_cast_results.hits[0].object_id)

        # Load next scene
        if gui_input.get_key_down(KeyCode.ZERO):
            self.next_scene()

    def sync_xr_local_state(self, xr_pose: XRPose = None):
        """
        Sets a local offset variable to transform XR headset and controller origin into the local robot space.
        Should be run once when the quest user reaches a comfortable pose.
        If no XRPose is provided, the current state is used.
        """
        if xr_pose is None:
            xr_pose = XRPose(
                remote_client_state=self._app_service.remote_client_state
            )

        if not xr_pose.valid:
            print(
                "Cannot set origin, no valid XR state. Is the XR user connected?"
            )
            return

        # the following lines construct a local correction frame which uses the headset pose as the origin
        # align the XR headset orientation with the x axis
        align_z_to_x = mn.Quaternion.rotation(
            mn.Rad(mn.math.pi / 2.0), mn.Vector3(0, 1, 0)
        )
        self.xr_origin_rotation = xr_pose.rot_head.inverted() * align_z_to_x
        # align the headset position with the origin
        self.xr_origin_offset = self.xr_origin_rotation.transform_vector(
            -xr_pose.pos_head
        )
        # TODO: try setting the VR user state to align with the robot
        # TODO: waiting for Unity control pathway for setting XR offset transformation
        # client_message_manager = self._app_service.client_message_manager
        # if client_message_manager is not None and self.robot is not None:
        #    client_message_manager.change_humanoid_position(self.robot.ao.translation)

    def handle_xr_input(self, dt: float):
        if self._app_service.remote_client_state is None:
            return

        xr_input = self._app_service.remote_client_state.get_xr_input(0)
        left = xr_input.left_controller
        right = xr_input.right_controller
        from habitat_hitl.core.xr_input import XRButton

        ###########################################################
        ###########################################################
        # Enumerate All XR Inputs here to avoid fragmentation and accidental overwriting
        # LEFT CONTROLLER BUTTONS
        if left.get_button_down(XRButton.ONE):
            print("pressed one left")
            self.sync_xr_local_state()
            print("synced headset state...")
        if left.get_button_up(XRButton.ONE):
            pass
        if left.get_button_down(XRButton.TWO):
            print("Resetting Robot Joint Positions")
            for reset_joint_values, arm_subset_key in [
                ("right_arm_in", "right_arm"),
                ("left_arm_in", "left_arm"),
            ]:
                self.robot.pos_subsets[arm_subset_key].set_cached_pose(
                    pose_name=reset_joint_values,
                    set_motor_targets=self.robot.using_joint_motors,
                    set_positions=not self.robot.using_joint_motors,
                )

        if left.get_button_up(XRButton.TWO):
            pass
        if left.get_button_down(XRButton.START):
            print("pressed START left")
            # NOTE: reserved by QuestReader for now...
        if left.get_button_up(XRButton.START):
            pass

        # RIGHT CONTROLLER BUTTONS
        if right.get_button_down(XRButton.ONE):
            print("pressed one right")
            print("Starting to record a new trajectory")
            self.xr_traj = XRTrajectory()
            self.recording_xr_traj = True
            self.replay_xr_traj = False
        if right.get_button_up(XRButton.ONE):
            print("released one right")
            self.recording_xr_traj = False
            self.xr_traj.save_json()
            print(
                f"Saved the trajectory with {len(self.xr_traj.traj)} poses to 'xr_pose.json'."
            )
        if right.get_button_down(XRButton.TWO):
            print("pressed two right")
            self.replay_xr_traj = not self.replay_xr_traj
            self.recording_xr_traj = False
            print(f"XR Traj playback = {self.replay_xr_traj}")
        if right.get_button_up(XRButton.TWO):
            pass
        # NOTE: XRButton.START is reserved for Quest menu functionality

        # NOTE: PRIMARY_HAND_TRIGGER mapped to IK toggle
        # NOTE: PRIMARY_INDEX_TRIGGER mapped to grasping functionality

        # TODO: PRIMARY_THUMBSTICK (click?) unmapped currently and available
        ###########################################################
        ###########################################################

        # construct the xr -> robot frame alignment transform
        # first construct a 4x4 with the local correction components from sync_xr_local_state
        local_correction_tform = mn.Matrix4().from_(
            self.xr_origin_rotation.to_matrix(), self.xr_origin_offset
        )
        # sync the aligned local XR state with the robot's orientation and cursor position
        local_to_robot = mn.Matrix4().from_(
            self.robot.ao.rotation.to_matrix(), self._cursor_pos
        )
        # cache the combined XR local to global transform
        self.xr_pose_adapter.xr_local_to_global = local_to_robot.__matmul__(
            local_correction_tform
        )

        # TODO: example of freezing the replay at a known frame
        # self.xr_replay_frame =1

        # test IK with replay motion frames
        if (
            self.replay_xr_traj
            and len(self.xr_traj.traj) > 0
            and self.xr_replay_frame < len(self.xr_traj.traj)
        ):
            xr_replay_pose = self.xr_traj.get_pose(self.xr_replay_frame)
            global_xr_pose = self.xr_pose_adapter.get_global_xr_pose(
                xr_replay_pose
            )
            _cur_angles = self.robot.ao.joint_positions
            # right hand
            robot_right_base_link = 47
            arm_base_link_t = self.robot.ao.get_link_scene_node(
                robot_right_base_link
            ).transformation.inverted()
            xr_pose_in_robot_frame = self.xr_pose_adapter.xr_pose_transformed(
                global_xr_pose, arm_base_link_t
            )
            pose_right = to_ik_pose(
                (
                    xr_pose_in_robot_frame.pos_right,
                    xr_pose_in_robot_frame.rot_right,
                )
            )
            _cur_angles[35:42] = self._ik.inverse_kinematics(
                pose_right, _cur_angles[35:42]
            )
            self.robot.ao.joint_positions = _cur_angles
            # TODO: activate motor control
            # self.robot.ao.update_all_motor_targets(_cur_angles)

            # left hand
            robot_left_base_link = 16
            arm_base_link_t = self.robot.ao.get_link_scene_node(
                robot_left_base_link
            ).transformation.inverted()
            xr_pose_in_robot_frame = self.xr_pose_adapter.xr_pose_transformed(
                global_xr_pose, arm_base_link_t
            )
            pose_left = to_ik_pose(
                (
                    xr_pose_in_robot_frame.pos_left,
                    xr_pose_in_robot_frame.rot_left,
                )
            )
            _cur_angles[12:19] = self._ik.inverse_kinematics(
                pose_left, _cur_angles[12:19]
            )
            self.robot.ao.joint_positions = _cur_angles
            # TODO: activate motor control
            # self.robot.ao.update_all_motor_targets(_cur_angles)

        # IK for each hand when trigger is pressed
        xr_pose = XRPose(
            remote_client_state=self._app_service.remote_client_state
        )
        if xr_pose.valid:
            global_xr_pose = self.xr_pose_adapter.get_global_xr_pose(xr_pose)

            # combine both hands' logic to reduce boilerplate code
            for xrcontroller, base_link_key, arm_subset_key in [
                (right, "right_arm_base", "right_arm"),
                (left, "left_arm_base", "left_arm"),
            ]:
                if xrcontroller.get_hand_trigger() > 0:
                    arm_base_link = self.robot.link_subsets[
                        base_link_key
                    ].link_ixs[0]
                    xr_pose_in_robot_frame = (
                        self.xr_pose_adapter.xr_pose_transformed(
                            global_xr_pose,
                            self.robot.ao.get_link_scene_node(
                                arm_base_link
                            ).transformation.inverted(),
                        )
                    )
                    arm_joint_subset = self.robot.pos_subsets[arm_subset_key]
                    cur_arm_pose = arm_joint_subset.get_pos()
                    if self.robot.using_joint_motors:
                        cur_arm_pose = arm_joint_subset.get_motor_pos()

                    # TODO: a bit hacky way to get the correct hand here
                    xr_pose = (
                        xr_pose_in_robot_frame.pos_left,
                        xr_pose_in_robot_frame.rot_left,
                    )
                    if "right" in base_link_key:
                        xr_pose = (
                            xr_pose_in_robot_frame.pos_right,
                            xr_pose_in_robot_frame.rot_right,
                        )

                    new_arm_pose = self._ik.inverse_kinematics(
                        to_ik_pose(xr_pose), cur_arm_pose
                    )
                    # using configuration subset to avoid accidental overwriting with noise
                    if self.robot.using_joint_motors:
                        self.robot.pos_subsets[arm_subset_key].set_motor_pos(
                            new_arm_pose
                        )
                    else:
                        self.robot.pos_subsets[arm_subset_key].set_pos(
                            new_arm_pose
                        )

            # grasping logic
            # TODO: actual constraints
            # NOTE: currently toggles between the grasp poses with trigger depth
            for xrcontroller, hand_subset_key in [
                (right, "right_hand"),
                (left, "left_hand"),
            ]:
                # active_pose = "hand_open"
                # if xrcontroller.get_index_trigger() > 0.5:
                #     active_pose = "grasp"
                # elif xrcontroller.get_index_trigger() > 0:
                #     active_pose = "pre_grasp"
                # self.robot.pos_subsets[hand_subset_key].set_cached_pose(
                #     pose_name=active_pose,
                #     set_motor_targets=self.robot.using_joint_motors,
                #     set_positions=not self.robot.using_joint_motors,
                # )
                cur_pose = LERP(
                    self.hand_grasp_poses[0],
                    self.hand_grasp_poses[1],
                    xrcontroller.get_index_trigger(),
                )
                if not self.robot.using_joint_motors:
                    self.robot.pos_subsets[hand_subset_key].set_pos(cur_pose)
                else:
                    self.robot.pos_subsets[hand_subset_key].set_motor_pos(
                        cur_pose
                    )

            # record XR state if necessary
            if self.recording_xr_traj:
                self.xr_traj.add_pose(xr_pose)

    def handle_mouse_press(self) -> None:
        """
        Uses GuiInput to catch and handle mouse press events independent from camera control.
        NOTE: Camera control UI is handled directly in the camera_helper.
        """
        if self._app_service.gui_input.get_mouse_button_down(
            MouseButton.RIGHT
        ):
            if (
                self.mouse_cast_results is not None
                and self.mouse_cast_results.has_hits()
            ):
                self.robot.place_robot(
                    base_pos=self.mouse_cast_results.hits[0].point
                )
        elif (
            self._app_service.gui_input.get_mouse_button_down(MouseButton.LEFT)
            and self.mouse_cast_results is not None
            and self.mouse_cast_results.has_hits
        ):
            # left click
            hit_link = self.mouse_cast_results.hit_link()
            if hit_link is not None:
                # hit an AO link, so create a DoFEditor
                self.dof_editor = DoFEditor(self.robot, hit_link)

        # destroy any DofEditor when releasing a LEFT click
        if self._app_service.gui_input.get_mouse_button_up(MouseButton.LEFT):
            self.dof_editor = None

        # update dof editor when dragging an active LEFT click
        if (
            self._app_service.gui_input.get_mouse_button(MouseButton.LEFT)
            and self.dof_editor is not None
        ):
            self.dof_editor.update(
                self._app_service.gui_input._relative_mouse_position[0] * 0.01,
                set_positions=not self.robot.using_joint_motors,
            )

    def get_cursor_cast(self) -> None:
        """
        Raycast in the scene to get the 3D point directly under the cursor.
        Updates self.cursor_cast_results
        """
        eye, cursor = self._camera_helper._get_eye_and_lookat(self._cursor_pos)
        ray = habitat_sim.geo.Ray(eye, (cursor - eye).normalized())
        self.cursor_cast_results = HitObjectInfo(
            self._sim.cast_ray(ray), self._sim
        )

    def get_mouse_cast(self) -> None:
        """
        Raycast in the scene to get the 3D point directly under the mouse.
        Updates self.mouse_cast_results
        """
        ray = self._app_service.gui_input.mouse_ray
        if ray is not None:
            self.mouse_cast_results = HitObjectInfo(
                self._sim.cast_ray(ray), self._sim
            )

    def debug_draw_quest(self) -> None:
        """
        Debug draw the quest controller and headset poses as axis frames.
        """
        dblr = self._app_service.gui_drawer

        if self.replay_xr_traj and len(self.xr_traj.traj) > 0:
            if self.xr_replay_frame >= len(self.xr_traj.traj):
                self.xr_replay_frame = 0
            xr_replay_pose = self.xr_traj.get_pose(self.xr_replay_frame)

            xr_replay_pose.draw_pose(
                dblr, transform=self.xr_pose_adapter.xr_local_to_global
            )

            self.xr_replay_frame += 1
        else:
            current_xr_pose = XRPose(
                remote_client_state=self._app_service.remote_client_state
            )
            if current_xr_pose.valid:
                self.xr_pose_adapter.get_global_xr_pose(
                    current_xr_pose
                ).draw_pose(dblr)

    # this is where connect with the thread for controller positions
    def sim_update(
        self, dt: float, post_sim_update_dict: Dict[str, Any]
    ) -> None:
        """
        The primary application loop function.
        Handle key presses, steps the simulator, updates the GUI, debug draw, etc...
        """

        self._sps_tracker.increment()

        # IO handling
        self.handle_keys(dt, post_sim_update_dict)
        self.handle_mouse_press()
        self.handle_xr_input(dt)
        self.move_robot_on_navmesh()
        self._update_cursor_pos()

        # step the simulator
        if self.robot is not None and self.robot.using_joint_motors:
            self._sim.step_physics(dt)

        # update robot finger raycast sensors
        if self.robot is not None:
            for finger_raycast_sensor in self.robot.finger_raycast_sensors:
                finger_raycast_sensor.update_sensor_raycasts()

        # update the camera position
        self._camera_helper.update(self._cursor_pos, dt)
        self._cam_transform = self._camera_helper.get_cam_transform()
        post_sim_update_dict["cam_transform"] = self._cam_transform

        dblr = self._app_service.gui_drawer
        # update the cursor raycast
        self.get_cursor_cast()
        if (
            self.cursor_cast_results is not None
            and self.cursor_cast_results.has_hits()
        ):
            dblr.draw_circle(
                translation=self.cursor_cast_results.hits[0].point,
                radius=0.005,
                color=mn.Color4(mn.Vector3(1.0), 1.0),
                normal=self.cursor_cast_results.hits[0].normal,
            )
        self.get_mouse_cast()
        if (
            self.mouse_cast_results is not None
            and self.mouse_cast_results.has_hits()
        ):
            dblr.draw_circle(
                translation=self.mouse_cast_results.hits[0].point,
                radius=0.005,
                color=mn.Color4(mn.Vector3(1.0), 1.0),
                normal=self.mouse_cast_results.hits[0].normal,
            )
            if self.robot is not None:
                hit_details = self.mouse_cast_results.get_hit_details()
                # hit the robot, so try to draw the link
                if (
                    hit_details.obj_name == self.robot.ao.handle
                    and hit_details.link_id is not None
                ):
                    self.robot.draw_dof(
                        dblr,
                        hit_details.link_id,
                        self._cam_transform.translation,
                    )

        # NOTE: do debug drawing here
        # draw lookat ring
        self.draw_lookat()
        self.debug_draw_quest()
        self.robot.draw_debug(dblr)
        if self.dof_editor is not None:
            self.dof_editor.debug_draw(dblr, self._cam_transform.translation)
        self.draw_navmesh_lines()
        self._update_help_text()


@hydra.main(version_base=None, config_path="./", config_name="robot_teleop")
def main(config):
    hitl_main(
        config, lambda app_service: AppStateRobotTeleopViewer(app_service)
    )


if __name__ == "__main__":
    register_hydra_plugins()
    main()
