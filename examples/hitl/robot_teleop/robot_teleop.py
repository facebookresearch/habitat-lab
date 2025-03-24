#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from habitat.datasets.rearrange.navmesh_utils import get_largest_island_index
import hydra
import magnum as mn
from hydra import compose
from omegaconf import DictConfig

import habitat.sims.habitat_simulator.sim_utilities as sutils
import habitat_sim  # unfortunately we can't import this earlier
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
from habitat_hitl.environment.camera_helper import CameraHelper
from habitat_sim.gfx import DebugLineRender

# path to this example app directory
dir_path = os.path.dirname(os.path.realpath(__file__))

# TODO LIST:
# - attach robots to navmesh using multi-point abstraction and configuration
# - define API for switching between saved poses (e.g. for grip poses)
# - setup constraint management API
# - setup pymomentum IK API
# - setup the Meta hand and define a few standard "grasp" poses
# - setup a VR interface for IK tracking EF
#   - use controllers for toggling things or mouse interchange
#   - use hand state for direct manipulation of the Meta Hand


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
        # expect a "urdf" config field with the filepath
        self.ao = self.sim.get_articulated_object_manager().add_articulated_object_from_urdf(
            self.robot_cfg.urdf,
            fixed_base=self.robot_cfg.fixed_base
            if hasattr(self.robot_cfg, "fixed_base")
            else False,
            force_reload=True,
        )

        # create joint motors
        self.motor_ids_to_link_ids: Dict[int, int] = {}
        if self.robot_cfg.create_joint_motors:
            self.create_joint_motors()

        # set initial pose
        self.set_cached_pose(
            pose_name=self.robot_cfg.initial_pose, set_positions=True
        )
        #self.init_ik()

    def init_ik(self):
        """
        Initialize pymomentum and load a model.
        """
        try:
            import pymomentum.geometry as pym_geo

            print("Initializing pymomentum IK library.")
            print(f"Loading URDF: {self.robot_cfg.urdf}")
            print(" ")
            print(" ")
            print(" ")
            self.momentum_character = pym_geo.Character.load_urdf(
                self.robot_cfg.urdf
            )
            # TODO: the above character is available for ik
        except:
            print("Could not initialize pymomentum IK library.")

    def create_joint_motors(self):
        """
        Creates a full set of joint motors for the robot.
        """
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

    def set_cached_pose(
        self,
        pose_file: str = os.path.join(dir_path, "robot_poses.json"),
        pose_name: str = "default",
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
                    f"Cannot load cached pose. No pose named {pose_name} cached for robot {self.ao.handle}."
                )
                return
            pose = poses[self.ao.handle][pose_name]
            if len(pose) != len(self.ao.joint_positions):
                print(
                    f"Cannot load cached pose (size {len(pose)}) as it does not match number of dofs ({len(self.ao.joint_positions)})"
                )
                return
            if self.robot_cfg.create_joint_motors:
                self.ao.update_all_motor_targets(pose)
            if set_positions:
                self.ao.joint_positions = pose

    def cache_pose(
        self,
        pose_file: str = os.path.join(dir_path, "robot_poses.json"),
        pose_name: str = "default",
    ) -> None:
        """
        Saves the current robot pose in a json cache file with the given name.
        """
        # create the directory if it doesn't exist
        dir = pose_file[: -len(pose_file.split("/")[-1])]
        os.makedirs(dir, exist_ok=True)
        poses = {}
        if os.path.exists(pose_file):
            with open(pose_file, "r") as f:
                poses = json.load(f)
        if self.ao.handle not in poses:
            poses[self.ao.handle] = {}
        poses[self.ao.handle][pose_name] = self.ao.joint_positions
        with open(pose_file, "w") as f:
            json.dump(
                poses,
                f,
                indent=4,
            )

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
        draw_at = center + (cam_pos - center).normalized() * size_to_camera

        link_T = self.ao.get_link_scene_node(link_ix).transformation
        global_link_pos = link_T.translation - link_T.transform_vector(
            self.ao.get_link_joint_to_com(link_ix)
        )

        joint_limits = self.ao.joint_position_limits
        joint_positions = self.ao.joint_positions

        for local_dof in range(self.ao.get_link_num_dofs(link_ix)):
            # this link has dofs

            dof = self.ao.get_link_joint_pos_offset(link_ix) + local_dof
            dof_value = joint_positions[dof]
            min_dof = joint_limits[0][dof]
            max_dof = joint_limits[1][dof]
            interp_dof = (dof_value - min_dof) / (max_dof - min_dof)

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


class DoFEditor:
    """
    A utility class for manipulating a robot DoF via GUI.
    """

    def __init__(self, robot: Robot, link_ix: int):
        self.robot = robot
        self.link_ix = link_ix

        self.joint_limits = self.robot.ao.joint_position_limits
        joint_positions = self.robot.ao.joint_positions
        self.dof = self.robot.ao.get_link_joint_pos_offset(
            link_ix
        )  # note this only handle single dof joints
        self.dof_value = joint_positions[self.dof]
        self.min_dof = self.joint_limits[0][self.dof]
        self.max_dof = self.joint_limits[1][self.dof]
        self.motor_id = None
        for motor_id, link_ix in self.robot.motor_ids_to_link_ids.items():
            if link_ix == self.link_ix:
                self.motor_id = motor_id

    def update(self, dt: float) -> None:
        """
        Attempt to increment the dof value by dt.
        """
        self.dof_value += dt
        # clamp to joint limits
        self.dof_value = min(self.max_dof, max(self.min_dof, self.dof_value))
        if self.motor_id is not None:
            jms = self.robot.ao.get_joint_motor_settings(self.motor_id)
            jms.position_target = self.dof_value
            self.robot.ao.update_joint_motor(self.motor_id, jms)
        else:
            # this joint has no motor, so directly manipulate the position
            cur_pos = self.robot.ao.joint_positions
            cur_pos[self.dof] = self.dof_value

    def debug_draw(self, dblr: DebugLineRender, cam_pos: mn.Vector3) -> None:
        """
        Draw a 1D number line showing the dof state vs. min and max limits.
        """
        # keep the dof drawn as the mouse moves
        self.robot.draw_dof(dblr, self.link_ix, cam_pos)

        link_obj_id = self.robot.ao.link_ids_to_object_ids[self.link_ix]
        obj_bb, transform = sutils.get_bb_for_object_id(
            self.robot.sim, link_obj_id
        )
        center = transform.transform_point(obj_bb.center())

        to_camera = (cam_pos - center).normalized()
        left_vec = mn.math.cross(to_camera, mn.Vector3(0, 1, 0))
        rel_up = mn.math.cross(to_camera, left_vec)

        size_to_camera, center = sutils.get_obj_size_along(
            self.robot.sim, link_obj_id, to_camera
        )
        draw_at = center + to_camera * (size_to_camera + 0.05)
        line_len = 0.3
        dash_height = 0.05
        frame_color = mn.Color3(0.8, 0.8, 0.4)
        end1 = draw_at - left_vec * (line_len / 2)
        end2 = draw_at + left_vec * (line_len / 2)
        dblr.draw_transformed_line(end1, end2, frame_color)
        dblr.draw_transformed_line(
            end1 + rel_up * dash_height,
            end1 - rel_up * dash_height,
            frame_color,
        )
        dblr.draw_transformed_line(
            end2 + rel_up * dash_height,
            end2 - rel_up * dash_height,
            frame_color,
        )
        # draw the current dof value
        interp_dof = (self.dof_value - self.min_dof) / (
            self.max_dof - self.min_dof
        )
        cur_dof_pos = end1 + (end2 - end1) * interp_dof
        dof_color = mn.Color3(0.4, 0.8, 0.4)
        dblr.draw_transformed_line(
            cur_dof_pos + rel_up * dash_height * 1.05,
            cur_dof_pos - rel_up * dash_height * 1.05,
            dof_color,
        )


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
        if self.has_hits:
            if self.hits[hit_ix] == habitat_sim.stage_id:
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

        # set with SPACE, when true cursor is locked on robot
        self.cursor_follow_robot = False

        self.robot: Optional[Robot] = None
        self.dof_editor: Optional[DoFEditor] = None
        self._cursor_pos = mn.Vector3(0.0, 0.0, 0.0)
        self.navmesh_lines: Optional[
            List[Tuple[mn.Vector3, mn.Vector3]]
        ] = None

        # setup the simulator
        self.set_scene(self._current_scene_index)

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
        self._cursor_pos = self.robot.ao.transformation.transform_point(
            self.robot.ao.aabb.center()
        )
        self._camera_helper.update(self._cursor_pos, 0.0)
        self.dof_editor = None
        self.navmesh_lines = None
        print(f"Loaded scene: {scene}.")

        
        client_message_manager = self._app_service._client_message_manager
        if client_message_manager is not None:
            client_message_manager.signal_scene_change()
            
            if True: # TODO: If in VR...
                self.recompute_navmesh()
                user_pos = AppStateRobotTeleopViewer._find_navmesh_position_near_target(
                    target=self._cursor_pos,
                    distance_from_target=1.2,
                    pathfinder=self._sim.pathfinder
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
        pts = self._sim.pathfinder.build_navmesh_vertices(
            largest_island_index
        )
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
        else:
            print("No robot configured.")

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
            self._cursor_pos[0] = self.robot.ao.translation[0]
            self._cursor_pos[2] = self.robot.ao.translation[2]
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
            candidate = mn.Vector3(target.x + dist_x, target.y, target.z + dist_z)
            if pathfinder.is_navigable(candidate, 3.0):
                dist_to_closest_obstacle = pathfinder.distance_to_closest_obstacle(
                    candidate, 2.0
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

        if gui_input.get_key(KeyCode.SPACE):
            self.cursor_follow_robot = not self.cursor_follow_robot
            # self.set_physics_paused(not self._do_pause_physics)

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
        if gui_input.get_key_down(KeyCode.P):
            if self.robot is not None:
                self.robot.cache_pose()
        if gui_input.get_key_down(KeyCode.O):
            if self.robot is not None:
                self.robot.set_cached_pose()

        # Load next scene
        if gui_input.get_key_down(KeyCode.ZERO):
            self.next_scene()

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
                self._app_service.gui_input._relative_mouse_position[0] * 0.01
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
        self._update_cursor_pos()
        self.move_robot_on_navmesh()

        # step the simulator
        self._sim.step_physics(dt)

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
                if hit_details.obj_name == self.robot.ao.handle:
                    # hit the robot, so try to draw the link
                    if hit_details.link_id is not None:
                        self.robot.draw_dof(
                            dblr,
                            hit_details.link_id,
                            self._cam_transform.translation,
                        )

        # NOTE: do debug drawing here
        # draw lookat ring
        self.draw_lookat()
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
