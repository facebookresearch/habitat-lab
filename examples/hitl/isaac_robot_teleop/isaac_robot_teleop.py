#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import TYPE_CHECKING, List

import hydra
import magnum as mn
import numpy as np

from habitat.isaac_sim import isaac_prim_utils
from habitat.isaac_sim.isaac_app_wrapper import IsaacAppWrapper
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
from scripts.ik import DifferentialInverseKinematics, to_ik_pose
from scripts.utils import LERP, debug_draw_axis
from scripts.xr_pose_adapter import XRPose, XRPoseAdapter

# path to this example app directory
dir_path = os.path.dirname(os.path.realpath(__file__))


if TYPE_CHECKING:
    from habitat.isaac_sim.isaac_rigid_object_manager import (
        IsaacRigidObjectWrapper,
    )
    from scripts.DoFeditor import DoFEditor

# unfortunately we can't import this earlier
# import habitat_sim  # isort:skip


def bind_physics_material_to_hierarchy(
    stage,
    root_prim,
    material_name,
    static_friction,
    dynamic_friction,
    restitution,
):
    """
    Recursively sets the friction properties of a prim and its child subtree to a uniform friction and restitution.
    """

    from omni.isaac.core.materials.physics_material import PhysicsMaterial
    from pxr import UsdShade

    # material_path = f"/PhysicsMaterials/{material_name}"
    # material_prim = stage.DefinePrim(material_path, "PhysicsMaterial")
    # material = UsdPhysics.MaterialAPI(material_prim)
    # material.CreateStaticFrictionAttr().Set(static_friction)
    # material.CreateDynamicFrictionAttr().Set(dynamic_friction)
    # material.CreateRestitutionAttr().Set(restitution)

    physics_material = PhysicsMaterial(
        prim_path=f"/PhysicsMaterials/{material_name}",
        name=material_name,
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=restitution,
    )

    binding_api = UsdShade.MaterialBindingAPI.Apply(root_prim)
    binding_api.Bind(
        physics_material.material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )


class AppStateIsaacSimViewer(AppState):
    """ """

    def __init__(self, app_service: AppService):
        self._app_service = app_service
        self._sim = app_service.sim

        self._app_cfg = omegaconf_to_object(
            app_service.config.isaac_robot_teleop
        )

        # todo: probably don't need video-recording stuff for this app
        self._video_output_prefix = "video"

        self._camera_helper = CameraHelper(
            self._app_service.hitl_config, self._app_service.gui_input
        )
        # not supported
        assert not self._app_service.hitl_config.camera.first_person_mode

        self.cursor_follow_robot = True
        self._cursor_pos: mn.Vector3 = mn.Vector3()
        self._xr_cursor_pos: mn.Vector3 = mn.Vector3()
        self._camera_helper.update(self._cursor_pos, 0.0)
        self.xr_pose_adapter = XRPoseAdapter()
        self.xr_origin_offset = mn.Vector3(0, 0, 0)
        self.xr_origin_rotation = mn.Quaternion()
        self.xr_origin_yaw_offset: float = -mn.math.pi / 2.0
        self.dof_editor: "DoFEditor" = None
        self._ik: DifferentialInverseKinematics = (
            DifferentialInverseKinematics()
        )

        # Either the HITL app is headless or Isaac is headless. They can't both spawn a window.
        do_isaac_headless = (
            not self._app_service.hitl_config.experimental.headless.do_headless
        )

        self._isaac_wrapper = IsaacAppWrapper(
            self._sim, headless=do_isaac_headless
        )
        isaac_world = self._isaac_wrapper.service.world
        self._usd_visualizer = self._isaac_wrapper.service.usd_visualizer

        self._isaac_physics_dt = 1.0 / 180
        # beware goofy behavior if physics_dt doesn't equal rendering_dt
        isaac_world.set_simulation_dt(
            physics_dt=self._isaac_physics_dt,
            rendering_dt=self._isaac_physics_dt,
        )

        # load the asset from config
        asset_path = os.path.join(dir_path, self._app_cfg.usd_scene_path)

        from omni.isaac.core.utils.stage import add_reference_to_stage

        add_reference_to_stage(
            usd_path=asset_path, prim_path="/World/test_scene"
        )
        self._usd_visualizer.on_add_reference_to_stage(
            usd_path=asset_path, prim_path="/World/test_scene"
        )

        self._rigid_objects: List["IsaacRigidObjectWrapper"] = []
        from habitat.isaac_sim.isaac_rigid_object_manager import (
            IsaacRigidObjectManager,
        )

        self._isaac_rom = IsaacRigidObjectManager(self._isaac_wrapper.service)
        # self.add_or_reset_rigid_objects()
        self._pick_target_rigid_object_idx = None

        stage = self._isaac_wrapper.service.world.stage
        prim = stage.GetPrimAtPath("/World")

        # TODO: improve estimated dynamic properties instead of max friction for everything
        bind_physics_material_to_hierarchy(
            stage=stage,
            root_prim=prim,
            material_name="my_material",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )

        isaac_world.reset()

        self._isaac_rom.post_reset()

        self.load_robot()

        self._hide_gui = False
        self._is_recording = False

        self._sps_tracker = AverageRateTracker(2.0)
        self._do_pause_physics = False
        self._timer = 0.0

        self.init_mouse_raycaster()

        # setup the configured navmesh
        self._sim.pathfinder.load_nav_mesh(self._app_cfg.navmesh_path)
        assert self._sim.pathfinder.is_loaded
        # place the robot on the navmesh
        self.robot.set_root_pose(
            pos=self._sim.pathfinder.get_random_navigable_point()
        )
        self._app_service.users.activate_user(0)

    def add_rigid_object(
        self, handle: str, bottom_pos: mn.Vector3 = None
    ) -> None:
        """
        Adds the specified rigid object to the scene and records it in self._rigid_objects.
        If specified, aligns the bottom most point of the object with bottom_pos.
        NOTE: intended to be used with raycasting to place objects on horizontal surfaces.
        """
        ro = self._isaac_rom.add_object_by_template_handle(handle)
        self._rigid_objects.append(ro)
        ro.rotation = mn.Quaternion.rotation(-mn.Deg(90), mn.Vector3.x_axis())

        # set a translation if specified offset such that the bottom most point of the object is coincident with bottom_pos
        if bottom_pos is not None:
            bounds = ro.get_aabb()
            obj_height = ro.translation[1] - bounds.bottom
            print(f"bounds = {bounds}")
            print(f"obj_height = {obj_height}")

            ro.translation = bottom_pos + mn.Vector3(0, obj_height, 0)

    def load_robot(self) -> None:
        """
        Loads a robot from USD file into the scene for testing.
        NOTE: resets the world when called, not safe for use during simulation
        """
        from scripts.robot import RobotAppWrapper

        robot_cfg = hydra.compose(config_name="robot_settings")
        self.robot = RobotAppWrapper(
            self._isaac_wrapper.service, self._sim, robot_cfg
        )
        # TODO: figure out how to initialize the robot in isolation instead of resetting the world. Doesn't work as expected.
        self._isaac_wrapper.service.world.reset()
        self.robot.post_init()
        self.hand_grasp_poses = [
            self.robot.pos_subsets["left_hand"].fetch_cached_pose(
                pose_name=hand_subset_key
            )
            for hand_subset_key in ["hand_open", "hand_closed"]
        ]

    def draw_lookat(self):
        if self._hide_gui:
            return

        lookat_ring_radius = 0.01
        lookat_ring_color = mn.Color3(1, 0.75, 0)
        self._app_service.gui_drawer.draw_circle(
            self._cursor_pos,
            lookat_ring_radius,
            lookat_ring_color,
        )

    def _get_controls_text(self):
        controls_str: str = ""
        controls_str += "ESC: exit\n"
        controls_str += "R + mousemove: rotate camera\n"
        controls_str += "mousewheel: cam zoom\n"
        controls_str += "WASD: move cursor\n"
        controls_str += "H: toggle GUI\n"
        controls_str += "P: pause physics\n"
        controls_str += "J: reset rigid objects\n"
        controls_str += "K: start recording\n"
        controls_str += "L: stop recording\n"
        if self._sps_tracker.get_smoothed_rate() is not None:
            controls_str += (
                f"server SPS: {self._sps_tracker.get_smoothed_rate():.1f}\n"
            )

        return controls_str

    def _get_status_text(self):
        status_str = ""
        cursor_pos = self._cursor_pos
        status_str += (
            f"({cursor_pos.x:.1f}, {cursor_pos.y:.1f}, {cursor_pos.z:.1f})\n"
        )
        if self._recent_mouse_ray_hit_info:
            status_str += self._recent_mouse_ray_hit_info["rigidBody"] + "\n"
        status_str += f"base_rot: {self.robot.base_rot}\n"
        return status_str

    def _update_help_text(self):
        if self._hide_gui:
            return

        controls_str = self._get_controls_text()
        if len(controls_str) > 0:
            self._app_service.text_drawer.add_text(
                controls_str, TextOnScreenAlignment.TOP_LEFT
            )

        status_str = self._get_status_text()
        if len(status_str) > 0:
            self._app_service.text_drawer.add_text(
                status_str,
                TextOnScreenAlignment.TOP_CENTER,
                text_delta_x=-120,
            )

    def _update_cursor_pos(self):
        """
        Consumes keyboard input to update the cursor position.
        keys: 'zxwasd'
        """
        gui_input = self._app_service.gui_input
        y_speed = 0.02
        if gui_input.get_key_down(KeyCode.Z):
            self._cursor_pos.y -= y_speed
        if gui_input.get_key_down(KeyCode.X):
            self._cursor_pos.y += y_speed

        xz_forward = self._camera_helper.get_xz_forward()
        xz_right = mn.Vector3(-xz_forward.z, 0.0, xz_forward.x)
        speed = (
            self._app_cfg.camera_move_speed * self._camera_helper.cam_zoom_dist
        )
        if gui_input.get_key(KeyCode.W):
            self._cursor_pos += xz_forward * speed
        if gui_input.get_key(KeyCode.S):
            self._cursor_pos -= xz_forward * speed
        if gui_input.get_key(KeyCode.D):
            self._cursor_pos += xz_right * speed
        if gui_input.get_key(KeyCode.A):
            self._cursor_pos -= xz_right * speed

    def sync_xr_local_state(self, xr_pose: XRPose = None):
        """
        Sets a local offset variable to transform XR headset and controller origin into the local robot space.
        Should be run once every time the quest user reaches a comfortable pose and wishes to re-align with the robot.
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

        # align the XR headset orientation with the x axis + user offset
        align_z_to_x = mn.Quaternion.rotation(
            mn.Rad(self.xr_origin_yaw_offset), mn.Vector3(0, 1, 0)
        )
        self.xr_origin_rotation = (
            align_z_to_x  # xr_pose.rot_head.inverted() * align_z_to_x
        )

        # get the local offset of the head from the origin for user correction
        origin_transform = mn.Matrix4.from_(
            xr_pose.rot_origin.to_matrix(), xr_pose.pos_origin
        )
        global_offset = xr_pose.pos_origin - xr_pose.pos_head
        self.xr_origin_offset = origin_transform.inverted().transform_vector(
            global_offset
        )
        self._sync_xr_user_to_robot_cursor()

    def _sync_xr_user_to_robot_cursor(self):
        """
        Updates the XR user 0's origin transform from the robot and cursor states.
        Call when robot base is moved or XR offsets are adjusted by user.
        """

        xr_pose = XRPose(
            remote_client_state=self._app_service.remote_client_state
        )
        client_message_manager = self._app_service.client_message_manager
        if (
            client_message_manager is not None
            and self.robot is not None
            and xr_pose.valid
        ):
            origin_transform = mn.Matrix4.from_(
                xr_pose.rot_origin.to_matrix(), xr_pose.pos_origin
            )
            # set the origin such that the headset is aligned with the cursor
            # NOTE: temporarily setting the mutable values of immutable xr_input.origin_position by reference until bugfix in unity app with pushing the state from client
            origin_position = list(
                self._xr_cursor_pos
                + origin_transform.transform_vector(self.xr_origin_offset)
            )
            for i in range(3):
                self._app_service.remote_client_state.get_xr_input(
                    0
                ).origin_position[i] = origin_position[i]

            _, robot_rot = self.robot.get_root_pose()
            origin_orientation = (
                robot_rot
                * mn.Quaternion.rotation(
                    mn.Rad(mn.math.pi / 2.0), mn.Vector3(1, 0, 0)
                )
                * self.xr_origin_rotation
            )

            self._app_service.remote_client_state.get_xr_input(
                0
            ).origin_rotation[0] = origin_orientation.scalar
            for i in range(3):
                self._app_service.remote_client_state.get_xr_input(
                    0
                ).origin_rotation[i + 1] = origin_orientation.vector[i]
            client_message_manager.set_xr_origin_transform(
                pos=origin_position,
                rot=self._app_service.remote_client_state.get_xr_input(
                    0
                ).origin_rotation,
            )

    def debug_draw_quest(self) -> None:
        """
        Debug draw the quest controller and headset poses as axis frames.
        """
        dblr = self._app_service.gui_drawer

        current_xr_pose = XRPose(
            remote_client_state=self._app_service.remote_client_state
        )
        if current_xr_pose.valid:
            self.xr_pose_adapter.get_global_xr_pose(current_xr_pose).draw_pose(
                dblr
            )

    def handle_xr_input(self, dt: float):
        if self._app_service.remote_client_state is None:
            return

        xr_input = self._app_service.remote_client_state.get_xr_input(0)
        left = xr_input.left_controller
        right = xr_input.right_controller
        from habitat_hitl.core.xr_input import XRButton

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
                    set_motor_targets=True,
                    set_positions=False,
                )

        if left.get_button_up(XRButton.TWO):
            pass

        if left.get_button_down(XRButton.START):
            print("pressed START left")
            # NOTE: reserved by QuestReader for now...
        if left.get_button_up(XRButton.START):
            pass

        # RIGHT CONTROLLER BUTTONS
        # TODO: trajectory recording?
        # if right.get_button_down(XRButton.ONE):
        #     print("pressed one right")
        #     print("Starting to record a new trajectory")
        #     self.xr_traj = XRTrajectory()
        #     self.recording_xr_traj = True
        #     self.replay_xr_traj = False
        # if right.get_button_up(XRButton.ONE):
        #     print("released one right")
        #     self.recording_xr_traj = False
        #     self.xr_traj.save_json()
        #     print(
        #         f"Saved the trajectory with {len(self.xr_traj.traj)} poses to 'xr_pose.json'."
        #     )
        # if right.get_button_down(XRButton.TWO):
        #     print("pressed two right")
        #     self.replay_xr_traj = not self.replay_xr_traj
        #     self.recording_xr_traj = False
        #     print(f"XR Traj playback = {self.replay_xr_traj}")
        if right.get_button_up(XRButton.TWO):
            pass
        # NOTE: XRButton.START is reserved for Quest menu functionality

        # NOTE: PRIMARY_HAND_TRIGGER mapped to IK toggle
        # NOTE: PRIMARY_INDEX_TRIGGER mapped to grasping functionality

        # TODO: PRIMARY_THUMBSTICK (click?) unmapped currently and available

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
                    (
                        arm_base_positions,
                        arm_base_rotations,
                    ) = self.robot.get_link_world_poses(
                        indices=[arm_base_link]
                    )
                    arm_base_transform = mn.Matrix4.from_(
                        arm_base_rotations[0].to_matrix(),
                        arm_base_positions[0],
                    )
                    xr_pose_in_robot_frame = (
                        self.xr_pose_adapter.xr_pose_transformed(
                            global_xr_pose,
                            arm_base_transform.inverted(),
                        )
                    )
                    arm_joint_subset = self.robot.pos_subsets[arm_subset_key]

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
                    debug_draw_axis(
                        self._app_service.gui_drawer,
                        arm_base_transform.__matmul__(self._ik.get_ee_T()),
                        scale=0.75,
                    )
                    # using configuration subset to avoid accidental overwriting with noise
                    self.robot.pos_subsets[arm_subset_key].set_motor_pos(
                        new_arm_pose
                    )

            # grasping logic
            for xrcontroller, hand_subset_key in [
                (right, "right_hand"),
                (left, "left_hand"),
            ]:
                cur_pose = LERP(
                    self.hand_grasp_poses[0],
                    self.hand_grasp_poses[1],
                    xrcontroller.get_index_trigger(),
                )

                self.robot.pos_subsets[hand_subset_key].set_motor_pos(cur_pose)

    def update_robot_base_control(self, dt: float):
        """
        Consume keyboard input to configure the robot's base controller.
        keys: 'ijkl'
        """

        gui_input = self._app_service.gui_input
        lin_speed = self.robot.robot_cfg.max_linear_speed
        ang_speed = self.robot.robot_cfg.max_angular_speed

        ##############################################
        # Waypoint base control via mouse right click
        if (
            gui_input.get_mouse_button_down(MouseButton.RIGHT)
            and self._recent_mouse_ray_hit_info is not None
        ):
            self.robot.base_vel_controller.track_waypoints = True
            self.robot.base_vel_controller._pause_track_waypoints = True
            hab_hit_pos = mn.Vector3(
                *isaac_prim_utils.usd_to_habitat_position(
                    self._recent_mouse_ray_hit_info["position"]
                )
            )
            if (
                self._sim.pathfinder.is_loaded
                and self._sim.pathfinder.is_navigable(hab_hit_pos)
            ):
                self.robot.base_vel_controller.target_position = (
                    self._sim.pathfinder.snap_point(hab_hit_pos)
                )
            else:
                print("Cannot set waypoint to non-navigable target.")

        if (
            gui_input.get_mouse_button(MouseButton.RIGHT)
            and self._recent_mouse_ray_hit_info is not None
        ):
            hab_hit_pos = mn.Vector3(
                *isaac_prim_utils.usd_to_habitat_position(
                    self._recent_mouse_ray_hit_info["position"]
                )
            )
            global_dir = mn.Vector3(1.0, 0, 0)
            frame_to_mouse = (
                hab_hit_pos - self.robot.base_vel_controller.target_position
            )
            norm_dir = mn.Vector3(
                [frame_to_mouse[0], 0, frame_to_mouse[2]]
            ).normalized()
            if not np.isnan(norm_dir).any():
                angle_to_target = self.robot.angle_to(
                    dir_target=frame_to_mouse, dir_init=global_dir
                )
                self.robot.base_vel_controller.target_rotation = (
                    angle_to_target
                )

        # start control on button release
        if (
            gui_input.get_mouse_button_up(MouseButton.RIGHT)
            and self._sim.pathfinder.is_loaded
            and self._sim.pathfinder.is_navigable(
                self.robot.base_vel_controller.target_position
            )
        ):
            self.robot.base_vel_controller._pause_track_waypoints = False

        # end waypoint control with mouse right click
        #####################################################################

        #####################################################################
        # Base velocity control with keyboard 'IJKL'
        # NOTE: interrupts waypoint control

        if not self.robot.base_vel_controller.track_waypoints:
            self.robot.base_vel_controller.reset()

        if gui_input.get_key(KeyCode.I):
            self.robot.base_vel_controller.track_waypoints = False
            self.robot.base_vel_controller.target_linear_vel = lin_speed
        if gui_input.get_key(KeyCode.K):
            self.robot.base_vel_controller.track_waypoints = False
            self.robot.base_vel_controller.target_linear_vel = -lin_speed
        if gui_input.get_key(KeyCode.J):
            # self.robot.base_rot += 0.1
            self.robot.base_vel_controller.track_waypoints = False
            self.robot.base_vel_controller.target_angular_vel = ang_speed
        if gui_input.get_key(KeyCode.L):
            self.robot.base_vel_controller.track_waypoints = False
            self.robot.base_vel_controller.target_angular_vel = -ang_speed
            # self.robot.base_rot -= 0.1

        # end base velocity control with keyboard
        #####################################################################

        #####################################################################
        # XR joystick base velocity and camera control
        # NOTE: interrupts waypoint control

        if self._app_service.remote_client_state is not None:
            xr_input = self._app_service.remote_client_state.get_xr_input(0)
            if xr_input is not None:
                left = xr_input.controllers[HAND_LEFT]
                right = xr_input.controllers[HAND_RIGHT]

                # use left thumbstick to move the robot
                left_thumbstick = left.get_thumbstick()
                if left_thumbstick[1] != 0:
                    self.robot.base_vel_controller.target_linear_vel = (
                        lin_speed * left_thumbstick[1]
                    )
                    self.robot.base_vel_controller.track_waypoints = False
                if left_thumbstick[0] != 0:
                    self.robot.base_vel_controller.target_angular_vel = (
                        -ang_speed * left_thumbstick[0]
                    )
                    self.robot.base_vel_controller.track_waypoints = False

                # use right thumbstick up/down to raise/lower the head point
                # use right thumbstick left/right to rotate the alignment
                right_thumbstick = right.get_thumbstick()
                yaw_scale = -0.06
                y_scale = 0.02
                self.xr_origin_yaw_offset += right_thumbstick[0] * yaw_scale
                self.xr_origin_rotation = mn.Quaternion.rotation(
                    mn.Rad(self.xr_origin_yaw_offset), mn.Vector3(0, 1, 0)
                )
                if abs(right_thumbstick[1]) > 0.1:
                    # vertical is z in isaac
                    self.robot.viewpoint_offset[2] += (
                        right_thumbstick[1] * y_scale
                    )

        # end base velocity control with XR joysticks
        #####################################################################

        self._xr_cursor_pos = self.robot.get_global_view_offset()
        if self.cursor_follow_robot:
            self._cursor_pos = self.robot.get_global_view_offset()

    def update_isaac(self, post_sim_update_dict):
        if self._isaac_wrapper:
            sim_app = self._isaac_wrapper.service.simulation_app
            if not sim_app.is_running():
                post_sim_update_dict["application_exit"] = True
            else:
                approx_app_fps = 30
                num_steps = int(
                    1.0 / (approx_app_fps * self._isaac_physics_dt)
                )
                self._isaac_wrapper.step(num_steps=num_steps)
                self._isaac_wrapper.pre_render()

    def set_physics_paused(self, do_pause_physics):
        self._do_pause_physics = do_pause_physics
        world = self._isaac_wrapper.service.world
        if do_pause_physics:
            world.pause()
        else:
            world.play()

    def handle_keys(self, dt, post_sim_update_dict):
        """
        Handle key presses which are not used for camera updates.
        NOTE: wasdzxr reserved for camera UI
        NOTE: ijkl reserved for robot control
        keys: 'ESC SPACE phon'
        """
        gui_input = self._app_service.gui_input
        if gui_input.get_key_down(KeyCode.ESC):
            post_sim_update_dict["application_exit"] = True

        if gui_input.get_key(KeyCode.SPACE):
            self._sim.step_physics(dt=1.0 / 60)

        if gui_input.get_key_down(KeyCode.P):
            self.set_physics_paused(not self._do_pause_physics)

        if gui_input.get_key_down(KeyCode.H):
            self._hide_gui = not self._hide_gui

        if (
            gui_input.get_key_down(KeyCode.O)
            and self._recent_mouse_ray_hit_info is not None
        ):
            # place an object at the mouse raycast endpoint
            hab_hit_pos = mn.Vector3(
                *isaac_prim_utils.usd_to_habitat_position(
                    self._recent_mouse_ray_hit_info["position"]
                )
            )
            self.add_rigid_object(
                handle="data/objects/ycb/configs/024_bowl.object_config.json",
                bottom_pos=hab_hit_pos,
            )

        if gui_input.get_key_down(KeyCode.N):
            self.robot.set_root_pose(
                pos=self._sim.pathfinder.get_random_navigable_point()
            )

        # cache a pose
        if gui_input.get_key_down(KeyCode.T):
            subset_input = input(
                f"Caching a pose subset.\n Enter the subset's name from {self.robot.pos_subsets.keys()}.\n Enter nothing to abort.\n >"
            )
            if subset_input != "":
                if subset_input not in self.robot.pos_subsets:
                    print("The desired subset does not exist, try again.")
                else:
                    pose_key_input = input(
                        "Now enter the desired pose key or nothing to abort:\n>"
                    )
                    if pose_key_input != "":
                        self.robot.pos_subsets[subset_input].cache_pose(
                            pose_name=pose_key_input
                        )
                        from scripts.robot import default_pose_cache_path

                        print(
                            f"Cached current pose of subset '{subset_input}' as entry '{pose_key_input}' in {default_pose_cache_path}"
                        )

        # load a cached pose
        if gui_input.get_key_down(KeyCode.G):
            subset_input = input(
                f"Loading a cached pose subset.\n Enter the subset's name from {self.robot.pos_subsets.keys()}.\n Enter nothing to abort.\n >"
            )
            if subset_input != "":
                if subset_input not in self.robot.pos_subsets:
                    print("The desired subset does not exist, try again.")
                else:
                    pose_key_input = input(
                        "Now enter the desired pose key or nothing to abort:\n>"
                    )
                    if pose_key_input != "":
                        self.robot.pos_subsets[subset_input].set_cached_pose(
                            pose_name=pose_key_input, set_motor_targets=True
                        )

    def handle_mouse_press(self) -> None:
        """
        TODO: add mouse controls
        NOTE: RIGHT click reserved for robot control
        """

        if (
            self._app_service.gui_input.get_mouse_button_down(MouseButton.LEFT)
            and self._recent_mouse_ray_hit_info is not None
        ):
            body_name = self._recent_mouse_ray_hit_info["rigidBody"]
            print(body_name)
            # if the robot joint is clicked try to create a DofEditor
            joint_for_rigid = self.robot.get_joint_for_rigid_prim(body_name)
            if joint_for_rigid is not None:
                from scripts.DoFeditor import DoFEditor

                self.dof_editor = DoFEditor(self.robot, joint_for_rigid)

        # mouse release
        if self._app_service.gui_input.get_mouse_button_up(MouseButton.LEFT):
            # release LEFT mouse destroys any active dof editor
            self.dof_editor = None

        # mouse is down (i.e. dragging/holding) but not the first click
        if (
            self._app_service.gui_input.get_mouse_button(MouseButton.LEFT)
            and self.dof_editor is not None
        ):
            self.dof_editor.update(
                self._app_service.gui_input._relative_mouse_position[0] * 0.01,
                set_positions=False,
            )

    def init_mouse_raycaster(self):
        self._recent_mouse_ray_hit_info = None

    def update_mouse_raycaster(self, dt):
        self._recent_mouse_ray_hit_info = None

        mouse_ray = self._app_service.gui_input.mouse_ray

        if not mouse_ray:
            return

        origin_usd = isaac_prim_utils.habitat_to_usd_position(mouse_ray.origin)
        dir_usd = isaac_prim_utils.habitat_to_usd_position(mouse_ray.direction)

        from omni.physx import get_physx_scene_query_interface

        hit_info = get_physx_scene_query_interface().raycast_closest(
            isaac_prim_utils.to_gf_vec3(origin_usd),
            isaac_prim_utils.to_gf_vec3(dir_usd),
            1000.0,
        )

        if not hit_info["hit"]:
            return

        # dist = hit_info['distance']
        hit_pos_usd = hit_info["position"]
        hit_normal_usd = hit_info["normal"]
        hit_pos_habitat = mn.Vector3(
            *isaac_prim_utils.usd_to_habitat_position(hit_pos_usd)
        )
        hit_normal_habitat = mn.Vector3(
            *isaac_prim_utils.usd_to_habitat_position(hit_normal_usd)
        )
        # collision_name = hit_info['collision']
        body_name = hit_info["rigidBody"]
        body_ix = self.robot.get_rigid_prim_ix(body_name)
        if body_ix is not None:
            self.robot.draw_dof(
                self._app_service.gui_drawer,
                body_ix,
                self._cam_transform.translation,
            )

        hit_radius = 0.05
        self._app_service.gui_drawer.draw_circle(
            hit_pos_habitat,
            hit_radius,
            mn.Color3(255, 0, 255),
            16,
            hit_normal_habitat,
        )

        self._recent_mouse_ray_hit_info = hit_info

        gui_input = self._app_service.gui_input
        if gui_input.get_key_down(KeyCode.Y):
            force_mag = 200.0
            import carb

            # instead of hit_normal_usd, consider dir_usd
            force_vec = carb.Float3(
                hit_normal_usd[0] * force_mag,
                hit_normal_usd[1] * force_mag,
                hit_normal_usd[2] * force_mag,
            )
            from omni.physx import get_physx_interface

            get_physx_interface().apply_force_at_pos(
                body_name, force_vec, hit_pos_usd
            )

    def sim_update(self, dt, post_sim_update_dict):
        self._sps_tracker.increment()

        self.handle_keys(dt, post_sim_update_dict)
        self.update_mouse_raycaster(dt)
        self.handle_mouse_press()
        self.handle_xr_input(dt)
        self._update_cursor_pos()
        self.update_robot_base_control(dt)
        self.update_isaac(post_sim_update_dict)
        self._sync_xr_user_to_robot_cursor()

        self._camera_helper.update(self._cursor_pos, dt)
        self._cam_transform = self._camera_helper.get_cam_transform()
        post_sim_update_dict["cam_transform"] = self._cam_transform

        # draw lookat ring
        self.draw_lookat()
        self.debug_draw_quest()
        # draw the robot frame
        self.robot.draw_debug(self._app_service.gui_drawer)
        if self.dof_editor is not None:
            self.dof_editor.debug_draw(
                self._app_service.gui_drawer, self._cam_transform.translation
            )

        self._update_help_text()


@hydra.main(
    version_base=None, config_path="./", config_name="isaac_robot_teleop"
)
def main(config):
    hitl_main(config, lambda app_service: AppStateIsaacSimViewer(app_service))


if __name__ == "__main__":
    register_hydra_plugins()
    main()
