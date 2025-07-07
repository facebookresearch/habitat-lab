#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import sys
import time
from collections import defaultdict
from typing import TYPE_CHECKING, List, Optional

import torch  # noqa: F401

# make sure we restore these flags after import (habitat_sim needs RTLD_GLOBAL but that breaks Isaac)
# hack: must import torch early, before habitat or isaac
from habitat_hitl.core.ui_elements import HorizontalAlignment
from habitat_hitl.core.user_mask import Mask

original_flags = sys.getdlopenflags()
import magnum

import habitat_sim  # noqa: F401

sys.setdlopenflags(original_flags)

import hydra
import magnum as mn
import numpy as np
from app_data import AppData
from app_state_base import AppStateBase
from app_states import (
    create_app_state_cancel_session,
    create_app_state_load_episode,
)
from session import Session

from habitat.datasets.rearrange.rearrange_dataset import (
    RearrangeDatasetV0,
    RearrangeEpisode,
)
from habitat.isaac_sim import isaac_prim_utils
from habitat.isaac_sim.isaac_app_wrapper import IsaacAppWrapper
from habitat_hitl._internal.networking.average_rate_tracker import (
    AverageRateTracker,
)
from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.core.hitl_main import hitl_main
from habitat_hitl.core.hydra_utils import (
    omegaconf_to_object,
    register_hydra_plugins,
)
from habitat_hitl.core.key_mapping import KeyCode, MouseButton
from habitat_hitl.core.text_drawer import TextOnScreenAlignment
from habitat_hitl.core.xr_input import HAND_LEFT, HAND_RIGHT
from habitat_hitl.environment.camera_helper import CameraHelper
from scripts.frame_recorder import FrameEvent, FrameRecorder
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

    from pxr import UsdPhysics, UsdShade

    # Isaac 4.5.0 mostly still supports our outdated 4.2.0-style imports but this one fails
    try:
        from omni.isaac.core.materials.physics_material import PhysicsMaterial
    except ImportError:
        from isaacsim.core.api.materials.physics_material import (
            PhysicsMaterial,
        )

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

    material = UsdPhysics.MaterialAPI(root_prim)

    # Set friction
    material.CreateStaticFrictionAttr().Set(static_friction)
    material.CreateDynamicFrictionAttr().Set(dynamic_friction)


def remove_stage_mesh_contact_sensors(root_prim):
    """
    Disables the contact sensor API on all rigids under the stage mesh root.
    NOTE: we do this after each contact check to recover perf for manipulation.
    """
    from pxr import PhysxSchema, Usd

    prim_range = Usd.PrimRange(root_prim)
    it = iter(prim_range)
    for prim in it:
        if prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
            prim.RemoveAPI(PhysxSchema.PhysxContactReportAPI)


def create_stage_mesh_contact_sensors(root_prim):
    """
    We need to manually apply the PhysxContactReportAPI to simplified stage meshes to detect contacts with them.
    It also helps to increase the ContactOffsetAttr which controls the distance at which collision detection occurs.
    """
    from pxr import PhysxSchema, Usd

    prim_range = Usd.PrimRange(root_prim)
    it = iter(prim_range)
    for prim in it:
        if prim.HasAPI(
            PhysxSchema.PhysxTriangleMeshSimplificationCollisionAPI
        ):
            # NOTE: useless except to identify stage meshes
            # mesh_simp_api = PhysxSchema.PhysxTriangleMeshSimplificationCollisionAPI(prim)

            collision_api = PhysxSchema.PhysxCollisionAPI(prim)
            # NOTE: tune this for application
            collision_api.CreateContactOffsetAttr(0.01)
            # collision_api.CreateRestOffsetAttr(0.5)

            PhysxSchema.PhysxContactReportAPI.Apply(prim)
            # TODO: could tune this?
            # _contactReportAPI.CreateThresholdAttr().Set(0)


class AppStateIsaacSimViewer(AppStateBase):
    """ """

    def __init__(
        self,
        app_service: AppService,
        app_data: Optional[AppData] = None,
        session: Optional[Session] = None,
    ):
        super().__init__(app_service, app_data)
        self._sim = app_service.sim
        self._session = session

        self._app_cfg = omegaconf_to_object(
            app_service.config.isaac_robot_teleop
        )

        # NOTE: replay mode requires specific configuration. See configs/replay_episode_record.yaml
        self.in_replay_mode = (
            hasattr(self._app_cfg, "replay_mode")
            and self._app_cfg.replay_mode == True
        )

        self._camera_helper = CameraHelper(
            self._app_service.hitl_config, self._app_service.gui_input
        )
        # not supported
        assert not self._app_service.hitl_config.camera.first_person_mode

        self.cursor_follow_robot = self._app_cfg.lock_pc_camera_to_robot
        self._draw_debug_shapes = self._app_cfg.draw_debug_shapes
        self._cursor_pos: mn.Vector3 = mn.Vector3()
        self._xr_cursor_pos: mn.Vector3 = mn.Vector3()
        self._camera_helper.update(self._cursor_pos, 0.0)
        self._first_xr_update = True
        self.xr_pose_adapter = XRPoseAdapter()
        self.xr_origin_offset = mn.Vector3(0, 0, 0)
        self.xr_origin_rotation = mn.Quaternion()
        self.xr_origin_yaw_offset: float = -mn.math.pi / 2.0
        self.dof_editor: "DoFEditor" = None
        self._ik: DifferentialInverseKinematics = (
            DifferentialInverseKinematics()
        )
        if app_data is None:
            self._isaac_wrapper = IsaacAppWrapper(
                self._sim,
                headless=True,
            )
        else:
            self._isaac_wrapper = app_data.isaac_wrapper
        isaac_world = self._isaac_wrapper.service.world
        self._usd_visualizer = self._isaac_wrapper.service.usd_visualizer

        self._isaac_physics_dt = 1.0 / 180
        # beware goofy behavior if physics_dt doesn't equal rendering_dt
        isaac_world.set_simulation_dt(
            physics_dt=self._isaac_physics_dt,
            rendering_dt=self._isaac_physics_dt,
        )

        self._frame_recorder = FrameRecorder(self)
        self._frame_events: List[FrameEvent] = []
        self.reverse_replay = False
        self.pause_replay = False

        # This variable is triggered by the user to indicate finished task
        self._task_finished_signaled = False
        # This value indicates that the task is really finished.
        self._task_finished = False
        # This value is set by the user after signaling that the task is finished.
        self._task_success = 0.0
        # This value is set on enter and when True, the language prompt is displayed on a splash screen.
        self._view_task_prompt = True
        # NOTE: this will be filled from the episode or contrived to fit the scenario
        self.task_prompt = ""

        usd_scenes_path = os.path.join(dir_path, self._app_cfg.usd_scene_path)
        navmeshes_path = self._app_cfg.navmesh_path

        self.episode_dataset: RearrangeDatasetV0 = None
        self.episode_index = 0
        self.episode: RearrangeEpisode = None
        scene_usd_file: str = None
        scene_navmesh_file: str = None
        scene_name = ""

        setup_from_episode = False

        if self.in_replay_mode:
            from examples.hitl.isaac_robot_teleop.record_post_process import (
                get_good_first_ep_frame,
                load_json_gz,
            )

            # Setup from the episode record
            self._frame_recorder.load_episode_record_json_gz(
                self._app_cfg.episode_record_filepath
            )
            self._frame_recorder.replaying = True
            good_start_frame = 0
            good_start_frame = get_good_first_ep_frame(
                self._frame_recorder.frame_data
            )
            self._frame_recorder.replay_frame = good_start_frame
            self._frame_recorder._start_frame = good_start_frame
            self.episode_metadata = load_json_gz(
                self._app_cfg.episode_record_filepath
            )["episode"]
            self.episode_index = self.episode_metadata["episode_index"]
            setup_from_episode = True

        elif self._app_data and self._session:
            setup_from_episode = True
            assert hasattr(self._app_cfg, "episode_dataset")

            # NOTE: recording is always active for session driven interactions
            self._frame_recorder.recording = True
            self.episode_index = self._session.current_episode_index

        else:
            # load from local yaml file
            if hasattr(self._app_cfg, "episode_dataset"):
                ######################
                ## YES EPISODES LOGIC (load the episode)
                # NOTE: initializes self.episode_dataset
                setup_from_episode = True
                if hasattr(self._app_cfg, "episode_index"):
                    self.episode_index = self._app_cfg.episode_index

            else:
                ######################
                ## NO EPISODES LOGIC (defaults)
                # load the asset from config
                scene_usd_file = (
                    usd_scenes_path + self._app_cfg.scene_name + ".usda"
                )
                scene_navmesh_file = (
                    navmeshes_path + self._app_cfg.scene_name + ".navmesh"
                )

        if setup_from_episode:
            self.load_episode_dataset(self._app_cfg.episode_dataset)

            self.episode = (
                self.episode_dataset.episodes[  # type:ignore[attr-defined]
                    self.episode_index
                ]
            )

            scene_name = self.episode.scene_id.split("/")[-1].split(".")[0]
            scene_usd_file = usd_scenes_path + scene_name + ".usda"
            scene_navmesh_file = navmeshes_path + scene_name + ".navmesh"

        ######################
        ## instantiate isaac world
        from omni.isaac.core.utils.stage import add_reference_to_stage

        add_reference_to_stage(
            usd_path=scene_usd_file, prim_path="/World/test_scene"
        )
        self._usd_visualizer.on_add_reference_to_stage(
            usd_path=scene_usd_file, prim_path="/World/test_scene"
        )

        self._rigid_objects: List["IsaacRigidObjectWrapper"] = []
        from habitat.isaac_sim.isaac_rigid_object_manager import (
            IsaacRigidObjectManager,
        )

        self._isaac_rom = IsaacRigidObjectManager(self._isaac_wrapper.service)
        # self.add_or_reset_rigid_objects()
        self._pick_target_rigid_object_idx = None

        stage = self._isaac_wrapper.service.world.stage
        self.root_prim = stage.GetPrimAtPath("/World")

        # physics properties for the stage and furniture
        bind_physics_material_to_hierarchy(
            stage=stage,
            root_prim=self.root_prim,
            material_name="stage_physics_properties_material",
            static_friction=self._app_cfg.stage_static_friction,
            dynamic_friction=self._app_cfg.stage_dynamic_friction,
            restitution=self._app_cfg.stage_restitution,
        )
        # setup the stage contact sensors
        create_stage_mesh_contact_sensors(root_prim=self.root_prim)

        isaac_world.reset()

        self._isaac_rom.post_reset()

        self.load_robot()

        # load episode contents
        if self.episode is not None:
            self.load_episode_contents()
        self.load_semantic_scene(
            "data/hssd-hab/semantics/scenes/"
            + scene_name
            + ".semantic_config.json"
        )

        self._hide_gui = False
        self._is_recording = False

        self._sps_tracker = AverageRateTracker(2.0)
        # NOTE: we pause physics for replay to avoid jittering and extra work between frame applications.
        self._do_pause_physics = self.in_replay_mode

        self.init_mouse_raycaster()

        # setup the configured navmesh
        self._sim.pathfinder.load_nav_mesh(scene_navmesh_file)
        assert self._sim.pathfinder.is_loaded

        # load or sample initial robot state
        self.set_robot_base_initial_state()

        self._app_service.users.activate_user(0)

        # from scripts.robot import unit_test_robot
        # unit_test_robot(self.robot)

        # this is when the application starts after initialization
        self._start_time = time.time()
        # this tracks the relative time from start_time
        self._timer = 0.0

    def load_semantic_scene(self, scene_descriptor: str) -> None:
        """
        Side-loads the semantic metadata from a JSON file to import region logic.
        """
        self._sim.load_semantic_scene_descriptor(scene_descriptor)

    def generate_contrived_prompt(self):
        """
        This function should create some reasonable task prompts for the episode.
        NOTE: This is a placeholder while episodes are without explicit language instructions.
        NOTE: This could have a semantic element where individual objects are queried and related at runtime to maximize episode re-use.
        """
        return random.choice(
            [
                "Pick any object and place it upright on another surface.",
                "Stack any two objects.",
                "Pick any object and place it back upside down.",
                "Pick any object with one hand \nthen transfer it to the other hand \nwithout setting it down.",
                "Pick any object twice with different grasps, \nplacing it back on a surface between picks.",
            ]
        )

    def load_episode_dataset(self, dataset_file: str):
        """
        Load a RearrangeDataset and setup local variables.
        """
        import gzip

        with gzip.open(dataset_file, "rt") as f:
            self.episode_dataset = RearrangeDatasetV0()
            self.episode_dataset.from_json(f.read())

    def load_episode_contents(self):
        """
        Loads the configured RearrangeEpisode's contents in the scene.
        Assumes the episode is already configured.
        """
        # NOTE: only loads rigid objects currently
        # TODO: handle initial ao states
        # TODO: handle targets
        # TODO: handle any additional metadata

        # NOTE: assume added objects come from configured "additional_obj_config_paths"
        self.episode_object_ids = []
        # cache any skipped object indices for more skipping later
        self.culled_object_ixs = []
        for ix, (obj_config_name, transform) in enumerate(
            self.episode.rigid_objs
        ):
            added_object = False
            try:
                ro_t = mn.Matrix4(
                    [[transform[j][i] for j in range(4)] for i in range(4)]
                )
                ro_t = mn.Matrix4.from_(
                    ro_t.rotation_normalized(), ro_t.translation
                )
            except ValueError as e:
                print(
                    f"Failed to add object '{obj_config_name}' with error: {e}"
                )
                self.culled_object_ixs.append(ix)
                continue
            for config_dir in self.episode.additional_obj_config_paths:
                candidate_object_path = os.path.join(
                    config_dir, obj_config_name
                )

                if os.path.exists(candidate_object_path):
                    ro = self._isaac_rom.add_object_by_template_handle(
                        candidate_object_path
                    )
                    self._rigid_objects.append(ro)
                    self.episode_object_ids.append(ro.object_id)
                    added_object = True
                    # set the friction parameters
                    stage = self._isaac_wrapper.service.world.stage
                    bind_physics_material_to_hierarchy(
                        stage=stage,
                        root_prim=ro._prim,
                        material_name="obj_physics_properties_material",
                        static_friction=self._app_cfg.object_static_friction,
                        dynamic_friction=self._app_cfg.object_dynamic_friction,
                        restitution=self._app_cfg.object_restitution,
                    )
                    break
            if not added_object:
                raise ValueError(
                    f"Object {obj_config_name} not found in paths {self.episode.additional_obj_config_paths}."
                )

        if self.episode.language_instruction != "":
            # load the episode's language prompt if not empty
            self.task_prompt = self.episode.language_instruction
        else:
            self.task_prompt = self.generate_contrived_prompt()

        # NOTE: this is where the states are set
        self.reset_episode_objects()

    def reset_episode_objects(self):
        """
        Places episode added objects back in their initial configurations.
        """
        obj_index = 0
        for ix, (_, transform) in enumerate(self.episode.rigid_objs):
            if ix not in self.culled_object_ixs:
                ro = self._isaac_rom.get_object_by_id(
                    self.episode_object_ids[obj_index]
                )
                obj_index += 1
                ro_t = mn.Matrix4(
                    [[transform[j][i] for j in range(4)] for i in range(4)]
                )

                ro_t = mn.Matrix4.from_(
                    ro_t.rotation_normalized(), ro_t.translation
                )
                # NOTE: this transform corrects for isaac's different coordinate system
                isaac_correction = mn.Matrix4.from_(
                    mn.Quaternion.rotation(
                        -mn.Deg(90), mn.Vector3.x_axis()
                    ).to_matrix(),
                    mn.Vector3(),
                )
                ro_t = ro_t @ isaac_correction

                ro.transformation = ro_t
                # NOTE: reset velocity after teleport to increase stability
                ro.clear_dynamics()
            else:
                # NOTE: this one had a bad transform and was skipped
                pass

    def add_rigid_object(
        self,
        handle: str,
        bottom_pos: mn.Vector3 = None,
        static_friction: float = 1.0,
        dynamic_friction: float = 1.0,
        restitution: float = 0.0,
        material_name: str = "test_material",
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
            stage = self._isaac_wrapper.service.world.stage
            prim = stage.GetPrimAtPath(
                "/World/rigid_objects/obj_" + str(ro.object_id)
            )
            bind_physics_material_to_hierarchy(
                stage=stage,
                root_prim=prim,
                material_name=material_name,
                static_friction=static_friction,
                dynamic_friction=dynamic_friction,
                restitution=restitution,
            )

    def highlight_added_objects(self):
        """
        Uses debug drawing to highlight rigid objects with a circle.
        """
        # TODO: should use the UI API highlight interfaces for better VR client highlighting
        dblr = self._app_service.gui_drawer

        # NOTE: This code crashes and disconnects the client. Is something wrong in the Unity app?
        # ro_ids = [ro.object_id for ro in self._rigid_objects]
        # self._app_service._gui_drawer._client_message_manager.draw_object_outline(
        #    priority=1,
        #    color=mn.Color4(0.8, 0.8, 0.1, 0.8), #yellow
        #    line_width=8.0,
        #    object_ids=ro_ids,
        #    destination_mask=Mask.ALL,
        # )

        # TODO: replace circles with client highlights
        for ro in self._rigid_objects:
            dblr.draw_circle(
                ro.translation,
                0.15,
                mn.Color4(0.7, 0.7, 0.3, 1.0),
                normal=(
                    self._cam_transform.translation - ro.translation
                ).normalized(),
            )
            # test linear vel by showing it here
            # dblr.draw_transformed_line(
            #    ro.translation,
            #    ro.translation + ro.linear_velocity,
            #    mn.Color3(1.0,0.5,1.0)
            # )
            # debug_draw_axis(dblr, mn.Matrix4.from_(mn.Matrix3(), ro.translation))
            # ang_vel = ro.angular_velocity
            # for axis in range(3):
            #     #for each angular velocity component, draw a line from that axis to illustrate
            #     axis_vec = mn.Vector3()
            #     axis_vec[axis] = 1.0
            #     color = mn.Color3(axis_vec)
            #     start_from = mn.Vector3()
            #     start_from[(axis+1)%3] = 1.0
            #     go_to = mn.Vector3()
            #     go_to[(axis+2)%3] = 1.0

            #     dblr.draw_transformed_line(
            #         ro.translation+start_from,
            #         ro.translation+start_from+go_to*ang_vel[axis],
            #         color
            #     )

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
        if self.in_replay_mode:
            # NOTE: This is a necessary workaround to accurately replay episodes before robot_settings were cached for the session
            # TODO: we'll remove this hack and retire the data in near future
            self.robot.ground_to_base_offset = 0
            print(
                "!! OVERWRITING self.robot.ground_to_base_offset = 0 FOR v0.1 EPISODE REPLAY !!"
            )
        # TODO: figure out how to initialize the robot in isolation instead of resetting the world. Doesn't work as expected.
        self._isaac_wrapper.service.world.reset()
        self.robot.post_init()

        # collect the open/closed grasp poses
        self.hand_grasp_poses = {}
        for hand in ["left", "right"]:
            hand_name = hand + "_hand"
            open_pose = self.robot.pos_subsets[hand_name].fetch_cached_pose(
                pose_name=hand_name + "_open"
            )
            closed_pose = self.robot.pos_subsets[hand_name].fetch_cached_pose(
                pose_name=hand_name + "_closed"
            )
            self.hand_grasp_poses[hand_name] = [open_pose, closed_pose]

        stage = self._isaac_wrapper.service.world.stage
        robot_root_prim = stage.GetPrimAtPath(self.robot._robot_prim_path)
        bind_physics_material_to_hierarchy(
            stage=stage,
            root_prim=robot_root_prim,
            material_name="robot_physics_properties_material",
            static_friction=self.robot.robot_cfg.link_static_friction,
            dynamic_friction=self.robot.robot_cfg.link_dynamic_friction,
            restitution=self.robot.robot_cfg.link_restitution,
        )

    def sample_valid_robot_base_state(self, reset_objects=True):
        """
        Resets robot joint state and attempts to set the robot base state to a collision free navmesh point within the target region.
        :param reset_objects: if true, reset object states from episode initial states
        """

        base_rot = random.uniform(-mn.math.pi, mn.math.pi)
        default_joint_state = self.robot.pos_subsets["full"].fetch_cached_pose(
            pose_name=self.robot.robot_cfg.initial_pose
        )

        # place robot in a region containing most objects
        obj_region_counts: defaultdict = defaultdict(int)
        max_region: int = None
        for obj in self._rigid_objects:
            possible_regions = self._sim.semantic_scene.get_regions_for_point(
                obj.translation
            )
            for rix in possible_regions:
                obj_region_counts[rix] += 1
                if (
                    max_region is None
                    or obj_region_counts[max_region] < obj_region_counts[rix]
                ):
                    max_region = rix
        assert max_region is not None

        # we have a region, now sample a navmesh point
        nav_point = self._sim.pathfinder.get_random_navigable_point()
        nav_samples = 0
        max_nav_samples = 1500
        # TODO: we use navmeshes closest obstacle as a proxy for collision detection. We should implement that for Isaac or pre-sample valid poses from Habitat-Bullet and cache them.
        while nav_samples < max_nav_samples and (
            max_region
            not in self._sim.semantic_scene.get_regions_for_point(
                nav_point + mn.Vector3(0, 0.1, 0)
            )
            or self.robot_contact_test(
                pos=nav_point,
                rot_angle=base_rot,
                joint_pos=default_joint_state,
            )
        ):
            nav_point = self._sim.pathfinder.get_random_navigable_point()
            # resample base rotation as necessary
            base_rot = random.uniform(-mn.math.pi, mn.math.pi)
            nav_samples += 1
        if nav_samples < max_nav_samples:
            # self.robot.set_root_pose(pos=nav_point)
            print("found collision free point?")
        else:
            print("Failed to find new nav point, max samples exceeded.")

        if reset_objects:
            self.reset_episode_objects()

    def set_robot_base_initial_state(self):
        """
        Either sets the configured initial state or samples a random point from the navmesh.
        """
        self.robot.set_cached_pose(
            pose_name=self.robot.robot_cfg.initial_pose,
            set_positions=True,
            set_motor_targets=True,
        )
        # get an initial base rotation from config
        base_rot = (
            random.uniform(-mn.math.pi, mn.math.pi)
            if not hasattr(self._app_cfg, "initial_robot_rotation")
            else self._app_cfg.initial_robot_rotation
        )

        if hasattr(self._app_cfg, "initial_robot_position"):
            # load initial base position if configured
            initial_pos = mn.Vector3(*self._app_cfg.initial_robot_position)
            self.robot.set_root_pose(pos=initial_pos)
            self.robot.base_rot = base_rot
        elif self.episode is not None and len(self._rigid_objects) > 0:
            self.sample_valid_robot_base_state()
        else:
            # place the robot on the navmesh
            self.robot.set_root_pose(
                pos=self._sim.pathfinder.get_random_navigable_point()
            )
            self.robot.base_rot = base_rot

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
        controls_str += "IJKL: move robot base\n"
        controls_str += "H: toggle GUI\n"
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
        status_str += f"base rot: {self.robot.base_rot} \n"
        status_str += f"time: {self._timer}\n"
        status_str += f"robot in contact: {self.robot.in_contact}\n"
        if self._frame_recorder.replaying:
            start_time = (
                self._frame_recorder.frame_data[0]["t"]
                if self._frame_recorder._start_frame is None
                else self._frame_recorder.frame_data[
                    self._frame_recorder._start_frame
                ]["t"]
            )
            status_str += f"Replaying {self._frame_recorder.replay_time:.2f} from range [{start_time:.2f}, {self._frame_recorder.frame_data[-1]['t']:.2f}]-\n"
        elif self._frame_recorder.recording:
            status_str += (
                f"-RECORDING FRAMES ({len(self._frame_recorder.frame_data)})-"
            )

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
                dblr, head=False
            )

    def handle_xr_input(self, dt: float):
        if self._app_service.remote_client_state is None:
            return
        elif self._first_xr_update:
            # when the XR user first join, do a sync automatically
            self._first_xr_update = False
            self.sync_xr_local_state()

        xr_input = self._app_service.remote_client_state.get_xr_input(0)
        left = xr_input.left_controller
        right = xr_input.right_controller
        from habitat_hitl.core.xr_input import XRButton

        if left.get_button_down(XRButton.ONE):
            print("pressed one left")
            self.sync_xr_local_state()
            self._frame_events.append(FrameEvent.SYNC_XR_OFFSET)
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
            self._frame_events.append(FrameEvent.RESET_ARMS_FINGERS)

        if left.get_button_up(XRButton.TWO):
            pass

        if left.get_button_down(XRButton.START):
            print("pressed START left")
            self._task_finished_signaled = True
        if left.get_button_up(XRButton.START):
            pass

        # RIGHT CONTROLLER BUTTONS
        # NOTE: recording toggle disabled when joining from session. Recording is required.
        if right.get_button_down(XRButton.ONE):
            # self._frame_recorder.recording = not self._frame_recorder.recording
            # print(
            #    f"pressed one right, recording = {self._frame_recorder.recording}"
            # )
            # TODO: hook this up to pass/fail
            self._view_task_prompt = True

        if right.get_button_up(XRButton.ONE):
            pass

        if right.get_button_down(XRButton.TWO):
            print("pressed two right, resetting episode objects")
            self.reset_episode_objects()
            self._frame_events.append(FrameEvent.RESET_OBJECTS)
        if right.get_button_up(XRButton.TWO):
            pass
        # NOTE: XRButton.START is reserved for Quest menu functionality

        # NOTE: PRIMARY_HAND_TRIGGER mapped to IK toggle
        # NOTE: PRIMARY_INDEX_TRIGGER mapped to grasping functionality

        # TODO: PRIMARY_THUMBSTICK (click?) unmapped currently and available
        if left.get_button_down(XRButton.PRIMARY_THUMBSTICK):
            # reset the robot state (e/g/ to unstick or restart)
            self.sample_valid_robot_base_state()
            self._frame_events.append(FrameEvent.TELEPORT)

        # IK for each hand when trigger is pressed
        xr_pose = XRPose(
            remote_client_state=self._app_service.remote_client_state
        )
        if xr_pose.valid:
            global_xr_pose = self.xr_pose_adapter.get_global_xr_pose(xr_pose)

            # combine both hands' logic to reduce boilerplate code
            for (
                xrcontroller,
                base_link_key,
                pivot_link_key,
                arm_subset_key,
            ) in [
                (right, "right_arm_base", "right_arm_pivot", "right_arm"),
                (left, "left_arm_base", "left_arm_pivot", "left_arm"),
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
                    ee_pos_global = global_xr_pose.pos_left
                    ee_rot_global = global_xr_pose.rot_left
                    if "right" in base_link_key:
                        xr_pose = (
                            xr_pose_in_robot_frame.pos_right,
                            xr_pose_in_robot_frame.rot_right,
                        )
                        ee_pos_global = global_xr_pose.pos_right
                        ee_rot_global = global_xr_pose.rot_right

                    cur_ef_T = arm_base_transform.__matmul__(
                        self._ik.get_ee_T(q=cur_arm_pose)
                    )
                    tar_disp = ee_pos_global - cur_ef_T.translation
                    tar_dist = (tar_disp).length()
                    max_tar_dist = 0.05
                    if tar_dist > max_tar_dist:
                        tar_ef = (
                            cur_ef_T.translation
                            + tar_disp.normalized() * max_tar_dist
                        )
                        xr_pose = (
                            arm_base_transform.inverted().transform_point(
                                tar_ef
                            ),
                            xr_pose[1],
                        )
                        tar_ef_T = mn.Matrix4.from_(
                            ee_rot_global.to_matrix(), tar_ef
                        )
                        debug_draw_axis(
                            self._app_service.gui_drawer,
                            tar_ef_T,
                            scale=0.25,
                        )

                    new_arm_pose = self._ik.inverse_kinematics(
                        to_ik_pose(xr_pose), cur_arm_pose
                    )

                    debug_draw_axis(
                        self._app_service.gui_drawer,
                        arm_base_transform.__matmul__(self._ik.get_ee_T()),
                        scale=0.75,
                    )

                    # draw the warning lines for arm length
                    arm_pivot_link = self.robot.link_subsets[
                        pivot_link_key
                    ].link_ixs[0]
                    arm_pivot_pos = self.robot.get_link_world_poses(
                        indices=[arm_pivot_link]
                    )[0][0]
                    target_dist_from_base = (
                        arm_pivot_pos - ee_pos_global
                    ).length()
                    if (
                        target_dist_from_base
                        > self.robot.approximate_arm_length
                    ):
                        self._app_service.gui_drawer.draw_transformed_line(
                            arm_pivot_pos, ee_pos_global, mn.Color4(1, 0, 0, 1)
                        )
                    elif (
                        target_dist_from_base
                        > self.robot.approximate_arm_length * 0.9
                    ):
                        self._app_service.gui_drawer.draw_transformed_line(
                            arm_pivot_pos, ee_pos_global, mn.Color4(1, 1, 0, 1)
                        )

                    # using configuration subset to avoid accidental overwriting with noise
                    self.robot.pos_subsets[arm_subset_key].set_motor_pos(
                        new_arm_pose
                    )
                    # self.robot.pos_subsets[arm_subset_key].set_pos(new_arm_pose)

            # grasping logic
            for xrcontroller, hand_subset_key in [
                (right, "right_hand"),
                (left, "left_hand"),
            ]:
                cur_pose = LERP(
                    self.hand_grasp_poses[hand_subset_key][0],
                    self.hand_grasp_poses[hand_subset_key][1],
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
                    dir_target=norm_dir, dir_init=global_dir
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
            # NOTE: hold Q while selecting the waypoint to teleport. Useful for crossing boundaries like doorframes.
            if gui_input.get_key(KeyCode.Q):
                target_pos = self.robot.base_vel_controller.target_position
                # this setter will reset the waypoint target so we cache first
                self.robot.base_rot = (
                    self.robot.base_vel_controller.target_rotation
                )
                self.robot.set_root_pose(target_pos)

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

    def single_step_isaac(self):
        """
        Do a single step of isaac simulation in order to update the collision structures.
        NOTE: This is intended to be used for instantaneous collision detection.
        """
        self._isaac_wrapper.step(num_steps=1)

    def robot_contact_test(
        self, pos=None, rot=None, rot_angle=None, joint_pos=None
    ):
        """
        Sets the robot's state from inputs and runs a single step of isaac simulation and then returns the robot's in_contact property.
        NOTE: Expected to be called immediately after a kinematic update to the robot's state in order to place the robot.
        Should not be called during active simulation, but rather used when modifying the state BEFORE acting.

        :param rot_angle: The scalar rotation angle around the Y axis, separate representation than 'rot', the rotation quaternion
        """
        # self.robot._contact_sensors_active = True
        if pos is not None or rot is not None:
            self.robot.set_root_pose(pos=pos, rot=rot)
        if rot_angle is not None:
            assert (
                rot is None
            ), "Can't set the base rotation by quaternion and scalar simultaneously."
            self.robot.base_rot = rot_angle
        if joint_pos is not None:
            self.robot.pos_subsets["full"].set_motor_pos(joint_pos)
            self.robot.pos_subsets["full"].set_pos(joint_pos)
            self.robot.pos_subsets["full"].clear_velocities()
        create_stage_mesh_contact_sensors(self.root_prim)
        self.robot.enable_contact_report_sensors()
        self.single_step_isaac()
        # self.robot._contact_sensors_active = False
        remove_stage_mesh_contact_sensors(self.root_prim)
        self.robot.disable_contact_report_sensors()
        return self.robot.in_contact

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
                if not self._do_pause_physics:
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

        if gui_input.get_key_down(KeyCode.SPACE):
            # self._frame_recorder.recording = not self._frame_recorder.recording
            # print(f"Set state recording = {self._frame_recorder.recording}")
            self.pause_replay = not self.pause_replay
            print(f"Pause Replay = {self.pause_replay}")

        if gui_input.get_key_down(KeyCode.TAB):
            self.reverse_replay = not self.reverse_replay
            print(f"Reverse Replay = {self.reverse_replay}")

        if gui_input.get_key_down(KeyCode.B):
            self._frame_recorder.save_json()

        if gui_input.get_key_down(KeyCode.V):
            self._frame_recorder.load_json()
            self._frame_recorder.replaying = True
            self._timer = self._frame_recorder.frame_data[0]["t"]
            self._start_time = time.time() - self._timer
            # NOTE: would be nice to turn physics subsets off, but this disables the robot state updates for some reason
            # self.set_physics_paused(True)

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
                handle="data/objects/ycb/configs/035_power_drill.object_config.json",
                bottom_pos=hab_hit_pos,
                static_friction=self._app_cfg.object_static_friction,
                dynamic_friction=self._app_cfg.object_dynamic_friction,
                restitution=self._app_cfg.object_restitution,
            )

        if gui_input.get_key_down(KeyCode.N):
            self.robot.set_root_pose(
                pos=self._sim.pathfinder.get_random_navigable_point()
            )
            self._frame_events.append(FrameEvent.TELEPORT)

        if gui_input.get_key_down(KeyCode.M):
            self.sample_valid_robot_base_state()
            self._frame_events.append(FrameEvent.TELEPORT)
            # self.set_robot_base_initial_state()

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
                        self._frame_events.append(
                            FrameEvent.RESET_ARMS_FINGERS
                        )

        if gui_input.get_key_down(KeyCode.ONE):
            self.reset_episode_objects()
            self._frame_events.append(FrameEvent.RESET_OBJECTS)

        if gui_input.get_key_down(KeyCode.THREE):
            print(f"Robot in contact = {self.robot_contact_test()}")

        if gui_input.get_key_down(KeyCode.F):
            self.cursor_follow_robot = not self.cursor_follow_robot
            print(f"Set cursor_follow_robot = {self.cursor_follow_robot}")

    def handle_mouse_press(self) -> None:
        """
        Mouse control UI for local server window.
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
        self._mouse_points_record = []

    def update_mouse_raycaster(self, dt):
        self._recent_mouse_ray_hit_info = None

        mouse_ray = self._app_service.gui_input.mouse_ray

        if self._app_service.gui_input.get_key(KeyCode.TWO):
            self._mouse_points_record = []

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
        # if self._app_service.gui_input.get_mouse_button(MouseButton.LEFT):
        #     self._mouse_points_record.append(
        #         (hit_pos_habitat, hit_normal_habitat)
        #     )
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

        for hit_point, hit_normal in self._mouse_points_record:
            self._app_service.gui_drawer.draw_circle(
                hit_point, 0.01, mn.Color3(0.1, 0.1, 0.1), 9, normal=hit_normal
            )

        self._recent_mouse_ray_hit_info = hit_info

        gui_input = self._app_service.gui_input
        if gui_input.get_key_down(KeyCode.Y):
            # example applying linear velocity
            # print(f" {body_name} in {self._isaac_rom._obj_wrapper_by_prim_path.keys()} = {body_name in self._isaac_rom._obj_wrapper_by_prim_path}")
            # if body_name in self._isaac_rom._obj_wrapper_by_prim_path:
            # this is a rigid object
            # ro = self._isaac_rom.get_object_by_prim_path(body_name)
            # ro.linear_velocity = mn.Vector3(0,1,0)
            # ro.angular_velocity = mn.Vector3(0,1,0)

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

    # def debug_draw_hands(self):
    #     """
    #     Draw finger tips as circles.
    #     NOTE: This is only necessary while finger tips are not rendering in the client.
    #     """
    #     dblr = self._app_service.gui_drawer
    #     for finger_subset_key in [
    #         "left_thumb_tip",
    #         "right_thumb_tip",
    #         "left_finger_tips",
    #         "right_finger_tips",
    #     ]:
    #         color = mn.Color3(0.6, 0.4, 0.6)
    #         # this distance is applied to offset the render shape from the root of the parent link
    #         finger_offset_dist = 0.043
    #         link_pos, link_rots = self.robot.get_link_world_poses(
    #             indices=self.robot.link_subsets[finger_subset_key].link_ixs
    #         )
    #         for pos, rot in zip(link_pos, link_rots):
    #             finger_normal = rot.transform_vector(mn.Vector3(0, 0, 1.0))
    #             dblr.draw_circle(
    #                 pos + finger_normal * finger_offset_dist,
    #                 0.01,
    #                 color,
    #                 normal=finger_normal,
    #             )
    #             dblr.draw_transformed_line(
    #                 pos, pos + finger_normal * finger_offset_dist, color
    #             )

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

        if self.in_replay_mode:
            self._frame_recorder.get_xr_pose_from_frame().draw_pose(
                self._app_service.gui_drawer
            )

        # need this in the app even if other debug lines are turned off
        self.debug_draw_quest()

        if self._draw_debug_shapes:
            # draw lookat ring
            self.draw_lookat()
            # draw the robot frame
            self.robot.draw_debug(self._app_service.gui_drawer)
            # self.debug_draw_hands()

            if self.dof_editor is not None:
                self.dof_editor.debug_draw(
                    self._app_service.gui_drawer,
                    self._cam_transform.translation,
                )
        self.highlight_added_objects()

        if self._view_task_prompt:
            if self.episode is not None:
                with self._app_service.ui_manager.update_canvas(
                    "center", destination_mask=Mask.ALL
                ) as ctx:
                    FONT_SIZE_LARGE = 32
                    FONT_SIZE_SMALL = 24
                    BTN_ID_DONE = "btn_done"
                    ctx.canvas_properties(
                        padding=12, background_color=[0.3, 0.3, 0.3, 0.7]
                    )
                    ctx.label(
                        text="Task Prompt",
                        font_size=FONT_SIZE_LARGE,
                        horizontal_alignment=HorizontalAlignment.CENTER,
                    )
                    ctx.separator()
                    ctx.label(
                        text=self.task_prompt,
                        font_size=FONT_SIZE_SMALL,
                        horizontal_alignment=HorizontalAlignment.CENTER,
                    )
                    ctx.separator()
                    ctx.button(
                        uid=BTN_ID_DONE,
                        text="Done",
                        enabled=self._view_task_prompt,
                    )
                if self._app_service.remote_client_state.ui_button_pressed(
                    0, BTN_ID_DONE
                ):
                    self._view_task_prompt = False
                    self._app_service.ui_manager.clear_all_canvases(Mask.ALL)
            else:
                self._view_task_prompt = False

        if self._task_finished_signaled:
            if self._session is None:
                # Skip the dialogue if outside of the state machine.
                self._task_finished = True
                self._task_success = 1.0
            else:
                with self._app_service.ui_manager.update_canvas(
                    "center", destination_mask=Mask.ALL
                ) as ctx:
                    FONT_SIZE_LARGE = 32
                    FONT_SIZE_SMALL = 24
                    BTN_ID_SUCCESS = "btn_success"
                    BTN_ID_FAILURE = "btn_failure"
                    BTN_ID_CANCEL = "btn_cancel"
                    ctx.canvas_properties(
                        padding=12, background_color=[0.3, 0.3, 0.3, 0.7]
                    )
                    ctx.label(
                        text="Finish Task?",
                        font_size=FONT_SIZE_LARGE,
                        horizontal_alignment=HorizontalAlignment.CENTER,
                    )
                    ctx.separator()
                    ctx.button(
                        uid=BTN_ID_SUCCESS,
                        text="Success",
                        enabled=not self._task_finished,
                    )
                    ctx.button(
                        uid=BTN_ID_FAILURE,
                        text="Failure",
                        enabled=not self._task_finished,
                    )
                    ctx.separator()
                    ctx.spacer()
                    ctx.button(
                        uid=BTN_ID_CANCEL,
                        text="Cancel",
                        enabled=not self._task_finished,
                    )
                if self._app_service.remote_client_state.ui_button_pressed(
                    0, BTN_ID_SUCCESS
                ):
                    self._task_finished = True
                    self._task_success = 1.0
                    self._app_service.ui_manager.clear_all_canvases(Mask.ALL)
                elif self._app_service.remote_client_state.ui_button_pressed(
                    0, BTN_ID_FAILURE
                ):
                    self._task_finished = True
                    self._task_success = 0.0
                    self._app_service.ui_manager.clear_all_canvases(Mask.ALL)
                elif self._app_service.remote_client_state.ui_button_pressed(
                    0, BTN_ID_CANCEL
                ):
                    self._task_finished_signaled = False
                    self._app_service.ui_manager.clear_all_canvases(Mask.ALL)

        self._update_help_text()
        self._timer = time.time() - self._start_time

        if not self.pause_replay:
            r = -1 if self.reverse_replay else 1
            self._frame_recorder.update(
                self._frame_recorder.replay_time + dt * r,
                [str(e) for e in self._frame_events],
            )
            self._frame_events = []

        # record state trajectory frames in the session object for cloud serialization
        if self._session is not None:
            assert self._frame_recorder.recording == True
            # most recent frame was recorded in update() above
            frame_data = self._frame_recorder.frame_data[-1]
            self._session.session_recorder.record_frame(frame_data)
            # Record the task prompt within the episode data
            if (
                self._session.session_recorder.episode_records[-1].episode_info
                is None
            ):
                self._session.session_recorder.episode_records[
                    -1
                ].episode_info = {}
            self._session.session_recorder.episode_records[-1].episode_info[
                "task_prompt"
            ] = self.task_prompt
            # record the robot settings
            # TODO: This is not yet implemented.
            """
            from omegaconf import OmegaConf

            self._session.session_recorder.session_record.config[
                "robot_settings"
            ] = OmegaConf.to_container(self.robot.robot_cfg)
            """

    def _is_episode_finished(self) -> bool:
        return self._task_finished

    def get_next_state(self) -> Optional[AppStateBase]:
        """When running from the state machine, this function determines whether the state must be changed."""
        assert self._app_data is not None
        assert self._session is not None
        assert self.episode is not None

        if self._cancel:
            return create_app_state_cancel_session(
                self._app_service,
                self._app_data,
                self._session,
                error="User disconnected",
            )
        elif self._is_episode_finished():
            return create_app_state_load_episode(
                self._app_service, self._app_data, self._session
            )
        return None

    def on_enter(self):
        """When running from the state machine, this function is called after construction."""
        assert self._app_data is not None
        assert self._session is not None
        assert self.episode is not None
        super().on_enter()

        episode = self.episode
        self._session.session_recorder.start_episode(
            episode_index=self._session.current_episode_index,
            episode_id=episode.episode_id,
            scene_id=episode.scene_id,
            dataset=episode.scene_dataset_config,
            episode_info=episode.info,
        )

    def on_exit(self):
        """When running from the state machine, this function is called before destruction."""
        super().on_exit()
        assert self._app_data is not None
        assert self._session is not None

        episode_finished = self._is_episode_finished() and not self._cancel

        self._session.session_recorder.end_episode(
            episode_finished=episode_finished,
            task_percent_complete=self._task_success,
            metrics={},
        )

        # TODO: Define UI.
        """
        for user_data in self._user_data:
            user_data.ui.reset()
        """


@hydra.main(
    version_base=None, config_path="./config", config_name="isaac_robot_teleop"
)
def main(config):
    hitl_main(config, lambda app_service: AppStateIsaacSimViewer(app_service))


if __name__ == "__main__":
    register_hydra_plugins()
    main()
