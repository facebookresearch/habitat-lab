#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Any, Dict

import magnum as mn
import numpy as np

from habitat.core.registry import registry
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.rearrange.rearrange_grasp_manager import (
    RearrangeGraspManager,
)
from habitat.tasks.rearrange.utils import (
    IkHelper,
    get_nav_mesh_settings,
    make_render_only,
)
from habitat_sim.physics import MotionType

# flake8: noqa
from habitat_sim.robots import FetchRobot, FetchRobotNoWheels


class MarkerInfo:
    def __init__(self, offset_position, link_node, ao_parent, link_id):
        self.offset_position = offset_position
        self.link_node = link_node
        self.link_id = link_id
        self.current_transform = None
        self.ao_parent = ao_parent

        self.joint_idx = ao_parent.get_link_joint_pos_offset(link_id)

        self.update()

    def set_targ_js(self, js):
        self.ao_parent.joint_positions[self.joint_idx] = js

    def get_targ_js(self):
        return self.ao_parent.joint_positions[self.joint_idx]

    def update(self):
        offset_T = mn.Matrix4.translation(mn.Vector3(self.offset_position))
        self.current_transform = self.link_node.transformation @ offset_T

    def get_current_position(self) -> np.ndarray:
        return np.array(self.current_transform.translation)

    def get_current_transform(self) -> mn.Matrix4:
        return self.current_transform


@registry.register_simulator(name="RearrangeSim-v0")
class RearrangeSim(HabitatSim):
    def __init__(self, config):
        super().__init__(config)

        agent_config = self.habitat_config
        self.navmesh_settings = get_nav_mesh_settings(self._get_agent_config())
        self.first_setup = True
        self._should_render_debug = False
        self.ep_info = None
        self.prev_loaded_navmesh = None
        self.prev_scene_id = None

        # Number of physics updates per action
        self.ac_freq_ratio = agent_config.AC_FREQ_RATIO
        # The physics update time step.
        self.ctrl_freq = agent_config.CTRL_FREQ
        # Effective control speed is (ctrl_freq/ac_freq_ratio)
        self._concur_render = self.habitat_config.get("CONCUR_RENDER", False)
        self._auto_sleep = self.habitat_config.get("AUTO_SLEEP", False)
        self._debug_render = self.habitat_config.get("DEBUG_RENDER", False)

        self.art_objs = []
        self._start_art_states = {}
        self._prev_obj_names = None
        self.cached_art_obj_ids = []
        self.scene_obj_ids = []
        # Used to get data from the RL environment class to sensors.
        self._goal_pos = None
        self.viz_ids: Dict[Any, Any] = defaultdict(lambda: None)
        self.ref_handle_to_rigid_obj_id = None
        robot_cls = eval(self.habitat_config.ROBOT_TYPE)
        self.robot = robot_cls(self.habitat_config.ROBOT_URDF, self)
        self._markers = {}

        self.ik_helper = None

        # Disables arm control. Useful if you are hiding the arm to perform
        # some scene sensing.
        self.ctrl_arm = True

        self.grasp_mgr: RearrangeGraspManager = RearrangeGraspManager(
            self, self.habitat_config
        )

    def _get_target_trans(self):
        """
        This is how the target transforms should be accessed since
        multiprocessing does not allow pickling.
        """
        # Preprocess the ep_info making necessary datatype conversions.
        target_trans = []
        rom = self.get_rigid_object_manager()
        for target_handle, trans in self.ep_info["targets"].items():
            targ_idx = self.scene_obj_ids.index(
                rom.get_object_by_handle(target_handle).object_id
            )
            target_trans.append((targ_idx, mn.Matrix4(trans)))
        return target_trans

    def _try_acquire_context(self):
        if self._concur_render:
            self.renderer.acquire_gl_context()

    def sleep_all_objects(self):
        """
        De-activate (sleep) all rigid objects in the scene, assuming they are already in a dynamically stable state.
        """
        rom = self.get_rigid_object_manager()
        for _, ro in rom.get_objects_by_handle_substring().items():
            ro.awake = False
        aom = self.get_articulated_object_manager()
        for _, ao in aom.get_objects_by_handle_substring().items():
            ao.awake = False

    def add_markers(self, ep_info):
        self._markers = {}
        aom = self.get_articulated_object_manager()
        for marker in ep_info["markers"]:
            p = marker["params"]
            ao = aom.get_object_by_handle(p["object"])
            name_to_link = {}
            name_to_link_id = {}
            for i in range(ao.num_links):
                name = ao.get_link_name(i)
                link = ao.get_link_scene_node(i)
                name_to_link[name] = link
                name_to_link_id[name] = i

            self._markers[marker["name"]] = MarkerInfo(
                p["offset"],
                name_to_link[p["link"]],
                ao,
                name_to_link_id[p["link"]],
            )

    def get_marker(self, name: str) -> MarkerInfo:
        return self._markers[name]

    def get_all_markers(self):
        return self._markers

    def _update_markers(self) -> None:
        for m in self._markers.values():
            m.update()

    def reconfigure(self, config):
        ep_info = config["ep_info"][0]
        self.instance_handle_to_ref_handle = ep_info["info"]["object_labels"]

        config["SCENE"] = ep_info["scene_id"]

        super().reconfigure(config)
        self.ref_handle_to_rigid_obj_id = {}

        self.ep_info = ep_info
        self._try_acquire_context()

        if self.prev_scene_id != ep_info["scene_id"]:
            self.grasp_mgr.reconfigure()
            # add and initialize the robot
            ao_mgr = self.get_articulated_object_manager()
            if self.robot.sim_obj is not None and self.robot.sim_obj.is_alive:
                ao_mgr.remove_object_by_id(self.robot.sim_obj.object_id)
            self.robot.reconfigure()
            self._prev_obj_names = None

        self.grasp_mgr.reset()

        # Only remove and re-add objects if we have a new set of objects.
        obj_names = [x[0] for x in ep_info["rigid_objs"]]
        should_add_objects = self._prev_obj_names != obj_names
        self._prev_obj_names = obj_names

        self._clear_objects(should_add_objects)

        self.prev_scene_id = ep_info["scene_id"]
        self._viz_templates = {}

        # Set the default articulated object joint state.
        for ao, set_joint_state in self._start_art_states.items():
            ao.clear_joint_states()
            ao.joint_positions = set_joint_state

        # Load specified articulated object states from episode config
        self._set_ao_states_from_ep(ep_info)

        self.robot.reset()
        # consume a fixed position from SIMUALTOR.AGENT_0 if configured
        if self.habitat_config.AGENT_0.IS_SET_START_STATE:
            self.robot.base_pos = mn.Vector3(
                self.habitat_config.AGENT_0.START_POSITION
            )
            agent_rot = self.habitat_config.AGENT_0.START_ROTATION
            self.robot.sim_obj.rotation = mn.Quaternion(
                mn.Vector3(agent_rot[:3]), agent_rot[3]
            )

            if "RENDER_CAMERA_OFFSET" in self.habitat_config:
                self.robot.params.cameras[
                    "robot_third"
                ].cam_offset_pos = mn.Vector3(
                    self.habitat_config.RENDER_CAMERA_OFFSET
                )
            if "RENDER_CAMERA_LOOKAT" in self.habitat_config:
                self.robot.params.cameras[
                    "robot_third"
                ].cam_look_at_pos = mn.Vector3(
                    self.habitat_config.RENDER_CAMERA_LOOKAT
                )

        # add episode clutter objects additional to base scene objects
        self._add_objs(ep_info, should_add_objects)

        self.add_markers(ep_info)

        # auto-sleep rigid objects as optimization
        if self._auto_sleep:
            self.sleep_all_objects()

        # recompute the NavMesh once the scene is loaded
        # NOTE: because ReplicaCADv3_sc scenes, for example, have STATIC objects with no accompanying NavMesh files
        self._recompute_navmesh()

        # Get the starting positions of the target objects.
        rom = self.get_rigid_object_manager()
        scene_pos = self.get_scene_pos()
        self.target_start_pos = np.array(
            [
                scene_pos[
                    self.scene_obj_ids.index(
                        rom.get_object_by_handle(t_handle).object_id
                    )
                ]
                for t_handle, _ in self.ep_info["targets"].items()
            ]
        )

        if self.first_setup:
            self.first_setup = False
            if self.habitat_config.get("IK_ARM_URDF", None) is not None:
                self.ik_helper = IkHelper(
                    self.habitat_config.IK_ARM_URDF,
                    np.array(self.robot.params.arm_init_params),
                )
            # Capture the starting art states
            self._start_art_states = {
                ao: ao.joint_positions for ao in self.art_objs
            }

    def get_nav_pos(self, pos):
        pos = mn.Vector3(*pos)
        height_thresh = 0.15
        z_min = -0.2
        use_vs = np.array(self.pathfinder.build_navmesh_vertices())

        if height_thresh is not None:
            use_vs = use_vs[use_vs[:, 1] < height_thresh]
        if z_min is not None:
            use_vs = use_vs[use_vs[:, 2] > z_min]
        dists = np.linalg.norm(
            use_vs[:, [0, 2]] - np.array(pos)[[0, 2]], axis=-1
        )

        closest_idx = np.argmin(dists)
        return use_vs[closest_idx]

    def _recompute_navmesh(self):
        """Generates the navmesh on the fly. This must be called
        AFTER adding articulated objects to the scene.
        """

        # cache current motiontype and set to STATIC for inclusion in the NavMesh computation
        motion_types = []
        for art_obj in self.art_objs:
            motion_types.append(art_obj.motion_type)
            art_obj.motion_type = MotionType.STATIC
        # compute new NavMesh
        self.recompute_navmesh(
            self.pathfinder,
            self.navmesh_settings,
            include_static_objects=True,
        )
        # optionally save the new NavMesh
        if self.habitat_config.get("SAVE_NAVMESH", False):
            scene_name = self.ep_info["scene_id"]
            inferred_path = scene_name.split(".glb")[0] + ".navmesh"
            self.pathfinder.save_nav_mesh(inferred_path)
        # reset cached MotionTypes
        for art_obj, motion_type in zip(self.art_objs, motion_types):
            art_obj.motion_type = motion_type

    def _clear_objects(self, should_add_objects):
        if should_add_objects:
            rom = self.get_rigid_object_manager()
            for scene_obj_id in self.scene_obj_ids:
                if rom.get_library_has_id(scene_obj_id):
                    rom.remove_object_by_id(scene_obj_id)
            self.scene_obj_ids = []

        # Do not remove the articulated objects from the scene, these are
        # managed by the underlying sim.
        self.art_objs = []

    def _set_ao_states_from_ep(self, ep_info):
        """
        Sets the ArticulatedObject states for the episode which are differ from base scene state.
        """
        aom = self.get_articulated_object_manager()
        # NOTE: ep_info["ao_states"]: Dict[str, Dict[int, float]] : {instance_handle -> {link_ix, state}}
        for aoi_handle, joint_states in ep_info["ao_states"].items():
            ao = aom.get_object_by_handle(aoi_handle)
            ao_pose = ao.joint_positions
            for link_ix, joint_state in joint_states.items():
                joint_position_index = ao.get_link_joint_pos_offset(
                    int(link_ix)
                )
                ao_pose[joint_position_index] = joint_state
            ao.joint_positions = ao_pose

    def _add_objs(self, ep_info, should_add_objects):
        # Load clutter objects:
        # NOTE: ep_info["rigid_objs"]: List[Tuple[str, np.array]]  # list of objects, each with (handle, transform)
        rom = self.get_rigid_object_manager()
        obj_counts: Dict[str, int] = defaultdict(int)

        for i, (obj_handle, transform) in enumerate(ep_info["rigid_objs"]):
            if should_add_objects:
                obj_attr_mgr = self.get_object_template_manager()
                matching_templates = (
                    obj_attr_mgr.get_templates_by_handle_substring(obj_handle)
                )
                assert (
                    len(matching_templates.values()) == 1
                ), "Duplicate object attributes matched to shortened handle. TODO: relative paths as handles should fix this. For now, try renaming objects to avoid collision."
                ro = rom.add_object_by_template_handle(
                    list(matching_templates.keys())[0]
                )
            else:
                ro = rom.get_object_by_id(self.scene_obj_ids[i])

            # The saved matrices need to be flipped when reloading.
            ro.transformation = mn.Matrix4(
                [[transform[j][i] for j in range(4)] for i in range(4)]
            )

            other_obj_handle = (
                obj_handle.split(".")[0] + f"_:{obj_counts[obj_handle]:04d}"
            )

            if other_obj_handle in self.instance_handle_to_ref_handle:
                ref_handle = self.instance_handle_to_ref_handle[
                    other_obj_handle
                ]
                # self.ref_handle_to_rigid_obj_id[ref_handle] = ro.object_id
                rel_idx = len(self.scene_obj_ids)
                self.ref_handle_to_rigid_obj_id[ref_handle] = rel_idx
            obj_counts[obj_handle] += 1

            if should_add_objects:
                self.scene_obj_ids.append(ro.object_id)

        ao_mgr = self.get_articulated_object_manager()
        for aoi_handle in ao_mgr.get_object_handles():
            self.art_objs.append(ao_mgr.get_object_by_handle(aoi_handle))

    def _create_obj_viz(self, ep_info):
        if self._debug_render:
            for marker_name, m in self._markers.items():
                m_T = m.get_current_transform()
                self.viz_ids[marker_name] = self.visualize_position(
                    m_T.translation, self.viz_ids[marker_name]
                )

        # TODO: refactor this
        # target_name_pos = [
        #     (ep_info["static_objs"][idx][0], self.scene_objs[idx], pos)
        #     for idx, pos in self._get_target_trans()
        # ]
        # self.viz_obj_ids = place_viz_objs(
        #     target_name_pos, self, self.viz_obj_ids
        # )

    def capture_state(self, with_robot_js=False):
        # Don't need to capture any velocity information because this will
        # automatically be set to 0 in `set_state`.
        robot_T = self.robot.sim_obj.transformation
        art_T = [ao.transformation for ao in self.art_objs]
        static_T = [self.get_transformation(i) for i in self.scene_obj_ids]
        art_pos = [ao.joint_positions for ao in self.art_objs]
        robot_js = self.robot.sim_obj.joint_positions

        ret = {
            "robot_T": robot_T,
            "art_T": art_T,
            "static_T": static_T,
            "art_pos": art_pos,
            "obj_hold": self.grasp_mgr.snap_idx,
        }
        if with_robot_js:
            ret["robot_js"] = robot_js
        return ret

    def set_state(self, state, set_hold=False):
        """
        - set_hold: If true this will set the snapped object from the `state`.
          This should probably be True by default, but I am not sure the effect
          it will have.
        """
        if state["robot_T"] is not None:
            self.robot.sim_obj.transformation = state["robot_T"]
            n_dof = len(self.robot.sim_obj.joint_forces)
            self.robot.sim_obj.joint_forces = np.zeros(n_dof)
            self.robot.sim_obj.joint_velocities = np.zeros(n_dof)

        if "robot_js" in state:
            self.robot.sim_obj.joint_positions = state["robot_js"]

        for T, ao in zip(state["art_T"], self.art_objs):
            ao.transformation = T

        for T, i in zip(state["static_T"], self.scene_obj_ids):
            self.reset_obj_T(i, T)

        for p, ao in zip(state["art_pos"], self.art_objs):
            ao.joint_positions = p

        if set_hold:
            if state["obj_hold"] is not None:
                self.internal_step(-1)
                self.grasp_mgr.snap_to_obj(state["obj_hold"])
            else:
                self.grasp_mgr.desnap(True)

    def reset_obj_T(self, i, T):
        self.set_transformation(T, i)
        self.set_linear_velocity(mn.Vector3(0, 0, 0), i)
        self.set_angular_velocity(mn.Vector3(0, 0, 0), i)

    def settle_sim(self, seconds):
        steps = int(seconds * self.ctrl_freq)
        for _ in range(steps):
            self.step_world(-1)

    def step(self, action):
        rom = self.get_rigid_object_manager()

        self._update_markers()

        if self._should_render_debug:
            self._try_acquire_context()
            for obj_handle, _ in self.ep_info["targets"].items():
                self.set_object_bb_draw(
                    False, rom.get_object_by_handle(obj_handle).object_id
                )

        add_back_viz_objs = {}
        for name, viz_id in self.viz_ids.items():
            if viz_id is None:
                continue

            before_pos = self.get_translation(viz_id)
            self.remove_object(viz_id)
            add_back_viz_objs[name] = before_pos
        self.viz_ids = defaultdict(lambda: None)
        self.grasp_mgr.update()

        if self._concur_render:
            self._prev_sim_obs = self.start_async_render()
            
            for _ in range(self.ac_freq_ratio):
                self.internal_step(-1)

            self._prev_sim_obs = self.get_sensor_observations_async_finish()
            obs = self._sensor_suite.get_observations(self._prev_sim_obs)
        else:
            for _ in range(self.ac_freq_ratio):
                self.internal_step(-1)

            self._prev_sim_obs = self.get_sensor_observations()
            obs = self._sensor_suite.get_observations(self._prev_sim_obs)

        # TODO: Make debug cameras more flexible.
        if "robot_third_rgb" in obs:
            self._should_render_debug = True
            self._try_acquire_context()
            for k, pos in add_back_viz_objs.items():
                self.viz_ids[k] = self.visualize_position(pos, self.viz_ids[k])

            # Also render debug information
            self._create_obj_viz(self.ep_info)

            # Always draw the target
            for obj_handle, _ in self.ep_info["targets"].items():
                self.set_object_bb_draw(
                    True, rom.get_object_by_handle(obj_handle).object_id
                )

            debug_obs = self.get_sensor_observations()
            obs["robot_third_rgb"] = debug_obs["robot_third_rgb"][:, :, :3]

        if self.habitat_config.HABITAT_SIM_V0.get(
            "ENABLE_GFX_REPLAY_SAVE", False
        ):
            self.gfx_replay_manager.save_keyframe()

        return obs

    def visualize_position(self, position, viz_id=None, r=0.05):
        """Adds the sphere object to the specified position for visualization purpose."""

        if viz_id is None:
            if r not in self._viz_templates:
                obj_mgr = self.get_object_template_manager()
                template = obj_mgr.get_template_by_handle(
                    obj_mgr.get_template_handles("sphere")[0]
                )
                template.scale = mn.Vector3(r, r, r)
                self._viz_templates[r] = obj_mgr.register_template(
                    template, "ball_new_viz"
                )
            viz_id = self.add_object(self._viz_templates[r])
            make_render_only(viz_id, self)
        self.set_translation(mn.Vector3(*position), viz_id)

        return viz_id

    def draw_obs(self):
        """Synchronously gets the observation at the current step"""
        # Update the world state to get most recent render
        self.internal_step(-1)

        prev_sim_obs = self.get_sensor_observations()
        obs = self._sensor_suite.get_observations(prev_sim_obs)
        return obs

    def internal_step(self, dt):
        """Never call sim.step_world directly."""

        #optionally step physics and update the robot for benchmarking purposes
        if self.habitat_config.get("STEP_PHYSICS", True):
            self.step_world(dt)
        if self.robot is not None and self.habitat_config.get(
            "UPDATE_ROBOT", True
        ):
            self.robot.update()

    def get_targets(self):
        """
        - Returns: ([idx: int], [goal_pos: list]) The index of the target object
          in self.scene_obj_ids and the 3D goal POSITION, rotation is IGNORED.
          Note that goal_pos is the desired position of the object, not the
          starting position.
        """
        targ_idx, targ_trans = list(zip(*self._get_target_trans()))

        a, b = np.array(targ_idx), [
            np.array(x.translation) for x in targ_trans
        ]
        return a, np.array(b)

    def get_n_targets(self):
        return len(self.ep_info["targets"])

    def get_target_objs_start(self):
        return np.array(self.target_start_pos)

    def get_scene_pos(self):
        return np.array(
            [self.get_translation(idx) for idx in self.scene_obj_ids]
        )
