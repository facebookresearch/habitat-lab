#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import magnum as mn
import numpy as np
import numpy.typing as npt

import habitat_sim
from habitat.config.default import Config
from habitat.core.registry import registry
from habitat.core.simulator import Observations
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.rearrange.marker_info import MarkerInfo
from habitat.tasks.rearrange.rearrange_grasp_manager import (
    RearrangeGraspManager,
)
from habitat.tasks.rearrange.utils import (
    IkHelper,
    get_aabb,
    is_pb_installed,
    make_render_only,
    rearrange_collision,
    rearrange_logger,
)
from habitat_sim.nav import NavMeshSettings
from habitat_sim.physics import JointMotorSettings, MotionType

# flake8: noqa
from habitat_sim.robots import FetchRobot, FetchRobotNoWheels
from habitat_sim.sim import SimulatorBackend


@registry.register_simulator(name="RearrangeSim-v0")
class RearrangeSim(HabitatSim):
    """
    :property ref_handle_to_rigid_obj_id: maps a handle name to the relative position of an object in `self.scene_obj_ids`.
    """

    ref_handle_to_rigid_obj_id: Optional[Dict[str, int]]

    def __init__(self, config: Config):
        super().__init__(config)

        agent_config = self.habitat_config

        agent_cfg = self._get_agent_config()

        self.first_setup = True
        self.ep_info: Optional[Config] = None
        self.prev_loaded_navmesh = None
        self.prev_scene_id = None
        self._is_pb_installed = is_pb_installed()

        # Number of physics updates per action
        self.ac_freq_ratio = agent_config.AC_FREQ_RATIO
        # The physics update time step.
        self.ctrl_freq = agent_config.CTRL_FREQ
        # Effective control speed is (ctrl_freq/ac_freq_ratio)

        self.art_objs: List[habitat_sim.physics.ManagedArticulatedObject] = []
        self._start_art_states: Dict[
            habitat_sim.physics.ManagedArticulatedObject, List[float]
        ] = {}
        self._prev_obj_names: Optional[List[str]] = None
        self.scene_obj_ids: List[int] = []
        # Used to get data from the RL environment class to sensors.
        self._goal_pos = None
        self.viz_ids: Dict[Any, Any] = defaultdict(lambda: None)
        self.ref_handle_to_rigid_obj_id = None
        robot_cls = eval(self.habitat_config.ROBOT_TYPE)
        self.robot = robot_cls(self.habitat_config.ROBOT_URDF, self)
        self._orig_robot_js_start = np.array(self.robot.params.arm_init_params)
        self._markers: Dict[str, MarkerInfo] = {}

        self._viz_templates: Dict[str, Any] = {}
        self._viz_handle_to_template: Dict[str, float] = {}
        self._viz_objs: Dict[str, Any] = {}

        self._ik_helper: Optional[IkHelper] = None

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
        for target_handle, trans in self._targets.items():
            targ_idx = self.scene_obj_ids.index(
                rom.get_object_by_handle(target_handle).object_id
            )
            target_trans.append((targ_idx, trans))
        return target_trans

    def _try_acquire_context(self):
        if self.habitat_config.CONCUR_RENDER:
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

    def add_markers(self, ep_info: Config):
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

    @property
    def ik_helper(self):
        if not self._is_pb_installed:
            raise ImportError(
                "Need to install PyBullet to use IK (`pip install pybullet==3.0.4`)"
            )
        return self._ik_helper

    def reset(self):
        SimulatorBackend.reset(self)
        for i in range(len(self.agents)):
            self.reset_agent(i)
        return None

    def reconfigure(self, config: Config):
        ep_info = config["ep_info"][0]
        self.instance_handle_to_ref_handle = ep_info["info"]["object_labels"]

        config["SCENE"] = ep_info["scene_id"]

        super().reconfigure(config, should_close_on_new_scene=False)

        self.ref_handle_to_rigid_obj_id = {}

        self.ep_info = ep_info
        self._try_acquire_context()

        new_scene = self.prev_scene_id != ep_info["scene_id"]

        if new_scene:
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
        self._viz_handle_to_template = {}

        # Set the default articulated object joint state.
        for ao, set_joint_state in self._start_art_states.items():
            ao.clear_joint_states()
            ao.joint_positions = set_joint_state

        # Load specified articulated object states from episode config
        self._set_ao_states_from_ep(ep_info)

        use_arm_start = self._orig_robot_js_start + (
            self.habitat_config.get("ROBOT_JOINT_START_NOISE", 0.0)
            * np.random.randn(self._orig_robot_js_start.shape[0])
        )
        self.robot.params.arm_init_params = use_arm_start
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
        self._setup_targets()

        self.add_markers(ep_info)

        # auto-sleep rigid objects as optimization
        if self.habitat_config.AUTO_SLEEP:
            self.sleep_all_objects()

        if new_scene:
            self._load_navmesh()

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
                for t_handle, _ in self._targets.items()
            ]
        )

        if self.first_setup:
            self.first_setup = False
            ik_arm_urdf = self.habitat_config.get("IK_ARM_URDF", None)
            if ik_arm_urdf is not None and self._is_pb_installed:
                self._ik_helper = IkHelper(
                    self.habitat_config.IK_ARM_URDF,
                    np.array(self.robot.params.arm_init_params),
                )
            # Capture the starting art states
            self._start_art_states = {
                ao: ao.joint_positions for ao in self.art_objs
            }

    def set_robot_base_to_random_point(
        self, max_attempts: int = 50
    ) -> Tuple[np.ndarray, float]:
        """
        :returns: The set base position and rotation
        """
        for attempt_i in range(max_attempts):
            start_pos = self.pathfinder.get_random_navigable_point()

            start_pos = self.safe_snap_point(start_pos)
            start_rot = np.random.uniform(0, 2 * np.pi)

            self.robot.base_pos = start_pos
            self.robot.base_rot = start_rot
            self.perform_discrete_collision_detection()
            did_collide, details = rearrange_collision(
                self,
                True,
                ignore_base=False,
            )
            if not did_collide:
                break
        if attempt_i == max_attempts - 1:
            rearrange_logger.warning(
                f"Could not find a collision free start for {self.ep_info['episode_id']}"
            )
        return start_pos, start_rot

    def _setup_targets(self):
        self._targets = {}
        for target_handle, transform in self.ep_info["targets"].items():
            self._targets[target_handle] = mn.Matrix4(
                [[transform[j][i] for j in range(4)] for i in range(4)]
            )

    def _load_navmesh(self):
        scene_name = self.ep_info["scene_id"].split("/")[-1].split(".")[0]
        base_dir = osp.join(*self.ep_info["scene_id"].split("/")[:2])

        navmesh_path = osp.join(base_dir, "navmeshes", scene_name + ".navmesh")
        self.pathfinder.load_nav_mesh(navmesh_path)

        self._navmesh_vertices = np.stack(
            self.pathfinder.build_navmesh_vertices(), axis=0
        )
        self._island_sizes = [
            self.pathfinder.island_radius(p) for p in self._navmesh_vertices
        ]
        self._max_island_size = max(self._island_sizes)

    def _clear_objects(self, should_add_objects: bool) -> None:
        rom = self.get_rigid_object_manager()

        # Clear all the rigid objects.
        if should_add_objects:
            for scene_obj_id in self.scene_obj_ids:
                if rom.get_library_has_id(scene_obj_id):
                    rom.remove_object_by_id(scene_obj_id)
            self.scene_obj_ids = []

        # Reset all marker visualization points
        for obj_id in self.viz_ids.values():
            if rom.get_library_has_id(obj_id):
                rom.remove_object_by_id(obj_id)
        self.viz_ids = defaultdict(lambda: None)

        # Remove all object mesh visualizations.
        for viz_obj in self._viz_objs.values():
            if rom.get_library_has_id(viz_obj.object_id):
                rom.remove_object_by_id(viz_obj.object_id)
        self._viz_objs = {}

        # Do not remove the articulated objects from the scene, these are
        # managed by the underlying sim.
        self.art_objs = []

    def _set_ao_states_from_ep(self, ep_info: Config) -> None:
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

    def safe_snap_point(self, pos: np.ndarray) -> np.ndarray:
        """
        snap_point can return nan which produces hard to catch errors.
        """
        new_pos = self.pathfinder.snap_point(pos)
        island_radius = self.pathfinder.island_radius(new_pos)

        if np.isnan(new_pos[0]) or island_radius != self._max_island_size:
            # The point is not valid or not in a different island. Find a
            # different point nearby that is on a different island and is
            # valid.
            for _ in range(10):
                new_pos = self.pathfinder.get_random_navigable_point_near(
                    pos, 1.5, 1000
                )
                island_radius = self.pathfinder.island_radius(new_pos)
                if island_radius == self._max_island_size:
                    break

        if np.isnan(new_pos[0]) or island_radius != self._max_island_size:
            # This is a last resort, take a navmesh vertex that is closest
            use_verts = [
                x
                for s, x in zip(self._island_sizes, self._navmesh_vertices)
                if s == self._max_island_size
            ]
            distances = np.linalg.norm(
                np.array(pos).reshape(1, 3) - use_verts, axis=-1
            )
            closest_idx = np.argmin(distances)
            new_pos = self._navmesh_vertices[closest_idx]

        return new_pos

    def _add_objs(self, ep_info: Config, should_add_objects: bool) -> None:
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
                ), f"Object attributes not uniquely matched to shortened handle. '{obj_handle}' matched to {matching_templates}. TODO: relative paths as handles should fix some duplicates. For now, try renaming objects to avoid collision."
                ro = rom.add_object_by_template_handle(
                    list(matching_templates.keys())[0]
                )
            else:
                ro = rom.get_object_by_id(self.scene_obj_ids[i])

            # The saved matrices need to be flipped when reloading.
            ro.transformation = mn.Matrix4(
                [[transform[j][i] for j in range(4)] for i in range(4)]
            )
            ro.angular_velocity = mn.Vector3.zero_init()
            ro.linear_velocity = mn.Vector3.zero_init()

            other_obj_handle = (
                obj_handle.split(".")[0] + f"_:{obj_counts[obj_handle]:04d}"
            )
            if should_add_objects:
                self.scene_obj_ids.append(ro.object_id)

            if other_obj_handle in self.instance_handle_to_ref_handle:
                ref_handle = self.instance_handle_to_ref_handle[
                    other_obj_handle
                ]
                rel_idx = self.scene_obj_ids.index(ro.object_id)
                self.ref_handle_to_rigid_obj_id[ref_handle] = rel_idx
            obj_counts[obj_handle] += 1

        ao_mgr = self.get_articulated_object_manager()
        for aoi_handle in ao_mgr.get_object_handles():
            self.art_objs.append(ao_mgr.get_object_by_handle(aoi_handle))

    def _create_obj_viz(self, ep_info: Config):
        """
        Adds a visualization of the goal for each of the target objects in the
        scene. This is the same as the target object, but is a render only
        object. This also places dots around the bounding box of the object to
        further distinguish the goal from the target object.
        """
        for marker_name, m in self._markers.items():
            m_T = m.get_current_transform()
            self.viz_ids[marker_name] = self.visualize_position(
                m_T.translation, self.viz_ids[marker_name]
            )

        rom = self.get_rigid_object_manager()
        obj_attr_mgr = self.get_object_template_manager()
        for target_handle, transform in self._targets.items():
            # Visualize the goal of the object
            if self.habitat_config.DEBUG_RENDER_GOAL:
                new_target_handle = (
                    target_handle.split("_:")[0] + ".object_config.json"
                )
                matching_templates = (
                    obj_attr_mgr.get_templates_by_handle_substring(
                        new_target_handle
                    )
                )
                ro = rom.add_object_by_template_handle(
                    list(matching_templates.keys())[0]
                )
                self.set_object_bb_draw(True, ro.object_id)
                ro.transformation = transform
                make_render_only(ro, self)
                bb = get_aabb(ro.object_id, self, True)
                bb_viz_name1 = target_handle + "_bb1"
                bb_viz_name2 = target_handle + "_bb2"
                viz_r = 0.01
                self.viz_ids[bb_viz_name1] = self.visualize_position(
                    bb.front_bottom_right, self.viz_ids[bb_viz_name1], viz_r
                )
                self.viz_ids[bb_viz_name2] = self.visualize_position(
                    bb.back_top_left, self.viz_ids[bb_viz_name2], viz_r
                )

                self._viz_objs[target_handle] = ro

            # Draw a bounding box around the target object
            self.set_object_bb_draw(
                True, rom.get_object_by_handle(target_handle).object_id
            )

    def capture_state(self, with_robot_js=False) -> Dict[str, Any]:
        """
        Record and return a dict of state info.

        :param with_robot_js: If true, state dict includes robot joint positions in addition.

        State info dict includes:
         - Robot transform
         - a list of ArticulatedObject transforms
         - a list of RigidObject transforms
         - a list of ArticulatedObject joint states
         - the object id of currently grasped object (or None)
         - (optionally) the robot's joint positions
        """
        # Don't need to capture any velocity information because this will
        # automatically be set to 0 in `set_state`.
        robot_T = self.robot.sim_obj.transformation
        art_T = [ao.transformation for ao in self.art_objs]
        rom = self.get_rigid_object_manager()
        static_T = [
            rom.get_object_by_id(i).transformation for i in self.scene_obj_ids
        ]
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

    def set_state(self, state: Dict[str, Any], set_hold=False) -> None:
        """
        Sets the simulation state from a cached state info dict. See capture_state().

          :param set_hold: If true this will set the snapped object from the `state`.

          TODO: This should probably be True by default, but I am not sure the effect
          it will have.
        """
        rom = self.get_rigid_object_manager()
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
            # reset object transform
            obj = rom.get_object_by_id(i)
            obj.transformation = T
            obj.linear_velocity = mn.Vector3()
            obj.angular_velocity = mn.Vector3()

        for p, ao in zip(state["art_pos"], self.art_objs):
            ao.joint_positions = p

        if set_hold:
            if state["obj_hold"] is not None:
                self.internal_step(-1)
                self.grasp_mgr.snap_to_obj(state["obj_hold"])
            else:
                self.grasp_mgr.desnap(True)

    def step(self, action: Union[str, int]) -> Observations:
        rom = self.get_rigid_object_manager()

        if self.habitat_config.DEBUG_RENDER:
            rom = self.get_rigid_object_manager()
            self._try_acquire_context()
            # Don't draw bounding boxes over target objects.
            for obj_handle, _ in self._targets.items():
                self.set_object_bb_draw(
                    False, rom.get_object_by_handle(obj_handle).object_id
                )

            # Remove viz objects
            for obj in self._viz_objs.values():
                if obj is not None and rom.get_library_has_id(obj.object_id):
                    rom.remove_object_by_id(obj.object_id)
            self._viz_objs = {}

            # Remove all visualized positions
            add_back_viz_objs = {}
            for name, viz_id in self.viz_ids.items():
                if viz_id is None:
                    continue
                viz_obj = rom.get_object_by_id(viz_id)
                before_pos = viz_obj.translation
                rom.remove_object_by_id(viz_id)
                r = self._viz_handle_to_template[viz_id]
                add_back_viz_objs[name] = (before_pos, r)
            self.viz_ids = defaultdict(lambda: None)

        self.grasp_mgr.update()
        if self.robot is not None and self.habitat_config.UPDATE_ROBOT:
            self.robot.update()

        if self.habitat_config.CONCUR_RENDER:
            self._prev_sim_obs = self.start_async_render()

            for _ in range(self.ac_freq_ratio):
                self.internal_step(-1, update_robot=False)

            self._prev_sim_obs = self.get_sensor_observations_async_finish()
            obs = self._sensor_suite.get_observations(self._prev_sim_obs)
        else:
            for _ in range(self.ac_freq_ratio):
                self.internal_step(-1, update_robot=False)
            self._prev_sim_obs = self.get_sensor_observations()
            obs = self._sensor_suite.get_observations(self._prev_sim_obs)

        if self.habitat_config.NEEDS_MARKERS:
            self._update_markers()

        # TODO: Make debug cameras more flexible
        if "robot_third_rgb" in obs and self.habitat_config.DEBUG_RENDER:
            self._try_acquire_context()
            for k, (pos, r) in add_back_viz_objs.items():
                viz_id = self.viz_ids[k]

                self.viz_ids[k] = self.visualize_position(
                    pos, self.viz_ids[k], r=r
                )

            # Also render debug information
            self._create_obj_viz(self.ep_info)

            debug_obs = self.get_sensor_observations()
            obs["robot_third_rgb"] = debug_obs["robot_third_rgb"][:, :, :3]

        if self.habitat_config.HABITAT_SIM_V0.get(
            "ENABLE_GFX_REPLAY_SAVE", False
        ):
            self.gfx_replay_manager.save_keyframe()

        return obs

    def visualize_position(
        self,
        position: np.ndarray,
        viz_id: Optional[int] = None,
        r: float = 0.05,
    ) -> int:
        """Adds the sphere object to the specified position for visualization purpose."""

        template_mgr = self.get_object_template_manager()
        rom = self.get_rigid_object_manager()
        viz_obj = None
        if viz_id is None:
            if r not in self._viz_templates:
                template = template_mgr.get_template_by_handle(
                    template_mgr.get_template_handles("sphere")[0]
                )
                template.scale = mn.Vector3(r, r, r)
                self._viz_templates[str(r)] = template_mgr.register_template(
                    template, "ball_new_viz_" + str(r)
                )
            viz_obj = rom.add_object_by_template_id(
                self._viz_templates[str(r)]
            )
            make_render_only(viz_obj, self)
            self._viz_handle_to_template[viz_obj.object_id] = r
        else:
            viz_obj = rom.get_object_by_id(viz_id)

        viz_obj.translation = mn.Vector3(*position)
        return viz_obj.object_id

    def internal_step(
        self, dt: Union[int, float], update_robot: bool = True
    ) -> None:
        """Step the world and update the robot.

        :param dt: Timestep by which to advance the world. Multiple physics substeps can be excecuted within a single timestep. -1 indicates a single physics substep.

        Never call sim.step_world directly or miss updating the robot.
        """

        # optionally step physics and update the robot for benchmarking purposes
        if self.habitat_config.get("STEP_PHYSICS", True):
            self.step_world(dt)
            if (
                update_robot
                and self.robot is not None
                and self.habitat_config.UPDATE_ROBOT
            ):
                self.robot.update()

    def get_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get a mapping of object ids to goal positions for rearrange targets.

        :return: ([idx: int], [goal_pos: list]) The index of the target object
          in self.scene_obj_ids and the 3D goal POSITION, rotation is IGNORED.
          Note that goal_pos is the desired position of the object, not the
          starting position.
        """
        targ_idx, targ_trans = list(zip(*self._get_target_trans()))

        a, b = np.array(targ_idx), [
            np.array(x.translation) for x in targ_trans
        ]
        return a, np.array(b)

    def get_n_targets(self) -> int:
        """Get the number of rearrange targets."""
        return len(self.ep_info["targets"])

    def get_target_objs_start(self) -> np.ndarray:
        """Get the initial positions of all objects targeted for rearrangement as a numpy array."""
        return self.target_start_pos

    def get_scene_pos(self) -> np.ndarray:
        """Get the positions of all clutter RigidObjects in the scene as a numpy array."""
        rom = self.get_rigid_object_manager()
        return np.array(
            [
                rom.get_object_by_id(idx).translation
                for idx in self.scene_obj_ids
            ]
        )
