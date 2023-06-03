#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
import time
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import magnum as mn
import numpy as np
import numpy.typing as npt

import habitat_sim

# flake8: noqa
from habitat.articulated_agents.robots import FetchRobot, FetchRobotNoWheels
from habitat.config import read_write
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, Observations
from habitat.datasets.rearrange.rearrange_dataset import RearrangeEpisode
from habitat.datasets.rearrange.samplers.receptacle import (
    AABBReceptacle,
    find_receptacles,
)
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.rearrange.articulated_agent_manager import (
    ArticulatedAgentManager,
)
from habitat.tasks.rearrange.marker_info import MarkerInfo
from habitat.tasks.rearrange.rearrange_grasp_manager import (
    RearrangeGraspManager,
)
from habitat.tasks.rearrange.utils import (
    get_aabb,
    make_render_only,
    rearrange_collision,
    rearrange_logger,
)
from habitat_sim.nav import NavMeshSettings
from habitat_sim.physics import CollisionGroups, JointMotorSettings, MotionType
from habitat_sim.sim import SimulatorBackend
from habitat_sim.utils.common import quat_from_magnum

if TYPE_CHECKING:
    from omegaconf import DictConfig


@registry.register_simulator(name="RearrangeSim-v0")
class RearrangeSim(HabitatSim):
    def __init__(self, config: "DictConfig"):
        if len(config.agents) > 1:
            with read_write(config):
                for agent_name, agent_cfg in config.agents.items():
                    # using list to create a copy of the sim_sensors keys since we will be
                    # editing the sim_sensors config
                    sensor_keys = list(agent_cfg.sim_sensors.keys())
                    for sensor_key in sensor_keys:
                        sensor_config = agent_cfg.sim_sensors.pop(sensor_key)
                        sensor_config.uuid = (
                            f"{agent_name}_{sensor_config.uuid}"
                        )
                        agent_cfg.sim_sensors[
                            f"{agent_name}_{sensor_key}"
                        ] = sensor_config

        super().__init__(config)

        self.first_setup = True
        self.ep_info: Optional[RearrangeEpisode] = None
        self.prev_loaded_navmesh = None
        self.prev_scene_id: Optional[str] = None

        # Number of physics updates per action
        self.ac_freq_ratio = self.habitat_config.ac_freq_ratio
        # The physics update time step.
        self.ctrl_freq = self.habitat_config.ctrl_freq
        # Effective control speed is (ctrl_freq/ac_freq_ratio)

        self.art_objs: List[habitat_sim.physics.ManagedArticulatedObject] = []
        self._start_art_states: Dict[
            habitat_sim.physics.ManagedArticulatedObject, List[float]
        ] = {}
        self._prev_obj_names: Optional[List[str]] = None
        self._scene_obj_ids: List[int] = []
        self._receptacles: Dict[str, mn.Range3D] = {}
        # Used to get data from the RL environment class to sensors.
        self._goal_pos = None
        self.viz_ids: Dict[Any, Any] = defaultdict(lambda: None)
        self._handle_to_object_id: Dict[str, int] = {}
        self._markers: Dict[str, MarkerInfo] = {}

        self._viz_templates: Dict[str, Any] = {}
        self._viz_handle_to_template: Dict[str, float] = {}
        self._viz_objs: Dict[str, Any] = {}
        self._draw_bb_objs: List[int] = []

        self.agents_mgr = ArticulatedAgentManager(self.habitat_config, self)

        self._debug_render_articulated_agent = (
            self.habitat_config.debug_render_articulated_agent
        )
        self._debug_render_goal = self.habitat_config.debug_render_goal
        self._debug_render = self.habitat_config.debug_render
        self._concur_render = self.habitat_config.concur_render
        self._enable_gfx_replay_save = (
            self.habitat_config.habitat_sim_v0.enable_gfx_replay_save
        )
        self._needs_markers = self.habitat_config.needs_markers
        self._update_articulated_agent = (
            self.habitat_config.update_articulated_agent
        )
        self._step_physics = self.habitat_config.step_physics
        self._additional_object_paths = (
            self.habitat_config.additional_object_paths
        )
        self._kinematic_mode = self.habitat_config.kinematic_mode
        self._backend_runtime_perf_stat_names = (
            # super().get_runtime_perf_stat_names()
        )
        self._extra_runtime_perf_stats: Dict[str, Any] = {}

    @property
    def receptacles(self) -> Dict[str, AABBReceptacle]:
        return self._receptacles

    @property
    def handle_to_object_id(self) -> Dict[str, int]:
        """
        Maps a handle name to the relative position of an object in `self._scene_obj_ids`.
        """
        return self._handle_to_object_id

    @property
    def draw_bb_objs(self) -> List[int]:
        """
        Simulator object indices of objects to draw bounding boxes around if
        debug render is enabled. By default, this is populated with all target
        objects.
        """
        return self._draw_bb_objs

    @property
    def scene_obj_ids(self) -> List[int]:
        """
        The simulator rigid body IDs of all objects in the scene.
        """
        return self._scene_obj_ids

    @property
    def articulated_agent(self):
        if len(self.agents_mgr) > 1:
            raise ValueError(
                f"Cannot access `sim.articulated_agent` with multiple articulated agents"
            )
        return self.agents_mgr[0].articulated_agent

    @property
    def grasp_mgr(self):
        if len(self.agents_mgr) > 1:
            raise ValueError(
                f"Cannot access `sim.grasp_mgr` with multiple articulated_agents"
            )
        return self.agents_mgr[0].grasp_mgr

    @property
    def grasp_mgrs(self):
        if len(self.agents_mgr) > 1:
            raise ValueError(
                f"Cannot access `sim.grasp_mgr` with multiple articulated_agents"
            )
        return self.agents_mgr[0].grasp_mgrs

    def _get_target_trans(self):
        """
        This is how the target transforms should be accessed since
        multiprocessing does not allow pickling.
        """
        # Preprocess the ep_info making necessary datatype conversions.
        target_trans = []
        rom = self.get_rigid_object_manager()
        for target_handle, trans in self._targets.items():
            targ_idx = self._scene_obj_ids.index(
                rom.get_object_by_handle(target_handle).object_id
            )
            target_trans.append((targ_idx, trans))
        return target_trans

    def _try_acquire_context(self):
        if self._concur_render:
            self.renderer.acquire_gl_context()

    def _sleep_all_objects(self):
        """
        De-activate (sleep) all rigid objects in the scene, assuming they are already in a dynamically stable state.
        """
        rom = self.get_rigid_object_manager()
        for _, ro in rom.get_objects_by_handle_substring().items():
            ro.awake = False

        aom = self.get_articulated_object_manager()
        for _, ao in aom.get_objects_by_handle_substring().items():
            ao.awake = False

    def _add_markers(self, ep_info: RearrangeEpisode):
        self._markers = {}
        aom = self.get_articulated_object_manager()
        for marker in ep_info.markers:
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

    def reset(self):
        SimulatorBackend.reset(self)
        for i in range(len(self.agents)):
            self.reset_agent(i)
        return None

    def reconfigure(self, config: "DictConfig", ep_info: RearrangeEpisode):
        self._handle_to_goal_name = ep_info.info["object_labels"]

        t_start = time.time()

        with read_write(config):
            config["scene"] = ep_info.scene_id

        self.ep_info = ep_info
        new_scene = self.prev_scene_id != ep_info.scene_id
        if new_scene:
            self._prev_obj_names = None

        # Only remove and re-add objects if we have a new set of objects.
        obj_names = [x[0] for x in ep_info.rigid_objs]
        should_add_objects = self._prev_obj_names != obj_names
        self._prev_obj_names = obj_names

        self._clear_objects(should_add_objects)

        super().reconfigure(config, should_close_on_new_scene=False)
        self._try_acquire_context()
        self.agents_mgr.reconfigure(new_scene)

        self.prev_scene_id = ep_info.scene_id
        self._viz_templates = {}
        self._viz_handle_to_template = {}

        # Set the default articulated object joint state.
        for ao, set_joint_state in self._start_art_states.items():
            ao.clear_joint_states()
            ao.joint_positions = set_joint_state

        # Load specified articulated object states from episode config
        self._set_ao_states_from_ep(ep_info)

        self.agents_mgr.post_obj_load_reconfigure()

        # add episode clutter objects additional to base scene objects
        if self.habitat_config.load_objs:
            self._add_objs(ep_info, should_add_objects)
        self._setup_targets(ep_info)

        self._add_markers(ep_info)

        # auto-sleep rigid objects as optimization
        if self.habitat_config.auto_sleep:
            self._sleep_all_objects()

        rom = self.get_rigid_object_manager()
        self._obj_orig_motion_types = {
            handle: ro.motion_type
            for handle, ro in rom.get_objects_by_handle_substring().items()
        }

        if new_scene:
            self._load_navmesh(ep_info)

        # Get the starting positions of the target objects.
        scene_pos = self.get_scene_pos()
        self.target_start_pos = np.array(
            [
                scene_pos[
                    self._scene_obj_ids.index(
                        rom.get_object_by_handle(t_handle).object_id
                    )
                ]
                for t_handle, _ in self._targets.items()
            ]
        )

        self._draw_bb_objs = [
            rom.get_object_by_handle(obj_handle).object_id
            for obj_handle in self._targets
        ]

        if self.first_setup:
            self.first_setup = False
            self.agents_mgr.first_setup()
            # Capture the starting art states
            self._start_art_states = {
                ao: ao.joint_positions for ao in self.art_objs
            }

        # Add the rigid object id for the semantic map
        rom = self.get_rigid_object_manager()
        for i, handle in enumerate(rom.get_object_handles()):
            obj = rom.get_object_by_handle(handle)
            for node in obj.visual_scene_nodes:
                node.semantic_id = (
                    obj.object_id + self.habitat_config.object_ids_start
                )
        self.add_perf_timing("reconfigure", t_start)

    def get_agent_data(self, agent_idx: Optional[int]):
        if agent_idx is None:
            return self.agents_mgr[0]
        else:
            return self.agents_mgr[agent_idx]

    @property
    def num_articulated_agents(self):
        return len(self.agents_mgr)

    def set_articulated_agent_base_to_random_point(
        self,
        max_attempts: int = 50,
        agent_idx: Optional[int] = None,
        filter_func: Optional[Callable[[np.ndarray, float], bool]] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        :returns: The set base position and rotation
        """
        articulated_agent = self.get_agent_data(agent_idx).articulated_agent

        for attempt_i in range(max_attempts):
            start_pos = self.pathfinder.get_random_navigable_point()

            start_pos = self.safe_snap_point(start_pos)
            start_rot = np.random.uniform(0, 2 * np.pi)

            if filter_func is not None and not filter_func(
                start_pos, start_rot
            ):
                continue

            articulated_agent.base_pos = start_pos
            articulated_agent.base_rot = start_rot
            self.perform_discrete_collision_detection()
            did_collide, _ = rearrange_collision(
                self, True, ignore_base=False, agent_idx=agent_idx
            )
            if not did_collide:
                break
        if attempt_i == max_attempts - 1:
            rearrange_logger.warning(
                f"Could not find a collision free start for {self.ep_info.episode_id}"
            )
        return start_pos, start_rot

    def _setup_targets(self, ep_info):
        self._targets = {}
        for target_handle, transform in ep_info.targets.items():
            self._targets[target_handle] = mn.Matrix4(
                [[transform[j][i] for j in range(4)] for i in range(4)]
            )

    def _load_navmesh(self, ep_info):
        scene_name = ep_info.scene_id.split("/")[-1].split(".")[0]

        if "fpss" in ep_info.scene_id.split("/"):
            # For FP scenes, we use different path structure than for other scenes.
            base_dir = osp.join(*ep_info.scene_id.split("/")[:3])
        else:
            base_dir = osp.join(*ep_info.scene_id.split("/")[:2])

        navmesh_path = osp.join(base_dir, "navmeshes", scene_name + ".navmesh")
        # If we cannot load the navmesh, try generarting navmesh on the fly.
        if osp.exists(navmesh_path):
            self.pathfinder.load_nav_mesh(navmesh_path)
        else:
            navmesh_settings = NavMeshSettings()
            navmesh_settings.set_defaults()

            if hasattr(self.habitat_config.agents, "agent_0"):
                radius = self.habitat_config.agents.agent_0.radius
                height = self.habitat_config.agents.agent_0.height
                max_climb = self.habitat_config.agents.agent_0.max_climb
            elif hasattr(self.habitat_config.agents, "main_agent"):
                radius = self.habitat_config.agents.main_agent.radius
                height = self.habitat_config.agents.main_agent.height
                max_climb = self.habitat_config.agents.main_agent.max_climb
            else:
                raise ValueError(f"Cannot find agent parameters.")
            navmesh_settings.agent_radius = radius
            navmesh_settings.agent_height = height
            navmesh_settings.agent_max_climb = max_climb
            self.recompute_navmesh(
                self.pathfinder,
                navmesh_settings,
                include_static_objects=True,
            )
            os.makedirs(osp.dirname(navmesh_path), exist_ok=True)
            self.pathfinder.save_nav_mesh(navmesh_path)

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
            for scene_obj_id in self._scene_obj_ids:
                if not rom.get_library_has_id(scene_obj_id):
                    continue
                rom.remove_object_by_id(scene_obj_id)
            self._scene_obj_ids = []

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

    def _set_ao_states_from_ep(self, ep_info: RearrangeEpisode) -> None:
        """
        Sets the ArticulatedObject states for the episode which are differ from base scene state.
        """
        aom = self.get_articulated_object_manager()
        for aoi_handle, joint_states in ep_info.ao_states.items():
            ao = aom.get_object_by_handle(aoi_handle)
            ao_pose = ao.joint_positions
            for link_ix, joint_state in joint_states.items():
                joint_position_index = ao.get_link_joint_pos_offset(
                    int(link_ix)
                )
                ao_pose[joint_position_index] = joint_state
            ao.joint_positions = ao_pose

    def is_point_within_bounds(self, pos):
        lower_bound, upper_bound = self.pathfinder.get_bounds()
        return all(lower_bound <= pos) and all(upper_bound >= pos)

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
            new_pos = self.pathfinder.get_random_navigable_point_near(
                pos, 1.5, 1000
            )
            island_radius = self.pathfinder.island_radius(new_pos)

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

    def _add_objs(
        self, ep_info: RearrangeEpisode, should_add_objects: bool
    ) -> None:
        # Load clutter objects:
        rom = self.get_rigid_object_manager()
        obj_counts: Dict[str, int] = defaultdict(int)

        self._handle_to_object_id = {}
        self._receptacles = {}
        if should_add_objects:
            self._scene_obj_ids = []

        for i, (obj_handle, transform) in enumerate(ep_info.rigid_objs):
            if should_add_objects:
                template = None
                for obj_path in self._additional_object_paths:
                    template = osp.join(obj_path, obj_handle)
                    if osp.isfile(template):
                        break
                assert (
                    template is not None
                ), f"Could not find config file for object {obj_handle}"

                ro = rom.add_object_by_template_handle(template)
            else:
                ro = rom.get_object_by_id(self._scene_obj_ids[i])

            # The saved matrices need to be flipped when reloading.
            ro.transformation = mn.Matrix4(
                [[transform[j][i] for j in range(4)] for i in range(4)]
            )
            ro.angular_velocity = mn.Vector3.zero_init()
            ro.linear_velocity = mn.Vector3.zero_init()

            other_obj_handle = (
                obj_handle.split(".")[0] + f"_:{obj_counts[obj_handle]:04d}"
            )
            if self._kinematic_mode:
                ro.motion_type = habitat_sim.physics.MotionType.KINEMATIC
                ro.collidable = False

            if should_add_objects:
                self._scene_obj_ids.append(ro.object_id)
            rel_idx = self._scene_obj_ids.index(ro.object_id)
            self._handle_to_object_id[other_obj_handle] = rel_idx

            if other_obj_handle in self._handle_to_goal_name:
                ref_handle = self._handle_to_goal_name[other_obj_handle]
                self._handle_to_object_id[ref_handle] = rel_idx

            obj_counts[obj_handle] += 1

        all_receps = find_receptacles(self)
        for recep in all_receps:
            recep = cast(AABBReceptacle, recep)
            local_bounds = recep.bounds
            global_T = recep.get_global_transform(self)
            self._receptacles[recep.name] = mn.Range3D(
                global_T.transform_point(local_bounds.min),
                global_T.transform_point(local_bounds.max),
            )

        ao_mgr = self.get_articulated_object_manager()
        articulated_agent_art_handles = [
            articulated_agent.sim_obj.handle
            for articulated_agent in self.agents_mgr.articulated_agents_iter
        ]
        for aoi_handle in ao_mgr.get_object_handles():
            ao = ao_mgr.get_object_by_handle(aoi_handle)
            if (
                self._kinematic_mode
                and ao.handle not in articulated_agent_art_handles
            ):
                ao.motion_type = habitat_sim.physics.MotionType.KINEMATIC
            self.art_objs.append(ao)

    def _create_obj_viz(self):
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

        # Enable BB render for the debug render call.
        for obj_id in self._draw_bb_objs:
            self.set_object_bb_draw(True, obj_id)

        if self._debug_render_goal:
            for target_handle, transform in self._targets.items():
                # Visualize the goal of the object
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

    def capture_state(self, with_articulated_agent_js=False) -> Dict[str, Any]:
        """
        Record and return a dict of state info.

        :param with_articulated_agent_js: If true, state dict includes articulated_agent joint positions in addition.

        State info dict includes:
         - Robot transform
         - a list of ArticulatedObject transforms
         - a list of RigidObject transforms
         - a list of ArticulatedObject joint states
         - the object id of currently grasped object (or None)
         - (optionally) the articulated_agent's joint positions
        """
        # Don't need to capture any velocity information because this will
        # automatically be set to 0 in `set_state`.
        articulated_agent_T = [
            articulated_agent.sim_obj.transformation
            for articulated_agent in self.agents_mgr.articulated_agents_iter
        ]
        art_T = [ao.transformation for ao in self.art_objs]
        rom = self.get_rigid_object_manager()
        static_T = [
            rom.get_object_by_id(i).transformation for i in self._scene_obj_ids
        ]
        art_pos = [ao.joint_positions for ao in self.art_objs]

        articulated_agent_js = [
            articulated_agent.sim_obj.joint_positions
            for articulated_agent in self.agents_mgr.articulated_agents_iter
        ]

        ret = {
            "articulated_agent_T": articulated_agent_T,
            "art_T": art_T,
            "static_T": static_T,
            "art_pos": art_pos,
            "obj_hold": [
                grasp_mgr.snap_idx for grasp_mgr in self.agents_mgr.grasp_iter
            ],
        }
        if with_articulated_agent_js:
            ret["articulated_agent_js"] = articulated_agent_js
        return ret

    def set_state(self, state: Dict[str, Any], set_hold=False) -> None:
        """
        Sets the simulation state from a cached state info dict. See capture_state().

          :param set_hold: If true this will set the snapped object from the `state`.

          TODO: This should probably be True by default, but I am not sure the effect
          it will have.
        """
        rom = self.get_rigid_object_manager()

        if state["articulated_agent_T"] is not None:
            for articulated_agent_T, robot in zip(
                state["articulated_agent_T"],
                self.agents_mgr.articulated_agents_iter,
            ):
                robot.sim_obj.transformation = articulated_agent_T
                n_dof = len(robot.sim_obj.joint_forces)
                robot.sim_obj.joint_forces = np.zeros(n_dof)
                robot.sim_obj.joint_velocities = np.zeros(n_dof)

        if "articulated_agent_js" in state:
            for articulated_agent_js, robot in zip(
                state["articulated_agent_js"],
                self.agents_mgr.articulated_agents_iter,
            ):
                robot.sim_obj.joint_positions = articulated_agent_js

        for T, ao in zip(state["art_T"], self.art_objs):
            ao.transformation = T

        for T, i in zip(state["static_T"], self._scene_obj_ids):
            # reset object transform
            obj = rom.get_object_by_id(i)
            obj.transformation = T
            obj.linear_velocity = mn.Vector3()
            obj.angular_velocity = mn.Vector3()

        for p, ao in zip(state["art_pos"], self.art_objs):
            ao.joint_positions = p

        if set_hold:
            if state["obj_hold"] is not None:
                for obj_hold_state, grasp_mgr in zip(
                    state["obj_hold"], self.agents_mgr.grasp_iter
                ):
                    self.internal_step(-1)
                    grasp_mgr.snap_to_obj(obj_hold_state)
            else:
                for grasp_mgr in self.agents_mgr.grasp_iter:
                    grasp_mgr.desnap(True)

    def get_agent_state(self, agent_id: int = 0) -> habitat_sim.AgentState:
        articulated_agent = self.get_agent_data(agent_id).articulated_agent
        rotation = mn.Quaternion.rotation(
            mn.Rad(articulated_agent.base_rot) - mn.Rad(0 * np.pi / 2),
            mn.Vector3(0, 1, 0),
        )
        rot_offset = mn.Quaternion.rotation(
            mn.Rad(-np.pi / 2), mn.Vector3(0, 1, 0)
        )
        return AgentState(
            articulated_agent.base_pos,
            quat_from_magnum(articulated_agent.sim_obj.rotation * rot_offset),
        )

    def step(self, action: Union[str, int]) -> Observations:
        rom = self.get_rigid_object_manager()

        if self._debug_render:
            if self._debug_render_articulated_agent:
                self.agents_mgr.update_debug()
            rom = self.get_rigid_object_manager()
            self._try_acquire_context()

            # Disable BB drawing for observation render
            for obj_id in self._draw_bb_objs:
                self.set_object_bb_draw(False, obj_id)

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

        self.maybe_update_articulated_agent()

        if self._concur_render:
            self._prev_sim_obs = self.start_async_render()

            for _ in range(self.ac_freq_ratio):
                self.internal_step(-1, update_articulated_agent=False)

            t_start = time.time()
            self._prev_sim_obs = self.get_sensor_observations_async_finish()
            self.add_perf_timing(
                "get_sensor_observations_async_finish", t_start
            )

            obs = self._sensor_suite.get_observations(self._prev_sim_obs)
        else:
            for _ in range(self.ac_freq_ratio):
                self.internal_step(-1, update_articulated_agent=False)

            t_start = time.time()
            self._prev_sim_obs = self.get_sensor_observations()
            self.add_perf_timing("get_sensor_observations", t_start)

            obs = self._sensor_suite.get_observations(self._prev_sim_obs)

        if self._enable_gfx_replay_save:
            self.gfx_replay_manager.save_keyframe()

        if self._needs_markers:
            self._update_markers()

        # TODO: Make debug cameras more flexible
        if "third_rgb" in obs and self._debug_render:
            self._try_acquire_context()
            for k, (pos, r) in add_back_viz_objs.items():
                viz_id = self.viz_ids[k]

                self.viz_ids[k] = self.visualize_position(
                    pos, self.viz_ids[k], r=r
                )

            # Also render debug information
            self._create_obj_viz()

            debug_obs = self.get_sensor_observations()
            obs["third_rgb"] = debug_obs["third_rgb"][:, :, :3]

        return obs

    def maybe_update_articulated_agent(self):
        """
        Calls the update agents method on the articulated agent manager if the
        `update_articulated_agent` configuration is set to True. Among other
        things, this will set the articulated agent's sensors' positions to their new
        positions.
        """
        if self._update_articulated_agent:
            self.agents_mgr.update_agents()

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
        self, dt: Union[int, float], update_articulated_agent: bool = True
    ) -> None:
        """Step the world and update the articulated_agent.

        :param dt: Timestep by which to advance the world. Multiple physics substeps can be excecuted within a single timestep. -1 indicates a single physics substep.

        Never call sim.step_world directly or miss updating the articulated_agent.
        """
        # Optionally step physics and update the articulated_agent for benchmarking purposes
        if self._step_physics:
            t_start = time.time()
            self.step_world(dt)
            self.add_perf_timing("step_world", t_start)

    def get_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get a mapping of object ids to goal positions for rearrange targets.

        :return: ([idx: int], [goal_pos: list]) The index of the target object
          in self._scene_obj_ids and the 3D goal position, rotation is IGNORED.
          Note that goal_pos is the desired position of the object, not the
          starting position.
        """
        target_trans = self._get_target_trans()
        if len(target_trans) == 0:
            return np.array([]), np.array([])
        targ_idx, targ_trans = list(zip(*self._get_target_trans()))

        a, b = np.array(targ_idx), [
            np.array(x.translation) for x in targ_trans
        ]
        return a, np.array(b)

    def get_n_targets(self) -> int:
        """Get the number of rearrange targets."""
        return len(self.ep_info.targets)

    def get_target_objs_start(self) -> np.ndarray:
        """Get the initial positions of all objects targeted for rearrangement as a numpy array."""
        return self.target_start_pos

    def get_scene_pos(self) -> np.ndarray:
        """Get the positions of all clutter RigidObjects in the scene as a numpy array."""
        rom = self.get_rigid_object_manager()
        return np.array(
            [
                rom.get_object_by_id(idx).translation
                for idx in self._scene_obj_ids
            ]
        )

    def add_perf_timing(self, desc, t_start):
        self._extra_runtime_perf_stats[desc] = time.time() - t_start

    def get_runtime_perf_stats(self):
        names = self._backend_runtime_perf_stat_names
        values = super().get_runtime_perf_stat_values()
        stats_dict = dict(zip(names, values))

        for name, value in self._extra_runtime_perf_stats.items():
            stats_dict[name] = value
        # clear this dict so we don't accidentally collect these twice
        self._extra_runtime_perf_stats = {}

        return stats_dict
