#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from collections import defaultdict
from typing import Any, Callable, Dict

import attr
import magnum as mn
import numpy as np

import habitat_sim
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.rearrange.obj_loaders import (
    add_obj,
    load_articulated_objs,
    load_objs,
    place_viz_objs,
)
from habitat.tasks.rearrange.utils import (
    convert_legacy_cfg,
    get_aabb,
    get_nav_mesh_settings,
)
from habitat_sim.gfx import LightInfo, LightPositionModel
from habitat_sim.physics import MotionType
from habitat_sim.robots import FetchRobot

# (unique id, Filename, BB size, BB offset, Robot base offset [can be none])
ART_BBS = [
    ("fridge", "fridge.urdf", [0.33, 0.9, 0.33], [0, 0.2, 0], [1.033, 0.0]),
    (
        "counter",
        "kitchen_counter.urdf",
        [0.28, 0.5, 1.53],
        [-0.067, 0.5, 0.0],
        None,
    ),
    # Counter R
    (
        "counter_R",
        "kitchen_counter.urdf",
        [0.28, 0.5, 0.55],
        [-0.067, 0.5, 1.0],
        [0.526, 1.107],
    ),
    # Counter L
    (
        "counter_L",
        "kitchen_counter.urdf",
        [0.28, 0.5, 0.75],
        [-0.067, 0.5, -0.7],
        [0.524, -0.896],
    ),
]


# temp workflow for loading lights into Habitat scene
def load_light_setup_for_glb(json_filepath):
    with open(json_filepath) as json_file:
        data = json.load(json_file)
        lighting_setup = []
        for light in data["lights"].values():
            t = light["position"]
            light_w = 1.0
            position = [float(t[0]), float(t[1]), float(t[2]), light_w]
            color_scale = float(light["intensity"])
            color = [float(c * color_scale) for c in light["color"]]
            lighting_setup.append(
                LightInfo(
                    vector=position,
                    color=color,
                    model=LightPositionModel.Global,
                )
            )

    return lighting_setup


@attr.s(auto_attribs=True, slots=True)
class SimEvent:
    is_ready: Callable[[], bool]
    run: Callable[[], None]


# Distance from the base of the end-effector to the actual end-effector
# position, which should be in the center of the gripper.
EE_GRIPPER_OFFSET = mn.Vector3(0.08, 0, 0)


@registry.register_simulator(name="RearrangeSim-v0")
class RearrangeSim(HabitatSim):
    def __init__(self, config):
        super().__init__(config)

        agent_config = self.habitat_config
        self.navmesh_settings = get_nav_mesh_settings(self._get_agent_config())
        self.first_setup = True
        self.is_render_obs = False
        self.pov_mode = agent_config.POV
        self.update_i = 0
        self.h_offset = 0.3
        self.ep_info = None
        self.do_grab_using_constraint = True
        self.snap_to_link_on_grab = True
        self.snapped_obj_id = None
        self.snapped_obj_constraint_id = []
        self.prev_loaded_navmesh = None
        self.prev_scene_id = None
        self.robot_name = agent_config.ROBOT_URDF.split("/")[-1].split(".")[0]
        self._force_back_pos = None

        # Number of physics updates per action
        self.ac_freq_ratio = agent_config.AC_FREQ_RATIO
        # The physics update time step.
        self.ctrl_freq = agent_config.CTRL_FREQ
        # Effective control speed is (ctrl_freq/ac_freq_ratio)

        self.art_objs = []
        self.start_art_states = {}
        self.cached_art_obj_ids = []
        self.scene_obj_ids = []
        self.viz_obj_ids = []
        self.event_callbacks = []
        # Used to get data from the RL environment class to sensors.
        self.track_markers = []
        self._goal_pos = None
        self.viz_ids: Dict[Any, Any] = defaultdict(lambda: None)

        # Disables arm control. Useful if you are hiding the arm to perform
        # some scene sensing.
        self.ctrl_arm = True

        self._light_setup = load_light_setup_for_glb(
            "data/replica_cad/configs/lighting/frl_apartment_stage.lighting_config.json"
        )
        obj_attr_mgr = self.get_object_template_manager()
        obj_attr_mgr.load_configs("data/objects/ycb")

        self.concur_render = self.habitat_config.get(
            "CONCUR_RENDER", True
        ) and hasattr(self, "get_sensor_observations_async_start")

    def _create_art_bbs(self):
        """
        Creates transformed bounding boxes for the articulated objects.
        """
        self.art_bbs = []
        for _, (name, urdf_name, bb_size, bb_pos, robo_pos) in enumerate(
            ART_BBS
        ):
            if urdf_name not in self.art_name_to_id:
                continue
            ao = self.art_name_to_id[urdf_name]
            art_T = ao.transformation

            if robo_pos is not None:
                robo_pos_vec = mn.Vector3([robo_pos[0], 0.5, robo_pos[1]])
                robo_pos_vec = art_T.transform_point(robo_pos_vec)
                robo_pos = np.array(robo_pos_vec)[[0, 2]]

            bb = mn.Range3D.from_center(
                mn.Vector3(*bb_pos), mn.Vector3(*bb_size)
            )
            bb = habitat_sim.geo.get_transformed_bb(bb, art_T)
            self.art_bbs.append((name, bb, robo_pos))

    def _get_target_trans(self):
        """
        This is how the target transforms should be accessed since
        multiprocessing does not allow pickling.
        """
        # Preprocess the ep_info making necessary datatype conversions.
        target_trans = []
        for i in range(len(self.ep_info["targets"])):
            targ_idx, trans = self.ep_info["targets"][i]
            if len(trans) == 3:
                # Legacy position only format.
                trans = mn.Matrix4.translation(mn.Vector3(*trans))
            else:
                trans = mn.Matrix4(trans)
            target_trans.append((targ_idx, trans))
        return target_trans

    def find_robo_for_art_name(self, art_name):
        """
        Gets the desired robot starting position for interacting with an
        articulated object.
        """
        for name, bb, robo_pos in self.art_bbs:
            if name == art_name and robo_pos is not None:
                return (bb, robo_pos)
        return None, None

    def get_nav_pos(self, pos, set_back: bool = False):
        """
        Gets the desired robot base position for an object. If the object is in
        an articulated object, a proper offset is applied.
        - set_back: (bool): Pushes the robot even further back if specified by
          offset. Used when spawning the robot in front of an open cabinet.
        """
        pos = mn.Vector3(*pos)
        force_spawn_pos = self.ep_info["force_spawn_pos"]
        if set_back and force_spawn_pos is not None:
            rel_art_bb_id, offset = force_spawn_pos
            _, urdf_name, _, _, robo_pos = ART_BBS[rel_art_bb_id]
            art_id = self.art_name_to_id[urdf_name]
            art_T = self.get_articulated_object_root_state(art_id)

            robo_pos = [robo_pos[0] + offset[0], 0.5, robo_pos[1] + offset[1]]
            robo_pos = art_T.transform_point(mn.Vector3(*robo_pos))
            robo_pos = np.array(robo_pos)
            robo_pos = robo_pos[[0, 2]]
            return np.array([robo_pos[0], 0.5, robo_pos[1]])

        set_pos = None
        for bb_info in self.art_bbs:
            bb = bb_info[1]
            if bb.contains(pos):
                set_pos = bb_info[2]
        if set_pos is None:
            if self.pathfinder.island_radius(pos) == 0.0:
                # TODO: Hack for points which somehow end up in 0 radius island.
                return np.array(
                    self.pathfinder.snap_point(pos - np.array([0, 0, 0.3]))
                )
            return np.array(self.pathfinder.snap_point(pos))
        else:
            return np.array([set_pos[0], 0.5, set_pos[1]])

    def _try_acquire_context(self):
        if self.concur_render:
            self.renderer.acquire_gl_context()

    def reconfigure(self, config):
        ep_info = config["ep_info"][0]

        config["SCENE"] = ep_info["scene_id"]
        super().reconfigure(config)

        self.ep_info = ep_info
        self.fixed_base = ep_info["fixed_base"]

        self.target_obj_ids = []
        self.event_callbacks = []

        if ep_info["scene_id"] != self.prev_scene_id:
            # Object instances are not valid between scenes.
            self.art_objs = []
            self.scene_obj_ids = []
            self.robot = None
            self.snapped_obj_constraint_id = []
            self.snapped_obj_id = None
        self.desnap_object(force=True)
        self.prev_scene_id = ep_info["scene_id"]

        self._try_acquire_context()

        self._add_objs(ep_info)
        if self.robot is None:
            self.robot = FetchRobot(self.habitat_config.ROBOT_URDF, self)
            self.robot.reconfigure()
        self.robot.reset()

        set_pos = {}
        # Set articulated object joint states.
        if self.habitat_config.get("LOAD_ART_OBJS", True):
            for i, art_state in self.start_art_states.items():
                set_pos[i] = art_state
            for i, art_state in ep_info["art_states"]:
                set_pos[self.art_objs[i]] = art_state

        # Get the positions after things have settled down.
        self.settle_sim(self.habitat_config.get("SETTLE_TIME", 0.1))

        # Get the starting positions of the target objects.
        scene_pos = self.get_scene_pos()
        self.target_start_pos = np.array(
            [scene_pos[idx] for idx, _ in self.ep_info["targets"]]
        )

        if self.first_setup:
            self.first_setup = False
            # Capture the starting art states
            self.start_art_states = {
                ao: ao.joint_positions for ao in self.art_objs
            }

        self.update_i = 0

    def _add_art_bbs(self):
        art_bb_ids = []
        for art_bb in ART_BBS:
            obj_s = "/BOX_" + "_".join([str(x) for x in art_bb[2]])
            urdf_name = art_bb[1]
            bb_pos = art_bb[3]
            if urdf_name not in self.art_name_to_id:
                continue

            ao = self.art_name_to_id[urdf_name]
            art_T = ao.transformation

            bb_T = art_T @ mn.Matrix4.translation(mn.Vector3(*bb_pos))

            obj_id = add_obj(obj_s, self)
            self.set_transformation(bb_T, obj_id)
            self.set_object_motion_type(MotionType.STATIC, obj_id)
            art_bb_ids.append(obj_id)
        return art_bb_ids

    def _load_navmesh(self):
        """
        Generates the navmesh if it was not specified. This must be called
        BEFORE adding any object / articulated objects to the scene.
        """
        art_bb_ids = self._add_art_bbs()
        # Add bounding boxes for articulated objects
        self.recompute_navmesh(
            self.pathfinder,
            self.navmesh_settings,
            include_static_objects=True,
        )
        for idx in art_bb_ids:
            self.remove_object(idx)
        if self.habitat_config.get("SAVE_NAVMESH", False):
            scene_name = self.ep_info["scene_id"]
            inferred_path = scene_name.split(".glb")[0] + ".navmesh"
            self.pathfinder.save_nav_mesh(inferred_path)
            print("Cached navmesh to ", inferred_path)

    def set_robot_pos(self, set_pos):
        """
        - set_pos: 2D coordinates of where the robot will be placed. The height
          will be same as current position.
        """
        pos = self.robot.sim_obj.translation
        new_set_pos = [set_pos[0], pos[1], set_pos[1]]
        self.robot.sim_obj.transformation.translation = new_set_pos

    def set_robot_rot(self, rot_rad):
        """
        Set the rotation of the robot along the y-axis. The position will
        remain the same.
        """
        cur_trans = self.robot.sim_obj.transformation
        pos = cur_trans.translation

        rot_trans = mn.Matrix4.rotation(mn.Rad(-1.56), mn.Vector3(1.0, 0, 0))
        add_rot_mat = mn.Matrix4.rotation(
            mn.Rad(rot_rad), mn.Vector3(0.0, 0, 1)
        )
        new_trans = rot_trans @ add_rot_mat
        new_trans.translation = pos
        self.robot.sim_obj.transformation = new_trans

    def reset(self):
        self.event_callbacks = []
        ret = super().reset()
        if self._light_setup:
            # Lighting reconfigure needs to be in the reset function and not
            # the reconfigure function.
            self.set_light_setup(self._light_setup)

        return ret

    def clear_objs(self, art_names=None):
        for scene_obj in self.scene_obj_ids:
            self.remove_object(scene_obj)
        self.scene_obj_ids = []

        if art_names is None or self.cached_art_obj_ids != art_names:
            ao_mgr = self.get_articulated_object_manager()
            ao_mgr.remove_all_objects()
            self.art_objs = []

    def _add_objs(self, ep_info):
        art_names = [x[0] for x in ep_info["art_objs"]]
        self.clear_objs(art_names)

        if self.habitat_config.get("LOAD_ART_OBJS", True):
            self.art_objs = load_articulated_objs(
                convert_legacy_cfg(ep_info["art_objs"]),
                self,
                self.art_objs,
                auto_sleep=self.habitat_config.get("AUTO_SLEEP", True),
            )
            self.cached_art_obj_ids = art_names
            self.art_name_to_id = {
                name.split("/")[-1]: art_id
                for name, art_id in zip(art_names, self.art_objs)
            }
            self._create_art_bbs()

        self._load_navmesh()

        if self.habitat_config.get("LOAD_OBJS", True):
            self.scene_obj_ids = load_objs(
                convert_legacy_cfg(ep_info["static_objs"]),
                self,
                obj_ids=self.scene_obj_ids,
                auto_sleep=self.habitat_config.get("AUTO_SLEEP", True),
            )

            for idx, _ in ep_info["targets"]:
                self.target_obj_ids.append(self.scene_obj_ids[idx])
        else:
            self.ep_info["targets"] = []

    def _create_obj_viz(self, ep_info):
        self.viz_obj_ids = []

        target_name_pos = [
            (ep_info["static_objs"][idx][0], self.scene_obj_ids[idx], pos)
            for idx, pos in self._get_target_trans()
        ]
        self.viz_obj_ids = place_viz_objs(
            target_name_pos, self, self.viz_obj_ids
        )

    def capture_state(self, with_robo_js=False):
        # Don't need to capture any velocity information because this will
        # automatically be set to 0 in `set_state`.
        robot_T = self.robot.sim_obj.transformation
        art_T = [ao.transformation for ao in self.art_objs]
        static_T = [self.get_transformation(i) for i in self.scene_obj_ids]
        art_pos = [ao.joint_positions for ao in self.art_objs]
        robo_js = self.robot.sim_obj.joint_positions

        return {
            "robot_T": robot_T,
            "robo_js": robo_js,
            "art_T": art_T,
            "static_T": static_T,
            "art_pos": art_pos,
            "obj_hold": self.snapped_obj_id,
        }

    def set_state(self, state, set_hold=False):
        """
        - set_hold: If true this will set the snapped object from the `state`.
          This should probably be True by default, but I am not sure the effect
          it will have.
        """
        if state["robot_T"] is not None:
            self.robot.sim_obj.transformation = state["robot_T"]
            self.robot.sim_obj.clear_joint_states()

        if "robo_js" in state:
            self.robot.arm_joint_pos = state["robo_js"]

        for T, ao in zip(state["art_T"], self.art_objs):
            ao.transformation = T

        for T, i in zip(state["static_T"], self.scene_obj_ids):
            self.reset_obj_T(i, T)

        for p, ao in zip(state["art_pos"], self.art_objs):
            ao.joint_positions = p

        if set_hold:
            if state["obj_hold"] is not None:
                self.internal_step(-1)
                self.full_snap(self.scene_obj_ids.index(state["obj_hold"]))
            elif state["marker_hold"] is not None:
                self.set_snapped_marker(state["marker_hold"])
            else:
                self.desnap_object(True)

    def reset_obj_T(self, i, T):
        self.set_transformation(T, i)
        self.set_linear_velocity(mn.Vector3(0, 0, 0), i)
        self.set_angular_velocity(mn.Vector3(0, 0, 0), i)

    def reset_art_obj_pos(self, i, p):
        self.set_articulated_object_positions(i, p)
        vel = self.get_articulated_object_velocities(i)
        forces = self.get_articulated_object_forces(i)
        self.set_articulated_object_velocities(i, np.zeros((len(vel),)))
        self.set_articulated_object_forces(i, np.zeros((len(forces),)))

    def settle_sim(self, seconds):
        steps = int(seconds * self.ctrl_freq)
        for _ in range(steps):
            self.step_world(-1)

    def full_snap(self, obj_id):
        """
        No matter where the object is in the scene, it will be moved to the
        robot's gripper and snapped. This should be used for "teleporting" the
        object to the robot's hand, it is not physically plausible.
        - obj_id: The index in scene_obj_ids, not absolute simulator index.
        """
        abs_idx = self.scene_obj_ids[obj_id]
        ee_T = self.get_end_effector_trans()
        obj_local_T = mn.Matrix4.translation(EE_GRIPPER_OFFSET)
        global_T = ee_T @ obj_local_T
        self.set_transformation(global_T, abs_idx)
        self.set_snapped_obj(obj_id)

    def set_snapped_obj(self, snapped_obj_id):
        """
        - snapped_obj_id: the index of the object in scene_obj_ids. Not
          absolute simulator index.
        """
        use_snap_obj_id = self.scene_obj_ids[snapped_obj_id]
        if use_snap_obj_id == self.snapped_obj_id:
            return
        if len(self.snapped_obj_constraint_id) != 0:
            self.desnap_object()

        self.snapped_obj_id = use_snap_obj_id

        max_impulse = 1000.0
        if self.snap_to_link_on_grab:
            # Set collision group to GraspedObject so that it doesn't collide
            # with the links of the robot.
            grasped_object_group = 16  # see Habitat-sim CollisionGroupHelper.h
            self.override_collision_group(
                use_snap_obj_id, grasped_object_group
            )
            if not self.do_grab_using_constraint:
                return

            # Set the transformation to be in the robot's hand already.
            ee_T = self.get_end_effector_trans()
            obj_local_T = mn.Matrix4.translation(EE_GRIPPER_OFFSET)
            global_T = ee_T @ obj_local_T
            self.set_transformation(global_T, use_snap_obj_id)

            def create_hold_constraint(pivot_in_link, pivot_in_obj):
                if hasattr(
                    self, "create_articulated_p2p_constraint_with_pivots"
                ):
                    return self.create_articulated_p2p_constraint_with_pivots(
                        self.robot.sim_obj.get_robot_sim_id(),
                        self.ee_link,
                        use_snap_obj_id,
                        pivot_in_link,
                        pivot_in_obj,
                        max_impulse,
                    )
                else:
                    return self.create_articulated_p2p_constraint(
                        self.robot.sim_obj.get_robot_sim_id(),
                        self.ee_link,
                        use_snap_obj_id,
                        pivot_in_link,
                        pivot_in_obj,
                        max_impulse,
                    )

            self.snapped_obj_constraint_id = [
                create_hold_constraint(
                    mn.Vector3(0.1, 0, 0), mn.Vector3(0, 0, 0)
                ),
                create_hold_constraint(
                    mn.Vector3(0.0, 0, 0), mn.Vector3(-0.1, 0, 0)
                ),
                create_hold_constraint(
                    mn.Vector3(0.1, 0.0, 0.1), mn.Vector3(0.0, 0.0, 0.1)
                ),
            ]
        else:
            self.snapped_obj_constraint_id = [
                self.create_articulated_p2p_constraint(
                    self.robot.sim_obj.get_robot_sim_id(),
                    self.ee_link,
                    use_snap_obj_id,
                    max_impulse,
                )
            ]
        if any([x == -1 for x in self.snapped_obj_constraint_id]):
            raise ValueError("Created bad constraint")

    def desnap_object(self, force=False):
        """
        Remove the constraint for holding an object OR articulated object.
        """
        if len(self.snapped_obj_constraint_id) == 0:
            # No constraints to unsnap
            self.snapped_obj_id = None
            return

        if self.snapped_obj_id is not None and self.snap_to_link_on_grab:
            obj_bb = get_aabb(self.snapped_obj_id, self)
            r = max(obj_bb.size_x(), obj_bb.size_y(), obj_bb.size_z())
            c = self.get_translation(self.snapped_obj_id)
            snap_obj_id = self.snapped_obj_id
            if force:
                self.override_collision_group(snap_obj_id, 1)
            else:

                def is_ready():
                    ee_pos = self.robot.ee_transform.translation
                    dist = np.linalg.norm(ee_pos - c)
                    return dist >= r

                self.event_callbacks.append(
                    SimEvent(
                        is_ready,
                        lambda: self.override_collision_group(snap_obj_id, 1),
                    )
                )
        if self.do_grab_using_constraint:
            for constraint_id in self.snapped_obj_constraint_id:
                self.remove_constraint(constraint_id)
            self.snapped_obj_constraint_id = []

        self.snapped_obj_id = None

    def path_to_point(self, point):
        trans = self.get_robot_transform()
        agent_pos = trans.translation
        closest_point = self.pathfinder.snap_point(point)
        path = habitat_sim.ShortestPath()
        path.requested_start = agent_pos
        path.requested_end = closest_point
        found_path = self.pathfinder.find_path(path)
        if not found_path:
            return [agent_pos, point]
        if len(path.points) == 1:
            return [agent_pos, path.points[0]]
        return path.points

    def step(self, action):
        self.update_i += 1

        if self.is_render_obs:
            self._try_acquire_context()
            for obj_idx, _ in self.ep_info["targets"]:
                self.set_object_bb_draw(False, self.scene_obj_ids[obj_idx])
        for viz_obj in self.viz_obj_ids:
            self.remove_object(viz_obj)

        add_back_viz_objs = {}
        for name, viz_id in self.viz_ids.items():
            if viz_id is None:
                continue

            before_pos = self.get_translation(viz_id)
            self.remove_object(viz_id)
            add_back_viz_objs[name] = before_pos
        self.viz_obj_ids = []
        self.viz_ids = defaultdict(lambda: None)

        remove_idxs = []
        for i, event in enumerate(self.event_callbacks):
            if event.is_ready():
                event.run()
                remove_idxs.append(i)

        for i in reversed(remove_idxs):
            del self.event_callbacks[i]

        if not self.concur_render:
            if self.habitat_config.get("STEP_PHYSICS", True):
                for _ in range(self.ac_freq_ratio):
                    self.internal_step(-1)

            self._prev_sim_obs = self.get_sensor_observations()
            obs = self._sensor_suite.get_observations(self._prev_sim_obs)

        else:
            self._prev_sim_obs = self.get_sensor_observations_async_start()

            if self.habitat_config.get("STEP_PHYSICS", True):
                for _ in range(self.ac_freq_ratio):
                    self.internal_step(-1)

            self._prev_sim_obs = self.get_sensor_observations_async_finish()
            obs = self._sensor_suite.get_observations(self._prev_sim_obs)

        if "high_rgb" in obs:
            self.is_render_obs = True
            self._try_acquire_context()
            for k, pos in add_back_viz_objs.items():
                self.viz_ids[k] = self.viz_pos(pos)

            # Also render debug information
            if self.habitat_config.get("RENDER_TARGS", True):
                self._create_obj_viz(self.ep_info)

            # Always draw the target
            for obj_idx, _ in self.ep_info["targets"]:
                self.set_object_bb_draw(True, self.scene_obj_ids[obj_idx])

            debug_obs = self.get_sensor_observations()
            obs["high_rgb"] = debug_obs["high_rgb"][:, :, :3]

        if self.habitat_config.HABITAT_SIM_V0.get(
            "ENABLE_GFX_REPLAY_SAVE", False
        ):
            self.gfx_replay_manager.save_keyframe()

        return obs

    def draw_obs(self):
        """Synchronously gets the observation at the current step"""
        # Update the world state to get most recent render
        self.internal_step(-1)

        prev_sim_obs = self.get_sensor_observations()
        obs = self._sensor_suite.get_observations(prev_sim_obs)
        return obs

    def internal_step(self, dt):
        """
        Never call sim.step_world directly.
        """

        self.step_world(dt)
        if self.robot is not None:
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

    def get_target_objs_start(self):
        return np.array(self.target_start_pos)

    def get_scene_pos(self):
        return np.array(
            [self.get_translation(idx) for idx in self.scene_obj_ids]
        )

    def get_collisions(self):
        return []

    def is_holding_obj(self):
        return self.snapped_obj_id is not None
