#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os.path as osp
from collections import defaultdict
from typing import Any, Dict

import magnum as mn
import numpy as np

from habitat.core.registry import registry
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.rearrange.obj_loaders import (
    load_articulated_objs,
    load_objs,
    place_viz_objs,
)
from habitat.tasks.rearrange.rearrange_grasp_manager import (
    RearrangeGraspManager,
)
from habitat.tasks.rearrange.utils import (
    IkHelper,
    convert_legacy_cfg,
    get_nav_mesh_settings,
    make_render_only,
)
from habitat_sim.gfx import LightInfo, LightPositionModel
from habitat_sim.physics import MotionType
from habitat_sim.robots import FetchRobot


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


@registry.register_simulator(name="RearrangeSim-v0")
class RearrangeSim(HabitatSim):
    def __init__(self, config):
        super().__init__(config)

        agent_config = self.habitat_config
        self.navmesh_settings = get_nav_mesh_settings(self._get_agent_config())
        self.first_setup = True
        self.is_render_obs = False
        self.ep_info = None
        self.prev_loaded_navmesh = None
        self.prev_scene_id = None

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
        # Used to get data from the RL environment class to sensors.
        self.track_markers = []
        self._goal_pos = None
        self.viz_ids: Dict[Any, Any] = defaultdict(lambda: None)

        self.ik_helper = None

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
        self.grasp_mgr = RearrangeGraspManager(self, self.habitat_config)

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

    def _try_acquire_context(self):
        if self.concur_render:
            self.renderer.acquire_gl_context()

    def _update_config(self, ep_info):
        """Updates config from legacy settings. Hopefully this can be
        deprecated soon.
        """
        scene = ep_info["scene_id"]
        if "replica_cad" in scene and "glb" in scene:
            parts = scene.split("/")
            base_replica_path = "/".join(parts[:-2])
            end_replica_path = "/".join(parts[-2:]).split(".")[0]
            ep_info["scene_id"] = osp.join(
                base_replica_path, "configs", end_replica_path
            )
        return ep_info

    def reconfigure(self, config):
        ep_info = config["ep_info"][0]
        ep_info = self._update_config(ep_info)

        config["SCENE"] = ep_info["scene_id"]
        super().reconfigure(config)

        self.ep_info = ep_info
        self.fixed_base = ep_info["fixed_base"]

        self.target_obj_ids = []

        if ep_info["scene_id"] != self.prev_scene_id:
            # Object instances are not valid between scenes.
            self.art_objs = []
            self.scene_obj_ids = []
            self.robot = None
            self.viz_ids = defaultdict(lambda: None)
            self.viz_obj_ids = []
        self.grasp_mgr.desnap(force=True)
        self.prev_scene_id = ep_info["scene_id"]

        self._try_acquire_context()

        self._add_objs(ep_info)
        if self.robot is None:
            self.robot = FetchRobot(self.habitat_config.ROBOT_URDF, self)
            self.robot.reconfigure()
        self.robot.reset()
        self.grasp_mgr.reset()

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
            if self.habitat_config.get("IK_ARM_URDF", None) is not None:
                self.ik_helper = IkHelper(
                    self.habitat_config.IK_ARM_URDF,
                    np.array(self.robot.params.arm_init_params),
                )
            # Capture the starting art states
            self.start_art_states = {
                ao: ao.joint_positions for ao in self.art_objs
            }

    def _load_navmesh(self):
        """Generates the navmesh on the fly. This must be called
        AFTER adding articulated objects to the scene.
        """

        motion_types = []
        for art_obj in self.art_objs:
            motion_types.append(art_obj.motion_type)
            art_obj.motion_type = MotionType.STATIC
        self.recompute_navmesh(
            self.pathfinder,
            self.navmesh_settings,
            include_static_objects=True,
        )
        if self.habitat_config.get("SAVE_NAVMESH", False):
            scene_name = self.ep_info["scene_id"]
            inferred_path = scene_name.split(".glb")[0] + ".navmesh"
            self.pathfinder.save_nav_mesh(inferred_path)
            print("Cached navmesh to ", inferred_path)
        for art_obj, motion_type in zip(self.art_objs, motion_types):
            art_obj.motion_type = motion_type

    def reset(self):
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

    def step(self, action):
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

        if "robot_third_rgb" in obs:
            self.is_render_obs = True
            self._try_acquire_context()
            for k, pos in add_back_viz_objs.items():
                self.viz_ids[k] = self.visualize_position(pos)

            # Also render debug information
            if self.habitat_config.get("RENDER_TARGS", True):
                self._create_obj_viz(self.ep_info)

            # Always draw the target
            for obj_idx, _ in self.ep_info["targets"]:
                self.set_object_bb_draw(True, self.scene_obj_ids[obj_idx])

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
            obj_mgr = self.get_object_template_manager()
            template = obj_mgr.get_template_by_handle(
                obj_mgr.get_template_handles("sphere")[0]
            )
            template.scale = mn.Vector3(r, r, r)
            new_template_handle = obj_mgr.register_template(
                template, "ball_new_viz"
            )
            viz_id = self.add_object(new_template_handle)
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

    def draw_sphere(self, r, template_name="ball_new"):
        obj_mgr = self.get_object_template_manager()
        template_handle = obj_mgr.get_template_handles("sphere")[0]
        template = obj_mgr.get_template_by_handle(template_handle)
        template.scale = mn.Vector3(r, r, r)
        new_template_handle = obj_mgr.register_template(template, "ball_new")
        obj_id = self.add_object(new_template_handle)
        self.set_object_motion_type(MotionType.KINEMATIC, obj_id)
        return obj_id
