import copy
import queue
import re
from collections import defaultdict
from typing import Callable

import attr
import magnum as mn
import numpy as np
import quaternion

import habitat_sim
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.nav.nav import NavigationTask
from habitat.tasks.rearrange.envs.obj_loaders import (
    add_obj,
    init_art_objs,
    load_articulated_objs,
    load_objs,
    place_viz_objs,
)
from habitat.tasks.rearrange.envs.utils import (
    IkHelper,
    convert_legacy_cfg,
    get_aabb,
    get_largest_island_point,
    get_nav_mesh_settings,
    make_render_only,
)
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


import json

from habitat_sim.gfx import LightInfo, LightPositionModel


# temp workflow for loading lights into Habitat scene
def load_light_setup_for_glb(json_filepath):
    lighting_setup = None

    with open(json_filepath) as json_file:
        data = json.load(json_file)
        lighting_setup = []
        for l in data["lights"]:
            t = l["position"]
            light_w = 1.0
            position = [float(t[0]), float(t[1]), float(t[2]), light_w]
            color_scale = float(l["color_scale"])
            color = [float(c * color_scale) for c in l["color"]]
            # print('position: {}'.format(position))
            # print('color: {}'.format(color))
            lighting_setup.append(
                LightInfo(
                    vector=position,
                    color=color,
                    model=LightPositionModel.Global,
                )
            )
        # print("loaded {} lights".format(len(data['lights'])))

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
        self.n_objs = config.N_OBJS

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
        self.snapped_marker_name = None
        self.snapped_obj_constraint_id = []
        self.prev_loaded_navmesh = None
        self.prev_scene_id = None
        self.robot_name = agent_config.ROBOT_URDF.split("/")[-1].split(".")[0]
        self._force_back_pos = None

        # A marker you can optionally render to visualize positions
        self.viz_marker = None
        self.move_cam_pos = np.zeros(3)
        if "CAM_START" in agent_config:
            self.move_cam_pos = np.array(agent_config.CAM_START)

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
        # Horrible hack to get data from the RL environment class to sensors.
        self.track_markers = []
        self._goal_pos = None

        # Disables arm control. Useful if you are hiding the arm to perform
        # some scene sensing.
        self.ctrl_arm = True

        self.viz_ids = defaultdict(lambda: None)
        self.viz_traj_ids = []
        self._light_setup = load_light_setup_for_glb(
            "data/misc_data/frl_apartment_stage_pvizplan_empty_lights.json"
        )
        obj_attr_mgr = self._sim.get_object_template_manager()
        obj_attr_mgr.load_configs("data/objects")

        self.concur_render = self.habitat_config.get(
            "CONCUR_RENDER", True
        ) and hasattr(self._sim, "get_sensor_observations_async_start")

    def _create_art_bbs(self):
        """
        Creates transformed bounding boxes for the articulated objects.
        """
        self.art_bbs = []
        for i, (name, urdf_name, bb_size, bb_pos, robo_pos) in enumerate(
            ART_BBS
        ):
            if urdf_name not in self.art_name_to_id:
                continue
            ao = self.art_name_to_id[urdf_name]
            art_T = ao.transformation

            if robo_pos is not None:
                robo_pos = [robo_pos[0], 0.5, robo_pos[1]]
                robo_pos = art_T.transform_point(mn.Vector3(*robo_pos))
                robo_pos = np.array(robo_pos)
                robo_pos = robo_pos[[0, 2]]

            bb = mn.Range3D.from_center(
                mn.Vector3(*bb_pos), mn.Vector3(*bb_size)
            )
            bb = habitat_sim.geo.get_transformed_bb(bb, art_T)
            self.art_bbs.append((name, bb, robo_pos))

    def get_track_markers_pos(self):
        """
        Gets the global positions of markers which are relevant for the task.
        Used so Sensors can get information about task-relevant markers. The
        track markers needs to first be set by the RL Environment.
        """
        return [self.markers[x]["global_pos"] for x in self.track_markers]

    def get_marker_positions(self):
        return {k: np.array(v["global_pos"]) for k, v in self.markers.items()}

    def set_gripper_state(self, gripper_state):
        self._gripper_state = gripper_state

    def is_gripper_open(self):
        # 0.0 is technically closed, but give some threshold.
        return self._gripper_state < 0.02

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

    def set_force_back(self, force_back_pos):
        self._force_back_pos = force_back_pos

    def get_nav_pos(self, pos, set_back: bool = False):
        """
        Gets the desired robot base position for an object. If the object is in
        an articulated object, a proper offset is applied.
        - set_back: (bool): Pushes the robot even further back if specified by
          offset. Used when spawning the robot in front of an open cabinet.
        """
        pos = mn.Vector3(*pos)
        if self._force_back_pos is not None:
            force_spawn_pos = self._force_back_pos
        else:
            force_spawn_pos = self.ep_info["force_spawn_pos"]
        if set_back and force_spawn_pos is not None:
            rel_art_bb_id, offset = force_spawn_pos
            _, urdf_name, _, _, robo_pos = ART_BBS[rel_art_bb_id]
            art_id = self.art_name_to_id[urdf_name]
            art_T = self._sim.get_articulated_object_root_state(art_id)

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
            if self._sim.pathfinder.island_radius(pos) == 0.0:
                # TODO: Hack for points which somehow end up in 0 radius island.
                return np.array(
                    self._sim.pathfinder.snap_point(
                        pos - np.array([0, 0, 0.3])
                    )
                )
            return np.array(self._sim.pathfinder.snap_point(pos))
        else:
            return np.array([set_pos[0], 0.5, set_pos[1]])

    def get_marker_nav_pos(self, marker_name):
        OFFSET = mn.Vector3(0.5, 0.0, 0.0)
        T = self.markers[marker_name]["T"]
        robo_pos = T.transform_point(OFFSET)
        return robo_pos

    def _try_acquire_context(self):
        if self.concur_render:
            self._sim.renderer.acquire_gl_context()

    def reconfigure(self, config):
        ep_info = config["ep_info"][0]

        # obj_order = list(range(len(ep_info['static_objs'])))
        # random.shuffle(obj_order)
        # ep_info['static_objs'] = [ep_info['static_objs'][i] for i in obj_order]

        # for j in range(len(ep_info['targets'])):
        #    ep_info['targets'][j][0] = obj_order.index(ep_info['targets'][j][0])

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
            self.viz_obj_ids = []
            self.snapped_obj_constraint_id = []
            self.snapped_obj_id = None
            self.snapped_marker_name = None
            self.viz_ids = defaultdict(lambda: None)
            self.viz_traj_ids = []
        self.desnap_object(force=True)
        self.prev_scene_id = ep_info["scene_id"]

        self._try_acquire_context()

        # Clear out all the viz ids
        for _, obj_idx in self.viz_ids.items():
            self._sim.remove_object(obj_idx)
        self.viz_ids = defaultdict(lambda: None)
        for viz_traj_id in self.viz_traj_ids:
            print("Viz traj id", viz_traj_id)
            self._sim.remove_object(viz_traj_id)
        self.viz_traj_ids = []

        self._add_objs(ep_info)
        if self.robot is None:
            self.robot = FetchRobot(self.habitat_config.ROBOT_URDF, self._sim)
            self.robot.reconfigure()
        self.robot.reset()

        set_pos = {}
        # Set articulated object joint states.
        if self.habitat_config.get("LOAD_ART_OBJS", True):
            for i, art_state in self.start_art_states.items():
                set_pos[i] = art_state
            for i, art_state in ep_info["art_states"]:
                set_pos[self.art_objs[i]] = art_state
            # TODO: NEED TO FIX
            # init_art_objs(
            #    set_pos.items(),
            #    self._sim,
            #    self.habitat_config.get("AUTO_SLEEP_ART_OBJS", True),
            # )

        # Get the positions after things have settled down.
        self.settle_sim(self.habitat_config.get("SETTLE_TIME", 0.1))

        # Get the starting positions of the target objects.
        scene_pos = self.get_scene_pos()
        self.target_start_pos = np.array(
            [scene_pos[idx] for idx, _ in self.ep_info["targets"]]
        )

        ###########################################################################
        # HACK. TEMPORARY FIX
        sn = self.ep_info["scene_config_path"]
        if "cab_top_left" in sn and "closed" not in sn:
            offset = mn.Vector3(-0.5, 0.0, 0.0)
            cab_T = self.art_objs[0].transformation
            offset = cab_T.transform_vector(offset)
            self.target_start_pos += offset

            self.art_objs[1].joint_positions = [0, 0]

        ###########################################################################

        if self.first_setup:
            self.first_setup = False
            # self._ik.setup_sim()
            # Capture the starting art states
            self.start_art_states = {
                ao: ao.joint_positions for ao in self.art_objs
            }

        self.update_i = 0
        self.allowed_region = ep_info["allowed_region"]
        self._load_markers(ep_info)

    def remove_traj_obj(self, traj_id):
        if traj_id not in self.viz_traj_ids:
            return
        del self.viz_traj_ids[self.viz_traj_ids.index(traj_id)]
        self._sim.remove_object(traj_id)

    def _add_art_bbs(self):
        art_bb_ids = []
        for art_bb in ART_BBS:
            obj_s = "/BOX_" + "_".join([str(x) for x in art_bb[2]])
            urdf_name = art_bb[1]
            bb_pos = art_bb[3]
            robo_pos = art_bb[4]
            if urdf_name not in self.art_name_to_id:
                continue

            ao = self.art_name_to_id[urdf_name]
            art_T = ao.transformation

            bb_T = art_T @ mn.Matrix4.translation(mn.Vector3(*bb_pos))

            obj_id = add_obj(obj_s, self._sim)
            self._sim.set_transformation(bb_T, obj_id)
            self._sim.set_object_motion_type(MotionType.STATIC, obj_id)
            art_bb_ids.append(obj_id)
        return art_bb_ids

    def _load_navmesh(self):
        """
        Generates the navmesh if it was not specified. This must be called
        BEFORE adding any object / articulated objects to the scene.
        """
        art_bb_ids = self._add_art_bbs()
        # Add bounding boxes for articulated objects
        self._sim.recompute_navmesh(
            self._sim.pathfinder,
            self.navmesh_settings,
            include_static_objects=True,
        )
        for idx in art_bb_ids:
            self._sim.remove_object(idx)
        if self.habitat_config.get("SAVE_NAVMESH", False):
            scene_name = self.ep_info["scene_id"]
            inferred_path = scene_name.split(".glb")[0] + ".navmesh"
            self._sim.pathfinder.save_nav_mesh(inferred_path)
            print("Cached navmesh to ", inferred_path)

    def _update_markers(self):
        raise ValueError("Not ready for release")
        # TODO: Not ready for Hab2.0 release
        # for marker_name, marker in self.markers.items():
        #    if "relative" not in marker:
        #        continue
        #    targ_idx, targ_link = marker["relative"]
        #    abs_targ_idx = self.art_obj_ids[targ_idx]
        #    link_state = self._sim.get_articulated_link_rigid_state(
        #        abs_targ_idx, targ_link
        #    )
        #    link_T = mn.Matrix4.from_(
        #        link_state.rotation.to_matrix(), link_state.translation
        #    )
        #    local_pos = marker["local_pos"]
        #    local_pos = mn.Vector3(*local_pos)
        #    global_pos = link_T.transform_point(local_pos)
        #    marker["T"] = link_T @ mn.Matrix4.translation(local_pos)
        #    marker["global_pos"] = global_pos

    def _load_markers(self, ep_info):
        self.markers = {}
        for marker in ep_info["markers"]:
            if "relative" in marker:
                self.markers[marker["name"]] = {
                    "local_pos": marker["local"],
                    "relative": marker["relative"],
                }
            else:
                self.markers[marker["name"]] = {
                    "global_pos": marker["global_pos"]
                }
        # self._update_markers()

    def reset(self):
        self.event_callbacks = []
        ret = super().reset()
        if self._light_setup:
            # Lighting reconfigure NEEDS to be in the reset function and NOT
            # the reconfigure function!
            self._sim.set_light_setup(self._light_setup)

        # Lag observations for N steps
        lag_n_steps = self.habitat_config.LAG_OBSERVATIONS
        self._sim_obs_queue = queue.Queue(lag_n_steps + 1)
        for _ in range(lag_n_steps):
            self._sim_obs_queue.put(ret)

        return ret

    def viz_pos(self, pos, viz_id=None, r=0.05):
        if viz_id is None:
            obj_mgr = self._sim.get_object_template_manager()
            template = obj_mgr.get_template_by_handle(
                obj_mgr.get_template_handles("sphere")[0]
            )
            template.scale = mn.Vector3(r, r, r)
            new_template_handle = obj_mgr.register_template(
                template, "ball_new_viz"
            )
            viz_id = self._sim.add_object(new_template_handle)
            make_render_only(viz_id, self._sim)
        self._sim.set_translation(mn.Vector3(*pos), viz_id)

        return viz_id

    @property
    def _sim(self):
        return self

    def clear_objs(self, art_names=None):
        # Clear the objects out.
        for scene_obj in self.scene_obj_ids:
            self._sim.remove_object(scene_obj)
        self.scene_obj_ids = []

        if art_names is None or self.cached_art_obj_ids != art_names:
            # for art_obj in self.art_obj_ids:
            #    self._sim.remove_articulated_object(art_obj)
            ao_mgr = self.get_articulated_object_manager()
            ao_mgr.remove_all_objects()
            self.art_objs = []

    def _add_objs(self, ep_info):
        art_names = [x[0] for x in ep_info["art_objs"]]
        self.clear_objs(art_names)

        if self.habitat_config.get("LOAD_ART_OBJS", True):
            self.art_objs = load_articulated_objs(
                convert_legacy_cfg(ep_info["art_objs"]),
                self._sim,
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
                self._sim,
                obj_ids=self.scene_obj_ids,
                auto_sleep=self.habitat_config.get("AUTO_SLEEP", True),
            )

            for idx, _ in ep_info["targets"]:
                self.target_obj_ids.append(self.scene_obj_ids[idx])
        else:
            self.ep_info["targets"] = []

    def set_robot_pos(self, set_pos):
        """
        - set_pos: 2D coordinates of where the robot will be placed. The height
          will be same as current position.
        """
        pos = self.robot._robot.translation
        new_set_pos = [set_pos[0], pos[1], set_pos[1]]
        self.robot._robot.transformation.translation = new_set_pos

    def set_robot_rot(self, rot_rad):
        """
        Set the rotation of the robot along the y-axis. The position will
        remain the same.
        """
        cur_trans = self.robot._robot.transformation
        pos = cur_trans.translation

        rot_trans = mn.Matrix4.rotation(mn.Rad(-1.56), mn.Vector3(1.0, 0, 0))
        add_rot_mat = mn.Matrix4.rotation(
            mn.Rad(rot_rad), mn.Vector3(0.0, 0, 1)
        )
        new_trans = rot_trans @ add_rot_mat
        new_trans.translation = pos
        self.robot._robot.transformation = new_trans

    def _create_obj_viz(self, ep_info):
        self.viz_obj_ids = []

        target_name_pos = [
            (ep_info["static_objs"][idx][0], self.scene_obj_ids[idx], pos)
            for idx, pos in self._get_target_trans()
        ]
        self.viz_obj_ids = place_viz_objs(
            target_name_pos, self._sim, self.viz_obj_ids
        )

    def capture_state(self, with_robo_js=False):
        # Don't need to capture any velocity information because this will
        # automatically be set to 0 in `set_state`.
        robot_T = self.robot._robot.transformation
        art_T = [ao.transformation for ao in self.art_objs]
        static_T = [
            self._sim.get_transformation(i) for i in self.scene_obj_ids
        ]
        art_pos = [ao.joint_positions for ao in self.art_objs]
        robo_js = self.robot._robot.joint_positions

        return {
            "robot_T": robot_T,
            "robo_js": robo_js,
            "art_T": art_T,
            "static_T": static_T,
            "art_pos": art_pos,
            "obj_hold": self.snapped_obj_id,
            "marker_hold": self.snapped_marker_name,
        }

    def set_state(self, state, set_hold=False):
        """
        - set_hold: If true this will set the snapped object from the `state`.
          This should probably be True by default, but I am not sure the effect
          it will have.
        """
        if state["robot_T"] is not None:
            self.robot._robot.transformation = state["robot_T"]
            self.robot._robot.clear_joint_states()

        if "robo_js" in state:
            self.robot._robo.joint_positions = state["robo_js"]

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
        self._sim.set_transformation(T, i)
        self._sim.set_linear_velocity(mn.Vector3(0, 0, 0), i)
        self._sim.set_angular_velocity(mn.Vector3(0, 0, 0), i)

    def reset_art_obj_pos(self, i, p):
        self._sim.set_articulated_object_positions(i, p)
        vel = self._sim.get_articulated_object_velocities(i)
        forces = self._sim.get_articulated_object_forces(i)
        self._sim.set_articulated_object_velocities(i, np.zeros((len(vel),)))
        self._sim.set_articulated_object_forces(i, np.zeros((len(forces),)))

    def settle_sim(self, seconds):
        steps = int(seconds * self.ctrl_freq)
        for _ in range(steps):
            self._sim.step_world(-1)

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
        self._sim.set_transformation(global_T, abs_idx)
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
            self._sim.override_collision_group(
                use_snap_obj_id, grasped_object_group
            )
            if not self.do_grab_using_constraint:
                return

            # Set the transformation to be in the robot's hand already.
            ee_T = self.get_end_effector_trans()
            obj_local_T = mn.Matrix4.translation(EE_GRIPPER_OFFSET)
            global_T = ee_T @ obj_local_T
            self._sim.set_transformation(global_T, use_snap_obj_id)

            def create_hold_constraint(pivot_in_link, pivot_in_obj):
                if hasattr(
                    self._sim, "create_articulated_p2p_constraint_with_pivots"
                ):
                    return self._sim.create_articulated_p2p_constraint_with_pivots(
                        self.robot._robot.get_robot_sim_id(),
                        self.ee_link,
                        use_snap_obj_id,
                        pivot_in_link,
                        pivot_in_obj,
                        max_impulse,
                    )
                else:
                    return self._sim.create_articulated_p2p_constraint(
                        self.robot._robot.get_robot_sim_id(),
                        self.ee_link,
                        use_snap_obj_id,
                        pivot_in_link,
                        pivot_in_obj,
                        max_impulse,
                    )

            self.snapped_obj_constraint_id = [
                # create_hold_constraint(mn.Vector3(0.0, 0, 0), mn.Vector3(0, 0, 0)),
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
                self._sim.create_articulated_p2p_constraint(
                    self.robot._robot.get_robot_sim_id(),
                    self.ee_link,
                    use_snap_obj_id,
                    max_impulse,
                )
            ]
        if any([x == -1 for x in self.snapped_obj_constraint_id]):
            raise ValueError("Created bad constraint")

    def set_snapped_marker(self, snapped_marker_name):
        """
        Create a constraint between the end-effector and the marker on the
        articulated object that is attempted to be grasped.
        """
        if len(self.snapped_obj_constraint_id) != 0:
            self.desnap_object()
        if snapped_marker_name == self.snapped_marker_name:
            return

        marker = self.markers[snapped_marker_name]

        push_pos = marker["local_pos"]
        marker_targ_idx, marker_targ_link = marker["relative"]
        marker_abs_art_idx = self.art_obj_ids[marker_targ_idx]

        constraint_id = self._sim.create_articulated_p2p_constraint(
            self.robot._robot.get_robot_sim_id(),
            self.ee_link,
            EE_GRIPPER_OFFSET,
            marker_abs_art_idx,
            marker_targ_link,
            mn.Vector3(*push_pos),
            100.0,
        )
        if constraint_id == -1:
            raise ValueError("Created bad constraint")
        self.snapped_obj_constraint_id = [constraint_id]
        self.snapped_marker_name = snapped_marker_name

    def desnap_object(self, force=False):
        """
        Remove the constraint for holding an object OR articulated object.
        """
        if len(self.snapped_obj_constraint_id) == 0:
            # No constraints to unsnap
            self.snapped_obj_id = None
            self.snapped_marker_name = None
            return

        if self.snapped_obj_id is not None and self.snap_to_link_on_grab:
            # todo: find a safe time to restore the collision group for the
            # grasped object. At the moment of release, it is probably
            # still overlapping the link, so now is not a good time to
            # re-enable collision between the object and robot.
            # free_object_group = 8
            # self._sim.override_collision_group(use_snap_obj_id, free_object_group)
            # TODO: This AABB will not work well for rotated objects.
            obj_bb = get_aabb(self.snapped_obj_id, self._sim)
            r = max(obj_bb.size_x(), obj_bb.size_y(), obj_bb.size_z())
            c = self._sim.get_translation(self.snapped_obj_id)
            snap_obj_id = self.snapped_obj_id
            if force:
                self._sim.override_collision_group(snap_obj_id, 1)
            else:

                def is_ready():
                    ee_pos = self.get_end_effector_pos()
                    dist = np.linalg.norm(ee_pos - c)
                    return dist >= r

                self.event_callbacks.append(
                    SimEvent(
                        is_ready,
                        lambda: self._sim.override_collision_group(
                            snap_obj_id, 1
                        ),
                    )
                )
        if self.do_grab_using_constraint:
            for constraint_id in self.snapped_obj_constraint_id:
                self._sim.remove_constraint(constraint_id)
            self.snapped_obj_constraint_id = []

        self.snapped_obj_id = None
        self.snapped_marker_name = None

    def path_to_point(self, point):
        trans = self.get_robot_transform()
        agent_pos = trans.translation
        closest_point = self._sim.pathfinder.snap_point(point)
        path = habitat_sim.ShortestPath()
        path.requested_start = agent_pos
        path.requested_end = closest_point
        found_path = self._sim.pathfinder.find_path(path)
        if not found_path:
            return [agent_pos, point]
        if len(path.points) == 1:
            return [agent_pos, path.points[0]]
        return path.points

    def step(self, action):
        self.update_i += 1

        if self.is_render_obs:
            self._sim._try_acquire_context()
            for obj_idx, _ in self.ep_info["targets"]:
                self._sim.set_object_bb_draw(
                    False, self.scene_obj_ids[obj_idx]
                )
        for viz_obj in self.viz_obj_ids:
            self._sim.remove_object(viz_obj)

        add_back_viz_objs = {}
        for name, viz_id in self.viz_ids.items():
            if viz_id is None:
                continue

            before_pos = self._sim.get_translation(viz_id)
            self._sim.remove_object(viz_id)
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
                for i in range(self.ac_freq_ratio):
                    self.internal_step(-1)

            self._prev_sim_obs = self._sim.get_sensor_observations()
            # self._sim_obs_queue.put(self._sensor_suite.get_observations(self._prev_sim_obs))
            # obs = self._sim_obs_queue.get()
            obs = self._sensor_suite.get_observations(self._prev_sim_obs)

        else:
            self._prev_sim_obs = (
                self._sim.get_sensor_observations_async_start()
            )

            if self.habitat_config.get("STEP_PHYSICS", True):
                for i in range(self.ac_freq_ratio):
                    self.internal_step(-1)

            self._prev_sim_obs = (
                self._sim.get_sensor_observations_async_finish()
            )
            # obs = self._sensor_suite.get_observations(self._prev_sim_obs)
            f_obs = self._sensor_suite.get_observations(self._prev_sim_obs)
            self._sim_obs_queue.put(f_obs)
            obs = self._sim_obs_queue.get()  #
            # print("obs: ", hash(frozenset(obs)), "put_obs", hash(frozenset(f_obs)), "queue: ", len(self._sim_obs_queue.queue))

        if "high_rgb" in obs:
            self.is_render_obs = True
            self._sim._try_acquire_context()
            for k, pos in add_back_viz_objs.items():
                self.viz_ids[k] = self.viz_pos(pos)

            # Also render debug information
            if self.habitat_config.get("RENDER_TARGS", True):
                self._create_obj_viz(self.ep_info)

            # Always draw the target
            for obj_idx, _ in self.ep_info["targets"]:
                self._sim.set_object_bb_draw(True, self.scene_obj_ids[obj_idx])

            debug_obs = self._sim.get_sensor_observations()
            obs["high_rgb"] = debug_obs["high_rgb"][:, :, :3]

        if self.habitat_config.HABITAT_SIM_V0.get(
            "ENABLE_GFX_REPLAY_SAVE", False
        ):
            self._sim.gfx_replay_manager.save_keyframe()

        return obs

    def draw_obs(self):
        """Synchronously gets the observation at the current step"""
        # Update the world state to get most recent render
        self.internal_step(-1)

        prev_sim_obs = self._sim.get_sensor_observations()
        obs = self._sensor_suite.get_observations(prev_sim_obs)
        return obs

    def internal_step(self, dt):
        """
        Never call sim.step_world directly.
        """

        self._sim.step_world(dt)
        if self.robot is not None:
            self.robot.update()

        # self._update_markers()

    def get_targets(self):
        """
        - Returns: ([idx: int], [goal_pos: list]) The index of the target object
          in self.scene_obj_ids and the 3D goal POSITION, rotation is IGNORED.
          Note that goal_pos is the desired position of the object, not the
          starting position.
        """
        targ_idx, targ_trans = [x for x in zip(*self._get_target_trans())]

        a, b = np.array(targ_idx), [
            np.array(x.translation) for x in targ_trans
        ]
        return a, np.array(b)

    def get_target_obj_idxs(self):
        """
        - Returns: [idx: int] where the idx is the simulator absolute idx.
        """
        if len(self.get_targets()) == 0:
            return []
        return [self.scene_obj_ids[x] for x in self.get_targets()[0]]

    def get_n_targets(self):
        return self.n_objs

    def get_target_objs_start(self):
        return np.array(self.target_start_pos)

    def get_scene_pos(self):
        return np.array(
            [self._sim.get_translation(idx) for idx in self.scene_obj_ids]
        )

    def get_collisions(self):
        # TODO: NEED TO FIX
        return []

        def extract_coll_info(coll, n_point):
            parts = coll.split(",")
            coll_type, name, link = parts[:3]
            return {
                "type": coll_type.strip(),
                "name": name.strip(),
                "link": link.strip(),
                "n_points": n_point,
            }

        sum_str = self._sim.get_physics_step_collision_summary()
        colls = sum_str.split("\n")
        if "no active" in colls[0]:
            return []
        n_points = [
            int(c.split(",")[-1].strip().split(" ")[0])
            for c in colls
            if c != ""
        ]
        colls = [
            [extract_coll_info(x, n) for x in re.findall("\[(.*?)\]", s)]
            for n, s in zip(n_points, colls)
        ]
        colls = [x for x in colls if len(x) != 0]
        return colls

    def is_holding_obj(self):
        return self.snapped_obj_id is not None

    def draw_sphere(self, r, template_name="ball_new"):
        obj_mgr = self._sim.get_object_template_manager()
        template_handle = obj_mgr.get_template_handles("sphere")[0]
        template = obj_mgr.get_template_by_handle(template_handle)
        template.scale = mn.Vector3(r, r, r)
        new_template_handle = obj_mgr.register_template(template, "ball_new")
        obj_id = self._sim.add_object(new_template_handle)
        self._sim.set_object_motion_type(MotionType.KINEMATIC, obj_id)
        return obj_id

    def get_agent_state(self, agent_id=0):
        prev_state = super().get_agent_state()
        trans = self.get_robot_transform()
        pos = np.array(trans.translation)
        rot = mn.Quaternion.from_matrix(trans.rotation())
        rot = quaternion.quaternion(*rot.vector, rot.scalar)
        new_state = copy.copy(prev_state)
        new_state.position = pos
        new_state.rotation = rot
        return new_state
