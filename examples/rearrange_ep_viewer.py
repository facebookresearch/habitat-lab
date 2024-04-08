# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
import math
import os
import string
import sys
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

flags = sys.getdlopenflags()
sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)

import magnum as mn
import numpy as np
from magnum import shaders, text
from magnum.platform.glfw import Application

import habitat.articulated_agents.robots.spot_robot as spot_robot
import habitat.datasets.rearrange.samplers.receptacle as hab_receptacle
import habitat_sim
from habitat.datasets.rearrange.navmesh_utils import get_largest_island_index
from habitat.datasets.rearrange.rearrange_dataset import (
    RearrangeDatasetV0,
    RearrangeEpisode,
)
from habitat.sims.habitat_simulator.sim_utilities import get_bb_corners
from habitat.tasks.rearrange.utils import (
    embodied_unoccluded_navmesh_snap,
    get_angle_to_pos,
)
from habitat_sim import ReplayRenderer, ReplayRendererConfiguration, physics
from habitat_sim.logging import LoggingContext, logger
from habitat_sim.utils.common import quat_from_angle_axis, quat_to_magnum
from habitat_sim.utils.settings import default_sim_settings, make_cfg


class HabitatSimInteractiveViewer(Application):
    # the maximum number of chars displayable in the app window
    # using the magnum text module. These chars are used to
    # display the CPU/GPU usage data
    MAX_DISPLAY_TEXT_CHARS = 256

    # how much to displace window text relative to the center of the
    # app window (e.g if you want the display text in the top left of
    # the app window, you will displace the text
    # window width * -TEXT_DELTA_FROM_CENTER in the x axis and
    # window height * TEXT_DELTA_FROM_CENTER in the y axis, as the text
    # position defaults to the middle of the app window)
    TEXT_DELTA_FROM_CENTER = 0.49

    # font size of the magnum in-window display text that displays
    # CPU and GPU usage info
    DISPLAY_FONT_SIZE = 16.0

    def __init__(self, sim_settings: Dict[str, Any]) -> None:
        self.sim_settings: Dict[str:Any] = sim_settings

        self.enable_batch_renderer: bool = self.sim_settings[
            "enable_batch_renderer"
        ]
        self.num_env: int = (
            self.sim_settings["num_environments"]
            if self.enable_batch_renderer
            else 1
        )

        # Compute environment camera resolution based on the number of environments to render in the window.
        window_size: mn.Vector2 = (
            self.sim_settings["window_width"],
            self.sim_settings["window_height"],
        )

        configuration = self.Configuration()
        configuration.title = "Habitat Sim Interactive Viewer"
        configuration.size = window_size
        Application.__init__(self, configuration)
        self.fps: float = 60.0

        # Compute environment camera resolution based on the number of environments to render in the window.
        grid_size: mn.Vector2i = ReplayRenderer.environment_grid_size(
            self.num_env
        )
        camera_resolution: mn.Vector2 = mn.Vector2(
            self.framebuffer_size
        ) / mn.Vector2(grid_size)
        self.sim_settings["width"] = camera_resolution[0]
        self.sim_settings["height"] = camera_resolution[1]

        # draw Bullet debug line visualizations (e.g. collision meshes)
        self.debug_bullet_draw = False
        # draw active contact point debug line visualizations
        self.contact_debug_draw = False
        # cache most recently loaded URDF file for quick-reload
        self.cached_urdf = ""

        # set up our movement map
        key = Application.KeyEvent.Key
        self.pressed = {
            key.UP: False,
            key.DOWN: False,
            key.LEFT: False,
            key.RIGHT: False,
            key.A: False,
            key.D: False,
            key.S: False,
            key.W: False,
            key.X: False,
            key.Z: False,
        }

        # set up our movement key bindings map
        key = Application.KeyEvent.Key
        self.key_to_action = {
            key.UP: "look_up",
            key.DOWN: "look_down",
            key.LEFT: "turn_left",
            key.RIGHT: "turn_right",
            key.A: "move_left",
            key.D: "move_right",
            key.S: "move_backward",
            key.W: "move_forward",
            key.X: "move_down",
            key.Z: "move_up",
        }

        # Load a TrueTypeFont plugin and open the font file
        self.display_font = text.FontManager().load_and_instantiate(
            "TrueTypeFont"
        )
        relative_path_to_font = "../data/fonts/ProggyClean.ttf"
        self.display_font.open_file(
            os.path.join(os.path.dirname(__file__), relative_path_to_font),
            13,
        )

        # Glyphs we need to render everything
        self.glyph_cache = text.GlyphCache(mn.Vector2i(256))
        self.display_font.fill_glyph_cache(
            self.glyph_cache,
            string.ascii_lowercase
            + string.ascii_uppercase
            + string.digits
            + ":-_+,.! %µ",
        )

        # magnum text object that displays CPU/GPU usage data in the app window
        self.window_text = text.Renderer2D(
            self.display_font,
            self.glyph_cache,
            HabitatSimInteractiveViewer.DISPLAY_FONT_SIZE,
            text.Alignment.TOP_LEFT,
        )
        self.window_text.reserve(
            HabitatSimInteractiveViewer.MAX_DISPLAY_TEXT_CHARS
        )

        # text object transform in window space is Projection matrix times Translation Matrix
        # put text in top left of window
        self.window_text_transform = mn.Matrix3.projection(
            self.framebuffer_size
        ) @ mn.Matrix3.translation(
            mn.Vector2(self.framebuffer_size)
            * mn.Vector2(
                -HabitatSimInteractiveViewer.TEXT_DELTA_FROM_CENTER,
                HabitatSimInteractiveViewer.TEXT_DELTA_FROM_CENTER,
            )
        )
        self.shader = shaders.VectorGL2D()

        # make magnum text background transparent
        mn.gl.Renderer.enable(mn.gl.Renderer.Feature.BLENDING)
        mn.gl.Renderer.set_blend_function(
            mn.gl.Renderer.BlendFunction.ONE,
            mn.gl.Renderer.BlendFunction.ONE_MINUS_SOURCE_ALPHA,
        )
        mn.gl.Renderer.set_blend_equation(
            mn.gl.Renderer.BlendEquation.ADD, mn.gl.Renderer.BlendEquation.ADD
        )

        # variables that track app data and CPU/GPU usage
        self.num_frames_to_track = 60

        # Cycle mouse utilities
        self.mouse_interaction = MouseMode.LOOK
        self.mouse_grabber: Optional[MouseGrabber] = None
        self.previous_mouse_point = None

        # toggle physics simulation on/off
        self.simulating = True

        # toggle a single simulation step at the next opportunity if not
        # simulating continuously.
        self.simulate_single_step = False

        # configure our simulator
        self.cfg: Optional[habitat_sim.simulator.Configuration] = None
        self.sim: Optional[habitat_sim.simulator.Simulator] = None
        self.tiled_sims: list[habitat_sim.simulator.Simulator] = None
        self.replay_renderer_cfg: Optional[ReplayRendererConfiguration] = None
        self.replay_renderer: Optional[ReplayRenderer] = None
        self.largest_indoor_island = -1
        self.reconfigure_sim()
        self.episode_dataset = None
        self.episode = None
        self.ep_objects = []
        self.selected_obj = 0
        self.show_obj_hints = True
        self.receptacles = []
        self.show_receptacles = False
        # map receptacle to parent objects
        self.rec_to_poh: Dict[hab_receptacle.Receptacle, str] = {}
        # contains filtering metadata and classification of meshes filtered automatically and manually
        self.rec_filter_data = None
        self.prev_test_batch_points: List[
            Tuple[mn.Vector3, float]
        ] = []  # points and radii

        self.ep_ix = 0
        self.load_rearrange_dataset(sim_settings["episode_dataset"])
        self.load_episode()
        self.spot = None
        self.spot_navmesh_offsets = [
            [0.0, 0.0],
            [0.25, 0.0],
            [-0.25, 0.0],
        ]  # from lab configs
        # NOTE: generator config doesn't work, different coordinate space. Refactor generator to match the action or modularize both
        # self.spot_navmesh_offsets = [[0, 0], [0, 0.15], [0, -0.15]] #from generator config
        self.load_spot()
        self.data_out_dict = {}  # used to cache debug values
        self.robot_orientation = 0  # radians
        self.robot_occlusion_height = 0.75
        self.orientation_noise = 0

        # compute NavMesh if not already loaded by the scene.
        if (
            not self.sim.pathfinder.is_loaded
            and self.cfg.sim_cfg.scene_id.lower() != "none"
        ):
            self.navmesh_config_and_recompute()

        self.time_since_last_simulation = 0.0
        LoggingContext.reinitialize_from_env()
        logger.setLevel("INFO")
        self.print_help_text()

    def load_spot(self):
        """
        Load a Spot robot via the robot wrapper interface for testing.
        """
        from omegaconf import DictConfig

        # add the robot to the world via the wrapper
        robot_path = "data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf"
        agent_config = DictConfig({"articulated_agent_urdf": robot_path})
        self.spot = spot_robot.SpotRobot(
            agent_config, self.sim, fixed_base=True
        )
        self.spot.reconfigure()
        self.spot.update()

    def load_rearrange_dataset(self, filename: str) -> None:
        """
        Load a RearrangeDataset from json.gz filepath.
        """
        self.episode_dataset = RearrangeDatasetV0()
        self.episode_dataset._load_from_file(fname=filename, scenes_dir=None)
        print(
            f"Loaded RearrangeDataset from {filename} with {len(self.episode_dataset.episodes)} episodes."
        )
        self.ep_ix = 0
        self.episode = self.episode_dataset.episodes[self.ep_ix]

    def load_ep_objects(self):
        """
        Initialize and place the objects for the episode.
        """
        self.selected_obj = 0
        self.ep_objects = []
        self.prev_test_batch_points = []
        rom = self.sim.get_rigid_object_manager()
        rotm = self.sim.metadata_mediator.object_template_manager
        for i, (obj_handle, transform) in enumerate(self.episode.rigid_objs):
            matching_template_handles = rotm.get_template_handles(obj_handle)
            assert len(matching_template_handles) == 1

            ro = rom.add_object_by_template_handle(
                matching_template_handles[0]
            )

            # The saved matrices need to be flipped when reloading.
            ro.transformation = mn.Matrix4(
                [[transform[j][i] for j in range(4)] for i in range(4)]
            )
            ro.angular_velocity = mn.Vector3.zero_init()
            ro.linear_velocity = mn.Vector3.zero_init()

            self.ep_objects.append(ro)

    def load_episode(self):
        self.sim_settings["scene"] = self.episode.scene_id
        print(
            f"Loading episode {self.ep_ix} in scene {self.episode.scene_id}."
        )
        print(f"scene = {self.episode.scene_id}")
        self.sim.metadata_mediator.active_dataset = (
            self.episode.scene_dataset_config
        )
        print(
            self.sim.metadata_mediator.ao_template_manager.get_num_templates()
        )
        print(
            self.sim.metadata_mediator.ao_template_manager.get_template_handles()
        )
        for obj_path in self.episode.additional_obj_config_paths:
            self.sim.get_object_template_manager().load_configs(obj_path)
        self.reconfigure_sim()
        self.navmesh_config_and_recompute()

        # load the objects
        self.load_ep_objects()
        self.load_receptacles()
        self.load_filtered_recs()

    def load_filtered_recs(self) -> None:
        """
        Load a Receptacle filtering metadata JSON to visualize the state of the scene.

        :param filepath: Defines the input filename for this JSON. If omitted, defaults to "./rec_filter_data.json".
        """
        scene_user_defined = self.sim.metadata_mediator.get_scene_user_defined(
            self.sim.curr_scene_name
        )
        if scene_user_defined is not None and scene_user_defined.has_value(
            "scene_filter_file"
        ):
            scene_filter_file = scene_user_defined.get("scene_filter_file")
            # construct the dataset level path for the filter data file
            filepath = os.path.join(
                os.path.dirname(self.sim.metadata_mediator.active_dataset),
                scene_filter_file,
            )
            import json

            if not os.path.exists(filepath):
                print(
                    f"Filtered rec metadata file {filepath} does not exist. Cannot load."
                )
                return
            with open(filepath, "r") as f:
                self.rec_filter_data = json.load(f)

            # assert the format is correct
            assert "active" in self.rec_filter_data
            assert "manually_filtered" in self.rec_filter_data
            assert "access_filtered" in self.rec_filter_data
            assert "stability_filtered" in self.rec_filter_data
            assert "height_filtered" in self.rec_filter_data
            print(f"Loaded filter annotations from {filepath}")
        else:
            print("No rec filter file configured for the scene...")

    def load_receptacles(self):
        """
        Load all receptacle data and setup helper datastructures.
        """
        self.receptacles = hab_receptacle.find_receptacles(self.sim)
        self.receptacles = [
            rec
            for rec in self.receptacles
            if "collision_stand-in" not in rec.parent_object_handle
        ]
        for receptacle in self.receptacles:
            if receptacle not in self.rec_to_poh:
                po_handle = (
                    self.sim.get_rigid_object_manager()
                    .get_object_by_handle(receptacle.parent_object_handle)
                    .creation_attributes.handle
                )
                self.rec_to_poh[receptacle] = po_handle

    def get_rec_instance_name(
        self, receptacle: hab_receptacle.Receptacle
    ) -> str:
        """
        Gets a unique string name for the Receptacle instance.
        Multiple Receptacles can share a name (e.g. if the object has multiple instances in the scene).
        The unique name is constructed as '<object instance name>|<receptacle name>'.
        """
        rec_unique_name = (
            receptacle.parent_object_handle + "|" + receptacle.name
        )
        return rec_unique_name

    def draw_contact_debug(self):
        """
        This method is called to render a debug line overlay displaying active contact points and normals.
        Yellow lines show the contact distance along the normal and red lines show the contact normal at a fixed length.
        """
        yellow = mn.Color4.yellow()
        red = mn.Color4.red()
        cps = self.sim.get_physics_contact_points()
        self.sim.get_debug_line_render().set_line_width(1.5)
        camera_position = (
            self.render_camera.render_camera.node.absolute_translation
        )
        # only showing active contacts
        # active_contacts = (x for x in cps if x.is_active)
        for cp in cps:
            # red shows the contact distance
            self.sim.get_debug_line_render().draw_transformed_line(
                cp.position_on_b_in_ws,
                cp.position_on_b_in_ws
                + cp.contact_normal_on_b_in_ws * -cp.contact_distance,
                red,
            )
            display_color = yellow
            if not cp.is_active:
                display_color = mn.Color4.magenta()
            # yellow shows the contact normal at a fixed length for visualization
            self.sim.get_debug_line_render().draw_transformed_line(
                cp.position_on_b_in_ws,
                # + cp.contact_normal_on_b_in_ws * cp.contact_distance,
                cp.position_on_b_in_ws + cp.contact_normal_on_b_in_ws * 0.1,
                display_color,
            )
            self.sim.get_debug_line_render().draw_circle(
                translation=cp.position_on_b_in_ws,
                radius=0.005,
                color=display_color,
                normal=camera_position - cp.position_on_b_in_ws,
            )

    def snap_point_is_occluded(
        self,
        target: mn.Vector3,
        snap_point: mn.Vector3,
        height: float,
        sim: habitat_sim.Simulator,
        granularity: float = 0.2,
        target_object_id: Optional[int] = None,
    ) -> bool:
        """
        Uses raycasting to check whether a target is occluded given a navmesh snap point.

        :property target: The 3D position which should be unoccluded from the snap point.
        :property snap_point: The navmesh snap point under consideration.
        :property height: The height of the agent. Given navmesh snap point is grounded, the maximum height from which a visibility check should indicate non-occlusion. First check starts from this height.
        :property sim: The Simulator instance.
        :property granularity: The distance between raycast samples. Finer granularity is more accurate, but more expensive.
        :property target_object_id: An optional object id which should be ignored in occlusion check.

        NOTE: If agent's eye height is known and only that height should be considered, provide eye height and granulatiry > height for fastest check.

        :return: whether or not the target is considered occluded from the snap_point.
        """
        # start from the top, assuming the agent's eyes are not at the bottom.
        cur_height = height
        while cur_height > 0:
            ray = habitat_sim.geo.Ray()
            ray.origin = snap_point + mn.Vector3(0, cur_height, 0)
            cur_height -= granularity
            ray.direction = target - ray.origin
            raycast_results = sim.cast_ray(ray)
            # distance of 1 is the displacement between the two points
            if (
                raycast_results.has_hits()
                and raycast_results.hits[0].ray_distance < 1
            ):
                if (
                    target_object_id is not None
                    and raycast_results.hits[0].object_id == target_object_id
                ):
                    # we hit an allowed object (i.e., the target object), so not occluded
                    return False
                # the ray hit a not-allowed object and is occluded
                continue
            else:
                # ray hit nothing, so not occluded
                return False
        return True

    def unoccluded_snap(
        self, pos: mn.Vector3, target_object_id: Optional[int] = None
    ) -> Optional[mn.Vector3]:
        """
        Consider point visibilty with raycast in snap.
        """
        self.prev_test_batch_points = []
        # first try the closest snap point
        snap_point = self.sim.pathfinder.snap_point(
            pos, self.largest_indoor_island
        )
        is_occluded = self.snap_point_is_occluded(
            target=pos,
            snap_point=snap_point,
            height=self.cfg.agents[self.agent_id].height,
            sim=self.sim,
            target_object_id=target_object_id,
        )
        is_occluded = True
        point_near_times = []

        # now sample and try different snap options
        if is_occluded:
            ###############################################
            # hyper-parameters for the search algorithm
            ###############################################
            # how much further than the minimum to search
            search_offset = 1.5
            # how many sampled points to test within the radius
            test_batch_size = 50
            # number of samples to attempt for finding the batch
            max_samples = 200
            # minimum distance allowed between samples in the batch
            min_sample_dist = 0.5
            ###############################################

            # distance to closest snap point is the absolute minimum
            min_radius = (snap_point - pos).length()
            search_radius = min_radius + search_offset

            # gather a test batch
            test_batch = []
            sample_count = 0
            while (
                len(test_batch) < test_batch_size
                and sample_count < max_samples
            ):
                start_time = time.time()
                sample = self.sim.pathfinder.get_random_navigable_point_near(
                    circle_center=pos,
                    radius=search_radius,
                    island_index=self.largest_indoor_island,
                )
                # print(f"sample = {sample}")
                point_near_times.append(time.time() - start_time)
                reject = False
                for batch_sample in test_batch:
                    if (
                        np.linalg.norm(sample - batch_sample[0])
                        < min_sample_dist
                    ):
                        reject = True
                        break
                if not reject:
                    test_batch.append((sample, np.linalg.norm(sample - pos)))
                sample_count += 1
            print(
                f"Avg. point near time = {np.sum(point_near_times)/len(point_near_times)}"
            )
            # new = 0.00023991942405700683
            # old = 5.588531494140625e-06
            # new2 = 7.155656814575195e-05
            # print(f"Found {len(test_batch)}/{test_batch_size} samples for the batch.")

            # pairwise_distances = np.array([np.linalg.norm(p1[0]-p2[0]) for p1 in test_batch for p2 in test_batch if not np.array_equal(p1[0],p2[0])])
            # print(pairwise_distances)
            # assert np.min(pairwise_distances) > min_sample_dist

            # sort the points by distance to the target
            test_batch.sort(key=lambda s: s[1])

            self.prev_test_batch_points = [
                (s[0], min_sample_dist / 2.0) for s in test_batch
            ]
            self.prev_test_batch_points.append((pos, search_radius))

            # search for an un-occluded option
            for batch_sample in test_batch:
                if not self.snap_point_is_occluded(
                    pos,
                    batch_sample[0],
                    self.cfg.agents[self.agent_id].height,
                    sim=self.sim,
                    target_object_id=target_object_id,
                ):
                    return batch_sample[0]

            return None

        print(f"snap_point = {snap_point}")

        return snap_point

    def look_at_object(
        self, obj: habitat_sim.physics.ManagedRigidObject
    ) -> None:
        """
        Re-position the camera agent to look at the object.
        """
        snap_point = self.unoccluded_snap(obj.translation, obj.object_id)
        if snap_point is None:
            print("Could not find an unoccluded snap point...")
            return

        self.default_agent.scene_node.translation = snap_point

        # turn agent toward the object
        agent_to_obj = obj.translation - snap_point
        agent_local_forward = np.array([0, 0, -1.0])
        flat_to_obj = np.array([agent_to_obj[0], 0.0, agent_to_obj[2]])
        flat_dist_to_obj = np.linalg.norm(flat_to_obj)
        flat_to_obj /= flat_dist_to_obj
        # unit y normal plane for rotation
        det = (
            flat_to_obj[0] * agent_local_forward[2]
            - agent_local_forward[0] * flat_to_obj[2]
        )
        turn_angle = math.atan2(det, np.dot(agent_local_forward, flat_to_obj))
        self.default_agent.scene_node.rotation = quat_to_magnum(
            quat_from_angle_axis(turn_angle, np.array([0, 1.0, 0]))
        )

        local_point = self.default_agent.scene_node.transformation.inverted().transform_point(
            obj.translation
        )

        self.render_camera.node.transformation = mn.Matrix4.look_at(
            eye=self.render_camera.node.translation,
            target=local_point,
            up=mn.Vector3(0, 1, 0),
        )

    def draw_object_highlights(self):
        """
        Draw circles around all rigid clutter objects for the episodes.
        """
        if self.show_obj_hints:
            render_cam_abs_translation = (
                self.render_camera.node.absolute_translation
            )
            for oix, obj in enumerate(self.ep_objects):
                color = mn.Color4(0.2, 0.2, 0.75, 1.0)
                if oix == self.selected_obj:
                    color = mn.Color4(0.75, 0.1, 0.1, 1.0)
                obj_size = obj.root_scene_node.cumulative_bb.size().max() / 2
                self.sim.get_debug_line_render().draw_circle(
                    translation=obj.translation,
                    radius=obj_size,
                    # pink
                    color=color,
                    normal=render_cam_abs_translation - obj.translation,
                )

    def draw_receptacles(self):
        """
        Draw receptacles meshes colored by filter status.
        """
        if self.receptacles is not None and self.show_receptacles:
            c_pos = self.render_camera.node.absolute_translation
            c_forward = self.render_camera.node.absolute_transformation().transform_vector(
                mn.Vector3(0, 0, -1)
            )
            for receptacle in self.receptacles:
                rom = self.sim.get_rigid_object_manager()
                rec_unique_name = self.get_rec_instance_name(receptacle)
                r_trans = receptacle.get_global_transform(self.sim)
                rec_obj = rom.get_object_by_handle(
                    receptacle.parent_object_handle
                )
                key_points = [r_trans.translation]
                key_points.extend(
                    get_bb_corners(rec_obj.root_scene_node.cumulative_bb)
                )

                in_view = False
                for ix, key_point in enumerate(key_points):
                    r_pos = key_point
                    if ix > 0:
                        r_pos = rec_obj.transformation.transform_point(
                            key_point
                        )
                    c_to_r = r_pos - c_pos
                    # only display receptacles within 4 meters centered in view
                    if (
                        c_to_r.length() < 4
                        and mn.math.dot((c_to_r).normalized(), c_forward) > 0.7
                    ):
                        in_view = True
                        break
                if in_view:
                    # handle coloring
                    rec_color = None
                    if self.rec_filter_data is not None:
                        # blue indicates no filter data for the receptacle, it may be newer than the filter file.
                        rec_color = mn.Color4.blue()
                        if rec_unique_name in self.rec_filter_data["active"]:
                            rec_color = mn.Color4.green()
                        elif (
                            rec_unique_name
                            in self.rec_filter_data["manually_filtered"]
                        ):
                            rec_color = mn.Color4.yellow()
                        elif (
                            rec_unique_name
                            in self.rec_filter_data["access_filtered"]
                        ):
                            rec_color = mn.Color4.red()
                        elif (
                            rec_unique_name
                            in self.rec_filter_data["stability_filtered"]
                        ):
                            rec_color = mn.Color4.magenta()
                        elif (
                            rec_unique_name
                            in self.rec_filter_data["height_filtered"]
                        ):
                            rec_color = mn.Color4.blue()

                    receptacle.debug_draw(self.sim, color=rec_color)
                    if False:
                        dblr = self.sim.get_debug_line_render()
                        t_form = receptacle.get_global_transform(self.sim)
                        dblr.push_transform(t_form)
                        dblr.draw_transformed_line(
                            mn.Vector3(0), receptacle.up, mn.Color4.cyan()
                        )
                        dblr.pop_transform()

    def draw_snap_points(self, points: List[mn.Vector3]) -> None:
        """
        Draw cirlces with Y normal for a list of points (e.g. from the navmesh).
        """
        for p in points:
            self.sim.get_debug_line_render().draw_circle(
                translation=p[0],
                radius=p[1],
                color=mn.Color4(0.7, 0.7, 0.7, 1.0),
                normal=mn.Vector3(0, 1, 0),
            )

    def draw_robot_embodiement_heuristic(self):
        """
        Draw cirlces with Y normal for a list of points (e.g. from the navmesh).
        """
        if self.spot is not None:
            robot_transform = self.spot.base_transformation
            robot_transform = self.spot.sim_obj.transformation
            for xz in self.spot_navmesh_offsets:
                xyz_local = mn.Vector3(xz[0], 0, xz[1])
                xyz_global = robot_transform.transform_point(xyz_local)

                circle_color = mn.Color4.cyan()
                if not (
                    self.sim.pathfinder.is_navigable(xyz_global)
                    and (
                        self.largest_indoor_island == -1
                        or self.sim.pathfinder.get_island(xyz_global)
                        == self.largest_indoor_island
                    )
                ):
                    # point not on navmesh
                    circle_color = mn.Color4.red()

                self.sim.get_debug_line_render().draw_circle(
                    translation=xyz_global,
                    radius=self.cfg.agents[self.agent_id].radius,
                    color=circle_color,
                    normal=mn.Vector3(0, 1, 0),
                )
            # glob_forward = None
            # for vec in [mn.Vector3(1,0,0), mn.Vector3(0,1,0), mn.Vector3(0,0,1)]:
            #     glob_vec = robot_transform.transform_vector(vec)
            #     self.sim.get_debug_line_render().draw_transformed_line(robot_transform.translation+glob_vec, robot_transform.translation, mn.Color4(vec))
            #     if glob_forward is None:
            #         glob_forward = glob_vec

            self.sim.get_debug_line_render().draw_transformed_line(
                self.spot.base_pos,
                self.spot.base_pos
                + mn.Vector3(0, self.robot_occlusion_height, 0),
                mn.Color4.green(),
            )

            # angle_test = get_angle_to_pos(glob_forward)
            # print(f"forward angle = {angle_test}")
            # rotation_2d = mn.Matrix3.rotation(-mn.Rad(angle_test))
            # p_front_local = mn.Vector2(1,0)
            # p_front_global = rotation_2d.transform_vector(p_front_local)
            # p_front_3d = mn.Vector3(p_front_global[0],0, p_front_global[1])
            # self.sim.get_debug_line_render().draw_transformed_line(p_front_3d + robot_transform.translation, robot_transform.translation, mn.Color4.yellow())
            # rotation_3d = mn.Matrix4.rotation(mn.Rad(angle_test), mn.Vector3(0,1,0))
            # #rotation_3d = mn.Quaternion.rotation(
            # #    mn.Rad(angle_test), mn.Vector3(0, 1, 0)
            # #)
            # p_front_local_3d = mn.Vector3(1.2,0,0)
            # p_front_global_3d = rotation_3d.transform_point(p_front_local_3d)
            # p_front_3d = p_front_global_3d# + robot_transform.translation
            # self.sim.get_debug_line_render().draw_transformed_line(p_front_3d + robot_transform.translation, robot_transform.translation, mn.Color4.magenta())

            if "offsets_3d" in self.data_out_dict:
                for xyz in self.data_out_dict["offsets_3d"]:
                    self.sim.get_debug_line_render().draw_circle(
                        translation=xyz,
                        radius=self.cfg.agents[self.agent_id].radius,
                        color=mn.Color4.yellow(),
                        normal=mn.Vector3(0, 1, 0),
                    )

    def draw_occlusion_rays(self, target_obj):
        height = abs(math.sin(self.sim.get_world_time()))
        snap_point = self.unoccluded_snap(
            target_obj.translation, target_obj.object_id
        )
        for p in self.prev_test_batch_points:
            is_occluded = self.snap_point_is_occluded(
                target=target_obj.translation,
                snap_point=p[0],
                height=height,
                sim=self.sim,
                granularity=2.0,
                target_object_id=target_obj.object_id,
            )
            color = mn.Color4.green()
            if is_occluded:
                color = mn.Color4.red()
            print(p[0])
            self.sim.get_debug_line_render().draw_transformed_line(
                p[0] + mn.Vector3(0, height, 0), target_obj.translation, color
            )

    def debug_draw(self):
        """
        Additional draw commands to be called during draw_event.
        """
        if self.debug_bullet_draw:
            render_cam = self.render_camera.render_camera
            proj_mat = render_cam.projection_matrix.__matmul__(
                render_cam.camera_matrix
            )
            self.sim.physics_debug_draw(proj_mat)
        if self.contact_debug_draw:
            self.draw_contact_debug()
        self.draw_object_highlights()
        self.draw_receptacles()
        # self.draw_occlusion_rays(self.ep_objects[self.selected_obj])
        self.draw_snap_points(self.prev_test_batch_points)
        self.draw_robot_embodiement_heuristic()

    def draw_event(
        self,
        simulation_call: Optional[Callable] = None,
        global_call: Optional[Callable] = None,
        active_agent_id_and_sensor_name: Tuple[int, str] = (0, "color_sensor"),
    ) -> None:
        """
        Calls continuously to re-render frames and swap the two frame buffers
        at a fixed rate.
        """
        agent_acts_per_sec = self.fps

        mn.gl.default_framebuffer.clear(
            mn.gl.FramebufferClear.COLOR | mn.gl.FramebufferClear.DEPTH
        )

        # Agent actions should occur at a fixed rate per second
        self.time_since_last_simulation += Timer.prev_frame_duration
        num_agent_actions: int = (
            self.time_since_last_simulation * agent_acts_per_sec
        )
        self.move_and_look(int(num_agent_actions))

        # Occasionally a frame will pass quicker than 1/60 seconds
        if self.time_since_last_simulation >= 1.0 / self.fps:
            if self.simulating or self.simulate_single_step:
                self.sim.step_world(1.0 / self.fps)
                self.simulate_single_step = False
                if simulation_call is not None:
                    simulation_call()
            if global_call is not None:
                global_call()

            # reset time_since_last_simulation, accounting for potential overflow
            self.time_since_last_simulation = math.fmod(
                self.time_since_last_simulation, 1.0 / self.fps
            )

        keys = active_agent_id_and_sensor_name

        if self.enable_batch_renderer:
            self.render_batch()
        else:
            self.sim._Simulator__sensors[keys[0]][keys[1]].draw_observation()
            agent = self.sim.get_agent(keys[0])
            self.render_camera = agent.scene_node.node_sensor_suite.get(
                keys[1]
            )
            self.debug_draw()
            self.render_camera.render_target.blit_rgba_to_default()

        # draw CPU/GPU usage data and other info to the app window
        mn.gl.default_framebuffer.bind()
        self.draw_text(self.render_camera.specification())

        self.swap_buffers()
        # if self.spot is not None and self.spot.sim_obj.contact_test():
        #    self.simulating = False
        #    print("!!!!!!!!!!!!!!!!CONTACT!!!!!!!!!!!!!!!!!!")
        #    self.contact_debug_draw = True
        Timer.next_frame()
        self.redraw()

    def default_agent_config(self) -> habitat_sim.agent.AgentConfiguration:
        """
        Set up our own agent and agent controls
        """
        make_action_spec = habitat_sim.agent.ActionSpec
        make_actuation_spec = habitat_sim.agent.ActuationSpec
        MOVE, LOOK = 0.07, 1.5

        # all of our possible actions' names
        action_list = [
            "move_left",
            "turn_left",
            "move_right",
            "turn_right",
            "move_backward",
            "look_up",
            "move_forward",
            "look_down",
            "move_down",
            "move_up",
        ]

        action_space: Dict[str, habitat_sim.agent.ActionSpec] = {}

        # build our action space map
        for action in action_list:
            actuation_spec_amt = MOVE if "move" in action else LOOK
            action_spec = make_action_spec(
                action, make_actuation_spec(actuation_spec_amt)
            )
            action_space[action] = action_spec

        sensor_spec: List[habitat_sim.sensor.SensorSpec] = self.cfg.agents[
            self.agent_id
        ].sensor_specifications

        agent_config = habitat_sim.agent.AgentConfiguration(
            height=1.5,
            radius=0.25,  # NOTE: Spot robot navmesh radius
            sensor_specifications=sensor_spec,
            action_space=action_space,
            body_type="cylinder",
        )
        return agent_config

    def reconfigure_sim(self) -> None:
        """
        Utilizes the current `self.sim_settings` to configure and set up a new
        `habitat_sim.Simulator`, and then either starts a simulation instance, or replaces
        the current simulator instance, reloading the most recently loaded scene
        """
        # configure our sim_settings but then set the agent to our default
        self.cfg = make_cfg(self.sim_settings)
        self.agent_id: int = self.sim_settings["default_agent"]
        self.cfg.agents[self.agent_id] = self.default_agent_config()

        if self.enable_batch_renderer:
            self.cfg.enable_batch_renderer = True
            self.cfg.sim_cfg.create_renderer = False
            self.cfg.sim_cfg.enable_gfx_replay_save = True

        if self.sim_settings["use_default_lighting"]:
            logger.info("Setting default lighting override for scene.")
            self.cfg.sim_cfg.override_scene_light_defaults = True
            self.cfg.sim_cfg.scene_light_setup = (
                habitat_sim.gfx.DEFAULT_LIGHTING_KEY
            )

        print(f"Configured scene = {self.cfg.sim_cfg.scene_id}")

        if self.sim is None:
            self.tiled_sims = []
            for _i in range(self.num_env):
                self.tiled_sims.append(habitat_sim.Simulator(self.cfg))
            self.sim = self.tiled_sims[0]
        else:  # edge case
            for i in range(self.num_env):
                if (
                    self.tiled_sims[i].config.sim_cfg.scene_id
                    == self.cfg.sim_cfg.scene_id
                ):
                    # we need to force a reset, so change the internal config scene name
                    self.tiled_sims[i].config.sim_cfg.scene_id = "NONE"
                self.tiled_sims[i].reconfigure(self.cfg)

        # post reconfigure
        self.default_agent = self.sim.get_agent(self.agent_id)
        self.render_camera = (
            self.default_agent.scene_node.node_sensor_suite.get("color_sensor")
        )

        # set sim_settings scene name as actual loaded scene
        self.sim_settings["scene"] = self.sim.curr_scene_name

        # Initialize replay renderer
        if self.enable_batch_renderer and self.replay_renderer is None:
            self.replay_renderer_cfg = ReplayRendererConfiguration()
            self.replay_renderer_cfg.num_environments = self.num_env
            self.replay_renderer_cfg.standalone = (
                False  # Context is owned by the GLFW window
            )
            self.replay_renderer_cfg.sensor_specifications = self.cfg.agents[
                self.agent_id
            ].sensor_specifications
            self.replay_renderer_cfg.gpu_device_id = (
                self.cfg.sim_cfg.gpu_device_id
            )
            self.replay_renderer_cfg.force_separate_semantic_scene_graph = (
                False
            )
            self.replay_renderer_cfg.leave_context_with_background_renderer = (
                False
            )
            self.replay_renderer = ReplayRenderer.create_batch_replay_renderer(
                self.replay_renderer_cfg
            )
            # Pre-load composite files
            if sim_settings["composite_files"] is not None:
                for composite_file in sim_settings["composite_files"]:
                    self.replay_renderer.preload_file(composite_file)

        Timer.start()
        self.step = -1

    def render_batch(self):
        """
        This method updates the replay manager with the current state of environments and renders them.
        """
        for i in range(self.num_env):
            # Apply keyframe
            keyframe = self.tiled_sims[i].gfx_replay_manager.extract_keyframe()
            self.replay_renderer.set_environment_keyframe(i, keyframe)
            # Copy sensor transforms
            sensor_suite = self.tiled_sims[i]._sensors
            for sensor_uuid, sensor in sensor_suite.items():
                transform = (
                    sensor._sensor_object.node.absolute_transformation()
                )
                self.replay_renderer.set_sensor_transform(
                    i, sensor_uuid, transform
                )
            # Render
            self.replay_renderer.render(mn.gl.default_framebuffer)

    def move_and_look(self, repetitions: int) -> None:
        """
        This method is called continuously with `self.draw_event` to monitor
        any changes in the movement keys map `Dict[KeyEvent.key, Bool]`.
        When a key in the map is set to `True` the corresponding action is taken.
        """
        # avoids unnecessary updates to grabber's object position
        if repetitions == 0:
            return

        key = Application.KeyEvent.Key
        agent = self.sim.agents[self.agent_id]
        press: Dict[key.key, bool] = self.pressed
        act: Dict[key.key, str] = self.key_to_action

        action_queue: List[str] = [act[k] for k, v in press.items() if v]

        for _ in range(int(repetitions)):
            [agent.act(x) for x in action_queue]

        # update the grabber transform when our agent is moved
        if self.mouse_grabber is not None:
            # update location of grabbed object
            self.update_grab_position(self.previous_mouse_point)

    def invert_gravity(self) -> None:
        """
        Sets the gravity vector to the negative of it's previous value. This is
        a good method for testing simulation functionality.
        """
        gravity: mn.Vector3 = self.sim.get_gravity() * -1
        self.sim.set_gravity(gravity)

    def key_press_event(self, event: Application.KeyEvent) -> None:
        """
        Handles `Application.KeyEvent` on a key press by performing the corresponding functions.
        If the key pressed is part of the movement keys map `Dict[KeyEvent.key, Bool]`, then the
        key will be set to False for the next `self.move_and_look()` to update the current actions.
        """
        key = event.key
        pressed = Application.KeyEvent.Key
        mod = Application.InputEvent.Modifier

        shift_pressed = bool(event.modifiers & mod.SHIFT)
        alt_pressed = bool(event.modifiers & mod.ALT)
        # warning: ctrl doesn't always pass through with other key-presses

        if key == pressed.ESC:
            event.accepted = True
            self.exit_event(Application.ExitEvent)
            return

        elif key == pressed.H:
            self.print_help_text()

        elif key == pressed.TAB:
            if shift_pressed:
                self.ep_ix = self.ep_ix - 1
                if self.ep_ix < 0:
                    self.ep_ix = len(self.episode_dataset.episodes) - 1
            else:
                self.ep_ix = (self.ep_ix + 1) % len(
                    self.episode_dataset.episodes
                )
            self.episode = self.episode_dataset.episodes[self.ep_ix]
            self.load_episode()

        elif key == pressed.SPACE:
            if not self.sim.config.sim_cfg.enable_physics:
                logger.warn("Warning: physics was not enabled during setup")
            else:
                self.simulating = not self.simulating
                logger.info(
                    f"Command: physics simulating set to {self.simulating}"
                )

        elif key == pressed.PERIOD:
            if self.simulating:
                logger.warn("Warning: physics simulation already running")
            else:
                self.simulate_single_step = True
                logger.info("Command: physics step taken")

        elif key == pressed.COMMA:
            self.debug_bullet_draw = not self.debug_bullet_draw
            logger.info(
                f"Command: toggle Bullet debug draw: {self.debug_bullet_draw}"
            )

        elif key == pressed.C:
            if shift_pressed:
                self.contact_debug_draw = not self.contact_debug_draw
                logger.info(
                    f"Command: toggle contact debug draw: {self.contact_debug_draw}"
                )
            else:
                # perform a discrete collision detection pass and enable contact debug drawing to visualize the results
                logger.info(
                    "Command: perform discrete collision detection and visualize active contacts."
                )
                self.sim.perform_discrete_collision_detection()
                self.contact_debug_draw = True
                # TODO: add a nice log message with concise contact pair naming.

        elif key == pressed.T:
            if shift_pressed:
                pos = self.default_agent.scene_node.translation
                rad = 0.5
                self.prev_test_batch_points = [(pos, rad)]
                snap_point = (
                    self.sim.pathfinder.get_random_navigable_point_near(
                        circle_center=pos,
                        radius=rad,
                        island_index=self.largest_indoor_island,
                    )
                )
                self.prev_test_batch_points.append((snap_point, 0.1))
            else:
                test_sample_size = 20
                found_occluded_target = False
                print("Testing for occluded objects:")
                # find any objects which are occluded and try to find valid viewpoints for them
                for oix, obj in enumerate(self.ep_objects):
                    basic_snap_point = self.sim.pathfinder.snap_point(
                        obj.translation, self.largest_indoor_island
                    )
                    if self.snap_point_is_occluded(
                        obj.translation,
                        basic_snap_point,
                        self.cfg.agents[self.agent_id].height,
                        sim=self.sim,
                        target_object_id=obj.object_id,
                    ):
                        found_occluded_target = True
                        print(
                            f"    Object {oix} is occluded, attempting snap and testing accuracy."
                        )
                        snap_successes = 0
                        for test_sample in range(test_sample_size):
                            snap_point = self.unoccluded_snap(
                                obj.translation, obj.object_id
                            )
                            if snap_point is not None:
                                snap_successes += 1
                        print(
                            f"    - snap success = {snap_successes}/{test_sample_size}"
                        )
                        self.selected_obj = oix
                if found_occluded_target:
                    self.look_at_object(self.ep_objects[self.selected_obj])
                else:
                    print(" ... no occluded objects.")

        elif key == pressed.M:
            self.cycle_mouse_mode()
            logger.info(f"Command: mouse mode set to {self.mouse_interaction}")

        elif key == pressed.V:
            if shift_pressed:
                self.show_obj_hints = not self.show_obj_hints
                print(f"Show object hints = {self.show_obj_hints}")
            else:
                self.show_receptacles = not self.show_receptacles
                print(f"Show receptacles = {self.show_receptacles}")

        if key == pressed.O:
            if len(self.ep_objects) > 0:
                if alt_pressed:
                    self.look_at_object(self.ep_objects[self.selected_obj])
                elif shift_pressed:
                    self.selected_obj = self.selected_obj - 1
                    if self.selected_obj < 0:
                        self.selected_obj = len(self.ep_objects) - 1
                else:
                    self.selected_obj = (self.selected_obj + 1) % len(
                        self.ep_objects
                    )

        if key == pressed.R:
            if self.spot is not None:
                robot_state: Optional[Tuple[mn.Vector3, float]] = None

                obj_pos = self.ep_objects[self.selected_obj].translation
                obj_id = self.ep_objects[self.selected_obj].object_id
                self.spot.reset()
                if shift_pressed and alt_pressed:
                    print("Both embodiements snapping.")
                    robot_state = embodied_unoccluded_navmesh_snap(
                        target_position=obj_pos,
                        island_id=self.largest_indoor_island,
                        height=self.robot_occlusion_height,
                        pathfinder=self.sim.pathfinder,
                        sim=self.sim,
                        target_object_id=obj_id,
                        orientation_noise=self.orientation_noise,
                        agent_embodiement=self.spot,
                        embodiement_heuristic_offsets=self.spot_navmesh_offsets,
                    )
                elif alt_pressed:
                    print("True embodiement snapping.")
                    robot_state = embodied_unoccluded_navmesh_snap(
                        target_position=obj_pos,
                        island_id=self.largest_indoor_island,
                        height=self.robot_occlusion_height,
                        pathfinder=self.sim.pathfinder,
                        sim=self.sim,
                        target_object_id=obj_id,
                        orientation_noise=self.orientation_noise,
                        agent_embodiement=self.spot,
                        data_out=self.data_out_dict,
                    )
                elif shift_pressed:
                    print("Heuristic embodiement snapping.")
                    robot_state = embodied_unoccluded_navmesh_snap(
                        target_position=obj_pos,
                        island_id=self.largest_indoor_island,
                        height=self.robot_occlusion_height,
                        pathfinder=self.sim.pathfinder,
                        sim=self.sim,
                        target_object_id=obj_id,
                        orientation_noise=self.orientation_noise,
                        embodiement_heuristic_offsets=self.spot_navmesh_offsets,
                        data_out=self.data_out_dict,
                    )
                else:
                    agent_local_forward = np.array([0, 0, -1.0])
                    agent_global_forward = self.default_agent.scene_node.transformation.transform_vector(
                        agent_local_forward
                    )
                    robot_state = (
                        self.default_agent.get_state().position,
                        get_angle_to_pos(agent_global_forward),
                    )
                    print("No embodiement snapping.")
                    robot_state = embodied_unoccluded_navmesh_snap(
                        target_position=obj_pos,
                        island_id=self.largest_indoor_island,
                        height=1.5,
                        pathfinder=self.sim.pathfinder,
                        sim=self.sim,
                        target_object_id=obj_id,
                        orientation_noise=self.orientation_noise,
                    )

                    # Turn the robot
                    # self.robot_orientation += 0.1
                    # if self.robot_orientation > np.pi*2:
                    #     self.robot_orientation = 0
                    # elif self.robot_orientation < 0:
                    #     self.robot_orientation = np.pi*2
                    # robot_state = (
                    #     self.spot.base_pos,
                    #     self.robot_orientation
                    # )

                if robot_state is not None and robot_state[2] == True:
                    self.spot.base_pos = robot_state[0]
                    self.spot.base_rot = robot_state[1]
                    # self.spot.reset()
                    if self.spot.sim_obj.contact_test():
                        print("!!!!!!!!!!!CONTACT!!!!!!!!!!!!")
                        self.spot.sim_obj.awake = True
                        self.sim.perform_discrete_collision_detection()
                        self.contact_debug_draw = True
                        self.simulating = False
                    # print(f"self.spot.base_rot = {self.spot.base_rot}")
                else:
                    print("Failed to get robot state.")

        elif key == pressed.F:
            # change orientation noise
            if shift_pressed:
                self.orientation_noise -= 0.1
                if self.orientation_noise < 0:
                    self.orientation_noise = 0
            else:
                self.orientation_noise += 0.1

        elif key == pressed.N:
            # (default) - toggle navmesh visualization
            # NOTE: (+ALT) - re-sample the agent position on the NavMesh
            # NOTE: (+SHIFT) - re-compute the NavMesh
            if alt_pressed:
                logger.info("Command: resample agent state from navmesh")
                if self.sim.pathfinder.is_loaded:
                    new_agent_state = habitat_sim.AgentState()
                    new_agent_state.position = (
                        self.sim.pathfinder.get_random_navigable_point(
                            self.largest_indoor_island
                        )
                    )
                    new_agent_state.rotation = quat_from_angle_axis(
                        self.sim.random.uniform_float(0, 2.0 * np.pi),
                        np.array([0, 1, 0]),
                    )
                    self.default_agent.set_state(new_agent_state)
                else:
                    logger.warning(
                        "NavMesh is not initialized. Cannot sample new agent state."
                    )
            elif shift_pressed:
                logger.info("Command: recompute navmesh")
                self.navmesh_config_and_recompute()
            else:
                if self.sim.pathfinder.is_loaded:
                    self.sim.navmesh_visualization = (
                        not self.sim.navmesh_visualization
                    )
                    logger.info("Command: toggle navmesh")
                else:
                    logger.warn("Warning: recompute navmesh first")

        # update map of moving/looking keys which are currently pressed
        if key in self.pressed:
            self.pressed[key] = True
        event.accepted = True
        self.redraw()

    def key_release_event(self, event: Application.KeyEvent) -> None:
        """
        Handles `Application.KeyEvent` on a key release. When a key is released, if it
        is part of the movement keys map `Dict[KeyEvent.key, Bool]`, then the key will
        be set to False for the next `self.move_and_look()` to update the current actions.
        """
        key = event.key

        # update map of moving/looking keys which are currently pressed
        if key in self.pressed:
            self.pressed[key] = False
        event.accepted = True
        self.redraw()

    def mouse_move_event(self, event: Application.MouseMoveEvent) -> None:
        """
        Handles `Application.MouseMoveEvent`. When in LOOK mode, enables the left
        mouse button to steer the agent's facing direction. When in GRAB mode,
        continues to update the grabber's object position with our agents position.
        """
        button = Application.MouseMoveEvent.Buttons
        # if interactive mode -> LOOK MODE
        if (
            event.buttons == button.LEFT
            and self.mouse_interaction == MouseMode.LOOK
        ):
            agent = self.sim.agents[self.agent_id]
            delta = self.get_mouse_position(event.relative_position) / 2
            action = habitat_sim.agent.ObjectControls()
            act_spec = habitat_sim.agent.ActuationSpec

            # left/right on agent scene node
            action(agent.scene_node, "turn_right", act_spec(delta.x))

            # up/down on cameras' scene nodes
            action = habitat_sim.agent.ObjectControls()
            sensors = list(
                self.default_agent.scene_node.subtree_sensors.values()
            )
            [
                action(s.object, "look_down", act_spec(delta.y), False)
                for s in sensors
            ]

        # if interactive mode is TRUE -> GRAB MODE
        elif self.mouse_interaction == MouseMode.GRAB and self.mouse_grabber:
            # update location of grabbed object
            self.update_grab_position(self.get_mouse_position(event.position))

        self.previous_mouse_point = self.get_mouse_position(event.position)
        self.redraw()
        event.accepted = True

    def mouse_press_event(self, event: Application.MouseEvent) -> None:
        """
        Handles `Application.MouseEvent`. When in GRAB mode, click on
        objects to drag their position. (right-click for fixed constraints)
        """
        button = Application.MouseEvent.Button
        physics_enabled = self.sim.get_physics_simulation_library()

        # if interactive mode is True -> GRAB MODE
        if self.mouse_interaction == MouseMode.GRAB and physics_enabled:
            render_camera = self.render_camera.render_camera
            ray = render_camera.unproject(
                self.get_mouse_position(event.position)
            )
            raycast_results = self.sim.cast_ray(ray=ray)

            if raycast_results.has_hits():
                hit_object, ao_link = -1, -1
                hit_info = raycast_results.hits[0]

                if hit_info.object_id >= 0:
                    # we hit an non-staged collision object
                    ro_mngr = self.sim.get_rigid_object_manager()
                    ao_mngr = self.sim.get_articulated_object_manager()
                    ao = ao_mngr.get_object_by_id(hit_info.object_id)
                    ro = ro_mngr.get_object_by_id(hit_info.object_id)

                    if ro:
                        # if grabbed an object
                        hit_object = hit_info.object_id
                        object_pivot = (
                            ro.transformation.inverted().transform_point(
                                hit_info.point
                            )
                        )
                        object_frame = ro.rotation.inverted()
                        for oix, e_ro in enumerate(self.ep_objects):
                            if ro.object_id == e_ro.object_id:
                                self.selected_obj = oix
                                print(f"selected object = {self.selected_obj}")
                    elif ao:
                        # if grabbed the base link
                        hit_object = hit_info.object_id
                        object_pivot = (
                            ao.transformation.inverted().transform_point(
                                hit_info.point
                            )
                        )
                        object_frame = ao.rotation.inverted()
                    else:
                        for (
                            ao_handle
                        ) in ao_mngr.get_objects_by_handle_substring():
                            ao = ao_mngr.get_object_by_handle(ao_handle)
                            link_to_obj_ids = ao.link_object_ids

                            if hit_info.object_id in link_to_obj_ids:
                                # if we got a link
                                ao_link = link_to_obj_ids[hit_info.object_id]
                                object_pivot = (
                                    ao.get_link_scene_node(ao_link)
                                    .transformation.inverted()
                                    .transform_point(hit_info.point)
                                )
                                object_frame = ao.get_link_scene_node(
                                    ao_link
                                ).rotation.inverted()
                                hit_object = ao.object_id
                                break
                    # done checking for AO

                    if hit_object >= 0:
                        node = self.default_agent.scene_node
                        constraint_settings = physics.RigidConstraintSettings()

                        constraint_settings.object_id_a = hit_object
                        constraint_settings.link_id_a = ao_link
                        constraint_settings.pivot_a = object_pivot
                        constraint_settings.frame_a = (
                            object_frame.to_matrix()
                            @ node.rotation.to_matrix()
                        )
                        constraint_settings.frame_b = node.rotation.to_matrix()
                        constraint_settings.pivot_b = hit_info.point

                        # by default use a point 2 point constraint
                        if event.button == button.RIGHT:
                            constraint_settings.constraint_type = (
                                physics.RigidConstraintType.Fixed
                            )

                        grip_depth = (
                            hit_info.point
                            - render_camera.node.absolute_translation
                        ).length()

                        self.mouse_grabber = MouseGrabber(
                            constraint_settings,
                            grip_depth,
                            self.sim,
                        )
                    else:
                        logger.warn(
                            "Oops, couldn't find the hit object. That's odd."
                        )
                # end if didn't hit the scene
            # end has raycast hit
        # end has physics enabled

        self.previous_mouse_point = self.get_mouse_position(event.position)
        self.redraw()
        event.accepted = True

    def mouse_scroll_event(self, event: Application.MouseScrollEvent) -> None:
        """
        Handles `Application.MouseScrollEvent`. When in LOOK mode, enables camera
        zooming (fine-grained zoom using shift) When in GRAB mode, adjusts the depth
        of the grabber's object. (larger depth change rate using shift)
        """
        scroll_mod_val = (
            event.offset.y
            if abs(event.offset.y) > abs(event.offset.x)
            else event.offset.x
        )
        if not scroll_mod_val:
            return

        # use shift to scale action response
        shift_pressed = bool(
            event.modifiers & Application.InputEvent.Modifier.SHIFT
        )
        alt_pressed = bool(
            event.modifiers & Application.InputEvent.Modifier.ALT
        )
        ctrl_pressed = bool(
            event.modifiers & Application.InputEvent.Modifier.CTRL
        )

        # if interactive mode is False -> LOOK MODE
        if self.mouse_interaction == MouseMode.LOOK:
            # use shift for fine-grained zooming
            mod_val = 1.01 if shift_pressed else 1.1
            mod = mod_val if scroll_mod_val > 0 else 1.0 / mod_val
            cam = self.render_camera
            cam.zoom(mod)
            self.redraw()

        elif self.mouse_interaction == MouseMode.GRAB and self.mouse_grabber:
            # adjust the depth
            mod_val = 0.1 if shift_pressed else 0.01
            scroll_delta = scroll_mod_val * mod_val
            if alt_pressed or ctrl_pressed:
                # rotate the object's local constraint frame
                agent_t = self.default_agent.scene_node.transformation_matrix()
                # ALT - yaw
                rotation_axis = agent_t.transform_vector(mn.Vector3(0, 1, 0))
                if alt_pressed and ctrl_pressed:
                    # ALT+CTRL - roll
                    rotation_axis = agent_t.transform_vector(
                        mn.Vector3(0, 0, -1)
                    )
                elif ctrl_pressed:
                    # CTRL - pitch
                    rotation_axis = agent_t.transform_vector(
                        mn.Vector3(1, 0, 0)
                    )
                self.mouse_grabber.rotate_local_frame_by_global_angle_axis(
                    rotation_axis, mn.Rad(scroll_delta)
                )
            else:
                # update location of grabbed object
                self.mouse_grabber.grip_depth += scroll_delta
                self.update_grab_position(
                    self.get_mouse_position(event.position)
                )
        self.redraw()
        event.accepted = True

    def mouse_release_event(self, event: Application.MouseEvent) -> None:
        """
        Release any existing constraints.
        """
        del self.mouse_grabber
        self.mouse_grabber = None
        event.accepted = True

    def update_grab_position(self, point: mn.Vector2i) -> None:
        """
        Accepts a point derived from a mouse click event and updates the
        transform of the mouse grabber.
        """
        # check mouse grabber
        if not self.mouse_grabber:
            return

        render_camera = self.render_camera.render_camera
        ray = render_camera.unproject(point)

        rotation: mn.Matrix3x3 = (
            self.default_agent.scene_node.rotation.to_matrix()
        )
        translation: mn.Vector3 = (
            render_camera.node.absolute_translation
            + ray.direction * self.mouse_grabber.grip_depth
        )
        self.mouse_grabber.update_transform(
            mn.Matrix4.from_(rotation, translation)
        )

    def get_mouse_position(
        self, mouse_event_position: mn.Vector2i
    ) -> mn.Vector2i:
        """
        This function will get a screen-space mouse position appropriately
        scaled based on framebuffer size and window size.  Generally these would be
        the same value, but on certain HiDPI displays (Retina displays) they may be
        different.
        """
        scaling = mn.Vector2i(self.framebuffer_size) / mn.Vector2i(
            self.window_size
        )
        return mouse_event_position * scaling

    def cycle_mouse_mode(self) -> None:
        """
        This method defines how to cycle through the mouse mode.
        """
        if self.mouse_interaction == MouseMode.LOOK:
            self.mouse_interaction = MouseMode.GRAB
        elif self.mouse_interaction == MouseMode.GRAB:
            self.mouse_interaction = MouseMode.LOOK

    def navmesh_config_and_recompute(self) -> None:
        """
        This method is setup to be overridden in for setting config accessibility
        in inherited classes.
        """
        self.navmesh_settings = habitat_sim.NavMeshSettings()
        self.navmesh_settings.set_defaults()
        self.navmesh_settings.agent_height = self.cfg.agents[
            self.agent_id
        ].height
        self.navmesh_settings.agent_radius = self.cfg.agents[
            self.agent_id
        ].radius
        self.navmesh_settings.include_static_objects = True
        self.sim.recompute_navmesh(
            self.sim.pathfinder,
            self.navmesh_settings,
        )
        self.largest_indoor_island = get_largest_island_index(
            self.sim.pathfinder, self.sim, allow_outdoor=False
        )
        self.default_agent.scene_node.translation = (
            self.sim.pathfinder.get_random_navigable_point(
                island_index=self.largest_indoor_island
            )
        )

    def exit_event(self, event: Application.ExitEvent):
        """
        Overrides exit_event to properly close the Simulator before exiting the
        application.
        """
        for i in range(self.num_env):
            self.tiled_sims[i].close(destroy=True)
            event.accepted = True
        exit(0)

    def draw_text(self, sensor_spec):
        self.shader.bind_vector_texture(self.glyph_cache.texture)
        self.shader.transformation_projection_matrix = (
            self.window_text_transform
        )
        self.shader.color = [1.0, 1.0, 1.0]

        sensor_type_string = str(sensor_spec.sensor_type.name)
        sensor_subtype_string = str(sensor_spec.sensor_subtype.name)
        if self.mouse_interaction == MouseMode.LOOK:
            mouse_mode_string = "LOOK"
        elif self.mouse_interaction == MouseMode.GRAB:
            mouse_mode_string = "GRAB"
        self.window_text.render(
            f"""
{self.fps} FPS
Sensor Type: {sensor_type_string}
Sensor Subtype: {sensor_subtype_string}
Mouse Interaction Mode: {mouse_mode_string}
Orientation Noise: {self.orientation_noise}
            """
        )
        self.shader.draw(self.window_text.mesh)

    def print_help_text(self) -> None:
        """
        Print the Key Command help text.
        """
        logger.info(
            """
=====================================================
Welcome to the Habitat-sim Python Viewer application!
=====================================================
Mouse Functions ('m' to toggle mode):
----------------
In LOOK mode (default):
    LEFT:
        Click and drag to rotate the agent and look up/down.
    WHEEL:
        Modify orthographic camera zoom/perspective camera FOV (+SHIFT for fine grained control)

In GRAB mode (with 'enable-physics'):
    LEFT:
        Click and drag to pickup and move an object with a point-to-point constraint (e.g. ball joint).
    RIGHT:
        Click and drag to pickup and move an object with a fixed frame constraint.
    WHEEL (with picked object):
        default - Pull gripped object closer or push it away.
        (+ALT) rotate object fixed constraint frame (yaw)
        (+CTRL) rotate object fixed constraint frame (pitch)
        (+ALT+CTRL) rotate object fixed constraint frame (roll)
        (+SHIFT) amplify scroll magnitude


Key Commands:
-------------
    esc:        Exit the application.
    'h':        Display this help message.
    'm':        Cycle mouse interaction modes.

    Agent Controls:
    'wasd':     Move the agent's body forward/backward and left/right.
    'zx':       Move the agent's body up/down.
    arrow keys: Turn the agent's body left/right and camera look up/down.

    Utilities:
    'r':        Reset the simulator with the most recently loaded scene.
    'n':        Show/hide NavMesh wireframe.
                (+SHIFT) Recompute NavMesh with default settings.
                (+ALT) Re-sample the agent(camera)'s position and orientation from the NavMesh.
    ',':        Render a Bullet collision shape debug wireframe overlay (white=active, green=sleeping, blue=wants sleeping, red=can't sleep).
    'c':        Run a discrete collision detection pass and render a debug wireframe overlay showing active contact points and normals (yellow=fixed length normals, red=collision distances).
                (+SHIFT) Toggle the contact point debug render overlay on/off.

    Object Interactions:
    SPACE:      Toggle physics simulation on/off.
    '.':        Take a single simulation step if not simulating continuously.
    'v':        (physics) Invert gravity.
    't':        Load URDF from filepath
                (+SHIFT) quick re-load the previously specified URDF
                (+ALT) load the URDF with fixed base
=====================================================
"""
        )


class MouseMode(Enum):
    LOOK = 0
    GRAB = 1
    MOTION = 2


class MouseGrabber:
    """
    Create a MouseGrabber from RigidConstraintSettings to manipulate objects.
    """

    def __init__(
        self,
        settings: physics.RigidConstraintSettings,
        grip_depth: float,
        sim: habitat_sim.simulator.Simulator,
    ) -> None:
        self.settings = settings
        self.simulator = sim

        # defines distance of the grip point from the camera for pivot updates
        self.grip_depth = grip_depth
        self.constraint_id = sim.create_rigid_constraint(settings)

    def __del__(self):
        self.remove_constraint()

    def remove_constraint(self) -> None:
        """
        Remove a rigid constraint by id.
        """
        self.simulator.remove_rigid_constraint(self.constraint_id)

    def updatePivot(self, pos: mn.Vector3) -> None:
        self.settings.pivot_b = pos
        self.simulator.update_rigid_constraint(
            self.constraint_id, self.settings
        )

    def update_frame(self, frame: mn.Matrix3x3) -> None:
        self.settings.frame_b = frame
        self.simulator.update_rigid_constraint(
            self.constraint_id, self.settings
        )

    def update_transform(self, transform: mn.Matrix4) -> None:
        self.settings.frame_b = transform.rotation()
        self.settings.pivot_b = transform.translation
        self.simulator.update_rigid_constraint(
            self.constraint_id, self.settings
        )

    def rotate_local_frame_by_global_angle_axis(
        self, axis: mn.Vector3, angle: mn.Rad
    ) -> None:
        """rotate the object's local constraint frame with a global angle axis input."""
        object_transform = mn.Matrix4()
        rom = self.simulator.get_rigid_object_manager()
        aom = self.simulator.get_articulated_object_manager()
        if rom.get_library_has_id(self.settings.object_id_a):
            object_transform = rom.get_object_by_id(
                self.settings.object_id_a
            ).transformation
        else:
            # must be an ao
            object_transform = (
                aom.get_object_by_id(self.settings.object_id_a)
                .get_link_scene_node(self.settings.link_id_a)
                .transformation
            )
        local_axis = object_transform.inverted().transform_vector(axis)
        R = mn.Matrix4.rotation(angle, local_axis.normalized())
        self.settings.frame_a = R.rotation().__matmul__(self.settings.frame_a)
        self.simulator.update_rigid_constraint(
            self.constraint_id, self.settings
        )


class Timer:
    """
    Timer class used to keep track of time between buffer swaps
    and guide the display frame rate.
    """

    start_time = 0.0
    prev_frame_time = 0.0
    prev_frame_duration = 0.0
    running = False

    @staticmethod
    def start() -> None:
        """
        Starts timer and resets previous frame time to the start time.
        """
        Timer.running = True
        Timer.start_time = time.time()
        Timer.prev_frame_time = Timer.start_time
        Timer.prev_frame_duration = 0.0

    @staticmethod
    def stop() -> None:
        """
        Stops timer and erases any previous time data, resetting the timer.
        """
        Timer.running = False
        Timer.start_time = 0.0
        Timer.prev_frame_time = 0.0
        Timer.prev_frame_duration = 0.0

    @staticmethod
    def next_frame() -> None:
        """
        Records previous frame duration and updates the previous frame timestamp
        to the current time. If the timer is not currently running, perform nothing.
        """
        if not Timer.running:
            return
        Timer.prev_frame_duration = time.time() - Timer.prev_frame_time
        Timer.prev_frame_time = time.time()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # optional arguments
    parser.add_argument(
        "--ep_dataset",
        type=str,
        help='Serialized RearrangeDataset file to load (e.g.: "data/datasets/floorplanner/rearrange/scratch/train/microtrain.json.gz")',
    )
    parser.add_argument(
        "--dataset",
        default="data/fpss/hssd-hab-uncluttered.scene_dataset_config.json",
        type=str,
        metavar="DATASET",
        help="dataset configuration file to use.",
    )
    parser.add_argument(
        "--disable-physics",
        action="store_true",
        help="disable physics simulation (default: False)",
    )
    parser.add_argument(
        "--use-default-lighting",
        action="store_true",
        help="Override configured lighting to use default lighting for the stage.",
    )
    parser.add_argument(
        "--hbao",
        action="store_true",
        help="Enable horizon-based ambient occlusion, which provides soft shadows in corners and crevices.",
    )
    parser.add_argument(
        "--enable-batch-renderer",
        action="store_true",
        help="Enable batch rendering mode. The number of concurrent environments is specified with the num-environments parameter.",
    )
    parser.add_argument(
        "--num-environments",
        default=1,
        type=int,
        help="Number of concurrent environments to batch render. Note that only the first environment simulates physics and can be controlled.",
    )
    parser.add_argument(
        "--composite-files",
        type=str,
        nargs="*",
        help="Composite files that the batch renderer will use in-place of simulation assets to improve memory usage and performance. If none is specified, the original scene files will be loaded from disk.",
    )
    parser.add_argument(
        "--width",
        default=1000,
        type=int,
        help="Horizontal resolution of the window.",
    )
    parser.add_argument(
        "--height",
        default=1000,
        type=int,
        help="Vertical resolution of the window.",
    )

    args = parser.parse_args()

    if args.num_environments < 1:
        parser.error("num-environments must be a positive non-zero integer.")
    if args.width < 1:
        parser.error("width must be a positive non-zero integer.")
    if args.height < 1:
        parser.error("height must be a positive non-zero integer.")

    # Setting up sim_settings
    sim_settings: Dict[str, Any] = default_sim_settings

    sim_settings["episode_dataset"] = args.ep_dataset
    sim_settings["scene_dataset_config_file"] = args.dataset
    sim_settings["enable_physics"] = not args.disable_physics
    sim_settings["use_default_lighting"] = args.use_default_lighting
    sim_settings["enable_batch_renderer"] = args.enable_batch_renderer
    sim_settings["num_environments"] = args.num_environments
    sim_settings["composite_files"] = args.composite_files
    sim_settings["window_width"] = args.width
    sim_settings["window_height"] = args.height
    sim_settings["default_agent_navmesh"] = False
    sim_settings["enable_hbao"] = args.hbao

    # start the application
    HabitatSimInteractiveViewer(sim_settings).exec()
