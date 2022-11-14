import os
import shutil
import cv2
import math
import numpy as np
from typing import Tuple, List
import skimage.morphology
from enum import Enum
import time

import agent.utils.pose_utils as pu
from .fmm_planner import FMMPlanner


# Same enum as HabitatSimActions without Habitat dependency
class DiscreteActions(Enum):
    stop = 0
    move_forward = 1
    turn_left = 2
    turn_right = 3


class DiscretePlanner:
    """
    This class translates planner inputs into a discrete low-level action
    using an FMM planner.
    """

    def __init__(
        self,
        turn_angle: float,
        collision_threshold: float,
        obs_dilation_selem_radius: int,
        goal_dilation_selem_radius: int,
        map_size_cm: int,
        map_resolution: int,
        visualize: bool,
        print_images: bool,
        dump_location: str,
        exp_name: str,
    ):
        """
        Arguments:
            turn_angle (float): agent turn angle (in degrees)
            collision_threshold (float): forward move distance under which we
             consider there's a collision (in meters)
            obs_dilation_selem_radius: radius (in cells) of obstacle dilation
             structuring element
            obs_dilation_selem_radius: radius (in cells) of goal dilation
             structuring element
            map_size_cm: global map size (in centimeters)
            map_resolution: size of map bins (in centimeters)
            visualize: if True, render planner internals for debugging
            print_images: if True, save visualization as images
        """
        self.visualize = visualize
        self.print_images = print_images
        self.default_vis_dir = f"{dump_location}/images/{exp_name}"
        os.makedirs(self.default_vis_dir, exist_ok=True)

        self.map_size_cm = map_size_cm
        self.map_resolution = map_resolution
        self.map_shape = (
            self.map_size_cm // self.map_resolution,
            self.map_size_cm // self.map_resolution,
        )
        self.turn_angle = turn_angle
        self.collision_threshold = collision_threshold
        self.start_obs_dilation_selem_radius = obs_dilation_selem_radius
        self.goal_dilation_selem_radius = goal_dilation_selem_radius

        self.vis_dir = None
        self.collision_map = None
        self.visited_map = None
        self.col_width = None
        self.last_pose = None
        self.curr_pose = None
        self.last_action = None
        self.timestep = None
        self.curr_obs_dilation_selem_radius = None
        self.obs_dilation_selem = None

    def reset(self):
        self.vis_dir = self.default_vis_dir
        self.collision_map = np.zeros(self.map_shape)
        self.visited_map = np.zeros(self.map_shape)
        self.col_width = 1
        self.last_pose = None
        self.curr_pose = [
            self.map_size_cm / 100.0 / 2.0,
            self.map_size_cm / 100.0 / 2.0,
            0.0,
        ]
        self.last_action = None
        self.timestep = 1
        self.curr_obs_dilation_selem_radius = self.start_obs_dilation_selem_radius
        self.obs_dilation_selem = skimage.morphology.disk(
            self.curr_obs_dilation_selem_radius
        )

    def set_vis_dir(self, scene_id: str, episode_id: str):
        self.print_images = True
        self.vis_dir = os.path.join(self.default_vis_dir, f"{scene_id}_{episode_id}")
        shutil.rmtree(self.vis_dir, ignore_errors=True)
        os.makedirs(self.vis_dir, exist_ok=True)

    def disable_print_images(self):
        self.print_images = False

    def plan(
        self,
        obstacle_map: np.ndarray,
        goal_map: np.ndarray,
        sensor_pose: np.ndarray,
        found_goal: bool,
    ) -> Tuple[int, np.ndarray]:
        """Plan a low-level action.

        Args:
            obstacle_map: (M, M) binary local obstacle map prediction
            goal_map: (M, M) binary array denoting goal location
            sensor_pose: (7,) array denoting global pose (x, y, o)
             and local map boundaries planning window (gx1, gx2, gy1, gy2)
            found_goal: whether we found the object goal category

        Returns:
            action: low-level action
            closest_goal_map: (M, M) binary array denoting closest goal
             location in the goal map in geodesic distance
        """
        self.last_pose = self.curr_pose
        obstacle_map = np.rint(obstacle_map)

        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = sensor_pose
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        start = [
            int(start_y * 100.0 / self.map_resolution - gx1),
            int(start_x * 100.0 / self.map_resolution - gy1),
        ]
        start = pu.threshold_poses(start, obstacle_map.shape)

        self.curr_pose = [start_x, start_y, start_o]
        self.visited_map[gx1:gx2, gy1:gy2][
            start[0] - 0 : start[0] + 1, start[1] - 0 : start[1] + 1
        ] = 1

        if self.last_action == DiscreteActions.move_forward.value:
            self._check_collision()

        # High-level goal -> short-term goal
        # t0 = time.time()
        short_term_goal, closest_goal_map, replan, stop = self._get_short_term_goal(
            obstacle_map, np.copy(goal_map), start, planning_window
        )
        # t1 = time.time()
        # print(f"[Planning] get_short_term_goal() time: {t1 - t0}")

        # We were not able to find a path to the high-level goal
        if replan:
            # Clean collision map
            self.collision_map *= 0

            # Reduce obstacle dilation
            if self.curr_obs_dilation_selem_radius > 1:
                self.curr_obs_dilation_selem_radius -= 1
                self.obs_dilation_selem = skimage.morphology.disk(
                    self.curr_obs_dilation_selem_radius
                )

        # Short-term goal -> deterministic local policy
        if stop and found_goal:
            action = DiscreteActions.stop.value
        else:
            stg_x, stg_y = short_term_goal
            angle_st_goal = math.degrees(math.atan2(stg_x - start[0], stg_y - start[1]))
            angle_agent = start_o % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.0
            if relative_angle > 180:
                relative_angle -= 360

            if relative_angle > self.turn_angle / 2.0:
                action = DiscreteActions.turn_right.value
            elif relative_angle < -self.turn_angle / 2.0:
                action = DiscreteActions.turn_left.value
            else:
                action = DiscreteActions.move_forward.value

        self.last_action = action
        return action, closest_goal_map

    def _get_short_term_goal(
        self,
        obstacle_map: np.ndarray,
        goal_map: np.ndarray,
        start: List[int],
        planning_window: List[int],
    ) -> Tuple[Tuple[int, int], np.ndarray, bool, bool]:
        """Get short-term goal.

        Args:
            obstacle_map: (M, M) binary local obstacle map prediction
            goal_map: (M, M) binary array denoting goal location
            start: start location (x, y)
            planning_window: local map boundaries (gx1, gx2, gy1, gy2)

        Returns:
            short_term_goal: short-term goal position (x, y) in map
            closest_goal_map: (M, M) binary array denoting closest goal
             location in the goal map in geodesic distance
            replan: binary flag to indicate we couldn't find a plan to reach
             the goal
            stop: binary flag to indicate we've reached the goal
        """
        gx1, gx2, gy1, gy2 = planning_window
        x1, y1, = (
            0,
            0,
        )
        x2, y2 = obstacle_map.shape

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1 : h + 1, 1 : w + 1] = mat
            return new_mat

        def remove_boundary(mat, value=1):
            return mat[value:-value, value:-value]

        obstacles = obstacle_map[x1:x2, y1:y2]

        # Dilate obstacles
        dilated_obstacles = cv2.dilate(obstacles, self.obs_dilation_selem, iterations=1)

        traversible = 1 - dilated_obstacles
        traversible[self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
        traversible[self.visited_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1
        traversible[
            int(start[0] - x1) - 1 : int(start[0] - x1) + 2,
            int(start[1] - y1) - 1 : int(start[1] - y1) + 2,
        ] = 1
        traversible = add_boundary(traversible)
        goal_map = add_boundary(goal_map, value=0)

        planner = FMMPlanner(
            traversible,
            vis_dir=self.vis_dir,
            visualize=self.visualize,
            print_images=self.print_images,
        )

        # Dilate the goal
        selem = skimage.morphology.disk(self.goal_dilation_selem_radius)
        dilated_goal_map = skimage.morphology.binary_dilation(goal_map, selem) != True
        dilated_goal_map = 1 - dilated_goal_map * 1.0

        planner.set_multi_goal(dilated_goal_map, self.timestep)
        self.timestep += 1

        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        stg_x, stg_y, replan, stop = planner.get_short_term_goal(state)
        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1
        short_term_goal = int(stg_x), int(stg_y)

        # Select closest point on goal map for visualization
        # TODO How to do this without the overhead of creating another FMM planner?
        vis_planner = FMMPlanner(traversible)
        curr_loc_map = np.zeros_like(goal_map)
        curr_loc_map[start[0], start[1]] = 1
        vis_planner.set_multi_goal(curr_loc_map)
        fmm_dist_ = vis_planner.fmm_dist.copy()
        goal_map_ = goal_map.copy()
        goal_map_[goal_map_ == 0] = 10000
        fmm_dist_[fmm_dist_ == 0] = 10000
        closest_goal_map = (goal_map_ * fmm_dist_) == (goal_map_ * fmm_dist_).min()
        closest_goal_map = remove_boundary(closest_goal_map)

        return short_term_goal, closest_goal_map, replan, stop

    def _check_collision(self):
        """Check whether we had a collision and update the collision map."""
        x1, y1, t1 = self.last_pose
        x2, y2, _ = self.curr_pose
        buf = 4
        length = 2

        if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
            self.col_width += 2
            if self.col_width == 7:
                length = 4
                buf = 3
            self.col_width = min(self.col_width, 5)
        else:
            self.col_width = 1

        dist = pu.get_l2_distance(x1, x2, y1, y2)

        if dist < self.collision_threshold:
            # We have a collision
            width = self.col_width

            # Add obstacles to the collision map
            for i in range(length):
                for j in range(width):
                    wx = x1 + 0.05 * (
                        (i + buf) * np.cos(np.deg2rad(t1))
                        + (j - width // 2) * np.sin(np.deg2rad(t1))
                    )
                    wy = y1 + 0.05 * (
                        (i + buf) * np.sin(np.deg2rad(t1))
                        - (j - width // 2) * np.cos(np.deg2rad(t1))
                    )
                    r, c = wy, wx
                    r, c = int(r * 100 / self.map_resolution), int(
                        c * 100 / self.map_resolution
                    )
                    [r, c] = pu.threshold_poses([r, c], self.collision_map.shape)
                    self.collision_map[r, c] = 1
