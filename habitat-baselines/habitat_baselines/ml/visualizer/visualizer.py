import os
import shutil
from typing import Optional

import cv2
import numpy as np
import skimage.morphology
from PIL import Image

import habitat_baselines.ml.utils.pose_utils as pu
import habitat_baselines.ml.utils.visualization_utils as vu

from .constants import GoalObjectMapping


class Visualizer:
    """
    This class is intended to visualize a single object goal navigation task.
    """

    def __init__(self, config):
        self.show_images = config.ml_environment.visualize
        self.print_images = config.ml_environment.print_images
        self.default_vis_dir = f"{config.ml_environment.dump_location}/images/{config.ml_environment.exp_name}"
        os.makedirs(self.default_vis_dir, exist_ok=True)
        self.episodes_data_path = config.dataset.data_path
        self.semantic_category_mapping = GoalObjectMapping()

        self.legend = cv2.imread(
            self.semantic_category_mapping.categories_legend_path
        )

        self.num_sem_categories = config.ml_environment.num_sem_categories
        self.map_resolution = config.ml_environment.map_resolution
        map_size_cm = config.ml_environment.map_size_cm
        self.map_shape = (
            map_size_cm // self.map_resolution,
            map_size_cm // self.map_resolution,
        )

        self.vis_dir = None
        self.image_vis = None
        self.visited_map_vis = None
        self.last_xy = None

    def reset(self):
        self.vis_dir = self.default_vis_dir
        self.image_vis = None
        self.visited_map_vis = np.zeros(self.map_shape)
        self.last_xy = None

    def set_vis_dir(self, scene_id: str, episode_id: str):
        self.print_images = True
        self.vis_dir = os.path.join(
            self.default_vis_dir,f"{scene_id}_{episode_id}"
        )
        shutil.rmtree(self.vis_dir, ignore_errors=True)
        os.makedirs(self.vis_dir, exist_ok=True)

    def disable_print_images(self):
        self.print_images = False

    def visualize(
        self,
        obstacle_map: np.ndarray,
        goal_map: np.ndarray,
        closest_goal_map: Optional[np.ndarray],
        sensor_pose: np.ndarray,
        found_goal: bool,
        explored_map: np.ndarray,
        semantic_map: np.ndarray,
        been_close_map: np.ndarray,
        semantic_frame: np.ndarray,
        third_person_rgb_frame: np.ndarray,
        goal_name: str,
        timestep: int,
        visualize_goal: bool = True,
        idx: int = 0,
    ):
        """Visualize frame input and semantic map.

        Args:
            obstacle_map: (M, M) binary local obstacle map prediction
            goal_map: (M, M) binary array denoting goal location
            closest_goal_map: (M, M) binary array denoting closest goal
             location in the goal map in geodesic distance
            sensor_pose: (7,) array denoting global pose (x, y, o)
             and local map boundaries planning window (gy1, gy2, gx1, gy2)
            found_goal: whether we found the object goal category
            explored_map: (M, M) binary local explored map prediction
            semantic_map: (M, M) local semantic map predictions
            semantic_frame: semantic frame visualization
            third_person_rgb_frame: third-person RGB frame visualization
            goal_name: semantic goal category
            timestep: time step within the episode
            visualize_goal: if True, visualize goal
        """
        if self.image_vis is None:
            self.image_vis = self._init_vis_image(goal_name)

        curr_x, curr_y, curr_o, gy1, gy2, gx1, gx2 = sensor_pose
        gy1, gy2, gx1, gx2 = int(gy1), int(gy2), int(gx1), int(gx2)

        # Update visited map with last visited area
        if self.last_xy is not None:
            last_x, last_y = self.last_xy
            last_pose = [
                int(last_y * 100.0 / self.map_resolution - gy1),
                int(last_x * 100.0 / self.map_resolution - gx1),
            ]
            last_pose = pu.threshold_poses(last_pose, obstacle_map.shape)
            curr_pose = [
                int(curr_y * 100.0 / self.map_resolution - gy1),
                int(curr_x * 100.0 / self.map_resolution - gx1),
            ]
            curr_pose = pu.threshold_poses(curr_pose, obstacle_map.shape)
            self.visited_map_vis[gy1:gy2, gx1:gx2] = vu.draw_line(
                last_pose, curr_pose, self.visited_map_vis[gy1:gy2, gx1:gx2]
            )
        self.last_xy = (curr_x, curr_y)

        semantic_map += 7

        # Obstacles, explored, and visited areas
        no_category_mask = (
            semantic_map == 7 + self.num_sem_categories - 1
        )  # Assumes the last category is "other"
        obstacle_mask = np.rint(obstacle_map) == 1
        explored_mask = np.rint(explored_map) == 1
        visited_mask = self.visited_map_vis[gy1:gy2, gx1:gx2] == 1
        semantic_map[no_category_mask] = 0
        semantic_map[np.logical_and(no_category_mask, explored_mask)] = 2
        semantic_map[np.logical_and(no_category_mask, obstacle_mask)] = 1
        semantic_map[visited_mask] = 3
        # Goal
        if visualize_goal:
            selem = skimage.morphology.disk(4)
            goal_mat = (
                1 - skimage.morphology.binary_dilation(goal_map, selem) != True
            )
            goal_mask = goal_mat == 1
            semantic_map[goal_mask] = 5
            if closest_goal_map is not None:
                closest_goal_mat = (
                    1
                    - skimage.morphology.binary_dilation(
                        closest_goal_map, selem
                    )
                    != True
                )
                closest_goal_mask = closest_goal_mat == 1
                semantic_map[closest_goal_mask] = 4

        # Semantic categories
        semantic_map_vis = Image.new(
            "P", (semantic_map.shape[1], semantic_map.shape[0])
        )
        semantic_map_vis.putpalette(
            self.semantic_category_mapping.map_color_palette
        )
        semantic_map_vis.putdata(semantic_map.flatten().astype(np.uint8))
        semantic_map_vis = semantic_map_vis.convert("RGB")
        semantic_map_vis = np.flipud(semantic_map_vis)
        semantic_map_vis = semantic_map_vis[:, :, [2, 1, 0]]

        # overlay the regions the agent has been close to
        been_close_map = np.flipud(np.rint(been_close_map) == 1)
        color_index = 18 + 3 * self.semantic_category_mapping.num_sem_categories
        color = self.semantic_category_mapping.map_color_palette[color_index: color_index + 3][::-1]
        semantic_map_vis[been_close_map] = (semantic_map_vis[been_close_map] + color) / 2

        semantic_map_vis = cv2.resize(
            semantic_map_vis, (480, 480), interpolation=cv2.INTER_NEAREST
        )

        self.image_vis[50:530, 670:1150] = semantic_map_vis

        # First-person semantic frame
        self.image_vis[50:530, 15:655] = cv2.resize(semantic_frame, (640, 480))

        # third-person rgb frame
        self.image_vis[50:530, 1165:1645] = cv2.resize(
            third_person_rgb_frame, (480, 480)
        )
        # Agent arrow
        pos = (
            (curr_x * 100.0 / self.map_resolution - gx1)
            * 480
            / obstacle_map.shape[0],
            (
                obstacle_map.shape[1]
                - curr_y * 100.0 / self.map_resolution
                + gy1
            )
            * 480
            / obstacle_map.shape[1],
            np.deg2rad(-curr_o),
        )
        agent_arrow = vu.get_contour_points(pos, origin=(670, 50))
        color = self.semantic_category_mapping.map_color_palette[9:12][::-1]
        cv2.drawContours(self.image_vis, [agent_arrow], 0, color, -1)

        if self.show_images:
            cv2.imshow("Visualization", self.image_vis)
            cv2.waitKey(1)

        if self.print_images:
            cv2.imwrite(
                os.path.join(
                    self.vis_dir,
                    str(idx) + "_snapshot_{:03d}.png".format(timestep),
                ),
                self.image_vis,
            )

    def _init_vis_image(self, goal_name: str):
        # vis_image = np.ones((655, 1165, 3)).astype(np.uint8) * 255
        vis_image = np.ones((540, 1660, 3)).astype(np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (20, 20, 20)  # BGR
        thickness = 2

        text = "Observations (Goal: {})".format(goal_name)
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = (640 - textsize[0]) // 2 + 15
        textY = (50 + textsize[1]) // 2
        vis_image = cv2.putText(
            vis_image,
            text,
            (textX, textY),
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        text = "Predicted Semantic Map"
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = 640 + (480 - textsize[0]) // 2 + 30
        textY = (50 + textsize[1]) // 2
        vis_image = cv2.putText(
            vis_image,
            text,
            (textX, textY),
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        text = "Third-person view"
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = 640 + 480 + (480 - textsize[0]) // 2 + 30
        textY = (50 + textsize[1]) // 2
        vis_image = cv2.putText(
            vis_image,
            text,
            (textX, textY),
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        # Draw outlines
        color = [100, 100, 100]
        vis_image[49, 15:655] = color
        vis_image[49, 670:1150] = color
        vis_image[50:530, 14] = color
        vis_image[50:530, 655] = color
        vis_image[50:530, 669] = color
        vis_image[50:530, 1150] = color
        vis_image[530, 15:655] = color
        vis_image[530, 670:1150] = color
        vis_image[530, 1165:1645] = color

        # Draw legend
        # lx, ly, _ = self.legend.shape
        # vis_image[537 : 537 + lx, 155 : 155 + ly, :] = self.legend

        return vis_image
