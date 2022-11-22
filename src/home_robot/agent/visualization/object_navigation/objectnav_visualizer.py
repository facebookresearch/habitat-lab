import os
import shutil
import numpy as np
import cv2
from PIL import Image
import skimage.morphology
from typing import Optional

import home_robot.agent.utils.visualization_utils as vu
import home_robot.agent.utils.pose_utils as pu
from home_robot.agent.perception.detection.coco_maskrcnn.coco_categories import (
    coco_categories_color_palette,
)


map_color_palette = [
    1.0,
    1.0,
    1.0,  # empty space
    0.6,
    0.6,
    0.6,  # obstacles
    0.95,
    0.95,
    0.95,  # explored area
    0.96,
    0.36,
    0.26,  # visited area
    0.12,
    0.46,
    0.70,  # closest goal
    0.63,
    0.78,
    0.95,  # rest of goal
    *coco_categories_color_palette,
]


class ObjectNavVisualizer:
    """
    This class is intended to visualize a single object goal navigation task.
    """

    def __init__(
        self,
        num_sem_categories: int,
        map_size_cm: int,
        map_resolution: int,
        show_images: bool,
        print_images: bool,
        dump_location: str,
        exp_name: str,
    ):
        """
        Arguments:
            num_sem_categories: number of semantic segmentation categories
            map_size_cm: global map size (in centimeters)
            map_resolution: size of map bins (in centimeters)
            show_images: if True, render observation and predicted semantic map
            print_images: if True, save visualization as images
        """
        self.show_images = show_images
        self.print_images = print_images
        self.default_vis_dir = f"{dump_location}/images/{exp_name}"
        os.makedirs(self.default_vis_dir, exist_ok=True)

        self.color_palette = [int(x * 255.0) for x in map_color_palette]
        self.legend = cv2.imread(
            "home_robot/agent/perception/detection/coco_maskrcnn/coco_categories_legend.png"
        )

        self.num_sem_categories = num_sem_categories
        self.map_resolution = map_resolution
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
        self.vis_dir = os.path.join(self.default_vis_dir, f"{scene_id}_{episode_id}")
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
        semantic_frame: np.ndarray,
        goal_name: str,
        timestep: int,
        visualize_goal: bool = True,
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

        semantic_map += 6

        # Obstacles, explored, and visited areas
        no_category_mask = semantic_map == 6 + self.num_sem_categories - 1
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
            goal_mat = 1 - skimage.morphology.binary_dilation(goal_map, selem) != True
            goal_mask = goal_mat == 1
            semantic_map[goal_mask] = 5
            if closest_goal_map is not None:
                closest_goal_mat = (
                    1 - skimage.morphology.binary_dilation(closest_goal_map, selem)
                    != True
                )
                closest_goal_mask = closest_goal_mat == 1
                semantic_map[closest_goal_mask] = 4

        # Semantic categories
        semantic_map_vis = Image.new(
            "P", (semantic_map.shape[1], semantic_map.shape[0])
        )
        semantic_map_vis.putpalette(self.color_palette)
        semantic_map_vis.putdata(semantic_map.flatten().astype(np.uint8))
        semantic_map_vis = semantic_map_vis.convert("RGB")
        semantic_map_vis = np.flipud(semantic_map_vis)
        semantic_map_vis = semantic_map_vis[:, :, [2, 1, 0]]
        semantic_map_vis = cv2.resize(
            semantic_map_vis, (480, 480), interpolation=cv2.INTER_NEAREST
        )
        self.image_vis[50:530, 670:1150] = semantic_map_vis

        # First-person semantic frame
        self.image_vis[50:530, 15:655] = cv2.resize(semantic_frame, (640, 480))

        # Agent arrow
        pos = (
            (curr_x * 100.0 / self.map_resolution - gx1) * 480 / obstacle_map.shape[0],
            (obstacle_map.shape[1] - curr_y * 100.0 / self.map_resolution + gy1)
            * 480
            / obstacle_map.shape[1],
            np.deg2rad(-curr_o),
        )
        agent_arrow = vu.get_contour_points(pos, origin=(670, 50))
        color = self.color_palette[9:12][::-1]
        cv2.drawContours(self.image_vis, [agent_arrow], 0, color, -1)

        if self.show_images:
            cv2.imshow("Visualization", self.image_vis)
            cv2.waitKey(1)

        if self.print_images:
            cv2.imwrite(
                os.path.join(self.vis_dir, f"snapshot_{timestep}.png"), self.image_vis
            )

    def _init_vis_image(self, goal_name: str):
        vis_image = np.ones((655, 1165, 3)).astype(np.uint8) * 255
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

        # Draw legend
        lx, ly, _ = self.legend.shape
        vis_image[537 : 537 + lx, 155 : 155 + ly, :] = self.legend

        return vis_image
