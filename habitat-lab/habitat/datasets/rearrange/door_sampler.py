#!/usr/bin/env python3

import sys
import argparse
import os
import os.path as osp
import time
from collections import defaultdict
from typing import Any, Dict, List, cast

import magnum as mn
import numpy as np

import habitat
import habitat.tasks.rearrange.rearrange_task
from habitat.articulated_agent_controllers import HumanoidRearrangeController
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    GfxReplayMeasureMeasurementConfig,
    PddlApplyActionConfig,
    ThirdRGBSensorConfig,
)
from habitat.core.logging import logger
from habitat.tasks.rearrange.actions.actions import ArmEEAction
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import euler_to_quat, write_gfx_replay
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
)
from habitat_sim.utils import viz_utils as vut
from habitat.utils.visualizations.utils import images_to_video
from habitat.utils.visualizations import maps

import threading

from IPython import embed
import cv2
import matplotlib.pyplot as plt


def to_grid(pathfinder, points, grid_dimensions):
    map_points = maps.to_grid(
                        points[2],
                        points[0],
                        grid_dimensions,
                        pathfinder=pathfinder,
                    )
    return ([map_points[1], map_points[0]])
def from_grid(pathfinder, points, grid_dimensions):
    floor_y = 0.0
    map_points = maps.from_grid(
                        points[1],
                        points[0],
                        grid_dimensions,
                        pathfinder=pathfinder,
                    )
    map_points_3d = np.array([map_points[1], floor_y, map_points[0]])
    map_points_3d = pathfinder.snap_point(map_points_3d)
    return map_points_3d

def door_samples(pathfinder, config) -> List:
    # Get top down map
    top_down_map = maps.get_topdown_map(
            pathfinder, height=0.0, meters_per_pixel=0.025
        )

    recolor_map = np.array(
        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
    )
    top_down_map = recolor_map[top_down_map]
    cv2.imwrite("top_down_map_generator.jpg", top_down_map)
    
    # Click to get the door coordinates
    image = plt.imread("top_down_map_generator.jpg")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image)
    plt.title('Click on the image to get coordinates')

    coords = []
    def onclick(event):
        ix, iy = int(event.xdata), int(event.ydata) # Convert float coordinates to integers
        print(f'x = {ix}, y = {iy}')

        coords.append([ix, iy])
        
        if len(coords) == 2:
            fig.canvas.mpl_disconnect(cid)
    # Connect the onclick function to the mouse click event
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    
    # Get the end point for door_line
    print("-------Coords for Door position----------")
    grid_dim = (top_down_map.shape[0], top_down_map.shape[1])
    door1 = from_grid(pathfinder, coords[0], grid_dim)
    door2 = from_grid(pathfinder, coords[1], grid_dim)
    
    # new_env = habitat.Env(config=config)
    # observations = new_env.reset()
    # z1 = observations['agent_0_localization_sensor'][1] #from robot_pos height
    z1 = 0.0
    # embed()
    door1_pos = np.array([door1[0], z1, door1[2]])
    door2_pos = np.array([door2[0], z1, door2[2]])
    mid_door_pos = (door1_pos + door2_pos) / 2.0
    print("Get Door position: ", door1_pos, door2_pos)
    print("Mid Door position: ", mid_door_pos)

    # Visualize door line
    cv2.line(
        top_down_map,
        [coords[0][0], coords[0][1]],
        [coords[1][0], coords[1][1]],
        1,
        thickness=2
        )
    mid_door_xy = to_grid(pathfinder, mid_door_pos, grid_dim)
    cv2.circle(top_down_map, [int(mid_door_xy[0]), int(mid_door_xy[1])], 10, (255, 255, 0), -1)
    cv2.imwrite("top_down_map_door_generator.jpg", top_down_map)

    # Generate One Valid pair of sampled points across the door
    sampled_points_a = []
    sampled_points_b = []
    def get_opposite_points():
        for i in range(10):
            sample = pathfinder.get_random_navigable_point_near(
                circle_center=mid_door_pos, radius=2 #TODO
            )
            print(sample)
            x = sample[0]
            y = sample[2]
            r1, r2 = door1[0], door2[0]
            c1, c2 = door1[2], door2[2]
            det = (x - r1) * (c2 - c1) - (y - c1) * (r2 - r1)
            if det > 0:
                sampled_points_a.append(sample)
            else:
                sampled_points_b.append(sample)
        print("TEST a, b groups: ", sampled_points_a, sampled_points_b)
        # Visualize the points
        for a in sampled_points_a:
            i, j = to_grid(pathfinder, a, grid_dim)
            cv2.circle(top_down_map, [i, j], 5, (0, 0, 255), -1)  #red points
        for b in sampled_points_b:
            i, j = to_grid(pathfinder, b, grid_dim)
            cv2.circle(top_down_map, [i, j], 5, (255, 0, 0), -1) #blue points
        cv2.imwrite("top_down_map_door_samples_generator.jpg", top_down_map)
        
    while(len(sampled_points_a) == 0 or len(sampled_points_b) < 2):
        get_opposite_points()
        print("Get sampled group points with one try ... ")

    start_sample = sampled_points_a[0]
    goal_sample = sampled_points_b[0]
    robot_start_sample = sampled_points_b[1]
    
    return [start_sample, goal_sample, robot_start_sample]