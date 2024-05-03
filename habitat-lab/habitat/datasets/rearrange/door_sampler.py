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
import math
import csv
import random


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

def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def vec_distance(x, y):
    return math.sqrt(x**2 + y**2)

def dot_product(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def angle(v1, v2):
    return math.acos(dot_product(v1, v2) / (vec_distance(*v1) * vec_distance(*v2)))

def angle_between_line_and_point(mid_door_xy, door_end_xy, point_xy):
    # xy: [x, y]
    # Vector representation of the line segment
    line_vector = [(mid_door_xy[0] - door_end_xy[0])*10, (mid_door_xy[1] - door_end_xy[1])*10]
    # Vector representation of the line segment to the point
    point_vector = [(point_xy[0] - mid_door_xy[0])*10, (point_xy[1] - mid_door_xy[1])*10]
    theta = angle(line_vector, point_vector)
    theta = theta if theta <= math.pi/2 else math.pi - theta
    return theta #in radian

def is_in_bound(mid_door_xy, door_end_xy, point_xy, threshold=2.0) -> bool:
    # Use pos_xy instead of xy in grid
    theta = angle_between_line_and_point(mid_door_xy, door_end_xy, point_xy)
    door_radius = distance(mid_door_xy[0], mid_door_xy[1], door_end_xy[0], door_end_xy[1])
    dist_point_mid_door = distance(mid_door_xy[0], mid_door_xy[1], point_xy[0], point_xy[1])
    line_proj = math.cos(theta) * dist_point_mid_door
    vert_proj = math.sin(theta) * dist_point_mid_door

    if line_proj > door_radius: return False
    if vert_proj < 0.5: return False
    return True

def append_to_csv(filename, x1, y1, x2, y2):
    # Open the CSV file in append mode and write the new data
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([x1, y1, x2, y2])

def generate_door_pos() -> List:
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
            plt.close()
    # Connect the onclick function to the mouse click event
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    print("TEST: coords: ", coords)
    return coords

def is_door_valid(coords, grid_dim, pathfinder) -> bool:
    z1 = 0.0
    door1 = from_grid(pathfinder, coords[0], grid_dim)
    door2 = from_grid(pathfinder, coords[1], grid_dim)
    door1_pos = np.array([door1[0], z1, door1[2]])
    door2_pos = np.array([door2[0], z1, door2[2]])
    mid_door_pos = (door1_pos + door2_pos) / 2.0
    dist_to_obs = pathfinder.distance_to_closest_obstacle(
        mid_door_pos,
        max_search_radius=10.0
        )
    print("Test distance door to obstacle: ", dist_to_obs)
    if dist_to_obs > 0.12 and dist_to_obs < 0.4:
        return True
    return False

def door_samples(pathfinder, config) -> List:
    print("TEST")
    # Get top down map
    top_down_map = maps.get_topdown_map(
            pathfinder, height=0.0, meters_per_pixel=0.025
        )

    recolor_map = np.array(
        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
    )
    top_down_map = recolor_map[top_down_map]
    cv2.imwrite("top_down_map_generator.jpg", top_down_map)

    # Check mid_door valid wrt obstacles
    grid_dim = (top_down_map.shape[0], top_down_map.shape[1])

    # Generate Valid Doors and Write into csv file
    #KL: modifying
    print("HERE")
    file_name = config.scene_sets[0].included_substrings[0] #substring
    file_path = f"test_door_dataset/data_{file_name}.csv"
    coords = []
    # Comment out this if producing dataset
    if os.path.exists(file_path):
        # os.remove(file_path)
        with open(file_path, 'r') as csvfile:
            # Create a CSV reader object
            csvreader = csv.reader(csvfile)
            # Convert CSV data to a list of lists
            data = list(csvreader)
            total_rows = len(data)
            row_list = data[0]
            coords = [[int(row_list[0]), int(row_list[1])], [int(row_list[2]), int(row_list[3])]]
    else:
        coords = generate_door_pos()
        while (not is_door_valid(coords, grid_dim, pathfinder)):
            coords = generate_door_pos()
        append_to_csv(file_path, coords[0][0], coords[0][1], coords[1][0], coords[1][1])
    ####till here
    
    
    # # append valid door pos to csv file
    # # if os.path.exists(file_path):
    # #     os.remove(file_path)
    # #make sure only 1 door in each dataset
    # coords = generate_door_pos()
    # while (not is_door_valid(coords, grid_dim, pathfinder)):
    #     coords = generate_door_pos()
    # append_to_csv(file_path, coords[0][0], coords[0][1], coords[1][0], coords[1][1])

    # Get door pos randomly from csv file
    # coords = []
    # file_name = config.scene_sets[0].included_substrings[0] #substring
    # file_path = f"data_{file_name}.csv"
    # with open(file_path, 'r') as csvfile:
    #     # Create a CSV reader object
    #     csvreader = csv.reader(csvfile)
    #     # Convert CSV data to a list of lists
    #     data = list(csvreader)
    #     total_rows = len(data)
    #     random_index = random.randint(0, total_rows - 1)
    #     row_list = data[random_index]
    #     coords = [[int(row_list[0]), int(row_list[1])], [int(row_list[2]), int(row_list[3])]]



    # Get the end point for door_line
    print("-------Coords for Valid Door position----------")
    door1 = from_grid(pathfinder, coords[0], grid_dim)
    door2 = from_grid(pathfinder, coords[1], grid_dim)
    door_end_xy = [door1[0], door1[2]]
    
    # new_env = habitat.Env(config=config)
    # observations = new_env.reset()
    # z1 = observations['agent_0_localization_sensor'][1] #from robot_pos height
    z1 = 0.0
    # embed()
    door1_pos = np.array([door1[0], z1, door1[2]])
    door2_pos = np.array([door2[0], z1, door2[2]])
    mid_door_pos = (door1_pos + door2_pos) / 2.0
    mid_door_pos_xy = [mid_door_pos[0], mid_door_pos[2]]
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
    print("----------Start Sampling points-----------")
    sampled_points_a = [] #left point
    sampled_points_b = [] #right point
    def get_opposite_points():
        samples_a = []
        samples_b = []
        for i in range(10):
            sample = pathfinder.get_random_navigable_point_near(
                circle_center=mid_door_pos, radius=2 #TODO
            )
            sample_xy = [sample[0], sample[2]]
            if not is_in_bound(mid_door_pos_xy, door_end_xy, sample_xy):
                continue
            # print("Is in bound: ", sample, sample_xy)
            
            x = sample[0]
            y = sample[2]
            r1, r2 = door1[0], door2[0]
            c1, c2 = door1[2], door2[2]
            det = (x - r1) * (c2 - c1) - (y - c1) * (r2 - r1)
            if det > 0:
                samples_a.append(sample)
            else:
                samples_b.append(sample)
        # print("TEST a, b groups: ", samples_a, samples_b)
        
        # Visualize the points
        for a in samples_a:
            i, j = to_grid(pathfinder, a, grid_dim)
            cv2.circle(top_down_map, [i, j], 5, (0, 0, 255), -1)  #blue points
        for b in samples_b:
            i, j = to_grid(pathfinder, b, grid_dim)
            cv2.circle(top_down_map, [i, j], 5, (255, 0, 0), -1) #red points
                
        return samples_a, samples_b
    
    print("-------Get sampled A B group points----------") 
    while(len(sampled_points_a) < 2 or len(sampled_points_b) < 2):
        sampled_points_a, sampled_points_b = get_opposite_points()
    

    # # make sure b group has more points to choose
    # if len(sampled_points_a) > len(sampled_points_b):
    #     tmp = sampled_points_a
    #     sampled_points_a = sampled_points_b
    #     sampled_points_b = tmp


    #sort sampled_points_b by distance to line(nearest-> furthest)
    def distance_to_line(point):
        point_pos_xy = point[0], point[2]
        theta = angle_between_line_and_point(mid_door_pos_xy, door1, point_pos_xy)
        dist_point_mid_door = distance(mid_door_pos_xy[0], mid_door_pos_xy[1], point_pos_xy[0], point_pos_xy[1])
        vert_proj = math.sin(theta) * dist_point_mid_door
        return vert_proj

    # Sort the points based on their distance to the line
    sorted_points_a = sorted(sampled_points_a, key=lambda point: distance_to_line(point))
    sorted_points_b = sorted(sampled_points_b, key=lambda point: distance_to_line(point))
    #TODO
    start_sample = sorted_points_a[0]
    goal_sample = sorted_points_b[-1]
    robot_start_sample = sorted_points_b[0]
    robot_goal_sample = sorted_points_a[-1]
    
    #visualize selected points
    for a in [start_sample, goal_sample, robot_start_sample, robot_goal_sample]:
        i, j = to_grid(pathfinder, a, grid_dim)
        cv2.circle(top_down_map, [i, j], 4, (0, 255, 0), -1)  #green points
    cv2.imwrite(f"test_door_images/top_down_map_{file_name}_generator.jpg", top_down_map)
    
    return [start_sample, goal_sample, robot_start_sample, robot_goal_sample]