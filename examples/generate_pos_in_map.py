#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
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
# Please reach out to the paper authors to obtain this file
DEFAULT_POSE_PATH = "data/humanoids/humanoid_data/walking_motion_processed.pkl"
DEFAULT_CFG = "benchmark/rearrange/play/play.yaml"
DEFAULT_RENDER_STEPS_LIMIT = 60
SAVE_VIDEO_DIR = "./data/vids"
SAVE_ACTIONS_DIR = "./data/interactive_play_replays"


lock = threading.Lock()

class sim_env(threading.Thread):
    _x_axis = 0
    _y_axis = 1
    _z_axis = 2
    _dt = 0.00478
    _sensor_rate = 50  # hz

    _current_episode = 0
    _total_number_of_episodes = 0
    control_frequency = 20
    time_step = 1.0 / (control_frequency)

    def __init__(self, config):
        threading.Thread.__init__(self)
        self.env = habitat.Env(config = config)
        self.observations = self.env.reset()

        self.linear_velocity = [0,0,0]
        self.angular_velocity = [0,0,0]

    def update_agent_pos_vel(self, mid_door_pos: np.ndarray): # KL:implement on this function
        embed()
        lin_vel = self.linear_velocity[2]
        ang_vel = self.angular_velocity[1]
        base_vel = [lin_vel, ang_vel]
        self.env._episode_over = False
        k = 'agent_1_oracle_nav_randcoord_action'
        # my_env.env.task.actions[k].coord_nav = self.observations['agent_0_localization_sensor'][:3]
        my_env.env.task.actions[k].coord_nav = np.array(mid_door_pos)
        # print("ROBOT position: ", self.observations['agent_0_localization_sensor'][:3])
        self.env.task.actions[k].step()
        self.observations.update(self.env.step({"action": 'agent_0_base_velocity', "action_args":{"agent_0_base_vel":base_vel}}))
    
    

def callback(vel, my_env):
    #### Robot Control ####
    my_env.linear_velocity = np.array([(1.0 * vel.linear.y), 0.0, (1.0 * vel.linear.x)])
    my_env.angular_velocity = np.array([0, vel.angular.z, 0])
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-render", action="store_true", default=True)
    parser.add_argument("--save-obs", action="store_true", default=False)
    parser.add_argument("--save-obs-fname", type=str, default="play.mp4")
    parser.add_argument("--save-actions", action="store_true", default=False)
    parser.add_argument(
        "--save-actions-fname", type=str, default="play_actions.txt"
    )
    parser.add_argument(
        "--save-actions-count",
        type=int,
        default=200,
        help="""
            The number of steps the saved action trajectory is clipped to. NOTE
            the episode must be at least this long or it will terminate with
            error.
            """,
    )
    parser.add_argument("--play-cam-res", type=int, default=512)
    parser.add_argument(
        "--skip-render-text", action="store_true", default=False
    )
    parser.add_argument(
        "--same-task",
        action="store_true",
        default=False,
        help="If true, then do not add the render camera for better visualization",
    )
    parser.add_argument(
        "--skip-task",
        action="store_true",
        default=False,
        help="If true, then do not add the render camera for better visualization",
    )
    parser.add_argument(
        "--never-end",
        action="store_true",
        default=False,
        help="If true, make the task never end due to reaching max number of steps",
    )
    parser.add_argument(
        "--disable-inverse-kinematics",
        action="store_true",
        help="If specified, does not add the inverse kinematics end-effector control.",
    )

    parser.add_argument(
        "--control-humanoid",
        action="store_true",
        default=False,
        help="Control humanoid agent.",
    )

    parser.add_argument(
        "--use-humanoid-controller",
        action="store_true",
        default=False,
        help="Control humanoid agent.",
    )

    parser.add_argument(
        "--gfx",
        action="store_true",
        default=False,
        help="Save a GFX replay file.",
    )
    parser.add_argument("--load-actions", type=str, default=None)
    parser.add_argument("--cfg", type=str, default=DEFAULT_CFG)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--walk-pose-path", type=str, default=DEFAULT_POSE_PATH
    )

    args = parser.parse_args()
    
    config = habitat.get_config(args.cfg, args.opts)
    with habitat.config.read_write(config):
        env_config = config.habitat.environment
        sim_config = config.habitat.simulator
        task_config = config.habitat.task

        if not args.same_task:
            sim_config.debug_render = True
            agent_config = get_agent_config(sim_config=sim_config)
            agent_config.sim_sensors.update(
                {
                    "third_rgb_sensor": ThirdRGBSensorConfig(
                        height=args.play_cam_res, width=args.play_cam_res
                    )
                }
            )
            if "pddl_success" in task_config.measurements:
                task_config.measurements.pddl_success.must_call_stop = False
            if "rearrange_nav_to_obj_success" in task_config.measurements:
                task_config.measurements.rearrange_nav_to_obj_success.must_call_stop = (
                    False
                )
            if "force_terminate" in task_config.measurements:
                task_config.measurements.force_terminate.max_accum_force = -1.0
                task_config.measurements.force_terminate.max_instant_force = (
                    -1.0
                )

        if args.gfx:
            sim_config.habitat_sim_v0.enable_gfx_replay_save = True
            task_config.measurements.update(
                {"gfx_replay_measure": GfxReplayMeasureMeasurementConfig()}
            )

        if args.never_end:
            env_config.max_episode_steps = 0

        if args.control_humanoid:
            args.disable_inverse_kinematics = True

        if not args.disable_inverse_kinematics:
            if "arm_action" not in task_config.actions:
                raise ValueError(
                    "Action space does not have any arm control so cannot add inverse kinematics. Specify the `--disable-inverse-kinematics` option"
                )
            sim_config.agents.main_agent.ik_arm_urdf = (
                "./data/robots/hab_fetch/robots/fetch_onlyarm.urdf"
            )
            task_config.actions.arm_action.arm_controller = "ArmEEAction"
        if task_config.type == "RearrangePddlTask-v0":
            task_config.actions["pddl_apply_action"] = PddlApplyActionConfig()
    
    
    my_env = sim_env(config)
    
    # save the top down map
    # top_down_map = maps.get_topdown_map_from_sim(
    #     cast("HabitatSim", my_env.env.sim), map_resolution=5000
    # )
    pathfinder = my_env.env.sim.pathfinder
    top_down_map = maps.get_topdown_map(
            pathfinder, height=0.0, meters_per_pixel=0.025
        )

    recolor_map = np.array(
        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
    )
    top_down_map = recolor_map[top_down_map]
    cv2.imwrite("top_down_map_test.jpg", top_down_map)
    
    
    image = plt.imread("top_down_map_test.jpg")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image)
    plt.title('Click on the image to get coordinates')

    coords = []
    def onclick(event):
        global coords
        ix, iy = int(event.xdata), int(event.ydata)  # Convert float coordinates to integers
        print(f'x = {ix}, y = {iy}')

        coords.append([ix, iy])
        
        if len(coords) == 2:
            fig.canvas.mpl_disconnect(cid)
    # Connect the onclick function to the mouse click event
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    # Display the image
    plt.show()


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
    
    # Get the end point for door_line
    
    print("-------Coords for Door position----------")
    grid_dim = (top_down_map.shape[0], top_down_map.shape[1])
    # embed()
    door1 = from_grid(pathfinder, coords[0], grid_dim)
    door2 = from_grid(pathfinder, coords[1], grid_dim)
    
    #TODO: test door distance
    z1 = my_env.observations['agent_0_localization_sensor'][1] #from robot_pos height
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
    robot_pos = my_env.observations['agent_0_localization_sensor'][:3]
    robot_xy = to_grid(pathfinder, robot_pos, grid_dim)
    human_pos = my_env.observations['agent_1_localization_sensor'][:3]
    human_xy = to_grid(pathfinder, human_pos, grid_dim)
    print("TEST ROBOT, HUMAN xy position: ", robot_xy, human_xy)
    cv2.circle(top_down_map, [int(robot_xy[0]), int(robot_xy[1])], 10, (0, 255, 0), -1) #green points
    cv2.circle(top_down_map, [int(human_xy[0]), int(human_xy[1])], 10, (255, 0, 255), -1) #green opposite points
    #visualize goal position
    mid_door_xy = to_grid(pathfinder, mid_door_pos, grid_dim)
    cv2.circle(top_down_map, [int(mid_door_xy[0]), int(mid_door_xy[1])], 10, (255, 255, 0), -1)
    cv2.imwrite("top_down_map_door.jpg", top_down_map)
    
    
    
    #check mid_door_pos to obstacle in realword
    im_1 = my_env.observations["agent_0_third_rgb"]
    cv2.imwrite("top_down_map_robot_view.jpg", im_1)
    dist_to_obs = pathfinder.distance_to_closest_obstacle(
        mid_door_pos,
        max_search_radius=10.0
        )
    print("Test distance door to obstacle: ", dist_to_obs)

    # def sample_points_helper(x, y, radius, num_points=10):
    #     # Generate random angles
    #     angles = np.linspace(0, 2*np.pi, num_points)
    #     # Calculate coordinates of sampled points
    #     points = [(x + radius * np.cos(angle), y + radius * np.sin(angle)) for angle in angles]

    #     return points
    
    def get_opposite_points():
        sampled_points_a = []
        sampled_points_b = []
        # samples = sample_points_helper(mid_door_pos[:1], mid_door_pos[1:2], 5)
        # for point in samples:
        #     x, y = point
        #     det = (x - r1) * (c2 - c1) - (y - c1) * (r2 - r1)
        #     if det > 0:
        #         sampled_points_a.append(point)
        #     else:
        #         sampled_points_b.append(point)
        
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
        cv2.imwrite("top_down_map_door_samples.jpg", top_down_map)
    
    get_opposite_points()
    
    
    # divide from and to points
    print("Generate pos from map finished")
    
    #-----Play agent
    human_images = []
    robot_images = []
    print("ROBOT position: ", my_env.observations['agent_0_localization_sensor'][:3])
    for i in range (1000):
   
        my_env.update_agent_pos_vel(mid_door_pos)
        im_0 = my_env.observations["agent_1_head_rgb"]
        im_1 = my_env.observations["agent_0_third_rgb"]

        human_images.append(im_0)
        robot_images.append(im_1)

    images_to_video(human_images, "test", "human_trajectory")
    images_to_video(robot_images, "test", "robot_trajectory")
    print("Episode finished")
        