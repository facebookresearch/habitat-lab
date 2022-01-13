#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# type: ignore

import argparse
import os
import random
import sys
import time
from math import pi

import numpy as np
import orbslam2
import PIL
import requests
import torch
from torch.nn import functional as F

import habitat
from habitat.config.default import get_config
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.config.default import get_config as cfg_baseline
from habitat_baselines.slambased.mappers import DirectDepthMapper
from habitat_baselines.slambased.monodepth import MonoDepthEstimator
from habitat_baselines.slambased.path_planners import DifferentiableStarPlanner
from habitat_baselines.slambased.reprojection import (
    angle_to_pi_2_minus_pi_2 as norm_ang,
)
from habitat_baselines.slambased.reprojection import (
    get_direction,
    get_distance,
    habitat_goalpos_to_mapgoal_pos,
    homogenize_p,
    planned_path2tps,
    project_tps_into_worldmap,
)
from habitat_baselines.slambased.utils import generate_2dgrid

GOAL_SENSOR_UUID = "pointgoal_with_gps_compass"


def download(url, filename):
    with open(filename, "wb") as f:
        response = requests.get(url, stream=True)
        total = response.headers.get("content-length")
        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(
                chunk_size=max(int(total / 1000), 1024 * 1024)
            ):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total)
                sys.stdout.write(
                    "\r[{}{}]".format("█" * done, "." * (50 - done))
                )
                sys.stdout.flush()
    sys.stdout.write("\n")


def ResizePIL2(np_img, size=256):
    im1 = PIL.Image.fromarray(np_img)
    return np.array(im1.resize((size, size)))


def make_good_config_for_orbslam2(config):
    config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
    config.SIMULATOR.RGB_SENSOR.WIDTH = 256
    config.SIMULATOR.RGB_SENSOR.HEIGHT = 256
    config.SIMULATOR.DEPTH_SENSOR.WIDTH = 256
    config.SIMULATOR.DEPTH_SENSOR.HEIGHT = 256
    config.TRAINER.ORBSLAM2.CAMERA_HEIGHT = (
        config.SIMULATOR.DEPTH_SENSOR.POSITION[1]
    )
    config.TRAINER.ORBSLAM2.H_OBSTACLE_MIN = (
        0.3 * config.TRAINER.ORBSLAM2.CAMERA_HEIGHT
    )
    config.TRAINER.ORBSLAM2.H_OBSTACLE_MAX = (
        1.0 * config.TRAINER.ORBSLAM2.CAMERA_HEIGHT
    )
    config.TRAINER.ORBSLAM2.MIN_PTS_IN_OBSTACLE = (
        config.SIMULATOR.DEPTH_SENSOR.WIDTH / 2.0
    )
    return


class RandomAgent:
    r"""Simplest agent, which returns random actions,
    until reach the goal
    """

    def __init__(self, config):
        super(RandomAgent, self).__init__()
        self.num_actions = config.NUM_ACTIONS
        self.dist_threshold_to_stop = config.DIST_TO_STOP
        self.reset()
        return

    def reset(self):
        self.steps = 0
        return

    def update_internal_state(self, habitat_observation):
        self.obs = habitat_observation
        self.steps += 1
        return

    def is_goal_reached(self):
        dist = self.obs[GOAL_SENSOR_UUID][0]
        return dist <= self.dist_threshold_to_stop

    def act(self, habitat_observation=None, random_prob=1.0):
        self.update_internal_state(habitat_observation)
        # Act
        # Check if we are done
        if self.is_goal_reached():
            action = HabitatSimActions.STOP
        else:
            action = random.randint(0, self.num_actions - 1)
        return {"action": action}


class BlindAgent(RandomAgent):
    def __init__(self, config):
        super(BlindAgent, self).__init__(config)
        self.pos_th = config.DIST_TO_STOP
        self.angle_th = config.ANGLE_TH
        self.reset()
        return

    def decide_what_to_do(self):
        distance_to_goal = self.obs[GOAL_SENSOR_UUID][0]
        angle_to_goal = norm_ang(np.array(self.obs[GOAL_SENSOR_UUID][1]))
        command = HabitatSimActions.STOP
        if distance_to_goal <= self.pos_th:
            return command
        if abs(angle_to_goal) < self.angle_th:
            command = HabitatSimActions.MOVE_FORWARD
        else:
            if (angle_to_goal > 0) and (angle_to_goal < pi):
                command = HabitatSimActions.TURN_LEFT
            elif (angle_to_goal > pi) or (
                angle_to_goal < 0 and angle_to_goal > -pi
            ):
                command = HabitatSimActions.TURN_RIGHT
            else:
                command = HabitatSimActions.TURN_LEFT

        return command

    def act(self, habitat_observation=None, random_prob=0.1):
        self.update_internal_state(habitat_observation)
        # Act
        if self.is_goal_reached():
            return HabitatSimActions.STOP
        command = self.decide_what_to_do()
        random_action = random.randint(0, self.num_actions - 1)
        act_randomly = np.random.uniform(0, 1, 1) < random_prob
        if act_randomly:
            action = random_action
        else:
            action = command
        return {"action": action}


class ORBSLAM2Agent(RandomAgent):
    def __init__(self, config, device=torch.device("cuda:0")):  # noqa: B008
        super(ORBSLAM2Agent, self).__init__(config)
        self.num_actions = config.NUM_ACTIONS
        self.dist_threshold_to_stop = config.DIST_TO_STOP
        self.slam_vocab_path = config.SLAM_VOCAB_PATH
        assert os.path.isfile(self.slam_vocab_path)
        self.slam_settings_path = config.SLAM_SETTINGS_PATH
        assert os.path.isfile(self.slam_settings_path)
        self.slam = orbslam2.System(
            self.slam_vocab_path, self.slam_settings_path, orbslam2.Sensor.RGBD
        )
        self.slam.set_use_viewer(False)
        self.slam.initialize()
        self.device = device
        self.map_size_meters = config.MAP_SIZE
        self.map_cell_size = config.MAP_CELL_SIZE
        self.pos_th = config.DIST_REACHED_TH
        self.next_wp_th = config.NEXT_WAYPOINT_TH
        self.angle_th = config.ANGLE_TH
        self.obstacle_th = config.MIN_PTS_IN_OBSTACLE
        self.depth_denorm = config.DEPTH_DENORM
        self.planned_waypoints = []
        self.mapper = DirectDepthMapper(
            camera_height=config.CAMERA_HEIGHT,
            near_th=config.D_OBSTACLE_MIN,
            far_th=config.D_OBSTACLE_MAX,
            h_min=config.H_OBSTACLE_MIN,
            h_max=config.H_OBSTACLE_MAX,
            map_size=config.MAP_SIZE,
            map_cell_size=config.MAP_CELL_SIZE,
            device=device,
        )
        self.planner = DifferentiableStarPlanner(
            max_steps=config.PLANNER_MAX_STEPS,
            preprocess=config.PREPROCESS_MAP,
            beta=config.BETA,
            device=device,
        )
        self.slam_to_world = 1.0
        self.timestep = 0.1
        self.timing = False
        self.reset()
        return

    def reset(self):
        super(ORBSLAM2Agent, self).reset()
        self.offset_to_goal = None
        self.tracking_is_OK = False
        self.waypointPose6D = None
        self.unseen_obstacle = False
        self.action_history = []
        self.planned_waypoints = []
        self.map2DObstacles = self.init_map2d()
        n, ch, height, width = self.map2DObstacles.size()
        self.coordinatesGrid = generate_2dgrid(height, width, False).to(
            self.device
        )
        self.pose6D = self.init_pose6d()
        self.action_history = []
        self.pose6D_history = []
        self.position_history = []
        self.planned2Dpath = torch.zeros((0))
        self.slam.reset()
        self.cur_time = 0
        self.toDoList = []
        self.waypoint_id = 0
        if self.device != torch.device("cpu"):
            torch.cuda.empty_cache()
        return

    def update_internal_state(self, habitat_observation):
        super(ORBSLAM2Agent, self).update_internal_state(habitat_observation)
        self.cur_time += self.timestep
        rgb, depth = self.rgb_d_from_observation(habitat_observation)
        t = time.time()
        try:
            self.slam.process_image_rgbd(rgb, depth, self.cur_time)
            if self.timing:
                print(time.time() - t, "ORB_SLAM2")
            self.tracking_is_OK = str(self.slam.get_tracking_state()) == "OK"
        except BaseException:
            print("Warning!!!! ORBSLAM processing frame error")
            self.tracking_is_OK = False
        if not self.tracking_is_OK:
            self.reset()
        t = time.time()
        self.set_offset_to_goal(habitat_observation)
        if self.tracking_is_OK:
            trajectory_history = np.array(self.slam.get_trajectory_points())
            self.pose6D = homogenize_p(
                torch.from_numpy(trajectory_history[-1])[1:]
                .view(3, 4)
                .to(self.device)
            ).view(1, 4, 4)
            self.trajectory_history = trajectory_history
            if len(self.position_history) > 1:
                previous_step = get_distance(
                    self.pose6D.view(4, 4),
                    torch.from_numpy(self.position_history[-1])
                    .view(4, 4)
                    .to(self.device),
                )
                if self.action_history[-1] == HabitatSimActions.MOVE_FORWARD:
                    self.unseen_obstacle = (
                        previous_step.item() <= 0.001
                    )  # hardcoded threshold for not moving
        current_obstacles = self.mapper(
            torch.from_numpy(depth).to(self.device).squeeze(), self.pose6D
        ).to(self.device)
        self.current_obstacles = current_obstacles
        self.map2DObstacles = torch.max(
            self.map2DObstacles, current_obstacles.unsqueeze(0).unsqueeze(0)
        )
        if self.timing:
            print(time.time() - t, "Mapping")
        return True

    def init_pose6d(self):
        return torch.eye(4).float().to(self.device)

    def map_size_in_cells(self):
        return int(self.map_size_meters / self.map_cell_size)

    def init_map2d(self):
        return (
            torch.zeros(
                1, 1, self.map_size_in_cells(), self.map_size_in_cells()
            )
            .float()
            .to(self.device)
        )

    def get_orientation_on_map(self):
        self.pose6D = self.pose6D.view(1, 4, 4)
        return torch.tensor(
            [
                [self.pose6D[0, 0, 0], self.pose6D[0, 0, 2]],
                [self.pose6D[0, 2, 0], self.pose6D[0, 2, 2]],
            ]
        )

    def get_position_on_map(self, do_floor=True):
        return project_tps_into_worldmap(
            self.pose6D.view(1, 4, 4),
            self.map_cell_size,
            self.map_size_meters,
            do_floor,
        )

    def act(self, habitat_observation, random_prob=0.1):
        # Update internal state
        t = time.time()
        cc = 0
        update_is_ok = self.update_internal_state(habitat_observation)
        while not update_is_ok:
            update_is_ok = self.update_internal_state(habitat_observation)
            cc += 1
            if cc > 2:
                break
        if self.timing:
            print(time.time() - t, " s, update internal state")
        self.position_history.append(
            self.pose6D.detach().cpu().numpy().reshape(1, 4, 4)
        )
        success = self.is_goal_reached()
        if success:
            action = HabitatSimActions.STOP
            self.action_history.append(action)
            return {"action": action}
        # Plan action
        t = time.time()
        self.planned2Dpath, self.planned_waypoints = self.plan_path()
        if self.timing:
            print(time.time() - t, " s, Planning")
        t = time.time()
        # Act
        if self.waypointPose6D is None:
            self.waypointPose6D = self.get_valid_waypoint_pose6d()
        if (
            self.is_waypoint_reached(self.waypointPose6D)
            or not self.tracking_is_OK
        ):
            self.waypointPose6D = self.get_valid_waypoint_pose6d()
            self.waypoint_id += 1
        action = self.decide_what_to_do()
        # May be random?
        random_action = random.randint(0, self.num_actions - 1)
        what_to_do = np.random.uniform(0, 1, 1)
        if what_to_do < random_prob:
            action = random_action
        if self.timing:
            print(time.time() - t, " s, get action")
        self.action_history.append(action)
        return {"action": action}

    def is_waypoint_good(self, pose6d):
        p_init = self.pose6D.squeeze()
        dist_diff = get_distance(p_init, pose6d)
        valid = dist_diff > self.next_wp_th
        return valid.item()

    def is_waypoint_reached(self, pose6d):
        p_init = self.pose6D.squeeze()
        dist_diff = get_distance(p_init, pose6d)
        reached = dist_diff <= self.pos_th
        return reached.item()

    def get_waypoint_dist_dir(self):
        angle = get_direction(
            self.pose6D.squeeze(), self.waypointPose6D.squeeze(), 0, 0
        )
        dist = get_distance(
            self.pose6D.squeeze(), self.waypointPose6D.squeeze()
        )
        return torch.cat(
            [
                dist.view(1, 1),
                torch.sin(angle).view(1, 1),
                torch.cos(angle).view(1, 1),
            ],
            dim=1,
        )

    def get_valid_waypoint_pose6d(self):
        p_next = self.planned_waypoints[0]
        while not self.is_waypoint_good(p_next):
            if len(self.planned_waypoints) > 1:
                self.planned_waypoints = self.planned_waypoints[1:]
                p_next = self.planned_waypoints[0]
            else:
                p_next = self.estimatedGoalPos6D.squeeze()
                break
        return p_next

    def set_offset_to_goal(self, observation):
        self.offset_to_goal = (
            torch.from_numpy(observation[GOAL_SENSOR_UUID])
            .float()
            .to(self.device)
        )
        self.estimatedGoalPos2D = habitat_goalpos_to_mapgoal_pos(
            self.offset_to_goal,
            self.pose6D.squeeze(),
            self.map_cell_size,
            self.map_size_meters,
        )
        self.estimatedGoalPos6D = planned_path2tps(
            [self.estimatedGoalPos2D],
            self.map_cell_size,
            self.map_size_meters,
            1.0,
        ).to(self.device)[0]
        return

    def rgb_d_from_observation(self, habitat_observation):
        rgb = habitat_observation["rgb"]
        depth = None
        if "depth" in habitat_observation:
            depth = self.depth_denorm * habitat_observation["depth"]
        return rgb, depth

    def prev_plan_is_not_valid(self):
        if len(self.planned2Dpath) == 0:
            return True
        pp = torch.cat(self.planned2Dpath).detach().cpu().view(-1, 2)
        binary_map = self.map2DObstacles.squeeze().detach() >= self.obstacle_th
        obstacles_on_path = (
            binary_map[pp[:, 0].long(), pp[:, 1].long()]
        ).long().sum().item() > 0
        return obstacles_on_path  # obstacles_nearby or  obstacles_on_path

    def rawmap2_planner_ready(self, rawmap, start_map, goal_map):
        map1 = (rawmap / float(self.obstacle_th)) ** 2
        map1 = (
            torch.clamp(map1, min=0, max=1.0)
            - start_map
            - F.max_pool2d(goal_map, 3, stride=1, padding=1)
        )
        return torch.relu(map1)

    def plan_path(self, overwrite=False):
        t = time.time()
        if (
            (not self.prev_plan_is_not_valid())
            and (not overwrite)
            and (len(self.planned_waypoints) > 0)
        ):
            return self.planned2Dpath, self.planned_waypoints
        self.waypointPose6D = None
        current_pos = self.get_position_on_map()
        start_map = torch.zeros_like(self.map2DObstacles).to(self.device)
        start_map[
            0, 0, current_pos[0, 0].long(), current_pos[0, 1].long()
        ] = 1.0
        goal_map = torch.zeros_like(self.map2DObstacles).to(self.device)
        goal_map[
            0,
            0,
            self.estimatedGoalPos2D[0, 0].long(),
            self.estimatedGoalPos2D[0, 1].long(),
        ] = 1.0
        path, cost = self.planner(
            self.rawmap2_planner_ready(
                self.map2DObstacles, start_map, goal_map
            ).to(self.device),
            self.coordinatesGrid.to(self.device),
            goal_map.to(self.device),
            start_map.to(self.device),
        )
        if len(path) == 0:
            return path, []
        if self.timing:
            print(time.time() - t, " s, Planning")
        t = time.time()
        planned_waypoints = planned_path2tps(
            path, self.map_cell_size, self.map_size_meters, 1.0, False
        ).to(self.device)
        return path, planned_waypoints

    def planner_prediction_to_command(self, p_next):
        command = HabitatSimActions.STOP
        p_init = self.pose6D.squeeze()
        d_angle_rot_th = self.angle_th
        pos_th = self.pos_th
        if get_distance(p_init, p_next) <= pos_th:
            return command
        d_angle = norm_ang(
            get_direction(p_init, p_next, ang_th=d_angle_rot_th, pos_th=pos_th)
        )
        if abs(d_angle) < d_angle_rot_th:
            command = HabitatSimActions.MOVE_FORWARD
        else:
            if (d_angle > 0) and (d_angle < pi):
                command = HabitatSimActions.TURN_LEFT
            elif (d_angle > pi) or (d_angle < 0 and d_angle > -pi):
                command = HabitatSimActions.TURN_RIGHT
            else:
                command = HabitatSimActions.TURN_LEFT
        return command

    def decide_what_to_do(self):
        action = None
        if self.is_goal_reached():
            action = HabitatSimActions.STOP
            return {"action": action}
        if self.unseen_obstacle:
            command = HabitatSimActions.TURN_RIGHT
            return command
        command = HabitatSimActions.STOP
        command = self.planner_prediction_to_command(self.waypointPose6D)
        return command


class ORBSLAM2MonodepthAgent(ORBSLAM2Agent):
    def __init__(
        self,
        config,
        device=torch.device("cuda:0"),  # noqa: B008
        monocheckpoint="habitat_baselines/slambased/data/mp3d_resnet50.pth",
    ):
        super(ORBSLAM2MonodepthAgent, self).__init__(config)
        self.num_actions = config.NUM_ACTIONS
        self.dist_threshold_to_stop = config.DIST_TO_STOP
        self.slam_vocab_path = config.SLAM_VOCAB_PATH
        assert os.path.isfile(self.slam_vocab_path)
        self.slam_settings_path = config.SLAM_SETTINGS_PATH
        assert os.path.isfile(self.slam_settings_path)
        self.slam = orbslam2.System(
            self.slam_vocab_path, self.slam_settings_path, orbslam2.Sensor.RGBD
        )
        self.slam.set_use_viewer(False)
        self.slam.initialize()
        self.device = device
        self.map_size_meters = config.MAP_SIZE
        self.map_cell_size = config.MAP_CELL_SIZE
        self.pos_th = config.DIST_REACHED_TH
        self.next_wp_th = config.NEXT_WAYPOINT_TH
        self.angle_th = config.ANGLE_TH
        self.obstacle_th = config.MIN_PTS_IN_OBSTACLE
        self.depth_denorm = config.DEPTH_DENORM
        self.planned_waypoints = []
        self.mapper = DirectDepthMapper(
            camera_height=config.CAMERA_HEIGHT,
            near_th=config.D_OBSTACLE_MIN,
            far_th=config.D_OBSTACLE_MAX,
            h_min=config.H_OBSTACLE_MIN,
            h_max=config.H_OBSTACLE_MAX,
            map_size=config.MAP_SIZE,
            map_cell_size=config.MAP_CELL_SIZE,
            device=device,
        )
        self.planner = DifferentiableStarPlanner(
            max_steps=config.PLANNER_MAX_STEPS,
            preprocess=config.PREPROCESS_MAP,
            beta=config.BETA,
            device=device,
        )
        self.slam_to_world = 1.0
        self.timestep = 0.1
        self.timing = False
        self.checkpoint = monocheckpoint
        if not os.path.isfile(self.checkpoint):
            mp3d_url = "http://cmp.felk.cvut.cz/~mishkdmy/navigation/mp3d_ft_monodepth_resnet50.pth"
            # suncg_me_url = "http://cmp.felk.cvut.cz/~mishkdmy/navigation/suncg_me_resnet.pth"
            # suncg_mf_url = "http://cmp.felk.cvut.cz/~mishkdmy/navigation/suncg_mf_resnet.pth"
            url = mp3d_url
            print("No monodepth checkpoint found. Downloading...", url)
            download(url, self.checkpoint)
        self.monodepth = MonoDepthEstimator(self.checkpoint)
        self.reset()
        return

    def rgb_d_from_observation(self, habitat_observation):
        rgb = habitat_observation["rgb"]
        depth = ResizePIL2(
            self.monodepth.compute_depth(
                PIL.Image.fromarray(rgb).resize((320, 320))
            ),
            256,
        )  # /1.75
        depth[depth > 3.0] = 0
        depth[depth < 0.1] = 0
        return rgb, np.array(depth).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent-type",
        default="orbslam2-rgbd",
        choices=["blind", "orbslam2-rgbd", "orbslam2-rgb-monod"],
    )
    parser.add_argument(
        "--task-config", type=str, default="tasks/pointnav_rgbd.yaml"
    )
    args = parser.parse_args()

    config = get_config()
    agent_config = cfg_baseline()
    config.defrost()
    config.BASELINE = agent_config.BASELINE
    make_good_config_for_orbslam2(config)

    if args.agent_type == "blind":
        agent = BlindAgent(config.TRAINER.ORBSLAM2)
    elif args.agent_type == "orbslam2-rgbd":
        agent = ORBSLAM2Agent(config.TRAINER.ORBSLAM2)
    elif args.agent_type == "orbslam2-rgb-monod":
        agent = ORBSLAM2MonodepthAgent(config.TRAINER.ORBSLAM2)
    else:
        raise ValueError(args.agent_type, "is unknown type of agent")
    benchmark = habitat.Benchmark(args.task_config)
    metrics = benchmark.evaluate(agent)
    for k, v in metrics.items():
        habitat.logger.info("{}: {:.3f}".format(k, v))


if __name__ == "__main__":
    main()
