#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import getpass
import json
import math
import os.path as osp
import random
import socket
import time
from time import sleep
from typing import List, NamedTuple, Tuple

import cv2
import numpy as np

import habitat
from examples.agent_demo.demo_blind_agent import DemoBlindAgent
from habitat.core.logging import logger
from habitat.tasks.nav.nav_task import NavigationEpisode, NavigationGoal
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps

OUTPUT_FILE = "{}-{}-results.json".format(
    socket.gethostname(), getpass.getuser()
)


class ActionKeyMapping(NamedTuple):
    name: str
    key: int
    action_id: int
    is_quit: bool = False


AGENT_ACTION_KEYS = [
    ActionKeyMapping(
        "FORWARD", ord("w"), habitat.SimulatorActions.MOVE_FORWARD
    ),
    ActionKeyMapping("LEFT", ord("a"), habitat.SimulatorActions.TURN_LEFT),
    ActionKeyMapping("RIGHT", ord("d"), habitat.SimulatorActions.TURN_RIGHT),
    ActionKeyMapping("DONE", ord(" "), "STOP"),
    ActionKeyMapping("QUIT", ord("q"), -1, True),
]

INSTRUCTIONS = [
    "Use W/A/D to move Forward/Left/Right",
    "Press <Space> when you reach the goal",
    "Q - Quit",
]

LINE_SPACING = 50

# TODO: Some of the functions below are potentially useful across other examples/demos
#       and should be moved to habitat/utils/visualizations


class Rect(NamedTuple):
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self):
        return self.left + self.width

    @property
    def bottom(self):
        return self.top + self.height

    @property
    def center(self):
        return (
            self.left + int(self.width / 2),
            self.top + int(self.height / 2),
        )


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def write_textlines(
    output, textlines, size=1, offset=(0, 0), fontcolor=(255, 255, 255)
):
    for i, text in enumerate(textlines):
        x = offset[1]
        y = offset[0] + int((i + 1) * size * LINE_SPACING) - 15
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            output, text, (x, y), font, size, fontcolor, 2, cv2.LINE_AA
        )


def draw_text(textlines=[], width=300, fontsize=0.8):
    text_height = int(fontsize * LINE_SPACING * len(textlines))
    text_img = np.zeros((text_height, width, 3), np.uint8)
    write_textlines(text_img, textlines, size=fontsize)
    return text_img


def add_text(img, textlines=[], fontsize=0.8, top=False):
    combined = img
    if len(textlines) > 0:
        text_img = draw_text(textlines, img.shape[1], fontsize)
        if top:
            combined = np.vstack((text_img, img))
        else:
            combined = np.vstack((img, text_img))
    return combined


def draw_gradient_circle(img, center, size, color, bgcolor):
    """ Draws a circle that fades from color (at the center)
        to bgcolor (at the boundaries)
    """
    for i in range(1, size):
        a = 1 - i / size
        c = np.add(
            np.multiply(color[0:3], a), np.multiply(bgcolor[0:3], 1 - a)
        )
        cv2.circle(img, center, i, c, 2)


def draw_gradient_wedge(
    img, center, size, color, bgcolor, start_angle, delta_angle
):
    """ Draws a wedge that fades from color (at the center)
        to bgcolor (at the boundaries)
    """
    for i in range(1, size):
        a = 1 - i / size
        c = np.add(np.multiply(color, a), np.multiply(bgcolor, 1 - a))
        cv2.ellipse(
            img,
            center,
            (i, i),
            start_angle,
            -delta_angle / 2,
            delta_angle / 2,
            c,
            2,
        )


def draw_goal_radar(
    pointgoal_with_gps_compass,
    img,
    r: Rect,
    start_angle=0,
    fov=0,
    goalcolor=(50, 0, 184, 255),
    wincolor=(0, 0, 0, 0),
    maskcolor=(85, 75, 70, 255),
    bgcolor=(255, 255, 255, 255),
    gradientcolor=(174, 112, 80, 255),
):
    """ Draws a radar that shows the goal as a dot
    """
    angle = pointgoal_with_gps_compass[1]  # angle
    mag = pointgoal_with_gps_compass[0]  # magnitude (>=0)
    nm = mag / (mag + 1)  # normalized magnitude (0 to 1)
    xy = (-math.sin(angle), -math.cos(angle))
    size = int(round(0.45 * min(r.width, r.height)))
    center = r.center
    target = (
        int(round(center[0] + xy[0] * size * nm)),
        int(round(center[1] + xy[1] * size * nm)),
    )
    if wincolor is not None:
        cv2.rectangle(
            img, (r.left, r.top), (r.right, r.bottom), wincolor, -1
        )  # Fill with window color
    cv2.circle(img, center, size, bgcolor, -1)  # Circle with background color
    if fov > 0:
        masked = 360 - fov
        cv2.ellipse(
            img,
            center,
            (size, size),
            start_angle + 90,
            -masked / 2,
            masked / 2,
            maskcolor,
            -1,
        )
    if gradientcolor is not None:
        if fov > 0:
            draw_gradient_wedge(
                img,
                center,
                size,
                gradientcolor,
                bgcolor,
                start_angle - 90,
                fov,
            )
        else:
            draw_gradient_circle(img, center, size, gradientcolor, bgcolor)
    cv2.circle(img, target, 4, goalcolor, -1)


def draw_top_down_map(info, heading, output_size):
    top_down_map = maps.colorize_topdown_map(info["top_down_map"]["map"])
    original_map_size = top_down_map.shape[:2]
    map_scale = np.array(
        (1, original_map_size[1] * 1.0 / original_map_size[0])
    )
    new_map_size = np.round(output_size * map_scale).astype(np.int32)
    # OpenCV expects w, h but map size is in h, w
    top_down_map = cv2.resize(top_down_map, (new_map_size[1], new_map_size[0]))

    map_agent_pos = info["top_down_map"]["agent_map_coord"]
    map_agent_pos = np.round(
        map_agent_pos * new_map_size / original_map_size
    ).astype(np.int32)
    top_down_map = maps.draw_agent(
        top_down_map,
        map_agent_pos,
        heading - np.pi / 2,
        agent_radius_px=top_down_map.shape[0] / 40,
    )
    return top_down_map


def get_goal_radius(env):
    return 0.2
    goal_radius = env.current_episode.goals[0].radius
    if goal_radius is None:
        goal_radius = config.SIMULATOR.FORWARD_STEP_SIZE
    return goal_radius


class Viewer:
    def __init__(
        self,
        initial_observations,
        overlay_goal_radar=None,
        goal_display_size=128,
        show_map=False,
    ):
        self.overlay_goal_radar = overlay_goal_radar
        self.show_map = show_map

        # What image sensors are active
        all_image_sensors = ["rgb", "depth"]
        self.active_image_sensors = [
            s for s in all_image_sensors if s in initial_observations
        ]
        total_width = 0
        total_height = 0
        for s in self.active_image_sensors:
            total_width += initial_observations[s].shape[1]
            total_height = max(initial_observations[s].shape[0], total_height)

        self.draw_info = {}
        if self.overlay_goal_radar:
            img = np.zeros((goal_display_size, goal_display_size, 4), np.uint8)
            self.draw_info["pointgoal_with_gps_compass"] = {
                "image": img,
                "region": Rect(0, 0, goal_display_size, goal_display_size),
            }
        else:
            side_img_height = max(total_height, goal_display_size)
            self.side_img = np.zeros(
                (side_img_height, goal_display_size, 3), np.uint8
            )
            self.draw_info["pointgoal_with_gps_compass"] = {
                "image": self.side_img,
                "region": Rect(0, 0, goal_display_size, goal_display_size),
            }
            total_width += goal_display_size
        self.window_size = (total_height, total_width)

    def draw_observations(self, observations, info=None):
        active_image_observations = [
            observations[s] for s in self.active_image_sensors
        ]
        for i, img in enumerate(active_image_observations):
            if img.shape[2] == 1:
                img *= 255.0 / img.max()  # naive rescaling for visualization
                active_image_observations[i] = cv2.cvtColor(
                    img, cv2.COLOR_GRAY2BGR
                ).astype(np.uint8)
            elif img.shape[2] == 3:
                active_image_observations[i] = transform_rgb_bgr(img)

        # draw pointgoal_with_gps_compass
        goal_draw_surface = self.draw_info["pointgoal_with_gps_compass"]
        # TODO: get fov from agent
        draw_goal_radar(
            observations["pointgoal_with_gps_compass"],
            goal_draw_surface["image"],
            goal_draw_surface["region"],
            start_angle=0,
            fov=90,
        )
        if self.overlay_goal_radar:
            goal_region = goal_draw_surface["region"]
            bottom = self.window_size[0]
            top = bottom - goal_region.height
            left = self.window_size[1] // 2 - goal_region.width // 2
            right = left + goal_region.width
            stacked = np.hstack(active_image_observations)
            alpha = 0.5 * (goal_draw_surface["image"][:, :, 3] / 255)
            rgb = goal_draw_surface["image"][:, :, 0:3]
            overlay = np.add(
                np.multiply(
                    stacked[top:bottom, left:right],
                    np.expand_dims(1 - alpha, axis=2),
                ),
                np.multiply(rgb, np.expand_dims(alpha, axis=2)),
            )
            stacked[top:bottom, left:right] = overlay
        else:
            stacked = np.hstack(active_image_observations + [self.side_img])
        if info is not None:
            if (
                self.show_map
                and info.get("top_down_map") is not None
                and "heading" in observations
            ):
                top_down_map = draw_top_down_map(
                    info, observations["heading"], stacked.shape[0]
                )
                stacked = np.hstack((top_down_map, stacked))
        return stacked


class Demo:
    def __init__(
        self,
        config,
        action_keys: List[ActionKeyMapping],
        instructions: List[str],
    ):
        self.window_name = "Habitat"
        self.config = config
        self.instructions = instructions
        self.action_keys = action_keys
        self.action_keys_map = {k.key: k for k in self.action_keys}
        self.is_quit = False

        self.env = habitat.Env(config=self.config)
        logger.info("Environment creation successful")

    def update(self, img, video_writer=None):
        self.window_shape = img.shape
        if video_writer is not None:
            video_writer.write(img)
        cv2.imshow(self.window_name, img)

    def do_episode(
        self, overlay_goal_radar=False, show_map=False, video_writer=None
    ):
        """ Have human controlled navigation for one episode
        """
        env = self.env
        action_keys_map = self.action_keys_map

        observations = env.reset()
        info = env.get_metrics()
        viewer = Viewer(
            observations,
            overlay_goal_radar=overlay_goal_radar,
            show_map=show_map,
        )
        img = viewer.draw_observations(observations, info)
        goal_radius = get_goal_radius(env)
        distance = observations["pointgoal_with_gps_compass"][0]
        self.update(
            add_text(
                img,
                [f"Distance {distance:.5}/{goal_radius:.5}"]
                + self.instructions,
            )
        )

        episode = env.current_episode
        logger.info("Agent stepping around inside environment.")
        actions = []
        while not env.episode_over:
            keystroke = -1
            while keystroke == -1:
                keystroke = cv2.waitKey(100)

            action = action_keys_map.get(keystroke)
            if action is not None:
                logger.info(action.name)
                if action.is_quit:
                    self.is_quit = True
                    return None
            else:
                logger.info("INVALID KEY")
                continue

            actions.append(action.action_id)
            observations = env.step(action.action_id)
            info = env.get_metrics()

            img = viewer.draw_observations(observations, info)
            distance = observations["pointgoal_with_gps_compass"][0]
            self.update(
                add_text(
                    img,
                    [f"Distance {distance:.5}/{goal_radius:.5}"]
                    + self.instructions,
                ),
                video_writer,
            )

        logger.info("Episode finished after {} steps.".format(len(actions)))
        return actions, info, observations, episode

    def run(self, args):
        while not self.is_quit:
            res = self.do_episode(
                overlay_goal_radar=True, show_map=False, video_writer=None
            )
            if self.is_quit:
                self.env.close()
                return

            actions, info, observations, episode = res
            if osp.exists(OUTPUT_FILE):
                with open(OUTPUT_FILE, "r") as f:
                    prev_res = json.load(f)
            else:
                prev_res = []

            logger.info("SPL: {}".format(info["spl"]))

            result = dict(
                spl=info["spl"],
                episode=dict(
                    scene_id=episode.scene_id,
                    episode_id=episode.episode_id,
                    geo_dist=episode.info["geodesic_distance"],
                ),
                actions=actions,
            )
            with open(OUTPUT_FILE, "w") as f:
                json.dump(prev_res + [result], f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Habitat API Demo")
    parser.add_argument(
        "--task-config",
        default="configs/tasks/pointnav_gibson.yaml",
        help="Task configuration file for initializing a Habitat environment",
    )

    args = parser.parse_args()
    opts = []
    config = habitat.get_config(args.task_config.split(","), opts)
    config.defrost()
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.TASK.TOP_DOWN_MAP.MAP_RESOLUTION = 6000
    config.TASK.SENSORS.append("HEADING_SENSOR")
    config.DATASET.SPLIT = "val"
    config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = True
    config.ENVIRONMENT.ITERATOR_OPTIONS.GROUP_BY_SCENE = True
    config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT = 2
    config.ENVIRONMENT.ITERATOR_OPTIONS.NUM_EPISODE_SAMPLE = 994
    config.SIMULATOR.RGB_SENSOR.WIDTH = 720
    config.SIMULATOR.RGB_SENSOR.HEIGHT = 720
    config.SEED = random.randint(0, int(1e5))
    config.freeze()

    demo = Demo(config, AGENT_ACTION_KEYS, INSTRUCTIONS)
    demo.run(args)
