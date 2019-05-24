#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import cv2
import habitat
import math
import numpy as np

from typing import NamedTuple, Tuple

class ActionKeyMapping(NamedTuple):
    name: str
    key: int
    action_id: int
    is_quit: bool = False

AGENT_ACTION_KEYS = [
    ActionKeyMapping("FORWARD", ord("w"), 0),
    ActionKeyMapping("LEFT", ord("a"), 1),
    ActionKeyMapping("RIGHT", ord("d"), 2),
    ActionKeyMapping("DONE", ord(" "), 3),
    ActionKeyMapping("QUIT", ord("q"), -1, True)
]  

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
        return (self.left + int(self.width/2), self.top + int(self.height/2))    

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

class ObsViewer:
    def __init__(self, initial_observations, 
            goal_display_size=128, overlay_goal=None):
        self.window_name = "Habitat"
        self.overlay_goal = overlay_goal

        # What image sensors are active
        all_image_sensors = ['rgb', 'depth'] 
        self.active_image_sensors = [s for s in all_image_sensors if s in initial_observations]
        total_width = 0
        total_height = 0
        for s in self.active_image_sensors:
            total_width += initial_observations[s].shape[1]
            total_height = max(initial_observations[s].shape[0], total_height)

        self.draw_info = {}
        if self.overlay_goal:
            img = np.zeros((goal_display_size, goal_display_size, 3), np.uint8)
            self.draw_info["pointgoal"] = { "image": img, "region": Rect(0,0,goal_display_size,goal_display_size) }
        else:
            side_img_height = max(total_height, goal_display_size)
            self.side_img = np.zeros((side_img_height, goal_display_size, 3), np.uint8)
            self.draw_info["pointgoal"] = { "image": self.side_img, "region": Rect(0,0,goal_display_size,goal_display_size) }
            total_width += goal_display_size
        self.window_size = (total_height, total_width)  

    def show(self, observations):
        img = self.draw_observations(observations)
        cv2.imshow(self.window_name, img)

    def draw_observations(self, observations):    
        active_image_observations = [observations[s] for s in self.active_image_sensors]
        for i,img in enumerate(active_image_observations):
            if img.shape[2] == 1:
                active_image_observations[i] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 3:
                active_image_observations[i] = transform_rgb_bgr(img)

        # draw pointgoal
        goal_draw_surface = self.draw_info['pointgoal']        
        self.draw_pointgoal_polar(observations['pointgoal'], goal_draw_surface['image'], goal_draw_surface['region'], fov=90)
        if self.overlay_goal:    
            goal_region = goal_draw_surface['region']
            bottom = self.window_size[0]
            top = bottom - goal_region.height
            left = self.window_size[1]//2 - goal_region.width//2
            right = left + goal_region.width
            stacked = np.hstack(active_image_observations)
            overlay=cv2.addWeighted(stacked[top:bottom, left:right],0.5,goal_draw_surface['image'],0.5,0)
            stacked[top:bottom, left:right] = overlay
        else:
            stacked = np.hstack(active_image_observations + [self.side_img])
        return stacked

    def draw_pointgoal_polar(self, pointgoal, img, 
                    r: Rect, 
                    fov=0,
                    color=(0,0,255), wincolor=(0,0,0,0), 
                    overlaycolor=(128,128,128), bgcolor=(255,255,255)):
        angle = pointgoal[1]
        mag = pointgoal[0]
        nm = mag/(mag+1)
        xy = (-math.sin(angle), -math.cos(angle))
        size = int(round(0.45*min(r.width, r.height)))
        center = r.center
        target = (int(round(center[0]+xy[0]*size*nm)), int(round(center[1]+xy[1]*size*nm)))
        cv2.rectangle(img,(r.left,r.top), (r.right,r.bottom), wincolor, -1)    # Fill with window color 
        cv2.circle(img, center, size, bgcolor, -1)  # Circle with background color
        if fov > 0:
            masked = 360-fov
            cv2.ellipse(img, center, (size,size), 90, -masked/2, masked/2, overlaycolor, -1) 
        #print(center)
        #print(target)
        cv2.line(img, center, target, color, 1)
        cv2.circle(img, target, 4, color, -1)

def example(config, action_keys, args):
    env = habitat.Env(
        config=config
    )

    print("Environment creation successful")
    observations = env.reset()
    viewer = ObsViewer(observations, overlay_goal=args.overlay)
    viewer.show(observations)

    action_keys_map = {k.key : k for k in action_keys}
    print("Agent stepping around inside environment.")
    count_steps = 0
    while not env.episode_over:
        # observations = env.step(env.action_space.sample())
        keystroke = cv2.waitKey(0)

        action = action_keys_map.get(keystroke)
        if action is not None:
            print(action.name)
            if action.is_quit:
                break
        else:
            print("INVALID KEY")
            continue

        observations = env.step(action.action_id)
        count_steps += 1

        viewer.show(observations)

    print(env.get_metrics())
    print("Episode finished after {} steps.".format(count_steps))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Habitat API Example")
    parser.add_argument("--task-config",
                        default="configs/tasks/pointnav.yaml",
                        help='Task configuration file for initializing a Habitat environment')
    parser.add_argument("--overlay",
                        default=False,
                        action="store_true",
                        help='Overlay pointgoal')
    args = parser.parse_args()
    opts = []
    config = habitat.get_config(args.task_config.split(","), opts)
    #print(config)
    example(config, AGENT_ACTION_KEYS, args)
