#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import cv2
import habitat
import math
import numpy as np

from habitat.tasks.nav.nav_task import NavigationEpisode, NavigationGoal
from habitat.utils.visualizations import maps

from time import sleep
from typing import List, NamedTuple, Tuple

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

INSTRUCTIONS = [
    "Use W/A/D to move Forward/Left/Right",
    "Press <Space> when you reach the goal",
    "Q - Quit"
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


def display_instructions(image, instructions, size=1, offset=(0,0), fontcolor=(255,255,255)):
    for i,instruction in enumerate(instructions):
        x = offset[1]
        y = offset[0] + int((i+1)*size*50) - 15
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, instruction, (x,y), font, size, fontcolor, 2, cv2.LINE_AA)


def draw_goal_radar(pointgoal, img, 
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


class Viewer:
    def __init__(self, initial_observations, 
            overlay_goal_radar=None,            
            goal_display_size=128,
            show_map=False):
        self.overlay_goal_radar = overlay_goal_radar
        self.show_map = show_map

        # What image sensors are active
        all_image_sensors = ['rgb', 'depth'] 
        self.active_image_sensors = [s for s in all_image_sensors if s in initial_observations]
        total_width = 0
        total_height = 0
        for s in self.active_image_sensors:
            total_width += initial_observations[s].shape[1]
            total_height = max(initial_observations[s].shape[0], total_height)

        self.draw_info = {}
        if self.overlay_goal_radar:
            img = np.zeros((goal_display_size, goal_display_size, 3), np.uint8)
            self.draw_info["pointgoal"] = { "image": img, "region": Rect(0,0,goal_display_size,goal_display_size) }
        else:
            side_img_height = max(total_height, goal_display_size)
            self.side_img = np.zeros((side_img_height, goal_display_size, 3), np.uint8)
            self.draw_info["pointgoal"] = { "image": self.side_img, "region": Rect(0,0,goal_display_size,goal_display_size) }
            total_width += goal_display_size
        self.window_size = (total_height, total_width)  


    def draw_observations(self, observations, info=None):    
        active_image_observations = [observations[s] for s in self.active_image_sensors]
        for i,img in enumerate(active_image_observations):
            if img.shape[2] == 1:
                img *= (255.0 / img.max())  # naive rescaling for visualization
                active_image_observations[i] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).astype(np.uint8)
            elif img.shape[2] == 3:
                active_image_observations[i] = transform_rgb_bgr(img)

        # draw pointgoal
        goal_draw_surface = self.draw_info['pointgoal']        
        draw_goal_radar(observations['pointgoal'], goal_draw_surface['image'], goal_draw_surface['region'], fov=90)
        if self.overlay_goal_radar:    
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
        if info is not None:
            if self.show_map and info.get('top_down_map') is not None and 'heading' in observations: 
                top_down_map = draw_top_down_map(info, observations['heading'], stacked.shape[0])
                stacked = np.hstack((top_down_map, stacked))
        return stacked


def get_video_writer(filename='output.avi', fps=20.0, resolution=(640,480)):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(filename, fourcc, fps, resolution)
    return writer


class VideoWriter:
    # wrapper around video writer that will create video writer and
    # initialize the resolution the first time write happens
    def __init__(self, filename='output.avi', fps=20.0):
        self.filename = filename
        self.fps = fps
        self.writer = None

    def write(self, frame):
        if self.writer is None:
            self.resolution = (frame.shape[1], frame.shape[0])
            self.writer = get_video_writer(self.filename, self.fps, self.resolution)
        else:
            res = (frame.shape[1], frame.shape[0])
            if res != self.resolution:
                print(f"Warning: video resolution mismatch expected={self.resolution}, frame={res}")
        self.writer.write(frame)

    def release(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None


class Demo:
    def __init__(self, config, action_keys: List[ActionKeyMapping], instructions: List[str]):
        self.window_name = "Habitat"
        self.config = config
        self.instructions = instructions
        self.action_keys = action_keys
        self.action_keys_map = {k.key : k for k in self.action_keys}
        self.is_quit = False

        self.env = habitat.Env(
            config=self.config
        )
        print("Environment creation successful")


    def update(self, img, video_writer=None):
        fontsize = 0.8
        instructions_height = int(fontsize*50*len(self.instructions))
        instructions_img = np.zeros((instructions_height, img.shape[1], 3), np.uint8)
        display_instructions(instructions_img, self.instructions, size=fontsize)
        combined = np.vstack((instructions_img, img))
        self.window_shape = combined.shape
        if video_writer is not None:
            video_writer.write(combined)
        cv2.imshow(self.window_name, combined)


    def run(self, overlay_goal_radar=False, show_map=False, video_writer=None):
        env = self.env
        action_keys_map = self.action_keys_map

        observations = env.reset()
        info = env.get_metrics()
        viewer = Viewer(observations, overlay_goal_radar=overlay_goal_radar, show_map=show_map)
        img = viewer.draw_observations(observations, info)
        self.update(img)
        print(env.current_episode)

        print("Agent stepping around inside environment.")
        count_steps = 0
        actions = []
        while not env.episode_over:
            # observations = env.step(env.action_space.sample())
            keystroke = cv2.waitKey(0)

            action = action_keys_map.get(keystroke)
            if action is not None:
                print(action.name)
                if action.is_quit:
                    self.is_quit = True
                    break
            else:
                print("INVALID KEY")
                continue

            actions.append(action.action_id)
            observations = env.step(action.action_id)
            info = env.get_metrics()
            count_steps += 1

            img = viewer.draw_observations(observations, info)
            self.update(img, video_writer)

        print("Episode finished after {} steps.".format(count_steps))
        return actions, info


    def replay(self, actions, overlay_goal_radar=False, delay=1, video_writer=None):
        # Set delay to 0 to wait for key presses before advancing
        env = self.env
        action_keys_map = self.action_keys_map

        observations = env.reset(keep_current_episode=True)
        info = env.get_metrics()
        viewer = Viewer(observations, overlay_goal_radar=overlay_goal_radar, show_map=True)
        img = viewer.draw_observations(observations, info)
        self.update(img)

        count_steps = 0
        for action_id in actions:
            # wait for key before advancing
            keystroke = cv2.waitKey(delay)
            action = action_keys_map.get(keystroke)
            if action is not None:
                if action.is_quit:
                    self.is_quit = True
                    break

            observations = env.step(action_id)
            info = env.get_metrics()
            count_steps += 1

            img = viewer.draw_observations(observations, info)
            self.update(img, video_writer)

        print("Episode finished after {} steps.".format(count_steps))


    def demo(self, args):
        video_writer = None
        #video_writer = VideoWriter('test1.avi') if args.save_video else None
        actions, info = self.run(overlay_goal_radar=args.overlay, show_map=args.show_map, video_writer=video_writer)
        #if video_writer is not None:
        #    video_writer.release()
        if not self.is_quit:
            # Display info about how well you did
            if info is not None:
                success = info['spl'] > 0
                if success:
                    print("You succeeded!")
                else:
                    print("You failed!")    
            # Hack to get size of video 
            video_writer = VideoWriter('output.avi') if args.save_video else None
            self.replay(actions, overlay_goal_radar=args.overlay, delay=1, video_writer=video_writer)
            if video_writer is not None:
                video_writer.release()
        if not self.is_quit:
            keystroke = cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Habitat API Example")
    parser.add_argument("--task-config",
                        default="configs/tasks/pointnav.yaml",
                        help='Task configuration file for initializing a Habitat environment')
    parser.add_argument("--overlay",
                        default=False,
                        action="store_true",
                        help='Overlay pointgoal')
    parser.add_argument("--scenes-dir",
                        help='Directory where scenes are found')
    parser.add_argument("--show-map",
                        default=False,
                        action="store_true",
                        help='Show top down map as agent is stepping')
    parser.add_argument("--save-video",
                        default=False,
                        action="store_true",
                        help='Save video as agent is stepping')
    args = parser.parse_args()
    opts = []
    config = habitat.get_config(args.task_config.split(","), opts)
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.TASK.SENSORS.append("HEADING_SENSOR")

    if args.scenes_dir is not None:
        config.defrost()
        config.DATASET.SCENES_DIR = args.scenes_dir
        config.freeze()

    print(config)
    demo = Demo(config, AGENT_ACTION_KEYS, INSTRUCTIONS)
    demo.demo(args)
    cv2.destroyAllWindows()
