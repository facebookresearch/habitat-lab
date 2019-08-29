#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
To run the demo
1. Simple demo on test scenes
- Download test scenes (http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip)
  and unzip into ${HABITAT_API_REPO}/data
- Update `configs/tasks/pointnav.yaml` to have higher resolution if you want bigger pictures
- `python examples/pointgoal_demo.py --task-config configs/tasks/pointnav.yaml --overlay`
2. Simple demo on test scenes with depth
- `python examples/pointgoal_demo.py --task-config configs/tasks/pointnav_rgbd.yaml --overlay`
3. Demo on replica scene with blind agent, with saving actions and videos
- Download pretrained blind agent 
  (get blind_agent_state.pth from https://www.dropbox.com/s/e63uf6joerkf7pe/agent_demo.zip?dl=0 and put into examples/agent_demo)
- Download replica dataset (https://github.com/facebookresearch/Replica-Dataset)
  (put under data/replica)
- Generate episodes for a replica scene (this takes a while to run)
  `mkdir data/replica_demo/pointnav`
  `python examples/gen_episodes.py -o data/replica_demo/pointnav/test.json --scenes data/replica/apartment_0/habitat/mesh_semantic.ply`
  `gzip data/replica_demo/pointnav/test.json`
- Create yaml config file for replica and put in `data/replica_demo/replica_test.yaml`
  DATASET:
  TYPE: PointNav-v1
  SPLIT: test
  POINTNAVV1:
    DATA_PATH: data/replica_demo/pointnav/{split}.json.gz 
- Run demo 
  `python examples/pointgoal_demo.py --task-config configs/tasks/pointnav.yaml,data/replica_demo/replica_test.yaml --agent blind --overlay --scenes-dir . --save-video --save-actions test.json`
  NOTE: video is saved to xyz.avi if you select to replay actions (select 1/2/3 for the agent to replay)
  NOTE: actions are save to simple json file

Future improvements to demo:
1. Selection of episodes
2. Precompute episodes for replica dataset and save action trace for shortest path follower / blind agents
3. Support new episodes (random/user specified)
"""

import argparse
import cv2
import habitat
import json
import math
import numpy as np

from habitat.tasks.nav.nav_task import NavigationEpisode, NavigationGoal
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from examples.agent_demo.demo_blind_agent import DemoBlindAgent

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

LINE_SPACING = 50

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


def write_textlines(output, textlines, size=1, offset=(0,0), fontcolor=(255,255,255)):
    for i,text in enumerate(textlines):
        x = offset[1]
        y = offset[0] + int((i+1)*size*LINE_SPACING) - 15
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(output, text, (x,y), font, size, fontcolor, 2, cv2.LINE_AA)


def draw_text(textlines=[], width=300, fontsize=0.8):
    text_height = int(fontsize*LINE_SPACING*len(textlines))
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
   #cv2.circle(img, center, size, color, 2)
   for i in range(1,size):
     a = 1-i/size
     c = np.add(np.multiply(color[0:3], a), np.multiply(bgcolor[0:3], 1-a))
     cv2.circle(img, center, i, c, 2);


def draw_gradient_wedge(img, center, size, color, bgcolor, start_angle, delta_angle):
   #cv2.circle(img, center, size, color, 2)
   for i in range(1,size):
     a = 1-i/size
     c = np.add(np.multiply(color, a), np.multiply(bgcolor, 1-a))
     #cv2.circle(img, center, i, c, 2);
     cv2.ellipse(img, center, (i,i), start_angle, -delta_angle/2, delta_angle/2, c, 2) 


def draw_goal_radar(pointgoal, img, 
                    r: Rect, 
                    start_angle=0, fov=0,
                    goalcolor=(50,0,184,255), 
                    wincolor=(0,0,0,0), 
                    #maskcolor=(128,128,128,255), 
                    maskcolor=(85,75,70,255),
                    bgcolor=(255,255,255,255),
                    gradientcolor=(174,112,80,255)):
    angle = pointgoal[1]
    mag = pointgoal[0]
    nm = mag/(mag+1)
    xy = (-math.sin(angle), -math.cos(angle))
    size = int(round(0.45*min(r.width, r.height)))
    center = r.center
    target = (int(round(center[0]+xy[0]*size*nm)), int(round(center[1]+xy[1]*size*nm)))
    if wincolor is not None:
        cv2.rectangle(img,(r.left,r.top), (r.right,r.bottom), wincolor, -1)    # Fill with window color 
    cv2.circle(img, center, size, bgcolor, -1)  # Circle with background color
    if fov > 0:
        masked = 360-fov
        cv2.ellipse(img, center, (size,size), start_angle+90, -masked/2, masked/2, maskcolor, -1) 
    if gradientcolor is not None:
        if fov > 0:
            draw_gradient_wedge(img, center, size, gradientcolor, bgcolor, start_angle-90, fov)
        else:
            draw_gradient_circle(img, center, size, gradientcolor, bgcolor)
    #print(center)
    #print(target)
    #cv2.line(img, center, target, goalcolor, 1)
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
    goal_radius = env.current_episode.goals[0].radius
    if goal_radius is None:
        goal_radius = config.SIMULATOR.FORWARD_STEP_SIZE
    return goal_radius


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
            img = np.zeros((goal_display_size, goal_display_size, 4), np.uint8)
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
        # TODO: get fov from agent
        draw_goal_radar(observations['pointgoal'], goal_draw_surface['image'], goal_draw_surface['region'],
                        start_angle=0, fov=90)
        if self.overlay_goal_radar:    
            goal_region = goal_draw_surface['region']
            bottom = self.window_size[0]
            top = bottom - goal_region.height
            left = self.window_size[1]//2 - goal_region.width//2
            right = left + goal_region.width
            stacked = np.hstack(active_image_observations)
            alpha = 0.5*(goal_draw_surface['image'][:,:,3]/255)
            rgb = goal_draw_surface['image'][:,:,0:3]
            overlay = np.add(
                np.multiply(stacked[top:bottom, left:right], np.expand_dims(1-alpha, axis=2)),
                np.multiply(rgb, np.expand_dims(alpha, axis=2))
            )
            #overlay=cv2.addWeighted(stacked[top:bottom, left:right],0.5,rgb,0.5,0)
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
        #self.env.reset()
        print("Environment creation successful")


    def update(self, img, video_writer=None):
        self.window_shape = img.shape
        if video_writer is not None:
            video_writer.write(img)
        cv2.imshow(self.window_name, img)


    def run(self, overlay_goal_radar=False, show_map=False, video_writer=None):
        env = self.env
        action_keys_map = self.action_keys_map

        observations = env.reset(keep_current_episode=False)
        info = env.get_metrics()
        viewer = Viewer(observations, overlay_goal_radar=overlay_goal_radar, show_map=show_map)
        img = viewer.draw_observations(observations, info)
        goal_radius = get_goal_radius(env)
        distance = observations['pointgoal'][0]
        self.update(add_text(img, [f'Distance {distance:.5}/{goal_radius:.5}'] + self.instructions))
        # print(env.current_episode)

        print("Agent stepping around inside environment.")
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

            img = viewer.draw_observations(observations, info)
            distance = observations['pointgoal'][0]
            self.update(add_text(img, [f'Distance {distance:.5}/{goal_radius:.5}'] + self.instructions), video_writer)

        print("Episode finished after {} steps.".format(len(actions)))
        return actions, info, observations


    def get_follower_actions(self, mode="geodesic_path"):
        env = self.env
        observations = env.reset(keep_current_episode=True)
        goal_radius = get_goal_radius(env)
        follower = ShortestPathFollower(env.sim, goal_radius, False)
        follower.mode = mode
        actions = []
        while not env.episode_over:
            best_action = follower.get_next_action(
                env.current_episode.goals[0].position
            )
            actions.append(best_action.value)
            observations = env.step(best_action.value)
            info = env.get_metrics()

        print("Episode finished after {} steps.".format(len(actions)))
        return actions, info, observations


    def get_agent_actions(self, agent):
        # NOTE: Action space for agent is hard coded (need to match our scenario)
        env = self.env
        observations = env.reset(keep_current_episode=True)
        agent.reset()
        actions = []
        while not env.episode_over:
            action = agent(observations)
            actions.append(action)
            observations = env.step(action)
            info = env.get_metrics()

        print("Episode finished after {} steps.".format(len(actions)))
        return actions, info, observations


    def replay(self, name, actions, overlay_goal_radar=False, delay=1, video_writer=None):
        # Set delay to 0 to wait for key presses before advancing
        env = self.env
        action_keys_map = self.action_keys_map

        observations = env.reset(keep_current_episode=True)
        info = env.get_metrics()
        viewer = Viewer(observations, overlay_goal_radar=overlay_goal_radar, show_map=True)
        img = viewer.draw_observations(observations, info)
        self.update(add_text(img, [name]))

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
            self.update(add_text(img, [name]), video_writer)

        print("Episode finished after {} steps.".format(count_steps))


    def get_comparisons(self, agents=None):
        comparisons = {}
        comparisons['Shortest'] = self.get_follower_actions(mode="geodesic_path")
        if agents is not None:
            for name, agent in agents.items():
                comparisons[name] = self.get_agent_actions(agent)
        # TODO: Get performance from other famous people
        return comparisons


    def show_comparisons(self, comparisons):
        imgs = []
        shortcut_keys = {}
        key = ord('1')
        for name, (actions, info, observations) in comparisons.items():
            top_down_map = draw_top_down_map(info, observations['heading'], 256)
            spl = info['spl']
            success = spl > 0
            distance = observations['pointgoal'][0]
            textlines = [
                f'({chr(key)}) {name}',
                'Success' if success else 'Failed',
                f'{len(actions)} steps, SPL={spl:.5}',
                f'dist={distance:.5}'
            ]
            imgs.append(add_text(top_down_map, textlines))
            shortcut_keys[key] = name
            key = key + 1
        stacked = np.hstack(imgs)
        cv2.imshow(self.window_name, stacked)
        return shortcut_keys

    def save_comparisons(self, comparisons, filename, save_info={}):
        save_info['traces'] = []
        for name, (actions, info, observations) in comparisons.items():
            m =  {k: info[k] for k in ['spl'] if k in info}
            m['success'] = 1 if info['spl'] > 0 else 0 
            m['pointgoal'] = observations['pointgoal'].tolist()
            save_info['traces'].append({
                'name': name,
                'summary': m,
                'actions': actions
            })
        with open(filename, 'w') as file:
            json.dump(save_info, file)


    def demo(self, args):
        video_writer = None
        #video_writer = VideoWriter('test1.avi') if args.save_video else None
        actions, info, observations = self.run(overlay_goal_radar=args.overlay, show_map=args.show_map, video_writer=video_writer)
        if not self.is_quit:
            agents = {}
            if ("blind" in args.agent):
                agents["Blind"] = DemoBlindAgent()
            comparisons = self.get_comparisons(agents)
            comparisons['Yours'] = actions, info, observations
        else:
            comparisons = {
                'Yours': (actions, info, observations)
            }

        # Save the actions
        if args.save_actions is not None:
            save_info = {
                'config': args.task_config,
            }
            self.save_comparisons(comparisons, args.save_actions, save_info)

        #if video_writer is not None:
        #    video_writer.release()
        while not self.is_quit:
            # Display info about how well you did
            viewer = Viewer(observations, overlay_goal_radar=args.overlay, show_map=True)

            # Show other people's route
            shortcut_keys = self.show_comparisons(comparisons)

            # Hack to get size of video 
            keystroke = cv2.waitKey(0)
            selected_name = shortcut_keys.get(keystroke)
            if selected_name is not None:
                (actions, info, observations) = comparisons[selected_name]
                print(f'Selected {selected_name}')
                video_writer = VideoWriter(f'{selected_name}.avi') if args.save_video else None
                self.replay(selected_name, actions, overlay_goal_radar=args.overlay, delay=1, video_writer=video_writer)
                if video_writer is not None:
                    video_writer.release()
            else:
                action = self.action_keys_map.get(keystroke)
                if action is not None and action.is_quit:
                   self.is_quit = True
                   break



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Habitat API Demo")
    parser.add_argument("--task-config",
                        default="configs/tasks/pointnav.yaml",
                        help='Task configuration file for initializing a Habitat environment')
    parser.add_argument("--overlay",
                        default=False,
                        action="store_true",
                        help='Overlay pointgoal')
    parser.add_argument("-a", "--agent",
                        default=[],
                        choices=['blind'],
                        action="append",
                        help='What agent to include')
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
    parser.add_argument("--save-actions",
                        default=None,
                        help='File to save actions to')
    args = parser.parse_args()
    opts = []
    config = habitat.get_config(args.task_config.split(","), opts)
    config.defrost()
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.TASK.TOP_DOWN_MAP.MAP_RESOLUTION = 6000
    config.TASK.SENSORS.append("HEADING_SENSOR")
    config.freeze()

    if args.scenes_dir is not None:
        config.defrost()
        config.DATASET.SCENES_DIR = args.scenes_dir
        config.freeze()

    # print(config)
    demo = Demo(config, AGENT_ACTION_KEYS, INSTRUCTIONS)
    demo.demo(args)
    cv2.destroyAllWindows()
