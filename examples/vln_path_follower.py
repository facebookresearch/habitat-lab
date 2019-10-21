#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import json
import cv2
import numpy as np
import argparse
import habitat
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video
from habitat.core.simulator import SimulatorActions
from habitat.core.registry import registry

IMAGE_DIR = os.path.join("examples", "images")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)


class SimpleRLEnv(habitat.RLEnv):
    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


def draw_top_down_map(info, heading, output_size):
    top_down_map = maps.colorize_topdown_map(
        info["top_down_map"]["map"], info["top_down_map"]["fog_of_war_mask"]
    )
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

def save_map(observations, info, images, instructions):
    # print(observations, reward, done, info)
    im = observations["rgb"]
    top_down_map = draw_top_down_map(
        info, observations["heading"], im.shape[0]
    )
    output_im = np.concatenate((im, top_down_map), axis=1)
    # cv2.imwrite("examples/images/i.jpg", output_im)
    shape = output_im.shape
    color = (255, 0, 0) 
    org = (5, shape[0] - 10) 

    fontScale = 0.5
    thickness = 1

    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    y0, dy = shape[0] - 80, 20
    for i, line in enumerate(instructions[0].split('.')):
        y = y0 + i*dy
        cv2.putText(output_im, line, (5, y), font, fontScale, color, thickness, cv2.LINE_AA)
    # cv2.putText(output_im, instructions[0], org, font, fontScale, color, thickness, cv2.LINE_AA)
    images.append(output_im)



def shortest_path_example(mode):
    config = habitat.get_config(config_paths="configs/tasks/vln_r2r_mp3d.yaml")
    config.defrost()
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.TASK.SENSORS.append("HEADING_SENSOR")
    
    # config.TASK.INSTRUCTION_SENSOR = habitat.Config()

    # config.TASK.INSTRUCTION_SENSOR.TYPE = "instruction_sensor"
    # # Add the sensor to the list of sensors in use
    # config.TASK.SENSORS.append("INSTRUCTION_SENSOR")

    config.freeze()
    env = SimpleRLEnv(config=config)
    # goal_radius = env.episodes[0].goals[0].radius
    goal_radius = 0.25
    # goal_radius = None
    if goal_radius is None: 
        goal_radius = config.SIMULATOR.FORWARD_STEP_SIZE
    
    print(goal_radius)

    follower = ShortestPathFollower(env.habitat_env.sim, goal_radius, False)
    follower.mode = mode
    error_list = []
    print("Environment creation successful")
    for episode in range(len(env.episodes)):
        env.reset()
        dirname = os.path.join(
            IMAGE_DIR, "shortest_path_example", mode, str(env.habitat_env.current_episode.episode_id),  "%02d" % episode
        )
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        os.makedirs(dirname)
        print("Agent stepping around inside environment.")
        images = []
        print("Episode Id : " + str(env.habitat_env.current_episode.episode_id))
        error = False
        current_episode_id  = env.habitat_env.current_episode.episode_id

        x = 0
        for path in env.habitat_env.current_episode.path:
            # done = False
            print("Path : " + str(x))
            print("Reachable? :" + str(env.habitat_env.sim.is_navigable(path)))
            # print(path) 
            steps = 0
            while True:
                best_action = follower.get_next_action(
                    path
                )
                # print(best_action)
                if best_action == SimulatorActions.STOP or steps > 200:
                    print("Reached The end")
                    # print(env.habitat_env.sim.get_agent_state().position)
                    break
                try :
                    observations, reward, done, info = env.step(best_action)
                    print(observations.keys())
                except:
                    error = True
                    break
                
                # print(observations, reward, done, info)
                save_map(observations, info, images, env.habitat_env.current_episode.goals[0].instruction)
                steps += 1
            print("Steps : " + str(steps))
            images_to_video(images, dirname, "path" + str(x))
            x += 1
            if error:
                error = True
                # break
        # if error:
        #     print("Error : " + str(current_episode_id))
        #     error_list.append(current_episode_id)
        #     print(str(current_episode_id), file=open("log.txt", "a"))
        #     continue


        done = False
        print("Goal!")
        # print(env.habitat_env.sim.is_navigable(env.habitat_env.current_episode.goals[0].position))
        steps = 0


        while not done:
            best_action = follower.get_next_action(
                    env.habitat_env.current_episode.goals[0].position
                )
            try :
                observations, reward, done, info = env.step(best_action)
                print(observations.keys())

            except:
                error = True
                break            
            
            save_map(observations, info, images, env.habitat_env.current_episode.goals[0].instruction)
            steps += 1

        if steps > 200 or error:
            print("Error : " + str(current_episode_id))
            error_list.append(current_episode_id)
            print(str(current_episode_id), file=open("log.txt", "a"))

        print("Iteration : " + str(steps))

        images_to_video(images, dirname, "final")
    print(error_list)


def main():
    # shortest_path_example("geodesic_path")
    shortest_path_example("greedy")

def load_r2r_from_folder(dataset_dir):
    input_file = open (dataset_dir)
    json_array = json.load(input_file)
    return json_array

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
        '''
        Run a particular episode from the R2R Dataset
        Example invocation:
          python r2r_runner.py -e episode_id -m mode
        The -e argument is required and specifies the path_id.
        The --m argument is required to point to the r2r dataset train, test, val_seen, val_unseen

        ''',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-e', '--episode_id', required=False, help='episode_id to load', default="all")
    parser.add_argument('-m', '--mode', required=True, help='Data mode')
    args = parser.parse_args()
    episode_id = args.episode_id
    mode = args.mode

        
    path = "data/datasets/R2R/hb_R2R_"
    single = "_single"
    json_name = ".json"
    json_array = load_r2r_from_folder(path + mode + json_name)

    if episode_id == "all":
        print("Episode Id : All")
        with open(path + "train" + single + json_name, 'w') as outfile:
            json.dump(json_array, outfile, indent=4)
    else:

        jsonObject = {}
        print("Episode Id : " + str(episode_id))
        jarray = []
        for jsonData in json_array["episodes"]:
            if int(jsonData["episode_id"]) == int(episode_id):
                jarray.append(jsonData)
                jsonObject["episodes"] = jarray
                break
        with open(path + "train" + single + json_name, 'w') as outfile:
            json.dump(jsonObject, outfile, indent=4)
    main()
