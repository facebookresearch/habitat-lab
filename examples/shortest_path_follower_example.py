#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil

import numpy as np

import habitat
from habitat.core.utils import try_cv2_import
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video

import json
import time
import tensorflow as tf
import tqdm
from tfrecordfeatures import tf_bytelist_feature, tf_bytes_feature, tf_int64_feature

cv2 = try_cv2_import()

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

#
# def get_floor_heights(sim):
#     floor_heights = []
#     heights = np.array([sim.sample_navigable_point()[1] for _ in range(5000)])
#
#     while len(heights) > 0:
#         mean_height = heights[0]
#         near_heights = heights[np.abs(heights - mean_height) < 0.3]
#         mean_height = np.mean(near_heights)
#         near_heights = heights[np.abs(heights - mean_height) < 0.5]
#         mean_height = np.mean(near_heights)
#         floor_heights.append(mean_height)
#         heights = heights[np.abs(heights - mean_height) >= 0.5]
#
#     print ("%d floors"%len(floor_heights))
#     return np.sort(floor_heights)


def get_all_floors_from_file(scene_id):
    floor_height_path = '../GibsonSim2RealChallenge/gibson-challenge-data/dataset/%s/floors.txt'%scene_id
    with open(floor_height_path, 'r') as f:
        floors = sorted(list(map(float, f.readlines())))
    print(scene_id + ' floors', floors)
    return floors


def get_floor(height, floor_heights):
    floor = np.argmin(np.abs(np.array(floor_heights) - height))
    return floor


def get_floor_from_file(sceen_id, height):
    floor_heights = get_all_floors_from_file(sceen_id)
    return get_floor(height, floor_heights)


def encode_image_to_string(image):
    if image.dtype != np.uint8:
        assert np.all(image <= 1.)
        image = (image * 255).astype(np.uint8)
    return cv2.imencode(".png", image)[1].tostring()


def generate_maps():
    # config = habitat.get_config(config_paths="configs/tasks/pointnav.yaml")
    config = habitat.get_config(
        "./configs/challenge_modified.yaml",
    )

    floors_for_scene = {}
    saved_maps = []

    # config = habitat.get_config(config_paths="./configs/ddppo_pointnav.yaml")
    config.defrost()
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    grid_cell_size = 0.05  # 5cm
    map_size = (maps.COORDINATE_MAX - maps.COORDINATE_MIN) / grid_cell_size
    config.TASK.TOP_DOWN_MAP.MAP_RESOLUTION = int(map_size)
    config.TASK.SENSORS.append("HEADING_SENSOR")
    config.freeze()
    with SimpleRLEnv(config=config) as env:
        # goal_radius = env.episodes[0].goals[0].radius
        # if goal_radius is None:
        #     goal_radius = config.SIMULATOR.FORWARD_STEP_SIZE
        # follower = ShortestPathFollower(
        #     env.habitat_env.sim, goal_radius, False
        # )

        episodes = []
        episode_iterator = env.habitat_env.episode_iterator

        print("Collecting episodes..")
        for episode_i in range(min(5000, len(env.habitat_env.episodes))):
            ep = next(episode_iterator)
            height = ep.start_position[1]

            sceen_id = ep.scene_id
            sceen_id = sceen_id.split('/')[-1]
            sceen_id = sceen_id.split('.')[0]
            if sceen_id not in floors_for_scene.keys():
                floor_heights = get_all_floors_from_file(sceen_id)
                floors_for_scene[sceen_id] = floor_heights
                floor_filename = './maps/%s_floors.json'%sceen_id
                with open(floor_filename, 'w') as file:
                    json.dump({'floor_heights': floor_heights}, file, indent=4)
            floor = get_floor(height, floors_for_scene[sceen_id])

            map_filename = './maps/' + sceen_id + '_%d_map.png'%floor
            if map_filename not in saved_maps:
                saved_maps.append(map_filename)
                episodes.append(ep)

        print("Collected %d episodes"%len(episodes))

        env.habitat_env.episodes = episodes
        iterator = iter(episodes)
        env.habitat_env.episode_iterator = iterator
        for episode in episodes:
            env.reset()
            assert env.current_episode == episode

            observations, reward, done, info = env.step(2)  # turn right
            height = env.habitat_env.sim.get_agent_state().position[1]

            sceen_id = env.current_episode.scene_id
            sceen_id = sceen_id.split('/')[-1]
            sceen_id = sceen_id.split('.')[0]
            floor = get_floor(height,floors_for_scene[sceen_id])
            map_filename = './maps/' + sceen_id + '_%d_map.png'%floor

            # Save map
            global_map = info['top_down_map']['map']
            global_map = (global_map > 0).astype(np.uint8) * 255
            cv2.imwrite(map_filename, global_map)
            print ('Saved %s'%map_filename)

        # print("Environment creation successful")
        # for episode in range(10000):
        #     env.reset()
        #     # dirname = os.path.join(
        #     #     IMAGE_DIR, "shortest_path_example", "%02d" % episode
        #     # )
        #     # if os.path.exists(dirname):
        #     #     shutil.rmtree(dirname)
        #     # os.makedirs(dirname)
        #     # print("Agent stepping around inside environment.")
        #     observations, reward, done, info = env.step(2)  # turn right
        #
        #     height = env.habitat_env.sim.get_agent_state().position[1]
        #
        #     sceen_id = env.current_episode.scene_id
        #     sceen_id = sceen_id.split('/')[-1]
        #     if sceen_id not in floors_for_scene.keys():
        #         floor_heights = get_floor_heights(env.habitat_env.sim)
        #         floors_for_scene[sceen_id] = floor_heights
        #         floor_filename = './maps/%s_floors.json'%sceen_id
        #         with open(floor_filename, 'w') as file:
        #             json.dump({'floor_heights': floor_heights}, file, indent=4)
        #     floor = get_floor(height, floors_for_scene[sceen_id])
        #
        #     map_filename = './maps/' + sceen_id + '_%d_map.png'%floor
        #     if map_filename not in saved_maps:
        #         # Save map
        #         global_map = info['top_down_map']['map']
        #         global_map = (global_map == 0).astype(np.uint8) * 255
        #         cv2.imwrite(map_filename, global_map)
        #         print ('Saved %s'%map_filename)

            # # global_map = cv2.imread(map_filename)
            # images = []
            # best_action = 1
            # while not env.habitat_env.episode_over:
            #     # best_action = follower.get_next_action(
            #     #     env.habitat_env.current_episode.goals[0].position
            #     # )
            #     if best_action is None:
            #         break
            #
            #     observations, reward, done, info = env.step(best_action)
            #     im = observations["rgb"]
            #     top_down_map = draw_top_down_map(
            #         info, observations["heading"][0], im.shape[0]
            #     )
            #     print (env.habitat_env.current_episode.goals[0].position)
            #     agent_state = env.habitat_env.sim.get_agent_state()
            #     agent_pos = agent_state.position
            #     xy = np.array((agent_pos[0], agent_pos[2]))
            #     xy_map = maps.to_grid(xy[0], xy[1], maps.COORDINATE_MIN, maps.COORDINATE_MAX, (5000, 5000), keep_float=True)
            #
            #     print (observations['gps'])
            #     best_action = follower.get_next_action(
            #         env.habitat_env.current_episode.goals[0].position
            #     )
            #     output_im = np.concatenate((im, top_down_map), axis=1)
            #     images.append(output_im)
            #
            #     tdmap = info['top_down_map']
            #     global_map = tdmap['map']
            #     xy = tdmap['agent_map_coord']
            #     print(xy, np.rad2deg(tdmap['agent_angle']))
            #     global_map[int(xy[0]), int(xy[1])] = 10

                # top_down_map = maps.get_topdown_map(self.env.sim, map_resolution=(5000, 5000))
                #
                # import matplotlib.pyplot as plt
                # plt.figure(1)
                # plt.imshow(output_im)
                # plt.figure(2)
                # plt.imshow(global_map)
                # plt.show()
                # import ipdb; ipdb.set_trace()



            # cv2.imwrite(dirname + '/map.png', tdmap['map'])


            # images_to_video(images, dirname, "trajectory")
            print("Episode finished")


def generate_scenarios(output_filename=None, num_episodes=50000, skip_first_n=0,
                       spin_every_forward_n=1000, spin_before_done=False, write_lock=None):
    # config = habitat.get_config(config_paths="configs/tasks/pointnav.yaml")
    config = habitat.get_config("./configs/challenge_modified.yaml",)

    floors_for_scene = {}

    config.defrost()
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    grid_cell_size = 0.05  # 5cm
    map_size = (maps.COORDINATE_MAX - maps.COORDINATE_MIN) / grid_cell_size
    config.TASK.TOP_DOWN_MAP.MAP_RESOLUTION = int(map_size)
    config.TASK.SENSORS.append("HEADING_SENSOR")
    config.freeze()

    with SimpleRLEnv(config=config) as env:
        # goal_radius = env.episodes[0].goals[0].radius
        # if goal_radius is None:
        #     goal_radius = config.SIMULATOR.FORWARD_STEP_SIZE
        goal_radius = 0.1
        follower = ShortestPathFollower(
            env.habitat_env.sim, goal_radius, False
        )

        if output_filename is None:
            output_filename = './data/habscenarios_{}.tfrecords'.format(time.strftime('%m-%d-%H-%M-%S', time.localtime()))
        tfwriter = tf.python_io.TFRecordWriter(output_filename)

        episodes = []
        episode_iterator = env.habitat_env.episode_iterator

        print ("Skipping %d episodes" % skip_first_n)
        for episode_i in tqdm.tqdm(range(skip_first_n)):
            next(episode_iterator)

        print("Generating %d episodes.."%num_episodes)
        for episode_i in tqdm.tqdm(range(num_episodes)):
            env.reset()
            ep = env.habitat_env.current_episode
            height = ep.start_position[1]
            sceen_id = ep.scene_id
            sceen_id = sceen_id.split('/')[-1]
            sceen_id = sceen_id.split('.')[0]
            if sceen_id not in floors_for_scene.keys():
                floors_for_scene[sceen_id] = get_all_floors_from_file(sceen_id)
            floor = get_floor(height, floors_for_scene[sceen_id])
            map_filename = './maps/%s_%d_map.png'%(sceen_id, floor)

            actions = []
            rgbs = []
            depths = []
            xys = []
            yaws = []

            num_forward_since_spun = 0
            spin_n_more = 0
            has_reached_goal = False
            spin_action = 2  # left

            other_dircetion = lambda a: (2 if a == 3 else 3)

            # Unroll episode
            while not env.habitat_env.episode_over:
                # Choose action
                if spin_n_more > 0:
                    action = spin_action
                    spin_n_more -= 1
                    num_forward_since_spun = 0
                elif num_forward_since_spun >= spin_every_forward_n:
                    spin_action = other_dircetion(spin_action)  # flip direction
                    action = spin_action
                    spin_n_more = 11
                else:
                    # shortest path
                    action = follower.get_next_action(
                        env.habitat_env.current_episode.goals[0].position
                    )
                # Overwrite at goal, choosing stop action for the first time
                if spin_before_done and action == 0 and not has_reached_goal:
                    has_reached_goal = True
                    spin_action = other_dircetion(spin_action)  # flip direction
                    action = spin_action
                    spin_n_more = 11  # trigger spin

                if action == 1:  # forward
                    num_forward_since_spun += 1

                observations, reward, done, info = env.step(action)
                xy = np.array(info['top_down_map']['agent_map_coord'])  # x: downwards; y: rightwars
                yaw = info['top_down_map']['agent_angle']  # 0 downwards, positive ccw. Forms stanard coord system with x and y.
                rgb = observations['rgb']
                depth = observations['depth']
                rgb = cv2.resize(rgb, (160, 90), )
                depth = cv2.resize(depth, (160, 90),) # interpolation=cv2.INTER_NEAREST)
                #TODO not sure what interpolation is best,
                # because depth has noise, nearest will be the worst;
                # but averaging (invalid) zero with sth else is not good either

                actions.append(action)
                rgbs.append(rgb)
                depths.append(depth)
                xys.append(xy)
                yaws.append(yaw)

            success = info['success']

            print ("ep %d/%d. %s %d. %d steps. Success: %d. Image shape: %s"%(
                episode_i, num_episodes, sceen_id, floor, len(rgbs), int(success),
                str(rgbs[-1].shape)))

            if not success:
                rgbs = []
                depths = []
                continue

            # store trajectory
            features = {
                'model_id': tf_bytes_feature(str(sceen_id).encode()),
                'floor': tf_int64_feature(floor),
                'episode_id': tf_int64_feature(int(ep.episode_id)),
                'xys': tf_bytes_feature(np.array(xys, np.float32).tostring()),
                'yaws': tf_bytes_feature(np.array(yaws, np.float32).tostring()),
                'actions': tf_bytes_feature(np.array(actions, np.float32).tostring()),  # TODO this should be int

                'depths': tf_bytelist_feature([encode_image_to_string(x) for x in depths]),
                'rgbs': tf_bytelist_feature([encode_image_to_string(x) for x in rgbs]),
            }

            example = tf.train.Example(features=tf.train.Features(feature=features))

            if write_lock is not None:
                write_lock.acquire()
            tfwriter.write(example.SerializeToString())
            tfwriter.flush()
            if write_lock is not None:
                write_lock.release()

            rgbs = []
            depths = []

        # print("Environment creation successful")
        # for episode in range(10000):
        #     env.reset()
        #     # dirname = os.path.join(
        #     #     IMAGE_DIR, "shortest_path_example", "%02d" % episode
        #     # )
        #     # if os.path.exists(dirname):
        #     #     shutil.rmtree(dirname)
        #     # os.makedirs(dirname)
        #     # print("Agent stepping around inside environment.")
        #     observations, reward, done, info = env.step(2)  # turn right
            # action = 1
            # while not env.habitat_env.episode_over:
            #     # action = follower.get_next_action(
            #     #     env.habitat_env.current_episode.goals[0].position
            #     # )
            #     if action is None:
            #         break
            #
            #     observations, reward, done, info = env.step(action)
            #     im = observations["rgb"]
            #     top_down_map = draw_top_down_map(
            #         info, observations["heading"][0], im.shape[0]
            #     )
            #     print (env.habitat_env.current_episode.goals[0].position)
            #     agent_state = env.habitat_env.sim.get_agent_state()
            #     agent_pos = agent_state.position
            #     xy = np.array((agent_pos[0], agent_pos[2]))
            #     xy_map = maps.to_grid(xy[0], xy[1], maps.COORDINATE_MIN, maps.COORDINATE_MAX, (5000, 5000), keep_float=True)
            #
            #     print (observations['gps'])
            #     action = follower.get_next_action(
            #         env.habitat_env.current_episode.goals[0].position
            #     )
            #     output_im = np.concatenate((im, top_down_map), axis=1)
            #     images.append(output_im)
            #
            #     tdmap = info['top_down_map']
            #     global_map = tdmap['map']
            #     xy = tdmap['agent_map_coord']
            #     print(xy, np.rad2deg(tdmap['agent_angle']))
            #     global_map[int(xy[0]), int(xy[1])] = 10

                # top_down_map = maps.get_topdown_map(self.env.sim, map_resolution=(5000, 5000))
                #
                # import matplotlib.pyplot as plt
                # plt.figure(1)
                # plt.imshow(output_im)
                # plt.figure(2)
                # plt.imshow(global_map)
                # plt.show()
                # import ipdb; ipdb.set_trace()



            # cv2.imwrite(dirname + '/map.png', tdmap['map'])


            # images_to_video(images, dirname, "trajectory")
            print("Episode finished")


def main(*args):
    # generate_maps()

    # Figured from maps folder. Training set: 72 scenes. Test set: 14 scenes.

    # train
    # num_episodes = 72 * 100 * 2  # 72 training scenes. 200 per scene. Actually there are total 88, training less. Ended with Brevort.
    #skip_first_n = 14400

    # val
    num_episodes = 15 * 100 * 2  # ? test scenes. 100 per scene. Actually there are total 88, training less. Ended with Brevort.
    skip_first_n = 0
    spin_every_forward_n = 12  # 3 meters
    spin_before_done = True

    if len(args) >= 2 and args[0] == '--num_episodes':
        num_episodes = args[1]

    generate_scenarios(num_episodes=num_episodes, skip_first_n=skip_first_n,
                       spin_every_forward_n=spin_every_forward_n,
                       spin_before_done=spin_before_done)


if __name__ == "__main__":
    main()
