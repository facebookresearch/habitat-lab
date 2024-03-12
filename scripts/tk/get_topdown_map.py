#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#This script geneartes a ros/rviz compatible map based on the specified Habitat scene's top-down map

import os

import imageio
import numpy as np
import argparse
import habitat
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.utils.visualizations import maps
from IPython import embed
# from habitat.utils.visualizations.maps import COORDINATE_MIN, COORDINATE_MAX


MAP_DIR = "/home/catkin_ws/src/habitat_ros_interface/maps/"
if not os.path.exists(MAP_DIR):
    print("Didi not find maps directory")
    os.makedirs(MAP_DIR)

def get_topdown_map(config_paths):

    config = habitat.get_config(config_paths)
    
    # dataset = habitat.make_dataset(
    #     id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
    # )
    env = habitat.Env(config=config)
    env.reset()
    scene = env.episodes[0].scene_id
    # scene.replace(".scene_instance.json", "")
    map_name = "sample_map"
    # print(map_name)
    # embed()
    meters_per_pixel =0.025
    hablab_topdown_map = maps.get_topdown_map(
            env._sim.pathfinder, 0.0, meters_per_pixel=meters_per_pixel
        )
    recolor_map = np.array(
        [ [128, 128, 128], [255, 255, 255], [0, 0, 0]], dtype=np.uint8
    )
    hablab_topdown_map = recolor_map[hablab_topdown_map]
    grid_dimensions = (hablab_topdown_map.shape[0]*meters_per_pixel, hablab_topdown_map.shape[1]*meters_per_pixel)
    imageio.imsave(os.path.join(MAP_DIR, map_name + ".pgm"), hablab_topdown_map)
    print("writing Yaml file! ", os.path.join(MAP_DIR, map_name + ".pgm"))
    complete_name = os.path.join(MAP_DIR, map_name + ".yaml")
    f = open(complete_name, "w+")

    f.write("image: " + map_name + ".pgm\n")
    f.write("resolution: " + str(meters_per_pixel) + "\n")
    f.write("origin: [" + str(-1) + "," + str(-grid_dimensions[0]+1) + ", 0.000000]\n")
    f.write("negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196")
    f.close()


def main():
    #first parameter is config path, second parameter is map name
    get_topdown_map("/habitat-lab/habitat-lab/habitat/config/benchmark/multi_agent/hssd_fetch_human_social_nav.yaml")


if __name__ == "__main__":
    main()