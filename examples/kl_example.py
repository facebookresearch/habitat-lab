#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import argparse
import os
import os.path as osp
import time
from collections import defaultdict
from typing import Any, Dict, List

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
sys.path.append("/usr/lib/python2.7/dist-packages")
sys.path.append("/opt/ros/kinetic/lib/python2.7/dist-packages/")
import rospy
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from nav_msgs.msg import Path
from visualization_msgs.msg import MarkerArray, Marker
import threading

from IPython import embed
# Please reach out to the paper authors to obtain this file
DEFAULT_POSE_PATH = "data/humanoids/humanoid_data/walking_motion_processed.pkl"
DEFAULT_CFG = "benchmark/rearrange/play/play.yaml"
DEFAULT_RENDER_STEPS_LIMIT = 60
SAVE_VIDEO_DIR = "./data/vids"
SAVE_ACTIONS_DIR = "./data/interactive_play_replays"


lock = threading.Lock()
rospy.init_node("robot_1", anonymous=False)

class sim_env(threading.Thread):
    _x_axis = 0
    _y_axis = 1
    _z_axis = 2
    _dt = 0.00478
    _sensor_rate = 50  # hz
    _r = rospy.Rate(_sensor_rate)
    _current_episode = 0
    _total_number_of_episodes = 0
    control_frequency = 20
    time_step = 1.0 / (control_frequency)
    _r_control = rospy.Rate(control_frequency)
    def __init__(self, config):
        threading.Thread.__init__(self)
        self.env = habitat.Env(config = config)
        self.observations = self.env.reset()
        self._pub_rgb = rospy.Publisher("~rgb", numpy_msg(Floats), queue_size=1)
        self._pub_rgb_2 = rospy.Publisher("~rgb2", numpy_msg(Floats), queue_size=1)
        self._pub_depth = rospy.Publisher("~depth", numpy_msg(Floats), queue_size=1)
        self._pub_pose = rospy.Publisher("~pose", PoseStamped, queue_size=1)
        self.linear_velocity = [0,0,0]
        self.angular_velocity = [0,0,0]

    def run(self):
        """Publish sensor readings through ROS on a different thread.
            This method defines what the thread does when the start() method
            of the threading class is called
        """
        while not rospy.is_shutdown():
            lock.acquire()
            rgb_with_res = np.concatenate(
                (
                    np.float32(self.observations["agent_0_third_rgb"].ravel()),
                    np.array(
                        [512,512]
                    ),
                )
            )
            rgb2_with_res = np.concatenate(
                (
                    np.float32(self.observations["agent_1_head_rgb"].ravel()),
                    np.array(
                        [224,224]
                    ),
                )
            )
            # multiply by 10 to get distance in meters
            depth_with_res = np.concatenate(
                (
                    np.float32(self.observations["agent_0_head_depth"].ravel() * 10),
                    np.array(
                        [
                            128,128
                        ]
                    ),
                )
            )       
            
            self._pub_rgb.publish(np.float32(rgb_with_res))
            self._pub_rgb_2.publish(np.float32(rgb2_with_res))
            self._pub_depth.publish(np.float32(depth_with_res))
            lock.release()
            self._r.sleep()

    def update_agent_pos_vel(self):
        lin_vel = self.linear_velocity[2]
        ang_vel = self.angular_velocity[1]
        base_vel = [lin_vel, ang_vel]
        self.env._episode_over = False
        k = 'agent_1_oracle_nav_randcoord_action'
        my_env.env.task.actions[k].coord_nav = self.observations['agent_0_localization_sensor'][:3]
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
    
    images = []

    my_env = sim_env(config)

    rospy.Subscriber("/cmd_vel", Twist, callback, (my_env), queue_size=1)
    for i in range (1000):
   
        my_env.update_agent_pos_vel()
        im_0 = my_env.observations["agent_1_head_rgb"]

        images.append(im_0)
    images_to_video(images, "test", "trajectory")
    print("Episode finished")
        # rospy.spin()
        
