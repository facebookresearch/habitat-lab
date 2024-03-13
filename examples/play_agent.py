#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
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
from habitat.utils.visualizations import maps

from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
)
from habitat_sim.utils import viz_utils as vut
from habitat.utils.visualizations.utils import images_to_video
# sys.path.append("/root/miniconda3/envs/robostackenv/lib/python3.9/site-packages")
sys.path.append("/opt/conda/envs/robostackenv/lib/python3.9/site-packages")
sys.path.append("/usr/lib/python2.7/dist-packages")
sys.path.append("/opt/ros/kinetic/lib/python2.7/dist-packages/")
import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist, TransformStamped
from geometry_msgs.msg import PointStamped, PoseStamped, PoseWithCovarianceStamped, PoseArray, Pose
from visualization_msgs.msg import Marker, MarkerArray
import geometry_msgs
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import tf
import tf2_ros
import threading

from IPython import embed
# Please reach out to the paper authors to obtain this file
DEFAULT_POSE_PATH = "data/humanoids/humanoid_data/walking_motion_processed.pkl"
DEFAULT_CFG = "benchmark/rearrange/play/play.yaml"
DEFAULT_RENDER_STEPS_LIMIT = 60
SAVE_VIDEO_DIR = "./data/vids"
SAVE_ACTIONS_DIR = "./data/interactive_play_replays"


lock = threading.Lock()
rospy.init_node("sim", anonymous=False)

def to_grid(pathfinder, points, grid_dimensions):
    map_points = maps.to_grid(
                        points[2],
                        points[0],
                        grid_dimensions,
                        pathfinder=pathfinder,
                    )
    return ([map_points[1]*0.025, map_points[0]*0.025])

def from_grid(pathfinder, points, grid_dimensions):
    floor_y = 0.0
    map_points = maps.from_grid(
                        points[1],
                        points[0],
                        grid_dimensions,
                        pathfinder=pathfinder,
                    )
    map_points_3d = np.array([map_points[1], floor_y, map_points[0]])
    # # agent_state.position = np.array(map_points_3d)  # in world space
    # # agent.set_state(agent_state)
    map_points_3d = pathfinder.snap_point(map_points_3d)
    return map_points_3d

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
        floor_y = 0.0
        top_down_map = maps.get_topdown_map(
            self.env._sim.pathfinder, height=floor_y, meters_per_pixel=0.025
        )
        self.grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
        self._pub_rgb = rospy.Publisher("~rgb", numpy_msg(Floats), queue_size=1)
        self._pub_rgb_2 = rospy.Publisher("~rgb2", numpy_msg(Floats), queue_size=1)
        self._pub_depth = rospy.Publisher("~depth", numpy_msg(Floats), queue_size=1)
        self._robot_pose = rospy.Publisher("~robot_pose", PoseStamped, queue_size = 1)
        self._pub_all_agents = rospy.Publisher("~agent_poses", PoseArray, queue_size = 1)
        self._pub_goal_marker = rospy.Publisher("~goal", Marker, queue_size = 1)
        self.br = tf.TransformBroadcaster()
        self.br_tf_2 = tf2_ros.TransformBroadcaster()
        rospy.Subscriber("/clicked_point", PointStamped,self.point_callback, queue_size=1)
        self.initial_state = []
        self.number_of_agents = len(self.env.sim.agents_mgr)
        self.objs = []
        for i in range(self.number_of_agents):
            agent_pos = self.env.sim.agents_mgr[i].articulated_agent.base_pos
            start_pos = [agent_pos[0], agent_pos[1], agent_pos[2]]
            initial_pos = list(to_grid(self.env._sim.pathfinder, start_pos, self.grid_dimensions))
            agents_goal_pos_3d = [[0,0,0]]
            agents_initial_velocity = [0.5,0.0]
            goal_pos = list(to_grid(self.env._sim.pathfinder, agents_goal_pos_3d[0], self.grid_dimensions))
            self.initial_state.append(initial_pos+agents_initial_velocity+goal_pos)
            self.objs.append(self.env.sim.agents_mgr[i].articulated_agent)
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
            for i in range(self.number_of_agents):
                agent_pos = self.objs[i].base_pos
                start_pos = [agent_pos[0], agent_pos[1], agent_pos[2]]
                initial_pos = list(to_grid(self.env._sim.pathfinder, start_pos, self.grid_dimensions))
                agents_goal_pos_3d = [[0,0,0]]
                agents_initial_velocity = [0.5,0.0]
                goal_pos = list(to_grid(self.env._sim.pathfinder, agents_goal_pos_3d[0], self.grid_dimensions))
                self.initial_state[i] = initial_pos+agents_initial_velocity+goal_pos
            self._pub_rgb.publish(np.float32(rgb_with_res))
            self._pub_rgb_2.publish(np.float32(rgb2_with_res))
            self._pub_depth.publish(np.float32(depth_with_res))
            self.map_to_base_link({'x': initial_pos[0], 'y': initial_pos[1], 'theta': self.get_object_heading(self.env.sim.agents_mgr[0].articulated_agent.base_transformation)})
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

    def get_object_heading(self,obj_transform):
        a = obj_transform
        b = a.transform_point([0.5,0.0,0.0])
        d = a.transform_point([0.0,0.0,0.0])
        c = np.array(to_grid(self.env._sim.pathfinder, [b[0],b[1],b[2]], self.grid_dimensions))
        e = np.array(to_grid(self.env._sim.pathfinder, [d[0],d[1],d[2]], self.grid_dimensions))
        vel = (c-e)*(0.5/np.linalg.norm(c-e)*np.ones([1,2]))[0]
        return mn.Rad(np.arctan2(vel[1], vel[0]))

    def map_to_base_link(self, msg):
        theta = msg['theta']
        use_tf_2 = True
        if (not use_tf_2):
            self.br.sendTransform((-self.initial_state[0][0]+1, -self.initial_state[0][1]+1,0.0),
                            tf.transformations.quaternion_from_euler(0, 0, 0.0),
                            rospy.Time(0),
                            "my_map_frame",
                            "interim_link"
            )
            self.br.sendTransform((0.0,0.0,0.0),
                            tf.transformations.quaternion_from_euler(0, 0, -theta),
                            rospy.Time(0),
                            "interim_link",
                            "base_link"
            )
        else:
            t = geometry_msgs.msg.TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "interim_link"
            t.child_frame_id = "my_map_frame"
            t.transform.translation.x = -self.initial_state[0][0]+1
            t.transform.translation.y = -self.initial_state[0][1]+1
            t.transform.translation.z = 0.0
            q = tf.transformations.quaternion_from_euler(0, 0, 0.0)
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]
            self.br_tf_2.sendTransform(t)

            t = geometry_msgs.msg.TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "base_link"
            t.child_frame_id = "interim_link"
            t.transform.translation.x = 0.0
            t.transform.translation.y = 0.0
            t.transform.translation.z = 0.0
            
            q = tf.transformations.quaternion_from_euler(0, 0, -theta)
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]
            self.br_tf_2.sendTransform(t)

        poseMsg = PoseStamped()
        poseMsg.header.stamp = rospy.Time.now()
        poseMsg.header.frame_id = "base_link"
        quat = tf.transformations.quaternion_from_euler(0, 0, 0.0)
        poseMsg.pose.orientation.x = quat[0]
        poseMsg.pose.orientation.y = quat[1]
        poseMsg.pose.orientation.z = quat[2]
        poseMsg.pose.orientation.w = quat[3]
        poseMsg.pose.position.x = 0.0
        poseMsg.pose.position.y = 0.0
        poseMsg.pose.position.z = 0.0
        self._robot_pose.publish(poseMsg)

        ##### Publish other agents 
        poseArrayMsg = PoseArray()
        poseArrayMsg.header.frame_id = "my_map_frame"
        poseArrayMsg.header.stamp = rospy.Time.now()
        # follower_pos = my_env.follower.rigid_state.translation
        # theta = my_env.get_object_heading(my_env.follower.transformation)
        # quat = tf.transformations.quaternion_from_euler(0, 0, theta)
        # follower_pose_2d = to_grid(my_env.env._sim.pathfinder, follower_pos, my_env.grid_dimensions)
        # follower_pose_2d = follower_pose_2d*(0.025*np.ones([1,2]))[0]
        
        for i in range(len(self.initial_state)-1):
            poseMsg = Pose()
            theta = self.get_object_heading(self.objs[i+1].base_transformation) #- mn.Rad(np.pi/2-0.97 +np.pi)
            quat = tf.transformations.quaternion_from_euler(0, 0, theta)
            poseMsg.orientation.x = quat[0]
            poseMsg.orientation.y = quat[1]
            poseMsg.orientation.z = quat[2]
            poseMsg.orientation.w = quat[3]
            poseMsg.position.x = self.initial_state[i+1][0]-1
            poseMsg.position.y = self.initial_state[i+1][1]-1
            poseMsg.position.z = 0.0
            poseArrayMsg.poses.append(poseMsg)
        self._pub_all_agents.publish(poseArrayMsg)

        goal_marker = Marker()
        goal_marker.header.frame_id = "my_map_frame"
        goal_marker.type = 2
        goal_marker.pose.position.x = self.initial_state[0][4]-1
        goal_marker.pose.position.y = self.initial_state[0][5]-1
        goal_marker.pose.position.z = 0.0
        goal_marker.pose.orientation.x = 0.0
        goal_marker.pose.orientation.y = 0.0
        goal_marker.pose.orientation.z = 0.0
        goal_marker.pose.orientation.w = 1.0
        goal_marker.scale.x = 0.5
        goal_marker.scale.y = 0.5
        goal_marker.scale.z = 0.5
        goal_marker.color.a = 1.0 
        goal_marker.color.r = 0.0
        goal_marker.color.g = 1.0
        goal_marker.color.b = 0.0
        self._pub_goal_marker.publish(goal_marker)

    def point_callback(self, msg):
        point_map = [msg.point.x, msg.point.y, msg.point.z]
        p = [point_map[0]+1, point_map[1]+1]
        point_3d = from_grid(self.env._sim.pathfinder, [p[0]/0.025, p[1]/0.025], self.grid_dimensions)
        print("Placing human at ",point_3d)
        my_env.objs[1].base_pos = point_3d
        

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
    my_env.start()

    rospy.Subscriber("/cmd_vel", Twist, callback, (my_env), queue_size=1)
    while not rospy.is_shutdown():
   
        my_env.update_agent_pos_vel()
        # rospy.spin()
        my_env._r_control.sleep()
