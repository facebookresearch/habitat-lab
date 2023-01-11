import argparse
import rospy
import sys
import timeit
import numpy as np

from home_robot.hw.ros.stretch_ros import HelloStretchROSInterface
from home_robot.agent.motion.robot import (
    STRETCH_HOME_Q,
    STRETCH_PREGRASP_Q,
    HelloStretchIdx,
)
from home_robot.agent.perception.constants import coco_categories
from home_robot.utils.pose import to_pos_quat

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser("collect_data")
    parser.add_argument("name", default="test", help="name of task to train")
    parser.add_argument(
        "-n", "--num-frames", help="number of frames per keypoint", default=50, type=int
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Create the robot
    print("--------------")
    print("Start example - hardware using ROS")
    rospy.init_node("hello_stretch_ros_test")
    args = parse_args()
    print("Create ROS interface")
    rob = HelloStretchROSInterface(visualize_planner=False)

    # Get cameras from the robot object
    t0 = timeit.default_timer()
    rgb_cam = rob.rgb_cam
    dpt_cam = rob.dpt_cam
    rgb_cam.wait_for_image()
    dpt_cam.wait_for_image()
    print("took", timeit.default_timer() - t0, "seconds to get images")

    home_q = STRETCH_PREGRASP_Q
    model = rob.get_model()
    q, _ = rob.update()
    print("q =", q)
    q = model.update_look_at_ee(home_q.copy())
    q = model.update_gripper(q, open=True)
    rob.goto(q, move_base=False, wait=True)
    model = rob.get_model()
    q, _ = rob.update()
    print("q =", q)

    # Main loop. collect waypoints and move the robot.
    while not done and not rospy.is_shutdown():

        valid_inp = False
        while not valid_inp and not rospy.is_shutdown():
            inp = input("Collect keypoint data? y/n:")
            if len(inp) > 0 and inp[0].lower() in ["y", "n"]:
                if inp[0].lower() == "y":
                    break
                else:
                    print("Quitting.")
                    done = True
                    break

        if not done and not rospy.is_shutdown():
            print("Capturing frames...")
            rate = rospy.Rate(10)
            for i in range(args.num_frames):
                # add frames and joint state info to buffer
                q, dq = rob.update()
                # Get the rgb and depth
                rgb, depth, xyz = rob.get_images()
                rate.sleep()
                if rospy.is_shutdown():
                    print("Abort.")
                    done = True
                    break
