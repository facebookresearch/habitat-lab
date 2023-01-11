#!/usr/bin/env python

import rospy
import os
from home_robot.hw.teleop.stretch_xbox_controller import StretchXboxController
from home_robot.agent.motion.robot import HelloStretch
from home_robot.hw.ros.path import get_package_path


if __name__ == '__main__':
    rospy.init_node("xbox_controller")

    stretch_planner_urdf_path = os.path.join(get_package_path(), "../assets/hab_stretch/urdf/planner_calibrated.urdf")
    model = HelloStretch(
        visualize=False,
        root="",
        urdf_path=stretch_planner_urdf_path,
    )
    controller = StretchXboxController(model)
    rospy.spin()
