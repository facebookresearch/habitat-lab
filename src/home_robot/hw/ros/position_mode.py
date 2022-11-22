#!/usr/bin/env python3

import rospy
from std_srvs.srv import Trigger


class SwitchToPositionMode(object):
    def __init__(self):
        self.switch = rospy.ServiceProxy("switch_to_position_mode", Trigger)
        print("Waiting for mode service...")
        self.switch.wait_for_service()
        print("Switching to position...", self.switch())

    def __call__(self):
        print("Switching to position...", self.switch())


if __name__ == "__main__":
    SwitchToPositionMode()
