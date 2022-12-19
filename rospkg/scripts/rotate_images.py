#!/usr/bin/env python3

import rospy
import numpy as np

from sensor_msgs.msg import Image

from home_robot.hw.ros.msg_numpy import image_to_numpy, numpy_to_image


pub_color = None
pub_rotated_color = None
pub_depth = None
pub_rotated_depth = None


def callback_color(msg):
    img = image_to_numpy(msg)
    img = np.rot90(np.rot90(np.rot90(img)))
    msg2 = numpy_to_image(img, msg.encoding)
    msg2.header = msg.header
    pub_color.publish(msg2)
    pub_rotated_color.publish(msg2)


def callback_depth(msg):
    img = image_to_numpy(msg)
    img = np.rot90(np.rot90(np.rot90(img)))
    msg2 = numpy_to_image(img, msg.encoding)
    msg2.header = msg.header
    pub_depth.publish(msg2)
    pub_rotated_depth.publish(msg2)


if __name__ == "__main__":
    rospy.init_node("rotate_images")
    pub_color = rospy.Publisher("/color/image_raw", Image, queue_size=2)
    pub_depth = rospy.Publisher("/depth/image_raw", Image, queue_size=2)
    pub_rotated_color = rospy.Publisher("/rotated_color", Image, queue_size=2)
    pub_rotated_depth = rospy.Publisher("/rotated_depth", Image, queue_size=2)
    sub_color = rospy.Subscriber("/camera/color/image_raw", Image, callback_color)
    sub_depth = rospy.Subscriber(
        "/camera/aligned_depth_to_color/image_raw", Image, callback_depth
    )
    rospy.spin()
