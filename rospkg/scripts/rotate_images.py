#!/usr/bin/env python3

import rospy
import numpy as np

from sensor_msgs.msg import Image

from home_robot.hw.ros.msg_numpy import image_to_numpy, numpy_to_image

pub = None


def callback(msg):
    img = image_to_numpy(msg)
    img = np.rot90(np.rot90(np.rot90(img)))
    msg2 = numpy_to_image(img, msg.encoding)
    msg2.header = msg.header
    pub.publish(msg2)


if __name__ == "__main__":
    rospy.init_node("rotate_images")
    pub = rospy.Publisher("/rotated_image", Image, queue_size=2)
    sub = rospy.Subscriber("/camera/color/image_raw", Image, callback)
    rospy.spin()
