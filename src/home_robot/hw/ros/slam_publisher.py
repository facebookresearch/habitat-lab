#!/usr/bin/env python

import rospy
import threading
import tf
import tf2_ros

from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped


class SlamPublisher(object):
    """republish robot pose so that we can add it to ROS"""

    def _cb(self, msg):
        """collect robot data here"""
        tform = TransformStamped()
        tform.header = msg.header
        tform.child_frame_id = "odom"  # "base_link"
        tform.transform.translation.x = msg.pose.pose.position.x
        tform.transform.translation.y = msg.pose.pose.position.y
        tform.transform.translation.z = msg.pose.pose.position.z
        tform.transform.rotation.x = msg.pose.pose.orientation.x
        tform.transform.rotation.y = msg.pose.pose.orientation.y
        tform.transform.rotation.z = msg.pose.pose.orientation.z
        tform.transform.rotation.w = msg.pose.pose.orientation.w
        self.tf_publisher.sendTransform(tform)

    def __init__(self):
        self._sub = rospy.Subscriber("/poseupdate", PoseWithCovarianceStamped, self._cb)
        self.tf_publisher = tf2_ros.TransformBroadcaster()

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    rospy.init_node("slam_republisher")
    SlamPublisher().spin()
