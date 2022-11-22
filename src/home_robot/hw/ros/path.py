import rospy
import rospkg


def get_package_path():
    r = rospkg.RosPack()
    return r.get_path("home_robot")
