import rospy
from home_robot.hw.ros.image_transport import ImageClient

if __name__ == "__main__":
    rospy.init_node("image_client")
    client = ImageClient(show_sizes=False)
    client.spin(rate=15)
