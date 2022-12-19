import rospy
from home_robot.hw.ros.image_transport import ImageServer

if __name__ == "__main__":
    rospy.init_node("local_republisher")
    server = ImageServer(show_images=True)
    server.spin(rate=15)
