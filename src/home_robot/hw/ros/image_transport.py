import rospy
import imagiz
import cv2
import numpy as np
import threading
from home_robot.hw.ros.msg_numpy import image_to_numpy, numpy_to_image
from home_robot.hw.ros.camera import RosCamera
from home_robot.utils.data_tools.image import img_from_bytes
from home_robot.utils.data_tools.image import img_to_bytes
from sensor_msgs.msg import Image, CameraInfo


class ImageServer(object):
    """Receives compressed images from remote - faster than ROS"""

    def __init__(self, show_images=False):
        self.show_images = show_images
        self.color_server = imagiz.TCP_Server(port=9990)
        self.depth_server = imagiz.TCP_Server(port=9991)

        self.pub_color = rospy.Publisher("/server/color/image_raw", Image, queue_size=2)
        self.pub_depth = rospy.Publisher("/server/depth/image_raw", Image, queue_size=2)

        # TODO: add rotated images if necessary
        # self.pub_rotated_color = rospy.Publisher("/server/rotated_color", Image, queue_size=2)
        # self.pub_rotated_depth = rospy.Publisher("/server/rotated_depth", Image, queue_size=2)

        # Create camera info publishers
        pub_color_cam_info = rospy.Publisher(
            "/server/color/camera_info", CameraInfo, queue_size=1
        )
        pub_depth_cam_info = rospy.Publisher(
            "/server/depth/camera_info", CameraInfo, queue_size=1
        )

        self.reference_frame = None
        self.sequence_id = 0

        cb_color_cam_info = lambda msg: self.send_and_receive_camera_info(
            msg, pub_color_cam_info
        )
        cb_depth_cam_info = lambda msg: self.send_and_receive_camera_info(
            msg, pub_depth_cam_info
        )
        self.sub_color_cam_info = rospy.Subscriber(
            "/camera/color/camera_info", CameraInfo, cb_color_cam_info
        )
        self.sub_depth_cam_info = rospy.Subscriber(
            "/camera/aligned_depth_to_color/camera_info", CameraInfo, cb_color_cam_info
        )

    def send_and_receive_camera_info(self, cam_info, publisher):
        if self.reference_frame is None:
            self.reference_frame = cam_info.header.frame_id
        cam_info.header.stamp = rospy.Time.now()
        cam_info.header.seq = self.sequence_id
        publisher.publish(cam_info)

    def spin(self, rate=15):
        servers = [self.color_server, self.depth_server]
        for server in servers:
            server.start()
        rate = rospy.Rate(rate)
        while not rospy.is_shutdown():
            for server, name in zip(servers, ["color", "depth"]):
                message = server.receive()
                if name == "color":
                    frame = img_from_bytes(message.image, format="webp")
                    # frame = cv2.imdecode(message.image, 1)
                    msg = numpy_to_image(frame, "8UC3")
                    msg.header.stamp = rospy.Time.now()
                    msg.header.frame_id = self.reference_frame
                    msg.header.seq = self.sequence_id
                    self.pub_color.publish(msg)
                elif name == "depth":
                    # frame = cv2.imdecode(message.image, 0)
                    frame = (
                        img_from_bytes(message.image, format="png") / 1000.0
                    ).astype(np.float32)
                    # Publish images to the new topic
                    msg = numpy_to_image((frame / 1000.0).astype(np.float32), "32FC1")
                    msg.header.stamp = rospy.Time.now()
                    msg.header.frame_id = self.reference_frame
                    msg.header.seq = self.sequence_id
                    self.pub_depth.publish(msg)
                if self.show_images:
                    cv2.imshow(name, frame)
                    cv2.waitKey(1)
            rate.sleep()
            self.sequence_id += 1
        print("Done.")


class ImageClient(object):
    """sends images to the server"""

    def __init__(self, show_sizes=False):
        self.color_client = imagiz.TCP_Client(
            client_name="color", server_ip="192.168.0.79", server_port=9990
        )
        self.depth_client = imagiz.TCP_Client(
            client_name="depth", server_ip="192.168.0.79", server_port=9991
        )
        self.clients = [self.color_client, self.depth_client]

        self.color_camera = RosCamera("/camera/color")
        self.depth_camera = RosCamera("/camera/aligned_depth_to_color")
        self.cameras = [self.color_camera, self.depth_camera]

        # Encode everything
        # JPEG compression
        # TODO currently unused - example code
        self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        self.show_sizes = show_sizes
        self.encoders = [self.encode_color, self.encode_depth]

    def encode_color(self, frame):
        # r, image = cv2.imencode(".jpg", frame, encode_param)
        image = img_to_bytes(frame, format="webp")
        # image = img_to_bytes(frame)
        if self.show_sizes:
            print("color len =", len(image))
        return image

    def encode_depth(self, frame):
        frame = (frame * 1000).astype(np.uint16)
        # r, image = cv2.imencode(".png", frame)
        image = img_to_bytes(frame, format="png")
        if self.show_sizes:
            print("depth len =", len(image))
        return image

    def spin(self, rate=15):
        print("Waiting for images from ROS...")
        rate = rospy.Rate(rate)
        while not rospy.is_shutdown():
            for camera, client, encode in zip(
                self.cameras, self.clients, self.encoders
            ):
                frame = camera.get().copy()
                if frame is not None:
                    client.send(encode(frame))
            rate.sleep()
        print("Done.")
