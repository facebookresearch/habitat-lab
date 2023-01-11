import numpy as np
import rospy
import threading
from collections import deque

from home_robot.utils.image import Camera
from home_robot.hw.ros.msg_numpy import image_to_numpy

from sensor_msgs.msg import CameraInfo, Image


class RosCamera(Camera):
    """compute camera parameters from ROS instead"""

    def _cb(self, msg):
        """capture the latest image and save it"""
        with self._lock:
            # print(msg.encoding)
            img = image_to_numpy(msg)
            if msg.encoding == "16UC1":
                # depth support goes here
                # Convert the image to metric (meters)
                img = img / 1000.0
            elif msg.encoding == "rgb8":
                # color support - do nothing
                pass
            self._img = img
            if self.buffer_size is not None:
                self._add_to_buffer(img)

    def _add_to_buffer(self, img):
        """add to buffer and remove old image if buffer size exceeded"""
        self._buffer.append(img)
        if len(self._buffer) > self.buffer_size:
            self._buffer.popleft()

    def valid_mask(self, depth):
        """return only valid pixels"""
        depth = depth.reshape(-1)
        return np.bitwise_and(depth > self.near_val, depth < self.far_val)

    def valid_pc(self, xyz, rgb, depth):
        mask = self.valid_mask(depth)
        xyz = xyz.reshape(-1, 3)[mask]
        rgb = rgb.reshape(-1, 3)[mask]
        return xyz, rgb

    def wait_for_image(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            with self._lock:
                if self.buffer_size is None:
                    if self._img is not None:
                        break
                else:
                    # Wait until we have a full buffer
                    if len(self._buffer) >= self.buffer_size:
                        break
                rate.sleep()

    def get(self, device=None):
        """return the current image associated with this camera"""
        with self._lock:
            if self._img is None:
                return None
            else:
                # We are using torch
                img = self._img.copy()
        if device is not None:
            # Convert to tensor and get the formatting right
            import torch

            img = torch.FloatTensor(img).to(device).permute(2, 0, 1)
        return img

    def get_filtered(self, std_threshold=0.005, device=None):
        """get image from buffer; do some smoothing"""
        if self.buffer_size is None:
            raise RuntimeError("no buffer")
        with self._lock:
            imgs = [img[None] for img in self._buffer]
        # median = np.median(np.concatenate(imgs, axis=0), axis=0)
        stacked = np.concatenate(imgs, axis=0)
        avg = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0)
        dims = avg.shape
        avg = avg.reshape(-1)
        avg[std.reshape(-1) > std_threshold] = 0
        img = avg.reshape(*dims)
        if device is not None:
            # Convert to tensor and get the formatting right
            import torch

            img = torch.FloatTensor(img).to(device).permute(2, 0, 1)
        return img

    def get_frame(self):
        return self.frame_id

    def get_K(self):
        return self.K.copy()

    def get_info(self):
        return {
            "D": self.D,
            "K": self.K,
            "fx": self.fx,
            "fy": self.fy,
            "px": self.px,
            "py": self.py,
            "near_val": self.near_val,
            "far_val": self.far_val,
            "R": self.R,
            "P": self.P,
            "height": self.height,
            "width": self.width,
        }

    def __init__(
        self, name="/camera/color", verbose=True, flipxy=False, buffer_size=None
    ):
        self.name = name
        self._img = None
        self._lock = threading.Lock()
        self._camera_info_topic = name + "/camera_info"
        print("Waiting for camera info on", self._camera_info_topic + "...")
        cam_info = rospy.wait_for_message(self._camera_info_topic, CameraInfo)
        print(cam_info)
        self.buffer_size = buffer_size
        if self.buffer_size is not None:
            # create buffer
            self._buffer = deque()
        self.height = cam_info.height
        self.width = cam_info.width
        self.pos, self.orn, self.pose_matrix = None, None, None
        # Get camera information and save it here
        self.distortion_model = cam_info.distortion_model
        self.D = np.array(cam_info.D)  # Distortion parameters
        self.K = np.array(cam_info.K).reshape(3, 3)
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.px = self.K[0, 2]
        self.py = self.K[1, 2]
        self.R = np.array(cam_info.R).reshape(3, 3)  # Rectification matrix
        self.P = np.array(cam_info.P).reshape(3, 4)  # Projection/camera matrix
        self.near_val = 0.1
        self.far_val = 5.0
        if verbose:
            print()
            print("---------------")
            print("Created camera with info:")
            print(cam_info)
            print("---------------")
        self.frame_id = cam_info.header.frame_id
        self.topic_name = name + "/image_raw"
        self._sub = rospy.Subscriber(self.topic_name, Image, self._cb, queue_size=10)
        print("Waiting for", self.topic_name)
        self.wait_for_image()
