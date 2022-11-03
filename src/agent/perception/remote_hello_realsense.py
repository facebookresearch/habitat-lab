"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
# python -m Pyro4.naming -n <MYIP>
import logging
import os
import json
import time
import copy
import math
from math import *

import pyrealsense2 as rs
import Pyro4
import numpy as np
import torch
import open3d as o3d
from droidlet.lowlevel.hello_robot.remote.utils import transform_global_to_base, goto
from slam_pkg.utils import depth_util as du
import obstacle_utils
from obstacle_utils import is_obstacle
from droidlet.dashboard.o3dviz import serialize as o3d_pickle
from data_compression import *
from segmentation.constants import coco_categories
from segmentation.detectron2_segmentation import Detectron2Segmentation
from home_robot.ros.camera import RosCamera


# Configure depth and color streams
CH = 480
CW = 640
FREQ = 30

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED.add("pickle")
Pyro4.config.ITER_STREAMING = True

# #####################################################


@Pyro4.expose
class RemoteHelloRealsense(object):
    """Hello Robot interface"""

    def __init__(self, bot, use_ros=False, use_ros_realsense=True):
        self.bot = bot
        if use_ros:
            self.use_ros = True
            from droidlet.lowlevel.hello_robot.remote.lidar_ros_driver import Lidar
        else:
            self.use_ros = False
            from droidlet.lowlevel.hello_robot.remote.lidar import Lidar
        self.use_ros_realsense = use_ros_realsense
        self._lidar = Lidar()
        self._lidar.start()
        self._done = True
        self._connect_to_realsense()
        # Slam stuff
        # uv_one_in_cam
        intrinsic_mat = np.asarray(self.get_intrinsics())
        intrinsic_mat_inv = np.linalg.inv(intrinsic_mat)
        img_resolution = self.get_img_resolution()
        img_pixs = np.mgrid[
            0 : img_resolution[1] : 1, 0 : img_resolution[0] : 1
        ]  # Camera on the hello is oriented vertically
        img_pixs = img_pixs.reshape(2, -1)
        img_pixs[[0, 1], :] = img_pixs[[1, 0], :]
        uv_one = np.concatenate((img_pixs, np.ones((1, img_pixs.shape[1]))))
        self.uv_one_in_cam = np.dot(intrinsic_mat_inv, uv_one)
        self.num_sem_categories = len(coco_categories)
        self.segmentation_model = Detectron2Segmentation(
            sem_pred_prob_thr=0.9, sem_gpu_id=-1, visualize=True
        )

    def get_base_state(self):
        return self.bot.get_base_state()

    def get_camera_transform(self):
        return self.bot.get_camera_transform()

    def get_lidar_scan(self):
        # returns tuple (timestamp, scan)
        return self._lidar.get_latest_scan()

    def is_lidar_obstacle(self):
        lidar_scan = self._lidar.get_latest_scan()
        if lidar_scan is None:
            print("no scan")
            return False
        return obstacle_utils.is_lidar_obstacle(lidar_scan)

    def _connect_to_realsense(self):
        if self.use_ros_realsense:
            print("Creating cameras...")
            self.rgb_cam = RosCamera('/camera/color')
            self.dpt_cam = RosCamera('/camera/aligned_depth_to_color', buffer_size=5)

            print("Waiting for camera images...")
            self.rgb_cam.wait_for_image()
            self.dpt_cam.wait_for_image()

            self.intrinsic_mat = np.array([
                [self.rgb_cam.fx, 0, self.rgb_cam.px],
                [0, self.rgb_cam.fy, self.rgb_cam.py],
                [0, 0, 1]]
            )
            self.intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
                CW, CH, self.rgb_cam.fx, self.rgb_cam.fy, self.rgb_cam.px, self.rgb_cam.py
            )

        else:
            config = rs.config()
            pipeline = rs.pipeline()
            config.enable_stream(rs.stream.color, CW, CH, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, CW, CH, rs.format.z16, 30)

            cfg = pipeline.start(config)
            dev = cfg.get_device()

            depth_sensor = dev.first_depth_sensor()
            # set high accuracy: https://github.com/IntelRealSense/librealsense/issues/2577#issuecomment-432137634
            depth_sensor.set_option(rs.option.visual_preset, 3)
            self.realsense = pipeline

            profile = pipeline.get_active_profile()
            # because we align the depth frame to the color frame, and only use the aligned depth frame,
            # we need to use the intrinsics of the color frame
            color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
            i = color_profile.get_intrinsics()
            self.intrinsic_mat = np.array([[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]])
            self.intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(CW, CH, i.fx, i.fy, i.ppx, i.ppy)

            align_to = rs.stream.color
            self.align = rs.align(align_to)

            self.decimate = rs.decimation_filter(2.0)
            self.threshold = rs.threshold_filter(0.1, 4.0)
            self.depth2disparity = rs.disparity_transform()
            self.spatial = rs.spatial_filter(0.5, 20.0, 2.0, 0.0)
            self.temporal = rs.temporal_filter(0.0, 100.0, 3)
            self.disparity2depth = rs.disparity_transform(False)
            self.hole_filling = rs.hole_filling_filter(
                2
            )  # Fill with neighboring pixel nearest to sensor

            print("connected to realsense")

    def get_intrinsics(self):
        return self.intrinsic_mat

    def get_img_resolution(self, rotate=True):
        if rotate:
            return (CW, CH)
        else:
            return (CH, CW)

    def test_connection(self):
        print("Connected!!")  # should print on server terminal
        return "Connected!"  # should print on client terminal

    def get_rgb_depth(self, rotate=True, compressed=False):
        if self.use_ros_realsense:
            depth_image = self.dpt_cam.get_filtered()
            depth_image[depth_image < 0.1] = 0.
            depth_image[depth_image > 4.0] = 0.
            color_image = self.rgb_cam.get()

        else:
            tm = time.time()
            frames = None
            while not frames:
                frames = self.realsense.wait_for_frames()

                # post-processing goes here
                decimated = self.decimate.process(frames).as_frameset()
                thresholded = self.threshold.process(decimated).as_frameset()
                disparity = self.depth2disparity.process(thresholded).as_frameset()
                spatial = self.spatial.process(disparity).as_frameset()
                # temporal = self.temporal.process(spatial).as_frameset() # TODO: re-enable
                postprocessed = self.disparity2depth.process(spatial).as_frameset()

                aligned_frames = self.align.process(postprocessed)
                # aligned_frames = self.align.process(frames)

                # Get aligned frames
                aligned_depth_frame = (
                    aligned_frames.get_depth_frame()
                )  # aligned_depth_frame is a 640x480 depth image
                color_frame = aligned_frames.get_color_frame()

                # Validate that both frames are valid
                if not aligned_depth_frame or not color_frame:
                    continue

                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                if not compressed:
                    depth_image = depth_image / 1000  # convert to meters

        if rotate:
            depth_image = np.rot90(depth_image, k=1, axes=(1, 0))
            color_image = np.rot90(color_image, k=1, axes=(1, 0))

        return color_image, depth_image

    def get_manipulation_pcd_from_depth(self, depth):
        def depth_to_xyz(depth):
            fx, px = self.intrinsic_mat[0, 0], self.intrinsic_mat[0, 2]
            fy, py = self.intrinsic_mat[1, 1], self.intrinsic_mat[1, 2]
            indices = np.indices((CH, CW), dtype=np.float32).transpose(1, 2, 0)
            z = depth
            # pixel indices start at top-left corner
            # for these equations, it starts at bottom-left
            x = (indices[:, :, 1] - px) * (z / fx)
            y = (indices[:, :, 0] - py) * (z / fy)
            # Should now be height x width x 3, after this:
            xyz = np.stack([x, y, z], axis=-1)
            return xyz

        depth = np.rot90(depth, k=1, axes=(0, 1))
        xyz = depth_to_xyz(depth)
        xyz = np.rot90(xyz, k=1, axes=(1, 0))
        xyz = xyz.reshape(-1, 3)
        return xyz

    def get_rgb_depth_optimized_for_habitat_transfer(self, rotate=True, compressed=False):
        tm = time.time()
        frames = None
        while not frames:
            frames = self.realsense.wait_for_frames()

            # post-processing goes here
            frames = self.decimate.process(frames).as_frameset()
            # thresholded = self.threshold.process(decimated).as_frameset()
            frames = self.depth2disparity.process(frames).as_frameset()
            frames = self.spatial.process(frames).as_frameset()
            # temporal = self.temporal.process(spatial).as_frameset() # TODO: re-enable
            frames = self.disparity2depth.process(frames).as_frameset()
            frames = self.hole_filling.process(frames).as_frameset()

            aligned_frames = self.align.process(frames)

            # Get aligned frames
            aligned_depth_frame = (
                aligned_frames.get_depth_frame()
            )  # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())[:, :, [2, 1, 0]]

            if not compressed:
                depth_image = depth_image / 1000  # convert to meters

            # rotate
            if rotate:
                depth_image = np.rot90(depth_image, k=1, axes=(1, 0))
                color_image = np.rot90(color_image, k=1, axes=(1, 0))

        return color_image, depth_image

    def get_semantics(self, rgb, depth):
        """Get semantic segmentation."""
        semantics, semantics_vis = self.segmentation_model.get_prediction(
            np.expand_dims(rgb[:, :, ::-1], 0), np.expand_dims(depth, 0)
        )
        semantics, semantics_vis = semantics[0], semantics_vis[0]
        unfiltered_semantics = semantics

        # given RGB and depth are rotated after the point cloud creation,
        # we rotate them back here to align to the point cloud
        depth = np.rot90(depth, k=1, axes=(0, 1))
        semantics = np.rot90(semantics, k=1, axes=(0, 1))

        # apply the same depth filter to semantics as we applied to the point cloud
        semantics = semantics.reshape(-1, self.num_sem_categories)
        valid = (depth > 0).flatten()
        semantics = semantics[valid]

        return semantics, unfiltered_semantics, semantics_vis

    def get_orientation(self):
        """Get discretized robot orientation."""
        # TODO yaw is in radians in [-3.14, 3.14] when using Hector SLAM
        # and in [0, 6.28] when not using Hector SLAM => make it consistent
        _, _, yaw_in_radians = self.get_base_state()
        # convert it to degrees in [0, 360]
        # yaw_in_degrees = int(yaw_in_radians * 180.0 / np.pi)
        yaw_in_degrees = int(yaw_in_radians * 180.0 / np.pi + 180.0)
        orientation = torch.tensor([yaw_in_degrees // 5])
        return orientation

    def get_open3d_pcd(self, rgb_depth=None, cam_transform=None, base_state=None):
        # get data
        if rgb_depth is None:
            rgb, depth = self.get_rgb_depth(rotate=False, compressed=False)
        else:
            rgb, depth = rgb_depth

        if cam_transform is None:
            cam_transform = self.get_camera_transform()
        if base_state is None:
            base_state = self.bot.get_base_state()
        intrinsic = self.intrinsic_o3d

        # convert to open3d RGBDImage
        rgb_u8 = np.ascontiguousarray(rgb[:, :, [2, 1, 0]], dtype=np.uint8)
        depth_f32 = np.ascontiguousarray(depth, dtype=np.float32) * 1000
        orgb = o3d.geometry.Image(rgb_u8)
        odepth = o3d.geometry.Image(depth_f32)
        orgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            orgb, odepth, depth_trunc=10.0, convert_rgb_to_intensity=False
        )

        # create transform matrix
        roty90 = o3d.geometry.get_rotation_matrix_from_axis_angle([0, math.pi / 2, 0])
        rotxn90 = o3d.geometry.get_rotation_matrix_from_axis_angle([-math.pi / 2, 0, 0])
        rotz = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, base_state[2]])
        rot_cam = cam_transform[:3, :3]
        trans_cam = cam_transform[:3, 3]
        final_rotation = rotz @ rot_cam @ rotxn90 @ roty90
        final_translation = [
            trans_cam[0] + base_state[0],
            trans_cam[1] + base_state[1],
            trans_cam[2] + 0,
        ]
        final_transform = cam_transform.copy()
        final_transform[:3, :3] = final_rotation
        final_transform[:3, 3] = final_translation
        extrinsic = np.linalg.inv(final_transform)
        # create point cloud

        opcd = o3d.geometry.PointCloud.create_from_rgbd_image(orgbd, intrinsic, extrinsic)
        return opcd

    def get_current_pcd(self):
        rgb, depth = self.get_rgb_depth(rotate=False, compressed=False)
        opcd = self.get_open3d_pcd(rgb_depth=[rgb, depth])
        pcd = np.asarray(opcd.points)

        # RGB and depth are rotated after the point cloud creation
        rgb = np.rot90(rgb, k=1, axes=(1, 0))
        depth = np.rot90(depth, k=1, axes=(1, 0))

        return pcd, rgb, depth

    def is_obstacle_in_front(self, return_viz=False):
        base_state = self.bot.get_base_state()
        lidar_scan = self.get_lidar_scan()
        pcd = self.get_open3d_pcd()
        ret = is_obstacle(
            pcd, base_state, lidar_scan=lidar_scan, max_dist=0.5, return_viz=return_viz
        )
        if return_viz:
            obstacle, cpcd, crop, bbox, rest = ret
            # cpcd = o3d_pickle(cpcd)
            crop = o3d_pickle(crop)
            bbox = o3d_pickle(bbox)
            rest = o3d_pickle(rest)
            # return obstacle, cpcd, crop, bbox, rest
            return obstacle, crop, bbox, rest
        else:
            obstacle = ret
            return obstacle

    def get_pcd_data(self, rotate=True):
        """Gets all the data to calculate the point cloud for a given rgb, depth frame."""
        rgb, depth = self.get_rgb_depth(rotate=rotate, compressed=True)
        # cap anything more than np.power(2,16)~ 64 meter
        depth[depth > np.power(2, 16) - 1] = np.power(2, 16) - 1
        T = self.get_camera_transform()
        rot = T[:3, :3]
        trans = T[:3, 3]
        base2cam_trans = np.array(trans).reshape(-1, 1)
        base2cam_rot = np.array(rot)

        rgb = jpg_encode(rgb)
        depth = blosc_encode(depth)
        return rgb, depth, base2cam_rot, base2cam_trans

    def calibrate_tilt(self):
        self.bot.set_tilt(math.radians(-60))
        time.sleep(2)
        pcd = self.get_open3d_pcd()
        plane, points = pcd.segment_plane(
            distance_threshold=0.03,
            ransac_n=3,
            num_iterations=1000,
        )
        angle = math.atan(plane[0] / plane[2])
        self.bot.set_tilt_correction(angle)

        self.bot.set_tilt(math.radians(0))
        time.sleep(2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pass in server device IP")
    parser.add_argument(
        "--ip",
        help="Server device (robot) IP. Default is 192.168.0.0",
        type=str,
        default="0.0.0.0",
    )
    parser.add_argument("--ros", action="store_true")
    parser.add_argument("--no-ros", dest="ros", action="store_false")
    parser.set_defaults(ros=False)

    args = parser.parse_args()

    np.random.seed(123)

    with Pyro4.Daemon(args.ip) as daemon:
        bot = Pyro4.Proxy("PYRONAME:hello_robot@" + args.ip)
        robot = RemoteHelloRealsense(bot, use_ros=args.ros)
        robot_uri = daemon.register(robot)
        with Pyro4.locateNS(host=args.ip) as ns:
            ns.register("hello_realsense", robot_uri)

        robot.calibrate_tilt()
        print("Server is started...")
        # try:
        #     while True:
        #         print(time.asctime(), "Waiting for requests...")

        #         sockets = daemon.sockets
        #         ready_socks = select.select(sockets, [], [], 0)
        #         events = []
        #         for s in ready_socks:
        #             events.append(s)
        #         daemon.events(events)
        #         time.sleep(0.0)
        # except KeyboardInterrupt:
        #     pass
        def callback():
            time.sleep(0.0)
            return True

        daemon.requestLoop(callback)
