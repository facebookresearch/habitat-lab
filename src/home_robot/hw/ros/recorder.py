import rospy
import numpy as np
import h5py

from tqdm import tqdm

from home_robot.hardware.stretch_ros import HelloStretchROSInterface
from data_tools.writer import DataWriter
from data_tools.image import img_from_bytes

import argparse
import cv2


class Recorder(object):
    """ROS object that subscribes from information from the robot and publishes it out."""

    def __init__(self, filename, *args, **kwargs):
        """Collect information"""
        print("robot")
        self.robot = HelloStretchROSInterface(visualize_planner=False)
        print("done")
        self.rgb_cam = self.robot.rgb_cam
        self.dpt_cam = self.robot.dpt_cam
        self.writer = DataWriter(filename)
        self.idx = 0

    def spin(self, rate=10):
        rate = rospy.Rate(rate)
        while not rospy.is_shutdown():
            print("...")
            # record rgb and depth
            rgb, depth = self.robot.get_images(filter_depth=False, compute_xyz=False)
            q, dq = self.robot.update()
            self.writer.add_img_frame(rgb=rgb, depth=(depth * 10000).astype(np.uint16))
            self.writer.add_frame(q=q, dq=dq)
        self.writer.write_trial(self.idx)
        self.idx += 1


def png_to_mp4(group: h5py.Group, key: str, name: str, fps=10):
    """
    Write key out as a gif
    """
    gif = []
    print("Writing gif to file:", name)
    img_stream = group[key]
    writer = None

    # for i,aimg in enumerate(tqdm(group[key], ncols=50)):
    for ki, k in tqdm(
        sorted([(int(j), j) for j in img_stream.keys()], key=lambda pair: pair[0]),
        ncols=50,
    ):

        bindata = img_stream[k][()]
        _img = img_from_bytes(bindata)
        w, h = _img.shape[:2]
        img = np.zeros_like(_img)
        img[:, :, 0] = _img[:, :, 2]
        img[:, :, 1] = _img[:, :, 1]
        img[:, :, 2] = _img[:, :, 0]

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
            writer = cv2.VideoWriter(name, fourcc, fps, (h, w))
        writer.write(img)
    writer.release()


def pngs_to_mp4(filename: str, key: str, vid_name: str, fps: int):
    h5 = h5py.File(filename, "r")
    for group_name, group in h5.items():
        png_to_mp4(group, key, str(vid_name) + "_" + group_name + ".mp4", fps=fps)


def parse_args():
    parser = argparse.ArgumentParser("data recorder v1")
    parser.add_argument("--filename", "-f", default="test-data.h5")
    parser.add_argument("--video", "-v", default="")
    parser.add_argument("--fps", "-r", default=10, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    rospy.init_node("record_data_from_stretch")
    print("ready...")
    args = parse_args()
    print(args)
    rec = Recorder(args.filename)
    rec.spin(rate=args.fps)
    if len(args.video) > 0:
        print("Writing video...")
        # Write to video
        pngs_to_mp4(args.filename, "rgb", args.video, fps=args.fps)
