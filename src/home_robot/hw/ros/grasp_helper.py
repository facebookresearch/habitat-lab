import rospy
import numpy as np
import threading

from sensor_msgs.msg import PointCloud, ChannelFloat32
from geometry_msgs.msg import PoseArray, Point
from std_msgs.msg import Float32MultiArray
from collections import deque
import trimesh.transformations as tra

from home_robot.srv import GraspRequest, GraspRequestResponse
from home_robot.hw.ros.utils import matrix_to_pose_msg
from home_robot.hw.ros.utils import matrix_from_pose_msg

# For debugging only
import tf2_ros
from geometry_msgs.msg import TransformStamped
from home_robot.hw.ros.utils import ros_pose_to_transform


def msg_to_segmented_point_cloud(msg):
    xyz = np.zeros((len(msg.points), 3))
    rgb = np.zeros((len(msg.points), 3))
    seg = np.zeros((len(msg.points),))

    print("got # pts =", len(msg.points))
    print("channel info:")
    for channel in msg.channels:
        print("\t", channel.name, len(channel.values))
    for i in range(len(msg.points)):
        xyz[i] = np.array([msg.points[i].x, msg.points[i].y, msg.points[i].z])
        seg[i] = msg.channels[0].values[i]
        rgb[i, 0] = msg.channels[1].values[i]
        rgb[i, 1] = msg.channels[2].values[i]
        rgb[i, 2] = msg.channels[3].values[i]
    return xyz, rgb, seg


class GraspClient(object):
    """send and receive grasp queries with no custom messages or anything else"""

    def __init__(
        self,
        topic="/grasping/request",
        offset=0.0,
        R=None,
        flip_grasps=True,
        debug=True,
    ):
        print("Initializing connection to ROS grasping server from the client...")
        # self.sub = rospy.Subscriber(recv_topic, PoseArray, queue_size=1, self._cb)
        # self.sub2 = rospy.Subscriber(score_topic, FloatArray, queue_size=1, self._cb2)
        # self.pub = rospy.Publisher(send_topic, PointCloud)
        rospy.wait_for_service(topic)
        self.proxy = rospy.ServiceProxy(topic, GraspRequest)
        self.grasp_lock = threading.Lock()
        self.score_lock = threading.Lock()
        self.poses = None
        self.req_id = 0

        self.offset = 0
        self.R = R
        self.flip_grasps = flip_grasps
        self.flip_grasps_R = tra.euler_matrix(0, 0, np.pi)

        self.debug = debug
        if self.debug:
            self.broadcaster = tf2_ros.TransformBroadcaster()

    def _cb(self, msg):
        # Convert rospy point cloud into a message that we care about
        with self.grasp_lock:
            self.grasp_id = msg.seq
            self.poses = msg_to_poses(msg)

    def msg_to_poses(self, msg, frame=None):
        grasps = []
        for i, pose in enumerate(msg.poses):
            grasp = matrix_from_pose_msg(pose)
            if self.debug:
                # Visualize in ROS with orientation
                t = TransformStamped()
                t.header.stamp = rospy.Time.now()
                t.child_frame_id = "grasp_" + str(i)
                if frame is not None:
                    t.header.frame_id = frame
                t.transform = ros_pose_to_transform(pose)
                self.broadcaster.sendTransform(t)
            grasps.append(grasp)
            if self.flip_grasps:
                # Rotate the grasps around the Z axis to give us some more options
                grasps.append(grasp @ self.flip_grasps_R)
        return grasps

    def _score_cb(self, msg):
        with self.score_lock:
            self.score_id = msg.header.seq
            self.scores = np.array([pt.x for x in msg.polygon.points])

    def segmented_point_cloud_to_msg(self, xyz, labels):
        msg = PointCloud()
        msg.header.stamp = self.req_id
        self.req_id += 1
        xyz = xyz.reshape(-1, 3)
        for i in range(xyz.shape[0]):
            msg.points.append(Point(xyz[i, 0], xyz[i, 1], xyz[i, 2]))
        msg.channels = ChannelFloat32(
            name="label", values=labels.astype(np.float32).tolist()
        )
        return msg

    def segmented_point_cloud_to_msg(self, xyz, rgb, labels):
        pc = PointCloud()
        xyz = xyz.reshape(-1, 3)
        rgb = rgb.reshape(-1, 3)
        labels = labels.reshape(-1)
        for i in range(xyz.shape[0]):
            pc.points.append(Point(*xyz[i]))
        r = ChannelFloat32("r", rgb[:, 0])
        g = ChannelFloat32("g", rgb[:, 1])
        b = ChannelFloat32("b", rgb[:, 2])
        label = ChannelFloat32("label", labels)
        pc.channels = [label, r, g, b]
        return pc

    def request(self, xyz, rgb, seg, frame=None):
        pc = self.segmented_point_cloud_to_msg(xyz, rgb, seg)
        if frame is not None:
            pc.header.frame_id = frame
        res = self.proxy(cloud=pc)
        objs = {}
        for obj_id, (grasps, scores) in enumerate(zip(res.grasps, res.scores)):
            # Get the grawps associated with a particular object ID
            grasps = self.msg_to_poses(grasps, frame)
            # Turn the scores into a numpy array
            scores = np.array(scores.data)
            # Get information for the objects
            objs[obj_id] = (grasps, scores)
        return objs

    def get_grasps(self, xyz, labels, timeout=10.0):
        msg = segmented_point_cloud_to_msg(xyz, labels)
        print("Sending grasp request...")
        with self.lock:
            self.pub.publish(msg)
        t0 = rospy.Time.now()

        # Wait for a response
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            with self.lock:
                poses = self.poses
                if poses is not None:
                    return poses
            if timeout is not None and (rospy.Time.now() - t0).to_sec() > timeout:
                break
            rate.sleep()
        print("... Grasp request timed out. No response received from server.")
        return None


class GraspServer(object):
    def __init__(
        self,
        handle_request_fn,
        topic="/grasping/request",
        dbg_topic="/grasping/all_grasps",
    ):
        print("Initializing ROS grasping server...")
        # self.sub = rospy.Subscriber(send_topic, PointCloud, queue_size=1, self._cb)
        # self.pub = rospy.Publisher(recv_topic, PointCloud)
        # self.queue = deque()
        self.handle_request_fn = handle_request_fn
        self.service = rospy.Service(topic, GraspRequest, self.process)
        self.pub = rospy.Publisher(dbg_topic, PoseArray)
        print("Waiting for requests...")

    def process(self, req):
        xyz, rgb, seg = msg_to_segmented_point_cloud(req.cloud)
        print()
        print("frame =", req.cloud.header.frame_id)
        # print(xyz)
        # print(seg)
        grasps, scores = self.handle_request_fn(xyz, rgb, seg)
        # print(grasps.keys())
        resp = GraspRequestResponse()
        all_grasps = PoseArray()
        all_grasps.header.frame_id = req.cloud.header.frame_id
        for k in grasps.keys():
            # print("----", k, "----")
            obj_grasps = grasps[k]
            # print(obj_grasps)
            print(k, "# grasps =", obj_grasps.shape)
            grasps_msg = PoseArray()
            for g in obj_grasps:
                pose_msg = matrix_to_pose_msg(g)
                grasps_msg.poses.append(pose_msg)
                all_grasps.poses.append(pose_msg)
            obj_scores = scores[k]
            grasps_msg.header.seq = int(k)
            grasps_msg.header.frame_id = req.cloud.header.frame_id
            resp.grasps.append(grasps_msg)
            resp.scores.append(Float32MultiArray(data=obj_scores))
        self.pub.publish(all_grasps)
        return resp

    def _cb(self, msg):
        # Process into the right format and make it into a message queue
        self.queue.push_right(msg_to_segmented_point_cloud(msg))

    def get(self):
        return self.queue.pop_left()

    def send_response(self, poses):
        self.pub.publish(poses_to_msg(poses))
