import pybullet as pb
import pybullet_data
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import open3d
import trimesh
import trimesh.transformations as tra

# Helpers
from home_robot.utils.image import z_from_opengl_depth
from home_robot.utils.image import opengl_depth_to_xyz
from home_robot.utils.image import show_point_cloud
from home_robot.utils.image import Camera
from home_robot.utils.image import T_CORRECTION


"""
This file contains simple tools for creating and loading objects in pybullet for easy simulation
and data generation.
"""

PbJointInfo = namedtuple(
    "JointInfo",
    [
        "index",
        "name",
        "type",
        "qindex",
        "uindex",
        "flags",
        "damping",
        "friction",
        "lower_limit",
        "upper_limit",
        "max_force",
        "max_velocity",
        "link_name",
        "axis",
        "parent_frame_pos",
        "parent_frame_rot",
        "parent_idx",
    ],
)


class PbObject(object):
    def __init__(
        self,
        name,
        filename,
        assets_path=None,
        start_pos=[0, 0, 0],
        start_rot=[0, 0, 0, 1],
        static=False,
        client=None,
    ):
        self.name = name
        self.filename = filename
        assert client is not None
        self.client = client
        self.assets_path = assets_path
        if self.assets_path is not None:
            pb.setAdditionalSearchPath(assets_path)
        self.id = pb.loadURDF(filename, start_pos, start_rot, useFixedBase=static)

    def set_pose(self, pos, rot):
        pb.resetBasePositionAndOrientation(
            self.id, pos, rot, physicsClientId=self.client
        )

    def get_aabb(self):
        mins, maxs = pb.getAABB(self.id, physicsClientId=self.client)
        return np.array(mins), np.array(maxs)

    def place_above(self, obj, offset=None):
        # get aabb for the other object
        bmin, bmax = obj.get_aabb()
        raise NotImplementedError()

    def get_pose(self):
        pos, orn = pb.getBasePositionAndOrientation(
            self.id, physicsClientId=self.client
        )
        return np.array(pos), np.array(orn)

    def is_colliding(self, other, distance=0.001):
        res = pb.getClosestPoints(self.id, other.id, distance)
        return len(res) > 0


class PbArticulatedObject(PbObject):
    def __init__(
        self,
        name,
        filename,
        assets_path=None,
        start_pos=[0, 0, 0],
        start_rot=[0, 0, 0, 1],
        static=False,
        client=None,
        *args,
        **kwargs
    ):
        super(PbArticulatedObject, self).__init__(
            name, filename, assets_path, start_pos, start_rot, static, client
        )
        self._link_idx = {}
        self._read_joint_info()

    def _read_joint_info(self):
        """get some joint info from pb for reproducing the robot"""
        self.num_joints = pb.getNumJoints(self.id, self.client)
        self.joint_infos = []
        for i in range(self.num_joints):
            self.joint_infos.append(
                PbJointInfo(*pb.getJointInfo(self.id, i, self.client))
            )
            # self.joint_infos[-1].name = self.joint_infos[-1].name.decode()
            self._link_idx[self.joint_infos[-1].link_name.decode()] = self.joint_infos[
                -1
            ].index

    def get_joint_info_by_name(self, name):
        for info in self.joint_infos:
            if info.name.decode() == name:
                return info
        else:
            return None

    def get_joint_names(self):
        return [info.name.decode() for info in self.joint_infos]

    def get_link_names(self):
        return [info.link_name.decode() for info in self.joint_infos]

    def set_joint_position(self, idx, pos):
        pb.resetJointState(
            self.id,
            idx,
            targetValue=pos,
            targetVelocity=0.0,
            physicsClientId=self.client,
        )

    def get_link_pose(self, name):
        """get link pose - forward kinematrics"""
        res = pb.getLinkState(
            self.id,
            self._link_idx[name],
            computeForwardKinematics=1,
            physicsClientId=self.client,
        )
        # Return the world positions of the URDF link
        # return res[4], res[5]
        return np.array(res[4]), np.array(res[5])


class PbCamera(Camera):
    def __init__(
        self,
        client,
        pos,
        orn,
        height=200,
        width=200,
        near_val=0.001,
        far_val=1000.0,
        fov=90,
    ):
        self.client = client
        self.height = height
        self.width = width
        self.near_val = near_val
        self.far_val = far_val
        self.fov = fov
        self.proj_matrix = pb.computeProjectionMatrixFOV(
            self.fov, self.width / self.height, self.near_val, self.far_val
        )
        self.max_depth = 5.0
        self.set_pose(pos, orn)
        self._set_params()

    def set_pose(self, pos, orn):
        self.pos = pos
        self.orn = orn
        if len(orn) == 3:
            x, y, z, w = pb.getQuaternionFromEuler(orn)
            # TODO - remove debugging code
            # print(x, y, z, w)
            # w, x, y, z = tra.quaternion_from_matrix(tra.euler_matrix(*orn))
            # print(x, y, z, w)
            # import pdb; pdb.set_trace()
        else:
            x, y, z, w = orn
        self.pose_matrix = tra.quaternion_matrix([w, x, y, z])
        self.pose_matrix[:3, 3] = pos
        T = np.eye(4)
        T[2, 3] = 2.0
        look_pose = self.pose_matrix @ T
        self.pos = pos
        self.view_matrix = pb.computeViewMatrix(self.pos, look_pose[:3, 3], (0, 0, 1))
        # self.view_matrix = pb.computeViewMatrix(self.pos, look_pose[:3, 3], self.pose_matrix[:3, 0])

    def _set_params(self):
        """
        from chris xie:
        https://github.com/chrisdxie/uois/blob/master/src/util/utilities.py#L204
        https://github.com/chrisdxie/uois/blob/master/LICENSE - MIT
        """
        aspect_ratio = self.width / self.height
        e = 1 / (np.tan(np.radians(self.fov / 2.0)))
        t = self.near_val / e
        b = -t
        r = t * aspect_ratio
        l = -r
        alpha = self.width / (r - l)  # pixels per meter
        focal_length = (
            self.near_val * alpha
        )  # focal length of virtual camera (frustum camera)
        fx = focal_length
        fy = focal_length
        self.fx = fx
        self.fy = fy
        self.px = self.width / 2.0
        self.py = self.height / 2.0

    def capture(self):
        res = pb.getCameraImage(
            self.width,
            self.height,
            self.view_matrix,
            self.proj_matrix,
            renderer=pb.ER_BULLET_HARDWARE_OPENGL,
            # renderer=pb.ER_TINY_RENDERER,
            physicsClientId=self.client,
        )
        w, h, rgb, depth, seg = res
        # fix them now
        depth = z_from_opengl_depth(depth, camera=self)
        depth = np.clip(depth, self.near_val, self.max_depth)
        return rgb[:, :, :3], depth, seg

    def capture_pc(self):
        """show xyz from current camera position"""
        rgb, depth, seg = self.capture()
        xyz = opengl_depth_to_xyz(depth, camera=self)
        xyz = xyz.reshape(-1, 3)
        rgb = rgb.reshape(-1, 3)
        seg = seg.reshape(-1)
        mask = np.bitwise_and(depth < 0.99 * self.max_depth, depth > 0.1).reshape(-1)
        xyz = xyz[mask]
        rgb = rgb[mask]
        xyz = trimesh.transform_points(xyz, self.pose_matrix)
        return rgb, xyz, seg

    def show(self, images=False, show_pc=True, test_id=2):
        rgb, depth, seg = self.capture()
        # rgb = np.flip(rgb, axis=0)
        # depth = np.flip(depth, axis=0)
        # seg = np.flip(seg, axis=0)
        xyz = opengl_depth_to_xyz(depth, camera=self)
        if images:
            plt.figure(1)
            plt.subplot(221)
            plt.imshow(rgb)
            plt.subplot(222)
            plt.imshow(depth)
            plt.subplot(223)
            plt.imshow(seg)
            plt.subplot(224)
            plt.imshow(xyz)
            plt.show()
        xyz = xyz.reshape(-1, 3)
        rgb = rgb.reshape(-1, 3)
        seg = seg.reshape(-1)
        mask = np.bitwise_and(depth < 0.99 * self.max_depth, depth > 0.1).reshape(-1)
        xyz = xyz[mask]
        rgb = rgb[mask]
        # xyz = trimesh.transform_points(xyz, self.pose_matrix)
        # TODO: default remove this
        if test_id > 0:
            red_mask = seg[mask] == test_id
            red_xyz = xyz[red_mask]
            mins = np.min(red_xyz, axis=0)
            maxs = np.max(red_xyz, axis=0)
            print("red size -", maxs - mins)

        xyz = trimesh.transform_points(xyz, self.pose_matrix)
        # SHow pc
        if show_pc:
            show_point_cloud(xyz, rgb / 255.0, orig=np.zeros(3))

    def get_pose(self):
        # return T_CORRECTION @ self.pose_matrix.copy()
        return self.pose_matrix.copy()


class PbClient(object):
    """
    Physics client; connects to backend.
    """

    def __init__(self, visualize=True, is_simulation=True, assets_path=None):
        self.is_simulation = is_simulation
        if visualize:
            self.id = pb.connect(pb.GUI)
        else:
            self.id = pb.connect(pb.DIRECT)

        self.objects = {}
        self.obj_name_to_id = {}
        self.assets_path = assets_path
        pb.setGravity(0, 0, -9.8)
        if self.assets_path is not None:
            pb.setAdditionalSearchPath(assets_path)
        self.camera = None

    def add_object(self, name, urdf_filename, assets_path=None, static=False):
        obj = PbObject(name, urdf_filename, assets_path, static=static, client=self.id)
        self.objects[name] = obj
        self.obj_name_to_id[name] = obj.id
        return obj

    def add_articulated_object(
        self, name, urdf_filename, assets_path=None, static=False
    ):
        obj = PbArticulatedObject(
            name, urdf_filename, assets_path, static=static, client=self.id
        )
        self.objects[name] = obj
        self.obj_name_to_id[name] = obj.id
        return obj

    def run_physics(self, t):
        raise NotImplementedError
        pass

    def add_camera(self, pos, orn, camera_params):
        """todo: create a camera in the bullet scene"""
        self.camera = PbCamera(self.id, pos, orn, **camera_params)
        return self.camera

    def add_ground_plane(self):
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = pb.loadURDF("plane.urdf")
