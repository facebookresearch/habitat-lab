import random
from abc import ABC, abstractmethod

import magnum as mn
import numpy as np

try:
    import pybullet as p
except ImportError:
    p = None


from habitat.tasks.rearrange.mp.projector.voxel_gen import VoxelMapper
from habitat.tasks.rearrange.mp.robot_target import ObjPlanningData
from habitat.tasks.rearrange.utils import check_pb_install, get_aabb
from habitat_sim.physics import MotionType


class MpSim(ABC):
    """
    The abstract simulator interface for the motion planner.
    """

    def __init__(self, sim):
        self._sim = sim
        self._ik = self._sim._ik

    def setup(self, use_prev):
        pass

    def should_ignore_first_collisions(self):
        return False

    @abstractmethod
    def set_targ_obj_idx(self, targ_obj_idx):
        pass

    @abstractmethod
    def unset_targ_obj_idx(self, targ_obj_idx):
        pass

    @abstractmethod
    def get_robot_transform(self):
        """
        Returns the robot to world transformation matrix.
        """

    @abstractmethod
    def get_collisions(self):
        """
        Returns a list of pairs that collided where each element in the pair is
        of the form:
            {
            "name": "body name",
            "link": "link name",
            }
        """

    @abstractmethod
    def capture_state(self):
        pass

    @abstractmethod
    def get_arm_pos(self):
        pass

    @abstractmethod
    def set_position(self, pos, obj_id):
        pass

    @abstractmethod
    def micro_step(self):
        pass

    @abstractmethod
    def add_sphere(self, radius, color):
        pass

    @abstractmethod
    def get_ee_pos(self):
        """
        Gets the end-effector position in GLOBAL coordinates
        """

    @abstractmethod
    def remove_object(self, obj_id):
        pass

    @abstractmethod
    def set_state(self, state):
        pass

    @abstractmethod
    def render(self):
        """
        Renders the current state of the simulator.
        """

    @abstractmethod
    def start_mp(self):
        pass

    @abstractmethod
    def end_mp(self):
        pass

    @abstractmethod
    def get_obj_info(self, obj_idx) -> ObjPlanningData:
        """
        Returns information about an object for the grasp planner
        """


class HabMpSim(MpSim):
    def get_collisions(self):
        return self._sim.get_collisions()

    @property
    def _hold_local_idx(self):
        if self._sim.snapped_obj_id is None:
            return None
        return self._sim.scene_obj_ids.index(self._sim.snapped_obj_id)

    def capture_state(self):
        env_state = self._sim.capture_state()
        return env_state

    def get_ee_pos(self):
        return self._sim.get_end_effector_pos()

    def set_state(self, state):
        if self._hold_local_idx is not None:
            # Auto-snap the held object to the robot's hand.
            state["static_T"][
                self._hold_local_idx
            ] = self._sim.robot.ee_transform
        self._sim.set_state(state)

    def set_arm_pos(self, joint_pos):
        self._sim.set_arm_pos(joint_pos)

    def get_robot_transform(self):
        return self._sim.get_robot_transform()

    def get_obj_info(self, obj_idx) -> ObjPlanningData:
        return ObjPlanningData(
            bb=get_aabb(obj_idx, self._sim._sim),
            trans=self._sim._sim.get_transformation(obj_idx),
        )

    def set_position(self, pos, obj_id):
        self._sim._sim.set_translation(pos, obj_id)

    def get_arm_pos(self):
        return self._sim.get_arm_pos()

    def micro_step(self):
        # self._sim.perform_discrete_collision_detection()
        self._sim.internal_step(-1)

    def add_sphere(self, radius, color):
        sphere_id = self._sim.draw_sphere(radius)
        self._sim._sim.override_collision_group(sphere_id, 64)
        return sphere_id

    def remove_object(self, obj_id):
        self._sim._sim.remove_object(obj_id)

    def set_targ_obj_idx(self, targ_obj_idx):
        if targ_obj_idx is not None:
            self._sim._sim.override_collision_group(targ_obj_idx, 128)

    def unset_targ_obj_idx(self, targ_obj_idx):
        if targ_obj_idx is not None:
            self._sim._sim.override_collision_group(targ_obj_idx, 8)

    def render(self):
        obs = self._sim.step(0)
        if "high_rgb" not in obs:
            raise ValueError(
                ("CHECKPOINT_RENDER_INTERVAL must be 1 " "to use mod_mp_")
            )
        pic = obs["high_rgb"]
        pic = np.flip(pic, 0)
        if pic.shape[-1] > 3:
            # Skip the depth part.
            pic = pic[:, :, :3]
        return pic

    def start_mp(self):
        self.prev_motion_types = {}
        self.hold_obj = self._hold_local_idx
        if self.hold_obj is not None:
            self._sim.desnap_object(force=True)
            self._sim.do_grab_using_constraint = False
            self._sim.set_snapped_obj(self.hold_obj)

        # Set everything to STATIC
        for obj_id in self._sim.scene_obj_ids:
            self.prev_motion_types[obj_id] = self._sim.get_object_motion_type(
                obj_id
            )
            if obj_id == self._sim.snapped_obj_id:
                pass
                # self._sim.set_object_motion_type(MotionType.KINEMATIC, obj_id)
            else:
                self._sim.set_object_motion_type(MotionType.STATIC, obj_id)

    def end_mp(self):
        # Set everything to how it was
        for obj_id, mt in self.prev_motion_types.items():
            self._sim.set_object_motion_type(mt, obj_id)

        if self.hold_obj is not None:
            self._sim.desnap_object(force=True)
            self._sim.do_grab_using_constraint = True
            self._sim.set_snapped_obj(self.hold_obj)


class PbMpSim(MpSim):
    def __init__(self, sim):
        super().__init__(sim)
        check_pb_install(p)
        self.pc_id = p.connect(p.DIRECT)
        load_urdf = "./orp/robots/opt_fetch/robots/fetch_no_base.urdf"
        self.robot_transform = mn.Matrix4.identity_init()
        self.robo_id = p.loadURDF(
            load_urdf,
            basePosition=[0, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.pc_id,
        )
        for i in range(p.getNumJoints(self.robo_id)):
            p.setCollisionFilterGroupMask(
                self.robo_id, i, 32, 1, physicsClientId=self.pc_id
            )
        if load_urdf.endswith("fetch_no_base.urdf"):
            self.ee_link = 15
            self.arm_start = 8
            self.hab_to_pb_js = {0: 0}
            for i in range(7):
                self.hab_to_pb_js[sim.arm_start + i] = self.arm_start + i
            for i in range(2):
                self.hab_to_pb_js[sim.arm_start + 7 + i] = (
                    self.arm_start + i + 8
                )

        else:
            raise ValueError("Unrecognized URDF")
        self.render_bodies = []

    def start_mp(self):
        pass

    def _set_hab_js(self, js):
        for hab_jid in range(len(js)):
            if hab_jid in self.hab_to_pb_js:
                pb_jid = self.hab_to_pb_js[hab_jid]
                p.resetJointState(
                    self.robo_id,
                    pb_jid,
                    js[hab_jid],
                    0.0,
                    physicsClientId=self.pc_id,
                )

    def setup(self, use_prev):
        """
        Captures the environment state from the depth camera and forms a point
        cloud.
        """
        take_count = 2000
        sphere_radius = 0.05
        mass = 0

        if use_prev and len(self.render_bodies) != 0:
            return

        for body_id in self.render_bodies:
            p.removeBody(body_id, physicsClientId=self.pc_id)
        self.render_bodies = []

        if not use_prev:
            # We are in a position where we CAN retract the arm.
            orig_arm = self._sim.get_arm_pos()
            self._sim.hack_retract_arm()

        obs = self._sim.step(0)

        depth = obs["depth"]
        voxel_mapper = VoxelMapper(False)
        voxel_mapper.compute_voxels(depth, self._sim)
        voxels = voxel_mapper.get_voxels()
        random.shuffle(voxels)
        voxels = np.array(voxels)
        voxels = voxels[:take_count, :3]

        if not use_prev:
            self._sim.set_arm_pos(orig_arm)

        sphere_id = p.createCollisionShape(
            p.GEOM_SPHERE, radius=sphere_radius, physicsClientId=self.pc_id
        )
        # sphere_id = -1
        sphere_reg = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            rgbaColor=[1, 0, 0, 1],
            radius=sphere_radius,
            physicsClientId=self.pc_id,
        )

        T = self._sim.get_robot_transform()
        self.robot_transform = T
        rot = mn.Quaternion.from_matrix(T.rotation())
        rot = [*rot.vector, rot.scalar]
        pos = list(T.translation)

        js = self._sim.get_joints_pos()

        self.fix_joint = js

        p.resetBasePositionAndOrientation(
            self.robo_id, pos, rot, physicsClientId=self.pc_id
        )

        p.setGravity(0, 0, 0, physicsClientId=self.pc_id)

        self._set_hab_js(js)

        for voxel in voxels:
            voxel_id = p.createMultiBody(
                mass, sphere_id, sphere_reg, voxel, physicsClientId=self.pc_id
            )
            self.render_bodies.append(voxel_id)

    def create_viz(self, viz_pos, r=0.05):
        sphere_id = p.createCollisionShape(
            p.GEOM_SPHERE, radius=r, physicsClientId=self.pc_id
        )
        sphere_targ = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            rgbaColor=[0, 0, 1, 1],
            radius=r,
            physicsClientId=self.pc_id,
        )
        return p.createMultiBody(
            0.0, sphere_id, sphere_targ, viz_pos, physicsClientId=self.pc_id
        )

    def get_robot_transform(self):
        return self.robot_transform

    def get_collisions(self):
        colls = p.getContactPoints(physicsClientId=self.pc_id)

        def convert_coll(x):
            linkA, bodyA = p.getBodyInfo(x[1], physicsClientId=self.pc_id)
            linkB, bodyB = p.getBodyInfo(x[2], physicsClientId=self.pc_id)
            bodyA = bodyA.decode("ascii")
            linkA = linkA.decode("ascii")
            bodyB = bodyB.decode("ascii")
            linkB = linkB.decode("ascii")
            if linkA == "base_link":
                # For some reason it can count the base link as the entire
                # robot, then base ignoring collision checks later are passed.
                linkA = ""
            if bodyA == "":
                bodyA = "%i" % x[1]
            if bodyB == "":
                bodyB = "%i" % x[2]

            # We cannot distinguish objects in the scene, so everything must be
            # classified as "type": "Stage"
            return [
                {
                    "name": bodyA,
                    "link": linkA,
                    "type": "Stage",
                },
                {
                    "name": bodyB,
                    "link": linkB,
                    "type": "Stage",
                },
            ]

        tmp = [convert_coll(x) for x in colls]
        return tmp

    def get_ee_pos(self):
        ls = p.getLinkState(
            self.robo_id,
            self.ee_link,
            computeForwardKinematics=1,
            physicsClientId=self.pc_id,
        )
        world_ee = ls[4]
        return world_ee

    def set_arm_pos(self, joint_pos):
        self._set_hab_js(self.fix_joint)

        # Then only move the joints we want to move.
        for i in range(7):
            jidx = self.arm_start + i
            p.resetJointState(
                self.robo_id, jidx, joint_pos[i], physicsClientId=self.pc_id
            )

    def get_arm_pos(self):
        return np.array(
            [
                p.getJointState(
                    self.robo_id, jidx, physicsClientId=self.pc_id
                )[0]
                for jidx in range(self.arm_start, self.arm_start + 7)
            ]
        )

    def add_sphere(self, radius, color):
        if color is None:
            color = [0, 1, 0, 1]
        sphere_coll = p.createCollisionShape(
            p.GEOM_SPHERE, radius=radius, physicsClientId=self.pc_id
        )

        # sphere_id = -1
        sphere_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            rgbaColor=color,
            radius=radius,
            physicsClientId=self.pc_id,
        )
        sphere_id = p.createMultiBody(
            0, sphere_coll, sphere_shape, [0, 0, 0], physicsClientId=self.pc_id
        )

        p.setCollisionFilterGroupMask(
            sphere_id, -1, 16, 1, physicsClientId=self.pc_id
        )
        return sphere_id

    def set_position(self, pos, obj_id):
        p.resetBasePositionAndOrientation(
            obj_id, pos, [0, 0, 0, 1], physicsClientId=self.pc_id
        )

    def micro_step(self):
        p.stepSimulation(physicsClientId=self.pc_id)

    def capture_state(self):
        # the state should always be static.
        return None

    def set_state(self, state):
        # the state should always be static.
        pass

    def render(self):
        cam_pos = [-0.65, 1.8, 4.35]
        look_at = np.array(self.get_robot_transform().translation)
        render_dim = 512

        view_mat = p.computeViewMatrix(cam_pos, look_at, [0.0, 1.0, 0.0])
        proj_mat = p.computeProjectionMatrixFOV(
            fov=90, aspect=1, nearVal=0.1, farVal=100.0
        )
        img = p.getCameraImage(
            render_dim,
            render_dim,
            viewMatrix=view_mat,
            projectionMatrix=proj_mat,
            physicsClientId=self.pc_id,
        )[2]
        img = img[:, :, :3]
        return img

    def set_targ_obj_idx(self, targ_obj_idx):
        pass

    def unset_targ_obj_idx(self, targ_obj_idx):
        pass

    def remove_object(self, obj_id):
        p.removeBody(obj_id, physicsClientId=self.pc_id)

    def end_mp(self):
        pass

    def get_obj_info(self, obj_idx) -> ObjPlanningData:
        # Assume unit bounding box for object.
        r = 0.15
        def_obj_bb = mn.Range3D.from_center(
            mn.Vector3(0, 0, 0), mn.Vector3(r, r, r)
        )
        obj_pos = self._sim._sim.get_translation(obj_idx)
        return ObjPlanningData(
            bb=def_obj_bb, trans=mn.Matrix4.translation(obj_pos)
        )
