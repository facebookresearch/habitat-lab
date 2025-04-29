import magnum as mn
import numpy as np

import habitat_sim
from habitat.core.registry import registry
from habitat.tasks.rearrange.actions.actions import ArmEEAction, BaseVelAction

# from gym import spaces


@registry.register_task_action
class BaseVelIsaacAction(BaseVelAction):
    def step(self, *args, **kwargs):
        lin_vel, ang_vel = kwargs[self._action_arg_prefix + "base_vel"]
        print(f"lin_vel: {lin_vel}; ang_vel: {ang_vel}")
        lin_vel = np.clip(lin_vel, -1, 1) * self._lin_speed
        ang_vel = np.clip(ang_vel, -1, 1) * self._ang_speed
        if not self._allow_back:
            lin_vel = np.maximum(lin_vel, 0)

        self.base_vel_ctrl.linear_velocity = mn.Vector3(lin_vel, 0, 0)
        self.base_vel_ctrl.angular_velocity = mn.Vector3(0, ang_vel, 0)
        # <omni.isaac.core.robots.robot.Robot object at 0x7fb613f91c00>
        self.cur_articulated_agent._robot_wrapper._robot.set_angular_velocity(
            [0, 0, ang_vel]
        )
        self.cur_articulated_agent._robot_wrapper._robot.set_linear_velocity(
            [lin_vel, 0, 0]
        )


@registry.register_task_action
class ArmReachEEAction(ArmEEAction):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

        self._robot_wrapper = self.cur_articulated_agent._robot_wrapper
        self.ee_rot_target = None
        self._use_ee_rot = self._config.get("use_ee_rot", False)

    def reset(self, *args, **kwargs):
        try:
            self.ee_target, self.ee_rot_target = self._ik_helper.calc_fk(
                np.array(
                    self._sim.articulated_agent._robot_wrapper.right_arm_joint_pos
                )
            )
        except Exception:
            self.ee_target = None
            self.ee_rot_target = None

    def calc_desired_joints(self):
        joint_pos = np.array(
            self._sim.articulated_agent._robot_wrapper.right_arm_joint_pos
        )
        joint_vel = np.zeros(joint_pos.shape)

        self._ik_helper.set_arm_state(joint_pos, joint_vel)

        des_joint_pos = self._ik_helper.calc_ik(
            self.ee_target, self.ee_rot_target
        )
        return np.array(des_joint_pos)

    def apply_joint_limits(self, des_joint_pos):
        murp_joint_limits_lower = np.deg2rad(
            np.array([-157, -102, -166, -174, -160, 31, -172])
        )
        murp_joint_limits_upper = np.deg2rad(
            np.array([157, 102, 166, -8, 160, 258, 172])
        )
        return np.clip(
            des_joint_pos, murp_joint_limits_lower, murp_joint_limits_upper
        )

    def step(self, *args, **kwargs):
        target_pos = kwargs[self._action_arg_prefix + "target_pos"]
        target_rot = kwargs[self._action_arg_prefix + "target_rot"]
        base_pos, base_rot = self._robot_wrapper.get_root_pose()

        def inverse_transform(pos_a, rot_b, pos_b):
            inv_pos = rot_b.inverted().transform_vector(pos_a - pos_b)
            return inv_pos

        target_rel_pos = inverse_transform(target_pos, base_rot, base_pos)
        self.ee_target = np.array(target_rel_pos)
        self.ee_rot_target = np.array(target_rot)
        des_joint_pos = self.calc_desired_joints()
        des_joint_pos = self.apply_joint_limits(des_joint_pos)

        self._robot_wrapper._target_right_arm_joint_positions = des_joint_pos


@registry.register_task_action
class BaseVelKinematicIsaacAction(BaseVelAction):
    def update_base(self):
        ctrl_freq = self._sim.ctrl_freq
        trans = self.cur_articulated_agent.base_transformation
        rigid_state = habitat_sim.RigidState(
            mn.Quaternion.from_matrix(trans.rotation()), trans.translation
        )

        target_rigid_state = self.base_vel_ctrl.integrate_transform(
            1 / ctrl_freq, rigid_state
        )
        target_trans = mn.Matrix4.from_(
            target_rigid_state.rotation.to_matrix(),
            target_rigid_state.translation,
        )
        self.cur_articulated_agent.base_transformation = target_trans
