import magnum as mn
import numpy as np

import habitat_sim
from examples.hitl.isaacsim_viewer.isaacsim_viewer import SpotPickHelper
from habitat.core.registry import registry
from habitat.tasks.rearrange.actions.actions import (
    ArticulatedAgentAction,
    BaseVelAction,
)


@registry.register_task_action
class BaseVelIsaacAction(BaseVelAction):
    def step(self, *args, **kwargs):
        lin_vel, ang_vel = kwargs[self._action_arg_prefix + "base_vel"]
        lin_vel = np.clip(lin_vel, -1, 1) * self._lin_speed
        ang_vel = np.clip(ang_vel, -1, 1) * self._ang_speed
        if not self._allow_back:
            lin_vel = np.maximum(lin_vel, 0)

        self.base_vel_ctrl.linear_velocity = mn.Vector3(lin_vel, 0, 0)
        self.base_vel_ctrl.angular_velocity = mn.Vector3(0, ang_vel, 0)
        self.cur_articulated_agent._robot_wrapper._robot.set_angular_velocity(
            [0, 0, ang_vel]
        )
        self.cur_articulated_agent._robot_wrapper._robot.set_linear_velocity(
            [lin_vel, 0, 0]
        )


@registry.register_task_action
class ArmReachAction(ArticulatedAgentAction):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

        self._spot_wrapper = self.cur_articulated_agent._robot_wrapper
        self._spot_pick_helper = SpotPickHelper(
            len(self._spot_wrapper._arm_joint_indices)
        )

    def step(self, *args, **kwargs):
        target_pos = kwargs[self._action_arg_prefix + "target_pos"]
        base_pos, base_rot = self._spot_wrapper.get_root_pose()

        def inverse_transform(pos_a, rot_b, pos_b):
            inv_pos = rot_b.inverted().transform_vector(pos_a - pos_b)
            return inv_pos

        target_rel_pos = inverse_transform(target_pos, base_rot, base_pos)
        dt = 0.5
        new_arm_joints = self._spot_pick_helper.update(dt, target_rel_pos)
        self._spot_wrapper._target_arm_joint_positions = new_arm_joints


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
