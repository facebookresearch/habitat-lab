import glob
import os
import os.path as osp
import sys
import uuid
from typing import List, Optional

import numpy as np
from gym import spaces
from PIL import Image

from habitat_sim.utils.viz_utils import save_video

matched_dir = glob.glob(
    osp.join(osp.expanduser("~"), "ompl-1.5.*/py-bindings")
)
if len(matched_dir) > 0:
    sys.path.insert(0, matched_dir[0])

try:
    from ompl import base as ob
    from ompl import util as ou
except ImportError:
    ou = None

from copy import copy
from functools import partial

from habitat.tasks.rearrange.utils import make_border_red
from habitat_baselines.motion_planning.grasp_generator import GraspGenerator
from habitat_baselines.motion_planning.mp_sim import HabMpSim
from habitat_baselines.motion_planning.mp_spaces import JsMpSpace


def is_ompl_installed():
    return ou is not None


class MotionPlanner:
    def __init__(self, sim, config):
        if not is_ompl_installed:
            raise ImportError("Need to install OMPL to use motion planning")
        self._config = config
        self._reach_for_obj = None
        self._should_render = False
        self._coll_check_count: int = 0
        self._num_calls = 0
        self._sphere_id: Optional[int] = None
        self._ignore_names: List[str] = []
        self.traj_viz_id = None
        self._sim = sim
        if not osp.exists(self._config.DEBUG_DIR):
            os.makedirs(self._config.DEBUG_DIR)

        self._use_sim = self._get_sim()
        self.grasp_gen = None

    def set_should_render(self, should_render):
        self._should_render = should_render
        if self._should_render:
            for f in glob.glob(f"{self._config.DEBUG_DIR}/*"):
                os.remove(f)

    def _log(self, txt):
        if self._config.VERBOSE:
            print("MP:", txt)

    @property
    def action_space(self):
        return spaces.Box(shape=(3,), low=0, high=1, dtype=np.float32)

    def photo(self, add_txt, before_txt="", should_save=True):
        """
        Render debug utility helper
        """
        pic = self._use_sim.render()
        if pic.shape[-1] > 3:
            pic = pic[:, :, :3]
        im = Image.fromarray(pic)
        save_name = "%s/%s%s_%s.jpeg" % (
            self._config.DEBUG_DIR,
            before_txt,
            str(uuid.uuid4())[:4],
            add_txt,
        )
        if should_save:
            im.save(save_name)
        return pic

    def get_mp_space(self):
        return JsMpSpace(
            self._use_sim,
            self._sim.ik_helper,
            self._num_calls,
            self._should_render,
        )

    def _is_state_valid(self, x, take_image=False):
        self._mp_space.set_arm(x)
        if self._ee_margin is not None and self._sphere_id is not None:
            self._use_sim.set_position(
                self._use_sim.get_ee_pos(), self._sphere_id
            )
        self._use_sim.micro_step()

        did_collide, coll_details = self._use_sim.get_collisions(
            self._config.COUNT_OBJ_COLLISIONS, self._ignore_names, True
        )
        if (
            self._ignore_first
            or self._use_sim.should_ignore_first_collisions()
        ) and self._coll_check_count == 0:
            self._ignore_names.extend(coll_details.robot_coll_ids)
            self._log(
                "First run, ignoring collisions from "
                + str(self._ignore_names)
            )
        self._coll_check_count += 1
        if take_image:
            self.photo(f"{did_collide}")

        if not self._use_sim.should_ignore_first_collisions():
            # We only want to continue to ignore collisions from this if we are
            # using a point cloud approach.
            self._ignore_names = []
        if did_collide and self._should_render:
            return False

        # Check we satisfy the EE margin, if there is one.
        if not self._check_ee_coll(
            self._ee_margin, self._sphere_id, coll_details
        ):
            return False

        return True

    def set_config(
        self,
        ee_margin,
        count_obj_collisions,
        grasp_thresh,
        n_gen_grasps,
        run_cfg,
        ignore_first: bool = False,
        use_prev=False,
    ):
        """
        Sets up the parameters of this motion planning call.
        """
        self._ee_margin = ee_margin
        self._count_obj_collisions = count_obj_collisions
        self._sphere_id = None
        self._run_cfg = run_cfg
        self._mp_space = self.get_mp_space()
        self._ignore_names = []
        self._ignore_first = ignore_first
        self._hold_id = self._sim.grasp_mgr.snap_idx
        self._use_sim.setup(use_prev)
        if self.traj_viz_id is not None:
            self._sim.remove_traj_obj(self.traj_viz_id)
            self.traj_viz_id = None
        self.grasp_gen = GraspGenerator(
            self._use_sim,
            self._mp_space,
            self._sim.ik_helper,
            self,
            self._should_render,
            grasp_thresh,
            n_gen_grasps,
            self._config.MP_SIM_TYPE == "Priv",
            self._config.DEBUG_DIR,
            self._config.GRASP_GEN_IS_VERBOSE,
        )

    def setup_ee_margin(self, obj_targ):
        """
        Adds a collision margin sphere around the end-effector if it was
        specified in the run config. This sphere intersects with everything but
        the robot and currently held object.
        """
        use_sim = self._use_sim
        if self._ee_margin is not None:
            self._sphere_id = use_sim.add_sphere(self._ee_margin)
            use_sim.set_targ_obj_idx(obj_targ)

    def remove_ee_margin(self, obj_targ):
        """
        Removes the collision margin sphere around the end-effector. If not
        called this object is never removed and will cause problems!
        """
        use_sim = self._use_sim
        if self._ee_margin is not None:
            use_sim.remove_object(self._sphere_id)
            use_sim.unset_targ_obj_idx(obj_targ)
            self._sphere_id = None

    def get_recent_plan_stats(self, plan, robo_targ, name=""):
        is_start_bad = False
        is_goal_bad = False
        if not robo_targ.is_guess and plan is None:
            # Planning failed, but was it the planner's fault?
            is_start_bad = self._is_state_valid(self._mp_space.used_js_start)
            is_goal_bad = self._is_state_valid(self._mp_space.used_js_goal)

        return {
            f"plan_{name}bad_coll": int(self.was_bad_coll),
            f"plan_{name}failure": int(plan is None),
            f"plan_{name}guess": robo_targ.is_guess,
            f"plan_{name}goal_bad": is_start_bad,
            f"plan_{name}start_bad": is_goal_bad,
            f"plan_{name}approx": self._is_approx_sol,
        }

    def motion_plan(
        self,
        start_js,
        robot_target,
        obs,
        timeout=30,
        use_prev_obs=False,
        ignore_names=None,
    ):
        if ignore_names is None:
            ignore_names = []
        use_sim = self._use_sim
        self.was_bad_coll = False
        self._is_approx_sol = False
        if robot_target.is_guess:
            return None

        self.hold_id = self._sim.grasp_mgr.snap_idx

        use_sim.start_mp()
        self._log("Starting plan from %s" % str(start_js))
        self._log("Target info %s" % str(robot_target))
        self._log(
            "Agent position" + str(use_sim.get_robot_transform().translation)
        )

        env_state = copy(use_sim.capture_state())
        self._mp_space.set_env_state(env_state)

        self._ignore_names = ["ball_new", *ignore_names]
        self._coll_check_count = 0

        self.setup_ee_margin(robot_target.obj_targ)

        joint_plan = self._get_path(
            partial(self._is_state_valid),
            start_js,
            robot_target,
            use_sim,
            self._mp_space,
            timeout,
        )

        if joint_plan is None:
            self._mp_space.render_start_targ(
                self._run_cfg.VIDEO_DIR,
                "mp_fail",
                robot_target.ee_targ,
                f"ep{self._sim.ep_info['episode_id']}",
            )

        if joint_plan is not None:
            self._render_verify_motion_plan(use_sim, robot_target, joint_plan)

            self._log("MP: Got plan of length %i" % len(joint_plan))

        self.remove_ee_margin(robot_target.obj_targ)

        self._num_calls += 1

        # Settle back to the regular environment
        use_sim.set_state(env_state)
        use_sim.set_arm_pos(start_js)
        use_sim.end_mp()
        for _ in range(100):
            use_sim.micro_step()
        use_sim.set_state(env_state)
        for _ in range(100):
            use_sim.micro_step()

        return joint_plan

    def _render_verify_motion_plan(self, use_sim, robot_target, joint_plan):
        """
        Renders the motion plan to a video by teleporting the arm to the
        planned joint states. Does not reset the environment state after
        finishing. Also sanity checks the motion plan to ensure each joint
        state is truely collision free.
        """
        all_frames = []

        # Visualize the target position.
        if robot_target.ee_targ is not None:
            robo_trans = use_sim.get_robot_transform()
            use_targ_state = robo_trans.transform_point(robot_target.ee_targ)
            targ_viz_id = use_sim.add_sphere(0.03, color=[0, 0, 1, 1])
            use_sim.set_position(use_targ_state, targ_viz_id)
        else:
            targ_viz_id = None

        all_ee_pos = []
        for i, js in enumerate(joint_plan):
            use_sim.set_arm_pos(js)
            all_ee_pos.append(use_sim.get_ee_pos())
            if self._ee_margin is not None:
                use_sim.set_position(
                    self._use_sim.get_ee_pos(), self._sphere_id
                )
            did_collide = not self._is_state_valid(js, True)
            if did_collide and self._should_render:
                # If this executes this means a state the motion planner
                # planned was not valid. This means a problem with the
                # simulator.
                print(
                    "Is state valid more?",
                    [self._is_state_valid(js) for _ in range(5)],
                )
                print("FAILED FAILED FAILED")
                self.was_bad_coll = True

            pic = self.photo("", f"{i}_{self._num_calls}", should_save=False)
            if did_collide:
                pic = make_border_red(pic)
            all_frames.append(pic)

        if targ_viz_id is not None:
            use_sim.remove_object(targ_viz_id)
            dist_to_goal = np.linalg.norm(
                use_targ_state - use_sim.get_ee_pos()
            )
        else:
            dist_to_goal = -1.0

        save_dir = osp.join(self._run_cfg.VIDEO_DIR, "mp_plans")
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        mp_name = "ep%s_%i_%.3f" % (
            self._sim.ep_info["episode_id"],
            self._num_calls,
            dist_to_goal,
        )
        save_video(osp.join(save_dir, mp_name + ".mp4"), all_frames, fps=5.0)

    def set_plan_ignore_obj(self, obj_id):
        self._reach_for_obj = obj_id

    def _get_sim(self):
        """
        The two different simulators used for planning.
        """
        if self._config.MP_SIM_TYPE == "Priv":
            return HabMpSim(self._sim)
        else:
            raise ValueError("Unrecognized simulator type")

    def _check_ee_coll(self, ee_margin, sphere_id, coll_details):
        if ee_margin is not None:
            obj_id = self.hold_id

            if obj_id is None:
                obj_id = self._reach_for_obj

            any_match = any([sphere_id in x for x in coll_details.all_colls])
            if any_match:
                return False

        return True

    def _get_path(
        self, is_state_valid, start_js, robot_targ, use_sim, mp_space, timeout
    ):
        """
        Does the low-level path planning with OMPL.
        """
        if not self._should_render:
            ou.setLogLevel(ou.LOG_ERROR)
        dim = mp_space.get_state_dim()
        space = ob.RealVectorStateSpace(dim)
        bounds = ob.RealVectorBounds(dim)

        lims = mp_space.get_state_lims()
        for i, lim in enumerate(lims):
            bounds.setLow(i, lim[0])
            bounds.setHigh(i, lim[1])
        space.setBounds(bounds)

        si = ob.SpaceInformation(space)
        si.setStateValidityChecker(ob.StateValidityCheckerFn(is_state_valid))
        si.setup()

        pdef = ob.ProblemDefinition(si)
        mp_space.set_problem(
            pdef, space, si, start_js, robot_targ, is_state_valid
        )

        planner = mp_space.get_planner(si)
        planner.setProblemDefinition(pdef)
        planner.setup()
        if mp_space.get_range() is not None:
            planner.setRange(mp_space.get_range())

        solved = planner.solve(timeout)

        if not solved:
            self._log("Could not find plan")
            return None
        objective = pdef.getOptimizationObjective()
        if objective is not None:
            cost = (
                pdef.getSolutionPath()
                .cost(pdef.getOptimizationObjective())
                .value()
            )
        else:
            cost = np.inf

        self._log(
            "Got a path of length %.2f and cost %.2f"
            % (pdef.getSolutionPath().length(), cost)
        )

        path = pdef.getSolutionPath()
        joint_plan = mp_space.convert_sol(path)
        self._is_approx_sol = pdef.hasApproximateSolution()

        return joint_plan
