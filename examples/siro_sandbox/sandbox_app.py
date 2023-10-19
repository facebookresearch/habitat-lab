#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
See README.md in this directory.
"""

import ctypes

# must call this before importing habitat or magnum! avoids EGL_BAD_ACCESS error on some platforms
import sys

flags = sys.getdlopenflags()
sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)

import argparse
from datetime import datetime
from functools import wraps
from typing import TYPE_CHECKING, Any, Dict, List, Set

import magnum as mn
import numpy as np
from controllers.controller_helper import ControllerHelper
from episode_helper import EpisodeHelper

# from hitl_tutorial import Tutorial, generate_tutorial
from magnum.platform.glfw import Application
from utils.gui.gui_application import GuiAppDriver, GuiApplication
from utils.gui.gui_input import GuiInput
from utils.gui.replay_gui_app_renderer import ReplayGuiAppRenderer
from utils.serialize_utils import (
    NullRecorder,
    StepRecorder,
    save_as_gzip,
    save_as_json_gzip,
    save_as_pickle_gzip,
)

import habitat
import habitat.gym
import habitat.tasks.rearrange.rearrange_task
import habitat_sim
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    HumanoidJointActionConfig,
    ThirdRGBSensorConfig,
)
from habitat_baselines.config.default import get_config as get_baselines_config

if TYPE_CHECKING:
    from habitat.core.environments import GymHabitatEnv

from app_states.app_state_abc import AppState
from app_states.app_state_fetch import AppStateFetch
from app_states.app_state_free_camera import AppStateFreeCamera
from app_states.app_state_rearrange import AppStateRearrange
from app_states.app_state_socialnav import AppStateSocialNav
from app_states.app_state_tutorial import AppStateTutorial
from sandbox_service import SandboxService

# Please reach out to the paper authors to obtain this file
DEFAULT_POSE_PATH = (
    "data/humanoids/humanoid_data/walking_motion_processed_smplx.pkl"
)

DEFAULT_CFG = "experiments_hab3/pop_play_kinematic_oracle_humanoid_spot.yaml"


def requires_habitat_sim_with_bullet(callable_):
    @wraps(callable_)
    def wrapper(*args, **kwds):
        assert (
            habitat_sim.built_with_bullet
        ), f"Habitat-sim is built without bullet, but {callable_.__name__} requires Habitat-sim with bullet."
        return callable_(*args, **kwds)

    return wrapper


@requires_habitat_sim_with_bullet
class SandboxDriver(GuiAppDriver):
    def __init__(self, args, config, gui_input, line_render, text_drawer):
        self._dataset_config = config.habitat.dataset
        self._play_episodes_filter_str = args.episodes_filter
        self._num_recorded_episodes = 0
        self._args = args
        self._gui_input = gui_input

        line_render.set_line_width(3)

        with habitat.config.read_write(config):  # type: ignore
            # needed so we can provide keyframes to GuiApplication
            config.habitat.simulator.habitat_sim_v0.enable_gfx_replay_save = (
                True
            )
            config.habitat.simulator.concur_render = False

        dataset = self._make_dataset(config=config)
        self.gym_habitat_env: "GymHabitatEnv" = (
            habitat.gym.make_gym_from_config(config=config, dataset=dataset)
        )
        self.habitat_env: habitat.Env = (  # type: ignore
            self.gym_habitat_env.unwrapped.habitat_env
        )

        if args.gui_controlled_agent_index is not None:
            sim_config = config.habitat.simulator
            gui_agent_key = sim_config.agents_order[
                args.gui_controlled_agent_index
            ]
            oracle_nav_sensor_key = f"{gui_agent_key}_has_finished_oracle_nav"
            if (
                oracle_nav_sensor_key
                in self.habitat_env.task.sensor_suite.sensors
            ):
                del self.habitat_env.task.sensor_suite.sensors[
                    oracle_nav_sensor_key
                ]

        self._save_filepath_base = args.save_filepath_base
        self._save_episode_record = args.save_episode_record
        self._step_recorder = (
            StepRecorder() if self._save_episode_record else NullRecorder()
        )
        self._episode_recorder_dict = None

        self._save_gfx_replay_keyframes: bool = args.save_gfx_replay_keyframes
        self._recording_keyframes: List[str] = []

        self.ctrl_helper = ControllerHelper(
            self.gym_habitat_env, config, args, gui_input, self._step_recorder
        )

        self._debug_images = args.debug_images

        self._viz_anim_fraction = 0.0
        self._pending_cursor_style = None

        self._episode_helper = EpisodeHelper(self.habitat_env)

        def local_end_episode(do_reset=False):
            self._end_episode(do_reset)

        self._sandbox_service = SandboxService(
            args,
            config,
            gui_input,
            line_render,
            text_drawer,
            lambda: self._viz_anim_fraction,
            self.habitat_env,
            self.get_sim(),
            lambda: self._compute_action_and_step_env(),
            self._step_recorder,
            lambda: self._get_recent_metrics(),
            local_end_episode,
            lambda: self._set_cursor_style,
            self._episode_helper,
        )

        self._app_states: List[AppState]
        if args.app_state == "fetch":
            self._app_states = [
                AppStateFetch(
                    self._sandbox_service,
                    self.ctrl_helper.get_gui_agent_controller(),
                )
            ]
        elif args.app_state == "rearrange":
            self._app_states = [
                AppStateRearrange(
                    self._sandbox_service,
                    self.ctrl_helper.get_gui_agent_controller(),
                )
            ]
            if args.show_tutorial:
                self._app_states.insert(
                    0,
                    AppStateTutorial(
                        self._sandbox_service,
                        self.ctrl_helper.get_gui_agent_controller(),
                    ),
                )
        elif args.app_state == "socialnav":
            self._app_states = [
                AppStateSocialNav(
                    self._sandbox_service,
                    self.ctrl_helper.get_gui_agent_controller(),
                )
            ]
        elif args.app_state == "free_camera":
            self._app_states = [AppStateFreeCamera(self._sandbox_service)]
        else:
            raise RuntimeError("Unexpected --app-state=", args.app_state)
        # Note that we expect SandboxDriver to create multiple AppStates in some
        # situations and manage the transition between them, e.g. tutorial -> rearrange.

        assert self._app_states
        self._app_state_index = None
        self._app_state = None

        self._reset_environment()

    def _make_dataset(self, config):
        from habitat.datasets import make_dataset  # type: ignore

        dataset_config = config.habitat.dataset
        dataset = make_dataset(
            id_dataset=dataset_config.type, config=dataset_config
        )

        if self._play_episodes_filter_str is not None:
            max_num_digits: int = len(str(len(dataset.episodes)))

            def get_play_episodes_ids(play_episodes_filter_str):
                play_episodes_ids: Set[str] = set()
                for ep_filter_str in play_episodes_filter_str.split(" "):
                    if ":" in ep_filter_str:
                        range_params = map(int, ep_filter_str.split(":"))
                        play_episodes_ids.update(
                            episode_id.zfill(max_num_digits)
                            for episode_id in map(str, range(*range_params))
                        )
                    else:
                        episode_id = ep_filter_str
                        play_episodes_ids.add(episode_id.zfill(max_num_digits))

                return play_episodes_ids

            play_episodes_ids_set: Set[str] = get_play_episodes_ids(
                self._play_episodes_filter_str
            )
            dataset.episodes = [
                ep
                for ep in dataset.episodes
                if ep.episode_id.zfill(max_num_digits) in play_episodes_ids_set
            ]

        return dataset

    def _get_recent_metrics(self):
        assert self._metrics
        return self._metrics

    def _env_step(self, action):
        (
            self._obs,
            reward,
            done,
            self._metrics,
        ) = self.gym_habitat_env.step(action)

    def _compute_action_and_step_env(self):
        action = self.ctrl_helper.update(self._obs)
        self._env_step(action)

        if self._save_episode_record:
            self._record_action(action)
            self._app_state.record_state()
            self._record_metrics(self._get_recent_metrics())
            self._step_recorder.finish_step()

    def _find_episode_save_filepath_base(self):
        retval = (
            self._save_filepath_base + "." + str(self._num_recorded_episodes)
        )
        return retval

    def _save_episode_recorder_dict(self):
        if not len(self._step_recorder._steps):
            return

        filepath_base = self._find_episode_save_filepath_base()

        json_filepath = filepath_base + ".json.gz"
        save_as_json_gzip(self._episode_recorder_dict, json_filepath)

        pkl_filepath = filepath_base + ".pkl.gz"
        save_as_pickle_gzip(self._episode_recorder_dict, pkl_filepath)

    def _reset_episode_recorder(self):
        assert self._step_recorder
        ep_dict: Any = dict()
        ep_dict["start_time"] = datetime.now()
        ep_dict["dataset"] = self._dataset_config
        ep_dict["scene_id"] = self._episode_helper.current_episode.scene_id
        ep_dict["episode_id"] = self._episode_helper.current_episode.episode_id

        self._step_recorder.reset()
        ep_dict["steps"] = self._step_recorder._steps

        self._episode_recorder_dict = ep_dict

    def _get_prev_app_state(self):
        return (
            self._app_states[self._app_state_index - 1]
            if self._app_state_index > 0
            else None
        )

    def _get_next_app_state(self):
        return (
            self._app_states[self._app_state_index + 1]
            if self._app_state_index < len(self._app_states) - 1
            else None
        )

    def _reset_environment(self):
        self._obs, self._metrics = self.gym_habitat_env.reset(return_info=True)

        self.ctrl_helper.on_environment_reset()

        if self._save_episode_record:
            self._reset_episode_recorder()

        # Reset all the app states
        for app_state in self._app_states:
            app_state.on_environment_reset(self._episode_recorder_dict)

        self._app_state_index = (
            0  # start from the first app state for each episode
        )
        self._app_state = self._app_states[self._app_state_index]
        self._app_state.on_enter(
            prev_state=self._get_prev_app_state(),
            next_state=self._get_next_app_state(),
        )

    def _check_save_episode_data(self, session_ended):
        saved_keyframes, saved_episode_data = False, False
        if self._save_gfx_replay_keyframes and session_ended:
            assert self._save_filepath_base
            self._save_recorded_keyframes_to_file()
            saved_keyframes = True
        if self._save_episode_record:
            assert self._save_filepath_base
            self._save_episode_recorder_dict()
            saved_episode_data = True

        if saved_keyframes or saved_episode_data:
            self._num_recorded_episodes += 1

    # trying to get around mypy complaints about missing sim attributes
    def get_sim(self) -> Any:
        return self.habitat_env.task._sim

    def _end_episode(self, do_reset=False):
        self._check_save_episode_data(session_ended=do_reset == False)
        if do_reset and self._episode_helper.next_episode_exists():
            self._reset_environment()

        self._episode_helper.increment_done_episode_counter()

    def _save_recorded_keyframes_to_file(self):
        if not self._recording_keyframes:
            return

        # Consolidate recorded keyframes into a single json string
        # self._recording_keyframes format:
        #     List['{"keyframe":{...}', '{"keyframe":{...}',...]
        # Output format:
        #     '{"keyframes":[{...},{...},...]}'
        json_keyframes = ",".join(
            keyframe[12:-1] for keyframe in self._recording_keyframes
        )
        json_content = '{{"keyframes":[{}]}}'.format(json_keyframes)

        # Save keyframes to file
        filepath = self._save_filepath_base + ".gfx_replay.json.gz"
        save_as_gzip(json_content.encode("utf-8"), filepath)

    def _record_action(self, action):
        if not isinstance(action, np.ndarray):
            action_args = action["action_args"]

            # These are large arrays and they massively bloat the record file size, so
            # let's exclude them.
            keys_to_clear = [
                "human_joints_trans",
                "agent_0_human_joints_trans",
                "agent_1_human_joints_trans",
            ]
            for key in keys_to_clear:
                if key in action_args:
                    action_args[key] = None
        else:
            # no easy way to remove the joints from the action ndarray
            pass

        self._step_recorder.record("action", action)

    def _record_metrics(self, metrics):
        # We don't want to include this.
        if "gfx_replay_keyframes_string" in metrics:
            del metrics["gfx_replay_keyframes_string"]

        self._step_recorder.record("metrics", metrics)

    def _set_cursor_style(self, cursor_style):
        self._pending_cursor_style = cursor_style

    def sim_update(self, dt):
        post_sim_update_dict: Dict[str, Any] = {}

        # _viz_anim_fraction goes from 0 to 1 over time and then resets to 0
        viz_anim_speed = 2.0
        self._viz_anim_fraction = (
            self._viz_anim_fraction + dt * viz_anim_speed
        ) % 1.0

        # Navmesh visualization only works in the debug third-person view
        # (--debug-third-person-width), not the main sandbox viewport. Navmesh
        # visualization is only implemented for simulator-rendering, not replay-
        # rendering.
        if self._gui_input.get_key_down(GuiInput.KeyNS.N):
            self.get_sim().navmesh_visualization = (  # type: ignore
                not self.get_sim().navmesh_visualization  # type: ignore
            )

        self._app_state.sim_update(dt, post_sim_update_dict)

        if self._app_state.is_app_state_done():
            self._try_next_state()

        if self._pending_cursor_style:
            post_sim_update_dict[
                "application_cursor"
            ] = self._pending_cursor_style
            self._pending_cursor_style = None

        keyframes = (
            self.get_sim().gfx_replay_manager.write_incremental_saved_keyframes_to_string_array()
        )

        if self._save_gfx_replay_keyframes:
            for keyframe in keyframes:
                self._recording_keyframes.append(keyframe)

        # Manually save recorded gfx-replay keyframes.
        if (
            self._gui_input.get_key_down(GuiInput.KeyNS.EQUAL)
            and self._save_gfx_replay_keyframes
        ):
            self._save_recorded_keyframes_to_file()

        if self._args.hide_humanoid_in_gui:
            # Hack to hide skinned humanoids in the GUI viewport. Specifically, this
            # hides all render instances with a filepath starting with
            # "data/humanoids/humanoid_data", by replacing with an invalid filepath.
            # Gfx-replay playback logic will print a warning to the terminal and then
            # not attempt to render the instance. This is a temp hack until
            # skinning is supported in gfx-replay.
            for i in range(len(keyframes)):
                keyframes[i] = keyframes[i].replace(
                    '"creation":{"filepath":"data/humanoids/humanoid_data',
                    '"creation":{"filepath":"invalid_filepath',
                )

        post_sim_update_dict["keyframes"] = keyframes

        def depth_to_rgb(obs):
            converted_obs = np.concatenate(
                [obs * 255.0 for _ in range(3)], axis=2
            ).astype(np.uint8)
            return converted_obs

        # reference code for visualizing a camera sensor in the app GUI
        assert set(self._debug_images).issubset(set(self._obs.keys())), (
            f"Camera sensors ids: {list(set(self._debug_images).difference(set(self._obs.keys())))} "
            f"not in available sensors ids: {list(self._obs.keys())}"
        )
        debug_images = (
            depth_to_rgb(self._obs[k]) if "depth" in k else self._obs[k]
            for k in self._debug_images
        )
        post_sim_update_dict["debug_images"] = [
            np.flipud(image) for image in debug_images
        ]

        return post_sim_update_dict

    def _try_next_state(self):
        self._app_state_index += 1
        if self._app_state_index >= len(self._app_states):
            return
        self._app_state = self._app_states[self._app_state_index]
        self._app_state.on_enter(
            prev_state=self._get_prev_app_state(),
            next_state=self._get_next_app_state(),
        )


def _parse_debug_third_person(args, framebuffer_size):
    viewport_multiplier = mn.Vector2(
        framebuffer_size.x / args.width, framebuffer_size.y / args.height
    )

    do_show = args.debug_third_person_width != 0

    width = args.debug_third_person_width
    # default to square aspect ratio
    height = (
        args.debug_third_person_height
        if args.debug_third_person_height != 0
        else width
    )

    width = int(width * viewport_multiplier.x)
    height = int(height * viewport_multiplier.y)

    return do_show, width, height


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-sps",
        type=int,
        default=30,
        help="Target rate to step the environment (steps per second); actual SPS may be lower depending on your hardware",
    )
    parser.add_argument(
        "--width",
        default=1280,
        type=int,
        help="Horizontal resolution of the window.",
    )
    parser.add_argument(
        "--height",
        default=720,
        type=int,
        help="Vertical resolution of the window.",
    )
    parser.add_argument(
        "--gui-controlled-agent-index",
        type=int,
        default=None,
        help=(
            "GUI-controlled agent index (must be >= 0 and < number of agents). "
            "Defaults to None, indicating that all the agents are policy-controlled. "
            "If none of the agents is GUI-controlled, the camera is switched to 'free camera' mode "
            "that lets the user observe the scene (instead of controlling one of the agents)"
        ),
    )
    parser.add_argument(
        "--disable-inverse-kinematics",
        action="store_true",
        help="If specified, does not add the inverse kinematics end-effector control. Only relevant for a user-controlled *robot* agent.",
    )
    parser.add_argument("--cfg", type=str, default=DEFAULT_CFG)
    parser.add_argument(
        "--cfg-opts",
        nargs="*",
        default=list(),
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--debug-images",
        nargs="*",
        default=list(),
        help=(
            "Visualize camera sensors (corresponding to `--debug-images` keys) in the app GUI."
            "For example, to visualize agent1's head depth sensor set: --debug-images agent_1_head_depth"
        ),
    )
    parser.add_argument(
        "--walk-pose-path", type=str, default=DEFAULT_POSE_PATH
    )
    parser.add_argument(
        "--lin-speed",
        type=float,
        default=10.0,
        help="GUI-controlled agent's linear speed",
    )
    parser.add_argument(
        "--ang-speed",
        type=float,
        default=10.0,
        help="GUI-controlled agent's angular speed",
    )
    parser.add_argument(
        "--never-end",
        action="store_true",
        default=False,
        help="If true, make the task never end due to reaching max number of steps",
    )
    parser.add_argument(
        "--use-batch-renderer",
        action="store_true",
        default=False,
        help="Choose between classic and batch renderer",
    )
    parser.add_argument(
        "--debug-third-person-width",
        default=0,
        type=int,
        help="If specified, enable the debug third-person camera (habitat.simulator.debug_render) with specified viewport width",
    )
    parser.add_argument(
        "--debug-third-person-height",
        default=0,
        type=int,
        help="If specified, use the specified viewport height for the debug third-person camera",
    )
    parser.add_argument(
        "--max-look-up-angle",
        default=15,
        type=float,
        help="Look up angle limit.",
    )
    parser.add_argument(
        "--min-look-down-angle",
        default=-60,
        type=float,
        help="Look down angle limit.",
    )
    parser.add_argument(
        "--first-person-mode",
        action="store_true",
        default=False,
        help="Choose between classic and batch renderer",
    )
    parser.add_argument(
        "--can-grasp-place-threshold",
        default=1.2,
        type=float,
        help="Object grasp/place proximity threshold",
    )
    parser.add_argument(
        "--episodes-filter",
        default=None,
        type=str,
        help=(
            "Episodes filter in the form '0:10 12 14:20:2', "
            "where single integer number (`12` in this case) represents an episode id, "
            "colon separated integers (`0:10' and `14:20:2`) represent start:stop:step ids range."
        ),
    )
    parser.add_argument(
        "--show-tutorial",
        action="store_true",
        default=False,
        help="Shows an intro sequence that helps familiarize the user to the scene and task in a HITL context.",
    )
    parser.add_argument(
        "--hide-humanoid-in-gui",
        action="store_true",
        default=False,
        help="Hide the humanoid in the GUI viewport. Note it will still be rendered into observations fed to policies. This option is a workaround for broken skinned humanoid rendering in the GUI viewport.",
    )
    parser.add_argument(
        "--save-gfx-replay-keyframes",
        action="store_true",
        default=False,
        help="Save the gfx-replay keyframes to file. Use --save-filepath-base to specify the filepath base.",
    )
    parser.add_argument(
        "--save-episode-record",
        action="store_true",
        default=False,
        help="Save recorded episode data to file. Use --save-filepath-base to specify the filepath base.",
    )
    parser.add_argument(
        "--save-filepath-base",
        default=None,
        type=str,
        help="Filepath base used for saving various session data files. Include a full path including basename, but not an extension.",
    )
    parser.add_argument(
        "--app-state",
        default="rearrange",
        type=str,
        help="'rearrange' (default) or 'fetch'",
    )
    parser.add_argument(
        "--remote-gui-mode",
        action="store_true",
        default=False,
        help="Observer mode, where the humanoid follows the VR headset pose provided by remote_gui_input",
    )

    args = parser.parse_args()
    if (
        args.save_gfx_replay_keyframes or args.save_episode_record
    ) and not args.save_filepath_base:
        raise ValueError(
            "--save-gfx-replay-keyframes and/or --save-episode-record flags are enabled, "
            "but --save-filepath-base argument is not set. Specify filepath base for the session episode data to be saved."
        )

    if args.show_tutorial and args.app_state != "rearrange":
        raise ValueError(
            "--show-tutorial is only supported for --app-state=rearrange"
        )

    if args.remote_gui_mode and args.app_state != "fetch":
        raise ValueError(
            "--remote-gui-mode is only supported for fetch app-state"
        )

    if (
        args.app_state == "free_camera"
        and args.gui_controlled_agent_index is not None
    ):
        raise ValueError(
            "--gui-controlled-agent-index is not supported for --app-state=free_camera"
        )

    glfw_config = Application.Configuration()
    glfw_config.title = "Sandbox App"
    glfw_config.size = (args.width, args.height)
    gui_app_wrapper = GuiApplication(glfw_config, args.target_sps)
    # on Mac Retina displays, this will be 2x the window size
    framebuffer_size = gui_app_wrapper.get_framebuffer_size()

    (
        show_debug_third_person,
        debug_third_person_width,
        debug_third_person_height,
    ) = _parse_debug_third_person(args, framebuffer_size)

    viewport_rect = None
    if show_debug_third_person:
        # adjust main viewport to leave room for the debug third-person camera on the right
        assert framebuffer_size.x > debug_third_person_width
        viewport_rect = mn.Range2Di(
            mn.Vector2i(0, 0),
            mn.Vector2i(
                framebuffer_size.x - debug_third_person_width,
                framebuffer_size.y,
            ),
        )

    # note this must be created after GuiApplication due to OpenGL stuff
    app_renderer = ReplayGuiAppRenderer(
        framebuffer_size,
        viewport_rect,
        args.use_batch_renderer,
    )

    config = get_baselines_config(args.cfg, args.cfg_opts)
    with habitat.config.read_write(config):  # type: ignore
        habitat_config = config.habitat
        env_config = habitat_config.environment
        sim_config = habitat_config.simulator
        task_config = habitat_config.task
        gym_obs_keys = habitat_config.gym.obs_keys

        agent_config = get_agent_config(sim_config=sim_config)

        if show_debug_third_person:
            sim_config.debug_render = True
            agent_config.sim_sensors.update(
                {
                    "third_rgb_sensor": ThirdRGBSensorConfig(
                        height=debug_third_person_height,
                        width=debug_third_person_width,
                    )
                }
            )
            agent_key = "" if len(sim_config.agents) == 1 else "agent_0_"
            agent_sensor_name = f"{agent_key}third_rgb"
            args.debug_images.append(agent_sensor_name)
            gym_obs_keys.append(agent_sensor_name)

        # Code below is ported from interactive_play.py. I'm not sure what it is for.
        if True:
            if "pddl_success" in task_config.measurements:
                task_config.measurements.pddl_success.must_call_stop = False
            if "rearrange_nav_to_obj_success" in task_config.measurements:
                task_config.measurements.rearrange_nav_to_obj_success.must_call_stop = (
                    False
                )
            if "force_terminate" in task_config.measurements:
                task_config.measurements.force_terminate.max_accum_force = -1.0
                task_config.measurements.force_terminate.max_instant_force = (
                    -1.0
                )

        if args.never_end:
            env_config.max_episode_steps = 0

        if not args.disable_inverse_kinematics:
            if "arm_action" not in task_config.actions:
                raise ValueError(
                    "Action space does not have any arm control so cannot add inverse kinematics. Specify the `--disable-inverse-kinematics` option"
                )
            sim_config.agents.main_agent.ik_arm_urdf = (
                "./data/robots/hab_fetch/robots/fetch_onlyarm.urdf"
            )
            task_config.actions.arm_action.arm_controller = "ArmEEAction"

        if args.gui_controlled_agent_index is not None:
            # make sure gui_controlled_agent_index is valid
            if not (
                args.gui_controlled_agent_index >= 0
                and args.gui_controlled_agent_index < len(sim_config.agents)
            ):
                print(
                    f"--gui-controlled-agent-index argument value ({args.gui_controlled_agent_index}) "
                    f"must be >= 0 and < number of agents ({len(sim_config.agents)})"
                )
                exit()

            # make sure chosen articulated_agent_type is supported
            gui_agent_key = sim_config.agents_order[
                args.gui_controlled_agent_index
            ]
            if (
                sim_config.agents[gui_agent_key].articulated_agent_type
                != "KinematicHumanoid"
            ):
                print(
                    f"Selected agent for GUI control is of type {sim_config.agents[gui_agent_key].articulated_agent_type}, "
                    "but only KinematicHumanoid is supported at the moment."
                )
                exit()

            # avoid camera sensors for GUI-controlled agents
            gui_controlled_agent_config = get_agent_config(
                sim_config, agent_id=args.gui_controlled_agent_index
            )
            gui_controlled_agent_config.sim_sensors.clear()

            lab_sensor_names = ["has_finished_oracle_nav"]
            for lab_sensor_name in lab_sensor_names:
                sensor_name = (
                    lab_sensor_name
                    if len(sim_config.agents) == 1
                    else (f"{gui_agent_key}_{lab_sensor_name}")
                )
                if sensor_name in task_config.lab_sensors:
                    task_config.lab_sensors.pop(sensor_name)

            task_measurement_names = [
                "does_want_terminate",
                "bad_called_terminate",
            ]
            for task_measurement_name in task_measurement_names:
                measurement_name = (
                    task_measurement_name
                    if len(sim_config.agents) == 1
                    else (f"{gui_agent_key}_{task_measurement_name}")
                )
                if measurement_name in task_config.measurements:
                    task_config.measurements.pop(measurement_name)

            sim_sensor_names = ["head_depth", "head_rgb"]
            for sensor_name in sim_sensor_names + lab_sensor_names:
                sensor_name = (
                    sensor_name
                    if len(sim_config.agents) == 1
                    else (f"{gui_agent_key}_{sensor_name}")
                )
                if sensor_name in gym_obs_keys:
                    gym_obs_keys.remove(sensor_name)

            # use humanoidjoint_action for GUI-controlled KinematicHumanoid
            # for example, humanoid oracle-planner-based policy uses following actions:
            # base_velocity, rearrange_stop, pddl_apply_action, oracle_nav_action
            task_actions = task_config.actions
            action_prefix = (
                "" if len(sim_config.agents) == 1 else f"{gui_agent_key}_"
            )
            gui_agent_actions = [
                action_key
                for action_key in task_actions.keys()
                if action_key.startswith(action_prefix)
            ]
            for action_key in gui_agent_actions:
                task_actions.pop(action_key)

            task_actions[
                f"{action_prefix}humanoidjoint_action"
            ] = HumanoidJointActionConfig()

    driver = SandboxDriver(
        args,
        config,
        gui_app_wrapper.get_sim_input(),
        app_renderer._replay_renderer.debug_line_render(0),
        app_renderer._text_drawer,
    )

    # sanity check if there are no agents with camera sensors
    if (
        len(config.habitat.simulator.agents) == 1
        and args.gui_controlled_agent_index is not None
    ):
        assert driver.get_sim().renderer is None

    gui_app_wrapper.set_driver_and_renderer(driver, app_renderer)

    gui_app_wrapper.exec()
