#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
See README.md in this directory.
"""

import abc
import json
from datetime import datetime
from functools import wraps
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np

import habitat
import habitat.gym
import habitat.tasks.rearrange.rearrange_task
import habitat_sim
from habitat_hitl._internal.networking.interprocess_record import (
    InterprocessRecord,
)
from habitat_hitl._internal.networking.networking_process import (
    launch_networking_process,
    terminate_networking_process,
)
from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.app_states.app_state_abc import AppState
from habitat_hitl.core.client_message_manager import ClientMessageManager
from habitat_hitl.core.hydra_utils import omegaconf_to_object
from habitat_hitl.core.remote_gui_input import RemoteGuiInput
from habitat_hitl.core.serialize_utils import (
    BaseRecorder,
    NullRecorder,
    StepRecorder,
    save_as_gzip,
    save_as_json_gzip,
    save_as_pickle_gzip,
)
from habitat_hitl.environment.controllers.controller_helper import (
    ControllerHelper,
)
from habitat_hitl.environment.episode_helper import EpisodeHelper

if TYPE_CHECKING:
    from habitat.core.environments import GymHabitatEnv


def requires_habitat_sim_with_bullet(callable_):
    @wraps(callable_)
    def wrapper(*args, **kwds):
        assert (
            habitat_sim.built_with_bullet
        ), f"Habitat-sim is built without bullet, but {callable_.__name__} requires Habitat-sim with bullet."
        return callable_(*args, **kwds)

    return wrapper


class AppDriver:
    # todo: rename to just "update"?
    @abc.abstractmethod
    def sim_update(self, dt):
        pass


@requires_habitat_sim_with_bullet
class HitlDriver(AppDriver):
    def __init__(
        self,
        config,
        gui_input,
        line_render,
        text_drawer,
        create_app_state_lambda,
    ):
        if "habitat_hitl" not in config:
            raise RuntimeError(
                "Required parameter 'habitat_hitl' not found in config. See hitl_defaults.yaml."
            )
        self._hitl_config = omegaconf_to_object(config.habitat_hitl)
        self._dataset_config = config.habitat.dataset
        self._play_episodes_filter_str = self._hitl_config.episodes_filter
        self._num_recorded_episodes = 0
        if (
            not self._hitl_config.experimental.headless
            and gui_input.is_stub_implementation
        ):
            raise RuntimeError(
                "HitlDriver with experimental.headless=False requires a non-stub-implementation GuiInput."
            )
        self._gui_input = gui_input

        line_render.set_line_width(self._hitl_config.debug_line_width)

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

        if self._hitl_config.gui_controlled_agent.agent_index is not None:
            sim_config = config.habitat.simulator
            gui_agent_key = sim_config.agents_order[
                self._hitl_config.gui_controlled_agent.agent_index
            ]
            oracle_nav_sensor_key = f"{gui_agent_key}_has_finished_oracle_nav"
            if (
                oracle_nav_sensor_key
                in self.habitat_env.task.sensor_suite.sensors
            ):
                del self.habitat_env.task.sensor_suite.sensors[
                    oracle_nav_sensor_key
                ]

        data_collection_config = self._hitl_config.data_collection
        if (
            data_collection_config.save_gfx_replay_keyframes
            or data_collection_config.save_episode_record
        ) and not data_collection_config.save_filepath_base:
            raise ValueError(
                "data_collection.save_gfx_replay_keyframes and/or data_collection.save_episode_record are enabled, "
                "but data_collection.save_filepath_base is not set."
            )

        self._save_filepath_base = data_collection_config.save_filepath_base
        self._save_episode_record = data_collection_config.save_episode_record
        self._step_recorder: BaseRecorder = (
            StepRecorder() if self._save_episode_record else NullRecorder()  # type: ignore
        )
        self._episode_recorder_dict = None

        self._save_gfx_replay_keyframes: bool = (
            data_collection_config.save_gfx_replay_keyframes
        )
        self._recording_keyframes: List[str] = []

        if not self._hitl_config.disable_policies_and_stepping:
            self.ctrl_helper = ControllerHelper(
                gym_habitat_env=self.gym_habitat_env,
                config=config,
                hitl_config=self._hitl_config,
                gui_input=gui_input,
                recorder=self._step_recorder,
            )
        else:
            self.ctrl_helper = None

        self._debug_images = self._hitl_config.debug_images

        self._viz_anim_fraction: float = 0.0
        self._pending_cursor_style = None

        self._episode_helper = EpisodeHelper(self.habitat_env)

        self._check_init_server(line_render)

        def local_end_episode(do_reset=False):
            self._end_episode(do_reset)

        self._client_message_manager = None
        if self.network_server_enabled:
            self._client_message_manager = ClientMessageManager()

        gui_agent_controller: Any = (
            self.ctrl_helper.get_gui_agent_controller()
            if self.ctrl_helper
            else None
        )

        self._app_service = AppService(
            config=config,
            hitl_config=self._hitl_config,
            gui_input=gui_input,
            remote_gui_input=self._remote_gui_input,
            line_render=line_render,
            text_drawer=text_drawer,
            get_anim_fraction=lambda: self._viz_anim_fraction,
            env=self.habitat_env,
            sim=self.get_sim(),
            reconfigure_sim=None,  # only for HitlBareSimDriver
            compute_action_and_step_env=lambda: self._compute_action_and_step_env(),
            step_recorder=self._step_recorder,
            get_metrics=lambda: self._get_recent_metrics(),
            end_episode=local_end_episode,
            set_cursor_style=self._set_cursor_style,
            episode_helper=self._episode_helper,
            client_message_manager=self._client_message_manager,
            gui_agent_controller=gui_agent_controller,
        )

        self._app_state: AppState = None
        assert create_app_state_lambda is not None
        self._app_state = create_app_state_lambda(self._app_service)

        self._reset_environment()

    def close(self):
        self._check_terminate_server()

    @property
    def network_server_enabled(self) -> bool:
        return self._hitl_config.networking.enable

    def _check_init_server(self, line_render):
        self._remote_gui_input = None
        self._interprocess_record = None
        if self.network_server_enabled:
            # How many frames we can simulate "ahead" of what keyframes have been sent.
            # A larger value increases lag on the client, while ensuring a more reliable
            # simulation rate in the presence of unreliable network comms.
            # See also server.py max_send_rate
            max_steps_ahead = 5
            self._interprocess_record = InterprocessRecord(
                self._hitl_config.networking, max_steps_ahead
            )
            launch_networking_process(self._interprocess_record)
            self._remote_gui_input = RemoteGuiInput(
                self._interprocess_record, line_render
            )

    def _check_terminate_server(self):
        if self.network_server_enabled:
            terminate_networking_process()

    def _make_dataset(self, config):
        from habitat.datasets import make_dataset  # type: ignore

        dataset_config = config.habitat.dataset
        dataset = make_dataset(
            id_dataset=dataset_config.type, config=dataset_config
        )
        if self._play_episodes_filter_str is not None:
            self._play_episodes_filter_str = str(
                self._play_episodes_filter_str
            )
            max_num_digits: int = len(str(len(dataset.episodes)))

            def get_play_episodes_ids(play_episodes_filter_str):
                play_episodes_ids: List[str] = []
                for ep_filter_str in play_episodes_filter_str.split(" "):
                    if ":" in ep_filter_str:
                        range_params = map(int, ep_filter_str.split(":"))
                        play_episodes_ids.extend(
                            episode_id.zfill(max_num_digits)
                            for episode_id in map(str, range(*range_params))
                        )
                    else:
                        episode_id = ep_filter_str
                        play_episodes_ids.append(
                            episode_id.zfill(max_num_digits)
                        )

                return play_episodes_ids

            play_episodes_ids_list = get_play_episodes_ids(
                self._play_episodes_filter_str
            )

            dataset.episodes = [
                ep
                for ep in dataset.episodes
                if ep.episode_id.zfill(max_num_digits)
                in play_episodes_ids_list
            ]

            dataset.episodes.sort(
                key=lambda x: play_episodes_ids_list.index(
                    x.episode_id.zfill(max_num_digits)
                )
            )

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
        if self._hitl_config.disable_policies_and_stepping:
            return

        action = self.ctrl_helper.update(self._obs)
        self._env_step(action)

        if self._save_episode_record:
            self._record_action(action)
            self._app_state.record_state()
            self._record_metrics(self._get_recent_metrics())
            self._step_recorder.finish_step()  # type: ignore

    def _find_episode_save_filepath_base(self):
        retval = (
            self._save_filepath_base + "." + str(self._num_recorded_episodes)
        )
        return retval

    def _save_episode_recorder_dict(self):
        if not len(self._step_recorder._steps):  # type: ignore
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

        self._step_recorder.reset()  # type: ignore
        ep_dict["steps"] = self._step_recorder._steps  # type: ignore

        self._episode_recorder_dict = ep_dict

    def _reset_environment(self):
        self._obs, self._metrics = self.gym_habitat_env.reset(return_info=True)

        if self.network_server_enabled:
            self._remote_gui_input.clear_history()

        # todo: fix duplicate calls to self.ctrl_helper.on_environment_reset() here
        if self.ctrl_helper:
            self.ctrl_helper.on_environment_reset()

        if self._save_episode_record:
            self._reset_episode_recorder()

        self._app_state.on_environment_reset(self._episode_recorder_dict)

        # hack: we have to reset controllers after AppState reset in case AppState reset overrides the start pose of agents
        # The reason is that the controller would need the latest agent's trans info, and we do agent init location in app reset
        if self.ctrl_helper:
            self.ctrl_helper.on_environment_reset()

        if self._hitl_config.disable_policies_and_stepping:
            # we need to manually save a keyframe since the Habitat env only does this
            # after an env step.
            self.get_sim().gfx_replay_manager.save_keyframe()

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

        if self._remote_gui_input:
            self._remote_gui_input.update()

        # _viz_anim_fraction goes from 0 to 1 over time and then resets to 0
        self._viz_anim_fraction = (
            self._viz_anim_fraction
            + dt * self._hitl_config.viz_animation_speed
        ) % 1.0

        self._app_state.sim_update(dt, post_sim_update_dict)

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

        if self._hitl_config.hide_humanoid_in_gui:
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

        if self._remote_gui_input:
            self._remote_gui_input.on_frame_end()

        if self.network_server_enabled:
            for keyframe_json in keyframes:
                obj = json.loads(keyframe_json)
                assert "keyframe" in obj
                keyframe_obj = obj["keyframe"]
                # Insert server->client message into the keyframe
                message = self._client_message_manager.get_message_dict()
                if len(message) > 0:
                    keyframe_obj["message"] = message
                    self._client_message_manager.clear_message_dict()
                # Send the keyframe
                self._interprocess_record.send_keyframe_to_networking_thread(
                    keyframe_obj
                )

        return post_sim_update_dict
