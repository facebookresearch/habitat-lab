#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
See README.md in this directory.
"""

import abc
import json
from typing import Any, Callable, Dict, List, Optional

import magnum as mn

import habitat
import habitat_sim
import habitat_sim.utils.settings
from habitat_hitl._internal.networking.interprocess_record import (
    InterprocessRecord,
)
from habitat_hitl._internal.networking.keyframe_utils import get_empty_keyframe
from habitat_hitl._internal.networking.networking_process import (
    launch_networking_process,
    terminate_networking_process,
)
from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.app_states.app_state_abc import AppState
from habitat_hitl.core.client_message_manager import ClientMessageManager
from habitat_hitl.core.gui_drawer import GuiDrawer
from habitat_hitl.core.gui_input import GuiInput
from habitat_hitl.core.hydra_utils import omegaconf_to_object
from habitat_hitl.core.remote_client_state import RemoteClientState
from habitat_hitl.core.serialize_utils import BaseRecorder, save_as_gzip
from habitat_hitl.core.text_drawer import AbstractTextDrawer
from habitat_hitl.core.types import KeyframeAndMessages
from habitat_hitl.core.user_mask import Users
from habitat_sim.gfx import DebugLineRender


# todo: define AppDriver in one place
class AppDriver:
    # todo: rename to just "update"?
    @abc.abstractmethod
    def sim_update(self, dt):
        pass


class HitlBareSimDriver(AppDriver):
    def __init__(
        self,
        config,
        gui_input: GuiInput,
        line_render: Optional[DebugLineRender],
        text_drawer: AbstractTextDrawer,
        create_app_state_lambda: Callable,
    ):
        if "habitat_hitl" not in config:
            raise RuntimeError(
                "Required parameter 'habitat_hitl' not found in config. See hitl_defaults.yaml."
            )
        self._hitl_config = omegaconf_to_object(config.habitat_hitl)
        self._gui_input = gui_input

        self.habitat_env: habitat.Env = None

        # todo: construct a sim with no renderer
        # hab_cfg = config.habitat.simulator
        cfg_settings = habitat_sim.utils.settings.default_sim_settings.copy()
        # keyword "NONE" initializes a scene with no scene mesh
        cfg_settings["scene"] = "NONE"
        cfg_settings[
            "scene_dataset_config_file"
        ] = "data/fpss/hssd-hab-siro.scene_dataset_config.json"
        cfg_settings["scene"] = "NONE"  # "102344022.scene_instance.json"
        cfg_settings["depth_sensor"] = False
        cfg_settings["color_sensor"] = False
        hab_cfg = habitat_sim.utils.settings.make_cfg(cfg_settings)
        # required for HITL apps
        hab_cfg.sim_cfg.enable_gfx_replay_save = True
        self._bare_sim = habitat_sim.Simulator(hab_cfg)

        assert self.get_sim().renderer is None

        data_collection_config = self._hitl_config.data_collection

        # There is no episode record when using HitlBareSimDriver
        assert not data_collection_config.save_episode_record

        # not yet supported
        assert not data_collection_config.save_gfx_replay_keyframes
        # assert data_collection_config.save_filepath_base is None

        self._save_filepath_base = data_collection_config.save_filepath_base
        self._save_episode_record = data_collection_config.save_episode_record
        self._step_recorder: BaseRecorder = None
        self._episode_recorder_dict = None

        self._save_gfx_replay_keyframes: bool = (
            data_collection_config.save_gfx_replay_keyframes
        )
        self._recording_keyframes: List[str] = []

        # unsupported for HitlBareSimDriver
        assert self._hitl_config.disable_policies_and_stepping
        self.ctrl_helper = None

        self._debug_images = self._hitl_config.debug_images

        self._viz_anim_fraction: float = 0.0
        self._pending_cursor_style: Optional[Any] = None

        self._episode_helper = None

        # gui_drawer = GuiDrawer(debug_line_drawer, self._client_message_manager)
        # gui_drawer.set_line_width(self._hitl_config.debug_line_width)
        if line_render:
            line_render.set_line_width(self._hitl_config.debug_line_width)

        # self._check_init_server(line_render)
        # Create a user container.
        # In local mode, there is always 1 active user.
        # In remote mode, use 'activate_user()' and 'deactivate_user()' when handling connections.
        users = Users(
            max_user_count=max(
                self._hitl_config.networking.max_client_count, 1
            ),
            activate_users=not self._hitl_config.networking.enable,
        )

        self._client_message_manager = None
        if self.network_server_enabled:
            self._client_message_manager = ClientMessageManager(users)

        gui_drawer = GuiDrawer(line_render, self._client_message_manager)
        gui_drawer.set_line_width(self._hitl_config.debug_line_width)

        self._check_init_server(gui_drawer, gui_input, users)

        # TODO: Dependency injection
        text_drawer._client_message_manager = self._client_message_manager

        gui_agent_controller: Any = (
            self.ctrl_helper.get_gui_agent_controller()
            if self.ctrl_helper
            else None
        )

        self._app_service = AppService(
            config=config,
            hitl_config=self._hitl_config,
            gui_input=gui_input,
            remote_client_state=self._remote_client_state,
            gui_drawer=gui_drawer,
            text_drawer=text_drawer,
            get_anim_fraction=lambda: self._viz_anim_fraction,
            env=self.habitat_env,
            sim=self.get_sim(),
            reconfigure_sim=lambda dataset, scene: self._reconfigure_sim(
                dataset, scene
            ),
            compute_action_and_step_env=None,
            step_recorder=self._step_recorder,
            get_metrics=None,
            end_episode=None,
            set_cursor_style=self._set_cursor_style,
            episode_helper=self._episode_helper,
            client_message_manager=self._client_message_manager,
            gui_agent_controllers=[gui_agent_controller],
            all_agent_controllers=[gui_agent_controller],
            ui_manager=None,
            users=users,
        )

        self._app_state: AppState = None
        assert create_app_state_lambda is not None
        self._app_state = create_app_state_lambda(self._app_service)

        # Limit the number of float decimals in JSON transmissions
        if hasattr(
            self.get_sim().gfx_replay_manager, "set_max_decimal_places"
        ):
            self.get_sim().gfx_replay_manager.set_max_decimal_places(4)

        self._reset_environment()

    def close(self):
        self._check_terminate_server()

    @property
    def network_server_enabled(self) -> bool:
        return self._hitl_config.networking.enable

    def _reconfigure_sim(self, dataset, scene):
        # todo: construct a sim with no renderer
        # hab_cfg = config.habitat.simulator
        cfg_settings = habitat_sim.utils.settings.default_sim_settings.copy()
        # keyword "NONE" initializes a scene with no scene mesh
        cfg_settings["scene"] = scene if scene else "NONE"
        cfg_settings["scene_dataset_config_file"] = (
            dataset if dataset else None
        )
        cfg_settings["depth_sensor"] = False
        cfg_settings["color_sensor"] = False
        hab_cfg = habitat_sim.utils.settings.make_cfg(cfg_settings)
        # required for HITL apps
        hab_cfg.sim_cfg.enable_gfx_replay_save = True
        self._bare_sim.reconfigure(hab_cfg)

        assert self.get_sim().renderer is None

    def _check_init_server(
        self, gui_drawer: GuiDrawer, server_gui_input: GuiInput, users: Users
    ):
        self._remote_client_state = None
        self._interprocess_record = None
        if self.network_server_enabled:
            self._interprocess_record = InterprocessRecord(
                self._hitl_config.networking
            )
            launch_networking_process(self._interprocess_record)
            self._remote_client_state = RemoteClientState(
                hitl_config=self._hitl_config,
                client_message_manager=self._client_message_manager,
                interprocess_record=self._interprocess_record,
                gui_drawer=gui_drawer,
                users=users,
            )
            # Bind the server input to user 0
            if self._hitl_config.networking.client_sync.server_input:
                self._remote_client_state.bind_gui_input(server_gui_input, 0)

    def _check_terminate_server(self):
        if self.network_server_enabled:
            terminate_networking_process()

    def _reset_environment(self):
        if self.network_server_enabled:
            self._remote_client_state.clear_history()

        self._app_state.on_environment_reset(self._episode_recorder_dict)

        self.get_sim().gfx_replay_manager.save_keyframe()

    # trying to get around mypy complaints about missing sim attributes
    def get_sim(self) -> Any:
        return self._bare_sim

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

    def _set_cursor_style(self, cursor_style):
        self._pending_cursor_style = cursor_style

    def sim_update(self, dt):
        post_sim_update_dict: Dict[str, Any] = {}

        if self._remote_client_state:
            self._remote_client_state.update()

        # _viz_anim_fraction goes from 0 to 1 over time and then resets to 0
        self._viz_anim_fraction = (
            self._viz_anim_fraction
            + dt * self._hitl_config.viz_animation_speed
        ) % 1.0

        self._app_state.sim_update(dt, post_sim_update_dict)

        # sloppy: we aren't currently stepping the sim for our current use case (gfx_replay_viewer), so we manually save a keyframe here after every app_state.sim_update (on the assumption that the app state updated the scene). Future use cases of HitlBareSimDriver may require stepping the sim (e.g. viewing a live physics simulation). We'll need to decide who is responsible for stepping the simulator (HitlBareSimDriver or the AppState). And we'll need to update this line of code. The equivalent logic in HitlDriver is AppService.compute_action_and_step_env, which is called by AppStates but implemented by the driver.
        self.get_sim().gfx_replay_manager.save_keyframe()

        if self._pending_cursor_style:
            post_sim_update_dict[
                "application_cursor"
            ] = self._pending_cursor_style
            self._pending_cursor_style = None

        keyframes: List[
            str
        ] = (
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

        if self._remote_client_state:
            self._remote_client_state.on_frame_end()

        if self.network_server_enabled:
            if (
                self._hitl_config.networking.client_sync.server_camera
                and "cam_transform" in post_sim_update_dict
            ):
                cam_transform: Optional[mn.Matrix4] = post_sim_update_dict[
                    "cam_transform"
                ]
                if cam_transform is not None:
                    self._client_message_manager.update_camera_transform(
                        cam_transform
                    )

            self._remote_client_state.on_frame_end()
            self._send_keyframes(keyframes)

        return post_sim_update_dict

    def _send_keyframes(self, keyframes_json: List[str]):
        assert self.network_server_enabled

        keyframes = []
        for keyframe_json in keyframes_json:
            obj = json.loads(keyframe_json)
            assert "keyframe" in obj
            keyframe_obj = obj["keyframe"]
            keyframes.append(keyframe_obj)

        # If messages need to be sent, but no keyframe is available, create an empty keyframe.
        if self._client_message_manager.any_message() and len(keyframes) == 0:
            keyframes.append(get_empty_keyframe())

        for keyframe in keyframes:
            # Remove rigs from keyframe if skinning is disabled.
            if not self._hitl_config.networking.client_sync.skinning:
                if "rigCreations" in keyframe:
                    del keyframe["rigCreations"]
                if "rigUpdates" in keyframe:
                    del keyframe["rigUpdates"]
            # Insert server->client message into the keyframe.
            messages = self._client_message_manager.get_messages()
            self._client_message_manager.clear_messages()
            # Send the keyframe.
            self._interprocess_record.send_keyframe_to_networking_thread(
                KeyframeAndMessages(keyframe, messages)
            )
