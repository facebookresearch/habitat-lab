#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
See README.md in this directory.
"""

from typing import Any, Callable, Dict, List, Optional

import magnum as mn

import habitat_sim
import habitat_sim.utils.settings
from habitat_hitl._internal.app_driver import AppDriver
from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.app_states.app_state_abc import AppState
from habitat_hitl.core.gui_input import GuiInput
from habitat_hitl.core.serialize_utils import BaseRecorder, save_as_gzip
from habitat_hitl.core.text_drawer import AbstractTextDrawer
from habitat_sim.gfx import DebugLineRender


class SimDriver(AppDriver):
    def __init__(
        self,
        config,
        gui_input: GuiInput,
        line_render: Optional[DebugLineRender],
        text_drawer: AbstractTextDrawer,
        create_app_state_lambda: Callable,
    ):
        """
        HITL application driver that instantiates a `habitat-sim` simulator, without a `habitat-lab` environment.
        """
        # Initialize simulator.
        cfg_settings = habitat_sim.utils.settings.default_sim_settings.copy()
        # keyword "NONE" initializes a scene with no scene mesh
        cfg_settings["scene"] = "NONE"
        cfg_settings[
            "scene_dataset_config_file"
        ] = "data/fpss/hssd-hab-siro.scene_dataset_config.json"
        cfg_settings["scene"] = "NONE"
        cfg_settings["depth_sensor"] = False
        cfg_settings["color_sensor"] = False
        hab_cfg = habitat_sim.utils.settings.make_cfg(cfg_settings)
        # required for HITL apps
        hab_cfg.sim_cfg.enable_gfx_replay_save = True
        sim = habitat_sim.Simulator(hab_cfg)

        # Initialize driver.
        super().__init__(
            config=config,
            gui_input=gui_input,
            line_render=line_render,
            text_drawer=text_drawer,
            sim=sim,
        )

        assert self.get_sim().renderer is None

        data_collection_config = self._hitl_config.data_collection

        # There is no episode record when using SimDriver
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

        # unsupported for SimDriver
        assert self._hitl_config.disable_policies_and_stepping

        self._debug_images = self._hitl_config.debug_images

        self._viz_anim_fraction: float = 0.0
        self._pending_cursor_style: Optional[Any] = None

        self._episode_helper = None

        self._app_service = AppService(
            config=config,
            hitl_config=self._hitl_config,
            gui_input=gui_input,
            remote_client_state=self._remote_client_state,
            gui_drawer=self._gui_drawer,
            text_drawer=text_drawer,
            ui_manager=self._ui_manager,
            get_anim_fraction=lambda: self._viz_anim_fraction,
            env=None,
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
            gui_agent_controllers=[],
            all_agent_controllers=[],
            users=self._users,
        )

        self._app_state: AppState = None
        assert create_app_state_lambda is not None
        self._app_state = create_app_state_lambda(self._app_service)

        self._reset_environment()

    def _reconfigure_sim(self, dataset: Optional[str], scene: Optional[str]):
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
        self.get_sim().reconfigure(hab_cfg)

        assert self.get_sim().renderer is None

    def _reset_environment(self):
        if self.network_server_enabled:
            self._remote_client_state.clear_history()

        self._app_state.on_environment_reset(self._episode_recorder_dict)

        self.get_sim().gfx_replay_manager.save_keyframe()

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

    def sim_update(self, dt: float):
        post_sim_update_dict: Dict[str, Any] = {}

        if self._remote_client_state:
            self._remote_client_state.update()

        # _viz_anim_fraction goes from 0 to 1 over time and then resets to 0
        self._viz_anim_fraction = (
            self._viz_anim_fraction
            + dt * self._hitl_config.viz_animation_speed
        ) % 1.0

        self._app_state.sim_update(dt, post_sim_update_dict)

        # sloppy: we aren't currently stepping the sim for our current use case (gfx_replay_viewer), so we manually save a keyframe here after every app_state.sim_update (on the assumption that the app state updated the scene). Future use cases of SimDriver may require stepping the sim (e.g. viewing a live physics simulation). We'll need to decide who is responsible for stepping the simulator (SimDriver or the AppState). And we'll need to update this line of code. The equivalent logic in LabDriver is AppService.compute_action_and_step_env, which is called by AppStates but implemented by the driver.
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
