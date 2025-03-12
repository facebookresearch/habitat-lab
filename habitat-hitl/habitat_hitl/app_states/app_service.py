#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Callable, Optional

from habitat import Env
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat_hitl.core.client_message_manager import ClientMessageManager
from habitat_hitl.core.gui_input import GuiInput
from habitat_hitl.core.remote_gui_input import RemoteGuiInput
from habitat_hitl.core.serialize_utils import BaseRecorder
from habitat_hitl.core.text_drawer import AbstractTextDrawer
from habitat_hitl.environment.controllers.controller_abc import GuiController
from habitat_hitl.environment.episode_helper import EpisodeHelper
from habitat_sim.gfx import DebugLineRender
from habitat_hitl._internal.video_recorder import FramebufferVideoRecorder

# Helpers to provide to AppState classes, provided by the underlying SandboxDriver
class AppService:
    def __init__(
        self,
        *,
        config,
        hitl_config,
        gui_input: GuiInput,
        remote_gui_input: RemoteGuiInput,
        line_render: DebugLineRender,
        text_drawer: AbstractTextDrawer,
        get_anim_fraction: Callable,
        env: Env,
        sim: RearrangeSim,
        reconfigure_sim: Callable,
        compute_action_and_step_env: Callable,
        step_recorder: BaseRecorder,
        get_metrics: Callable,
        end_episode: Callable,
        set_cursor_style: Callable,
        episode_helper: EpisodeHelper,
        client_message_manager: ClientMessageManager,
        gui_agent_controller: Optional[GuiController],
        video_recorder: FramebufferVideoRecorder,
    ):
        self._config = config
        self._hitl_config = hitl_config
        self._gui_input = gui_input
        self._remote_gui_input = remote_gui_input
        self._line_render = line_render
        self._text_drawer = text_drawer
        self._get_anim_fraction = get_anim_fraction
        self._env = env
        self._sim = sim
        self._reconfigure_sim = reconfigure_sim
        self._compute_action_and_step_env = compute_action_and_step_env
        self._step_recorder = step_recorder
        self._get_metrics = get_metrics
        self._end_episode = end_episode
        self._set_cursor_style = set_cursor_style
        self._episode_helper = episode_helper
        self._client_message_manager = client_message_manager
        self._gui_agent_controller = gui_agent_controller
        self._video_recorder = video_recorder

    @property
    def config(self):
        return self._config

    @property
    def hitl_config(self):
        return self._hitl_config

    @property
    def gui_input(self) -> GuiInput:
        return self._gui_input

    @property
    def remote_gui_input(self) -> RemoteGuiInput:
        return self._remote_gui_input

    @property
    def line_render(self) -> DebugLineRender:
        return self._line_render

    @property
    def text_drawer(self) -> AbstractTextDrawer:
        return self._text_drawer

    @property
    def get_anim_fraction(self) -> Callable:
        return self._get_anim_fraction

    @property
    def env(self) -> Env:
        return self._env

    @property
    def sim(self) -> RearrangeSim:
        return self._sim
    
    @property
    def reconfigure_sim(self) -> Callable:
        return self._reconfigure_sim

    @property
    def compute_action_and_step_env(self) -> Callable:
        return self._compute_action_and_step_env

    @property
    def step_recorder(self) -> BaseRecorder:
        return self._step_recorder

    @property
    def get_metrics(self) -> Callable:
        return self._get_metrics

    @property
    def end_episode(self) -> Callable:
        return self._end_episode

    @property
    def set_cursor_style(self) -> Callable:
        return self._set_cursor_style

    @property
    def episode_helper(self) -> EpisodeHelper:
        return self._episode_helper

    @property
    def client_message_manager(self) -> ClientMessageManager:
        return self._client_message_manager

    @property
    def gui_agent_controller(self) -> Optional[GuiController]:
        return self._gui_agent_controller

    @property
    def video_recorder(self) -> FramebufferVideoRecorder:
        return self._video_recorder