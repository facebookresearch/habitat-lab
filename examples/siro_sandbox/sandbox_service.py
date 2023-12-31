#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# Helpers to provide to AppState classes, provided by the underlying SandboxDriver
class SandboxService:
    def __init__(
        self,
        args,
        config,
        gui_input,
        remote_gui_input,
        line_render,
        text_drawer,
        get_anim_fraction,
        env,
        sim,
        compute_action_and_step_env,
        step_recorder,
        get_metrics,
        end_episode,
        set_cursor_style,
        episode_helper,
        client_message_manager,
    ):
        self._args = args
        self._config = config
        self._gui_input = gui_input
        self._remote_gui_input = remote_gui_input
        self._line_render = line_render
        self._text_drawer = text_drawer
        self._get_anim_fraction = get_anim_fraction
        self._env = env
        self._sim = sim
        self._compute_action_and_step_env = compute_action_and_step_env
        self._step_recorder = step_recorder
        self._get_metrics = get_metrics
        self._end_episode = end_episode
        self._set_cursor_style = set_cursor_style
        self._episode_helper = episode_helper
        self._client_message_manager = client_message_manager

    @property
    def args(self):
        return self._args

    @property
    def config(self):
        return self._config

    @property
    def gui_input(self):
        return self._gui_input

    @property
    def remote_gui_input(self):
        return self._remote_gui_input

    @property
    def line_render(self):
        return self._line_render

    @property
    def text_drawer(self):
        return self._text_drawer

    @property
    def get_anim_fraction(self):
        return self._get_anim_fraction

    @property
    def env(self):
        return self._env

    @property
    def sim(self):
        return self._sim

    @property
    def compute_action_and_step_env(self):
        return self._compute_action_and_step_env

    @property
    def step_recorder(self):
        return self._step_recorder

    @property
    def get_metrics(self):
        return self._get_metrics

    @property
    def end_episode(self):
        return self._end_episode

    @property
    def set_cursor_style(self):
        return self._set_cursor_style

    @property
    def episode_helper(self):
        return self._episode_helper

    @property
    def client_message_manager(self):
        return self._client_message_manager
