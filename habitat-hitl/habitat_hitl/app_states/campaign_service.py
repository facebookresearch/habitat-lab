#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import random
import string
from enum import Enum
from typing import Any, Callable, Dict

import requests

from habitat_hitl.environment.episode_helper import EpisodeHelper


class TaskStatus(Enum):
    INIT = "initialized"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class CampaignService:
    def __init__(
        self,
        *,
        hitl_config,
        get_metrics: Callable,
        episode_helper: EpisodeHelper,
    ):
        self._hitl_config = hitl_config
        self._get_metrics = get_metrics
        self._episode_helper = episode_helper

        self.server_url = self._hitl_config.campaign_server.url
        self.session_meta: Dict[str, Any] = {}

    def post(self, url, data):
        response = requests.post(url, data=json.dumps(data))
        return response

    def get(self, url):
        response = requests.get(url)
        return response

    @staticmethod
    def random_id(max_len: int = 10):
        return "".join(
            random.choice(string.ascii_uppercase + string.digits)
            for _ in range(max_len)
        )

    def initialize_session(self):
        self.session_meta["session_id"] = CampaignService.random_id()
        self.session_meta["worker_id"] = CampaignService.random_id()
        self.session_meta["mode"] = "sandbox"
        response = self.initialize_task(
            {
                "scene_id": "dummy",
                "episode_id": 0,
                "task_status": TaskStatus.INIT.value,
            }
        )
        if response.status_code == 200:
            self.session_meta.update(response.json()["data"])

    def set_task_status(self, status: TaskStatus):
        endpoint = self._hitl_config.campaign_server.endpoints.update_task
        url = f"{self.server_url}/{endpoint}"

        response = self.post(url, {"status": status, **self.session_meta})
        return response

    def initialize_task(self, data: Dict[str, Any]):
        endpoint = self._hitl_config.campaign_server.endpoints.initialize_task
        url = f"{self.server_url}/{endpoint}"

        response = self.post(url, {"data": data, **self.session_meta})
        return response

    def end_task(self, data: Dict[str, Any]):
        endpoint = self._hitl_config.campaign_server.endpoints.end_task
        url = f"{self.server_url}/{endpoint}"

        response = self.post(url, {"data": data, **self.session_meta})
        return response

    @property
    def max_episodes_per_session(self):
        return self.session_meta.get("max_episodes_per_session", 100)
