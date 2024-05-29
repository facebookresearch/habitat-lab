#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing

from habitat.datasets.rearrange.rearrange_dataset import RearrangeEpisode

try:
    from habitat_llm.agent.env import dataset  # noqa: F401
    from habitat_llm.agent.env import (
        register_actions,
        register_measures,
        register_sensors,
    )
    from habitat_llm.agent.env.dataset import CollaborationEpisode

    collaboration_episode_enabled = True
except ImportError:
    print("Unable to load CollaborationDataset episode format.")
    collaboration_episode_enabled = False


class CollaborationEpisodeData:
    def __init__(self):
        self.instruction: str = ""


if collaboration_episode_enabled:

    def load_collaboration_episode_data(
        episode: RearrangeEpisode,
    ) -> CollaborationEpisodeData:
        episode_data = CollaborationEpisodeData()

        if not isinstance(episode, CollaborationEpisode):
            return episode_data

        episode = typing.cast(CollaborationEpisode, episode)
        episode_data.instruction = episode.instruction

        return episode_data

    def register_habitat_llm_extensions(config):
        try:
            register_actions(config)
            register_measures(config)
            register_sensors(config)
        except Exception as e:
            print(f"Config incompatible with LLM. {e}")

else:

    def load_collaboration_episode_data(
        episode: RearrangeEpisode,
    ) -> CollaborationEpisodeData:
        return CollaborationEpisodeData()

    def register_habitat_llm_extensions(config):
        pass
