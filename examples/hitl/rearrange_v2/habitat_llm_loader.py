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

    habitat_llm_enabled = True
except ImportError as e:
    print(f"Unable to load 'habitat_llm'. {e}")
    habitat_llm_enabled = False


class CollaborationEpisodeData:
    def __init__(self):
        self.instruction: str = ""


if habitat_llm_enabled:

    def load_collaboration_episode_data(
        episode: RearrangeEpisode,
    ) -> CollaborationEpisodeData:
        """
        Load the data contained within a 'CollaborationEpisode'.
        """
        episode_data = CollaborationEpisodeData()

        if not isinstance(episode, CollaborationEpisode):
            return episode_data

        episode = typing.cast(CollaborationEpisode, episode)
        episode_data.instruction = episode.instruction

        return episode_data

    def register_habitat_llm_extensions(config):
        """
        Register habitat-llm actions, sensors and measures.
        """
        register_actions(config)

        if hasattr(config.habitat.dataset, "metadata"):
            register_measures(config)

        if hasattr(config, "habitat_llm") and config.habitat_llm.enable:
            register_sensors(config)

else:

    def load_collaboration_episode_data(
        episode: RearrangeEpisode,
    ) -> CollaborationEpisodeData:
        return CollaborationEpisodeData()

    def register_habitat_llm_extensions(config):
        pass
