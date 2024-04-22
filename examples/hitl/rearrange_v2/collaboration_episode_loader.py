#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Set, Tuple
import typing

from habitat.datasets.rearrange.rearrange_dataset import RearrangeEpisode

try:
    from habitat_llm.agent.env import dataset # noqa: F401
    from habitat_llm.agent.env.dataset import CollaborationEpisode
    collaboration_episode_enabled = True
except ImportError:
    print("Unable to load CollaborationDataset episode format.")
    collaboration_episode_enabled = False

class CollaborationEpisodeData:
    instruction: str = ""
    object_goal_pairs: Dict[str, str] = {}

if collaboration_episode_enabled:
    def load_collaboration_episode_data(episode: RearrangeEpisode) -> CollaborationEpisodeData:
        episode_data = CollaborationEpisodeData()

        if not isinstance(episode, CollaborationEpisode):
            return episode_data
        
        episode = typing.cast(CollaborationEpisode, episode)
        episode_data.instruction = episode.instruction
        name_to_receptacle: Dict[str, str] = episode.name_to_receptacle
        for key, value in name_to_receptacle.items():
            episode_data.object_goal_pairs[key] = value.split("|")[0]
        
        return episode_data
else:
    def load_collaboration_episode_data(episode: RearrangeEpisode) -> CollaborationEpisodeData:
        return CollaborationEpisodeData()