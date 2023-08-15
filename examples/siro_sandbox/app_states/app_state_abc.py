#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC


# See also SandboxDriver. SandboxDriver loads a habitat env, including scene/episode/task/agent/policy. A derived AppState class drives use-case-specific behavior for the HITL tool during each env step, including processing user input and manipulating the sim state. In the future, SandboxDriver may manage the transition between AppStates, e.g. transition from a non-interactive tutorial state to a user-controlled-agent state.
#
# AppState is conceptually an ABC, but all the current methods are optional so there's no use of @abstractmethod here yet.
class AppState(ABC):
    # This is where the main app state logic should go. The AppState should update the sim state and also populate post_sim_update_dict["cam_transform"], at a minimum. The AppState can also record to _sandbox_service.step_recorder here.
    def sim_update(self, dt, post_sim_update_dict):
        pass

    # The AppState should reset internal members as necessary for the new episode. It can record arbitrary per-episode data to episode_recorder_dict, e.g. episode_recorder_dict["mymetric"] = my_metric, but beware episode_recorder_dict may be None.
    def on_environment_reset(self, episode_recorder_dict):
        pass

    # This is a good place to record using _sandbox_service.step_recorder because this function will be skipped if the app isn't recording. However, an AppState is free to record at any time (in other methods, too).
    def record_state(self):
        pass
