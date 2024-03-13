#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC


class AppState(ABC):
    """
    Base class for implementing a HITL app. See habitat-hitl/README.md.
    """

    def sim_update(self, dt, post_sim_update_dict):
        """
        A hook called continuously (for each "frame"), before updating/rendering the app's GUI window.
        This is where the main app state logic should go. The AppState should update the sim state and also populate post_sim_update_dict["cam_transform"], at a minimum. The AppState can also record to _app_service.step_recorder here.
        """

    def on_environment_reset(self, episode_recorder_dict):
        """
        A hook called on environment reset (starting a new episode).
        The AppState should reset internal members as necessary. It can record arbitrary per-episode data to episode_recorder_dict, e.g. episode_recorder_dict["mymetric"] = my_metric, but beware episode_recorder_dict will be None when data-collection is not enabled/configured.
        """

    def record_state(self):
        """
        A hook called on each environment step.
        This is a good place to record using _app_service.step_recorder because this function will be skipped if the app isn't recording. However, an AppState is free to record at any time (in other methods, too).
        """
