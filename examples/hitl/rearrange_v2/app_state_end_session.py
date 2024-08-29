#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
from typing import Optional

from app_data import AppData
from app_state_base import AppStateBase
from app_states import create_app_state_reset
from s3_upload import (
    generate_unique_session_id,
    upload_file_to_s3,
    validate_experiment_name,
)
from session import Session
from util import get_top_down_view

from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.core.serialize_utils import save_as_json_gzip
from habitat_hitl.core.user_mask import Mask

# Duration of the end session message, before users are kicked.
SESSION_END_DELAY = 5.0


class AppStateEndSession(AppStateBase):
    """
    * Indicate users that the session is terminated.
    * Upload collected data.
    """

    def __init__(
        self, app_service: AppService, app_data: AppData, session: Session
    ):
        super().__init__(app_service, app_data)
        self._session = session
        self._elapsed_time = 0.0
        self._save_keyframes = False

        self._status = "Session ended."
        if len(session.error) > 0:
            self._status += f"\nError: {session.error}"

    def get_next_state(self) -> Optional[AppStateBase]:
        connected_user_count = len(self._app_data.connected_users)
        if self._elapsed_time > SESSION_END_DELAY or connected_user_count == 0:
            self._end_session()
            return create_app_state_reset(self._app_service, self._app_data)
        return None

    def sim_update(self, dt: float, post_sim_update_dict):
        # Top-down view.
        cam_matrix = get_top_down_view(self._app_service.sim)
        post_sim_update_dict["cam_transform"] = cam_matrix
        self._app_service._client_message_manager.update_camera_transform(
            cam_matrix, destination_mask=Mask.ALL
        )

        self._status_message(self._status)
        self._elapsed_time += dt

    def _end_session(self):
        session = self._session
        if session is None:
            print("Null session. Skipping S3 upload.")
            return

        # Finalize session.
        if self._session.error == "":
            session.finished = True
        session.session_recorder.end_session(self._session.error)

        # Get data collection parameters.
        config = self._app_service.config

        # Set the base S3 path as the experiment name.
        s3_path: str = "default"
        if len(session.connection_records) > 0:
            experiment_name: Optional[str] = session.connection_records[0].get(
                "experiment", None
            )
            if validate_experiment_name(experiment_name):
                s3_path = experiment_name
            else:
                print(f"Invalid experiment name: '{experiment_name}'")

        # Generate unique session ID
        session_id = generate_unique_session_id(
            session.episode_indices, session.connection_records
        )
        s3_path = os.path.join(s3_path, session_id)

        # Use the port as a discriminator for when there are multiple concurrent servers.
        output_folder_suffix = str(config.habitat_hitl.networking.port)
        output_folder = f"hitl_output_{output_folder_suffix}"

        # Delete previous output directory
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)

        # Create new output directory
        os.makedirs(output_folder)

        # Create a session metadata file.
        session_json_path = os.path.join(output_folder, "session.json.gz")
        save_as_json_gzip(
            session.session_recorder.get_session_output(), session_json_path
        )

        # Create one file per episode.
        episode_outputs = session.session_recorder.get_episode_outputs()
        for episode_output in episode_outputs:
            episode_json_path = os.path.join(
                output_folder,
                f"{episode_output.episode.episode_index}.json.gz",
            )
            save_as_json_gzip(
                episode_output,
                episode_json_path,
            )

        # Upload output directory
        orig_file_names = [
            f
            for f in os.listdir(output_folder)
            if os.path.isfile(os.path.join(output_folder, f))
        ]
        for orig_file_name in orig_file_names:
            local_file_path = os.path.join(output_folder, orig_file_name)
            s3_file_name = orig_file_name
            print(
                f"Uploading '{local_file_path}' to '{s3_path}' as '{s3_file_name}'."
            )
            upload_file_to_s3(local_file_path, s3_file_name, s3_path)
