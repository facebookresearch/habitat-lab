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
from s3_upload import upload_file_to_s3
from session import Session
from util import get_top_down_view, timestamp

from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.core.serialize_utils import save_as_json_gzip
from habitat_hitl.core.user_mask import Mask

SESSION_END_DELAY = 5.0


class AppStateEndSession(AppStateBase):
    """
    Indicate users that the session is terminated.
    """

    def __init__(
        self, app_service: AppService, app_data: AppData, session: Session
    ):
        super().__init__(app_service, app_data)
        self._session = session
        self._elapsed_time = 0.0
        self._next_state: Optional[AppStateBase] = None
        self._auto_save_keyframes = True

        self._status = "Session ended."
        if len(session.status) > 0:
            self._status += f"\nError: {session.status}"

    def get_next_state(self) -> Optional[AppStateBase]:
        return self._next_state

    def sim_update(self, dt: float, post_sim_update_dict):
        # Top-down view.
        cam_matrix = get_top_down_view(self._app_service.sim)
        post_sim_update_dict["cam_transform"] = cam_matrix
        self._app_service._client_message_manager.update_camera_transform(
            cam_matrix, destination_mask=Mask.ALL
        )

        self._status_message(self._status)
        self._elapsed_time += dt
        if self._elapsed_time > SESSION_END_DELAY:
            self._end_session()
            self._next_state = create_app_state_reset(
                self._app_service, self._app_data
            )

    def _end_session(self):
        session = self._session
        if session is None:
            print("Null session. Skipping S3 upload.")
            return

        # Finalize session.
        if self._session.status == "":
            session.success = True
        session.session_recorder.end_session(self._session.status)

        # Find S3 params.
        data_collection_config = (
            self._app_service.config.rearrange_v2.data_collection
        )
        s3_path = data_collection_config.s3_path
        if s3_path[-1] != "/":
            s3_path += "/"
        s3_subdir = "complete" if session.success else "incomplete"
        s3_path += s3_subdir

        # Delete previous output directory
        output_folder = session.output_folder
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)

        # Create new output directory
        os.makedirs(output_folder)
        json_path = os.path.join(output_folder, "session.json.gz")
        save_as_json_gzip(session.session_recorder, json_path)

        # Generate unique session ID
        if len(self._session.connection_records) == 0:
            print("No connection record. Aborting upload.")
            return
        episodes_str = ""
        user_id_str = ""
        for _, connection_record in self._session.connection_records.items():
            episodes_str = connection_record.get(
                "episodes", "unknown_episodes"
            )

            if user_id_str != "":
                user_id_str += "_"
            if "user_id" in connection_record:
                uid = str(connection_record["user_id"])
                if uid.isalnum():
                    user_id_str += uid
                else:
                    user_id_str += "unknown_user"
            else:
                user_id_str += "unknown_user"
        session_id = f"{episodes_str}_{user_id_str}_{timestamp()}"
        # TODO Filter str

        # Upload output directory
        output_files = [
            f
            for f in os.listdir(output_folder)
            if os.path.isfile(os.path.join(output_folder, f))
        ]
        for output_file in output_files:
            local_file_path = os.path.join(output_folder, output_file)
            s3_file_name = f"{session_id}_{output_file}"
            upload_file_to_s3(local_file_path, s3_file_name, s3_path)
