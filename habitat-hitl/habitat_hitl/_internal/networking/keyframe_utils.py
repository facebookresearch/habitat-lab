#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any


def update_consolidated_keyframe(consolidated_keyframe, inc_keyframe):
    """
    A "consolidated" keyframe is several incremental keyframes merged together.
    See nearly duplicate logic in habitat-sim Recorder::addLoadsCreationsDeletions.
    Note the simplification logic here: if an instance is created and then
    later deleted, the latest consolidated keyframe won't have either the
    creation, update, or deletion.
    """
    assert consolidated_keyframe is not None
    assert inc_keyframe is not None

    def ensure_list(keyframe, key):
        if key not in keyframe:
            keyframe[key] = []

    def ensure_dict(keyframe, key):
        if key not in keyframe:
            keyframe[key] = {}

    # append loads
    if "loads" in inc_keyframe:
        ensure_list(consolidated_keyframe, "loads")
        consolidated_keyframe["loads"] += inc_keyframe["loads"]

    # add or update stateUpdates based on instanceKey
    if "stateUpdates" in inc_keyframe:
        ensure_list(consolidated_keyframe, "stateUpdates")
        for state_update in inc_keyframe["stateUpdates"]:
            key = state_update["instanceKey"]
            state = state_update["state"]
            found = False
            for con_state_update in consolidated_keyframe["stateUpdates"]:
                if con_state_update["instanceKey"] == key:
                    con_state_update["state"] = state
                    found = True
            if not found:
                consolidated_keyframe["stateUpdates"].append(state_update)

    # add or update rigUpdates
    if "rigUpdates" in inc_keyframe:
        for rig_update in inc_keyframe["rigUpdates"]:
            key = rig_update["id"]
            pose = rig_update["pose"]
            found = False
            for con_rig_update in consolidated_keyframe["rigUpdates"]:
                if con_rig_update["id"] == key:
                    con_rig_update["pose"] = pose
                    found = True
            if not found:
                consolidated_keyframe["rigUpdates"].append(rig_update)

    # append creations
    if "creations" in inc_keyframe:
        ensure_list(consolidated_keyframe, "creations")
        consolidated_keyframe["creations"] += inc_keyframe["creations"]

    # append rigCreations if skinning transmission is enabled
    if "rigCreations" in inc_keyframe:
        ensure_list(consolidated_keyframe, "rigCreations")
        consolidated_keyframe["rigCreations"] += inc_keyframe["rigCreations"]

    # for a deletion, just remove all references to this instanceKey
    if "deletions" in inc_keyframe:
        inc_deletions = inc_keyframe["deletions"]
        for key in inc_deletions:
            # If we find a corresponding creation in the con keyframe, we can remove
            # the creation and otherwise skip this deletion. This logic ensures
            # consolidated keyframes don't get bloated as many items are added
            # and removed over time.
            con_creations = consolidated_keyframe["creations"]
            found = False
            for entry in con_creations:
                if entry["instanceKey"] == key:
                    con_creations.remove(entry)
                    found = True
                    break
            if not found:
                # if we didn't find the creation, then we should still include the deletion
                ensure_list(consolidated_keyframe, "deletions")
                consolidated_keyframe["deletions"].append(key)

        # remove stateUpdates for the deleted keys
        if "stateUpdates" in consolidated_keyframe:
            consolidated_keyframe["stateUpdates"] = [
                entry
                for entry in consolidated_keyframe["stateUpdates"]
                if entry["instanceKey"] not in inc_deletions
            ]

    if "message" in inc_keyframe:
        inc_message = inc_keyframe["message"]
        # add/update all messages
        for message_key in inc_message:
            ensure_dict(consolidated_keyframe, "message")
            consolidated_keyframe["message"][message_key] = inc_message[
                message_key
            ]

    # todo: lights, userTransforms


def get_empty_keyframe():
    keyframe: Any = dict()
    keyframe["loads"] = []
    keyframe["creations"] = []
    keyframe["rigCreations"] = []
    keyframe["stateUpdates"] = []
    keyframe["rigUpdates"] = []
    keyframe["deletions"] = []
    keyframe["lightsChanged"] = False
    keyframe["lights"] = []
    # todo: lights, userTransforms
    return keyframe
