#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from os import makedirs
from os import path as osp
from typing import Any, Dict, List, Tuple

from habitat.core.logging import logger


def check_make_dir(directory_path: str) -> bool:
    """
    Check for the existence of the provided directory_path and create it if not found.
    """
    # if output directory doesn't exist, create it
    if not osp.exists(directory_path):
        try:
            makedirs(directory_path)
        except OSError:
            logger.error(
                f"check_make_dir: Failed to create the specified directory_path: {directory_path}"
            )
            return False
        logger.info(
            f"check_make_dir: directory_path did not exist and was created: {directory_path}"
        )
    return True


def cull_string_list_by_substrings(
    full_list: List[str],
    included_substrings: List[str],
    excluded_substrings: List[str],
) -> List[str]:
    """
    Cull a list of strings to the subset of strings containing any of the "included_substrings" and none of the "excluded_substrings".
    Returns the culled list, does not modify the input list.
    """
    culled_list: List[str] = []
    for string in full_list:
        excluded = False
        for excluded_substring in excluded_substrings:
            if excluded_substring in string:
                excluded = True
                break
        if not excluded:
            for included_substring in included_substrings:
                if included_substring in string:
                    culled_list.append(string)
                    break
    return culled_list


def flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict:
    r"""Flattens nested dict.

    Source: https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys

    :param d: Nested dict.
    :param parent_key: Parameter to set parent dict key.
    :param sep: Nested keys separator.
    :return: Flattened dict.
    """
    items: List[Tuple[str, Any]] = []
    for k, v in d.items():
        new_key = parent_key + sep + str(k) if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, parent_key=new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)
