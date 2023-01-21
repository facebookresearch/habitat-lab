from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

# These metrics are not scalars and cannot be easily reported
# (unless using videos)
NON_SCALAR_METRICS = {"top_down_map", "collisions.is_collision"}


def extract_scalars_from_info(info: Dict[str, Any]) -> Dict[str, float]:
    r"""From an environment info dictionary, returns a flattened
    dictionary of string to floats by filtering all non-scalar
    metrics.

        Args:
            info: A gym.Env  info dict

        Returns:
            dictionary of scalar values
    """
    result = {}
    for k, v in info.items():
        if not isinstance(k, str) or k in NON_SCALAR_METRICS:
            continue
        if k.split(".")[0] in NON_SCALAR_METRICS:
            continue
        # Things that are scalar-like will have an np.size of 1.
        # Strings also have an np.size of 1, so explicitly ban those
        if np.size(v) == 1 and not isinstance(v, str):
            result[k] = float(v)

    return result


def extract_scalars_from_infos(
    infos: List[Dict[str, Any]]
) -> Dict[str, List[float]]:
    r"""From alist of gym.Env info dictionary, returns a
    dictionary of string to list of floats. Also filters
    all non-scalar metrics.

        Args:
            info: A list of gym.Env type info dict

        Returns:
            dict of list of scalar values
    """
    results = defaultdict(list)
    for i in range(len(infos)):
        for k, v in extract_scalars_from_info(infos[i]).items():
            results[k].append(v)

    return results
