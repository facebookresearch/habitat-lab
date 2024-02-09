from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

import numpy as np

# These metrics are not scalars and cannot be easily reported
# (unless using videos)
NON_SCALAR_METRICS = {"top_down_map", "collisions.is_collision"}


def extract_scalars_from_info(
    info: Dict[str, Any], ignore_keys: Optional[Set[str]] = None
) -> Dict[str, float]:
    r"""From an environment info dictionary, returns a flattened
    dictionary of string to floats by filtering all non-scalar
    metrics.

        Args:
            info: A gym.Env  info dict
            ignore_keys: The list of info key names to exclude in the result.

        Returns:
            dictionary of scalar values
    """
    if ignore_keys is None:
        ignore_keys = set()
    ignore_keys.update(NON_SCALAR_METRICS)
    result = {}
    for k, v in info.items():
        if not isinstance(k, str) or k in ignore_keys:
            continue

        if isinstance(v, dict):
            result.update(
                {
                    k + "." + subk: subv
                    for subk, subv in extract_scalars_from_info(v).items()
                    if isinstance(subk, str)
                    and k + "." + subk not in ignore_keys
                }
            )
        # Things that are scalar-like will have an np.size of 1.
        # Strings also have an np.size of 1, so explicitly ban those
        elif np.size(v) == 1 and not isinstance(v, str):
            result[k] = float(v)

    return result


def extract_scalars_from_infos(
    infos: List[Dict[str, Any]],
    ignore_keys: Optional[Set[str]] = None,
) -> Dict[str, List[float]]:
    r"""From alist of gym.Env info dictionary, returns a
    dictionary of string to list of floats. Also filters
    all non-scalar metrics.

        Args:
            infos: A list of gym.Env type info dict
            ignore_keys: The list of info key names to exclude in the result.

        Returns:
            dict of list of scalar values
    """
    results = defaultdict(list)
    for i in range(len(infos)):
        for k, v in extract_scalars_from_info(infos[i], ignore_keys).items():
            results[k].append(v)

    return results
