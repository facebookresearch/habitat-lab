# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from habitat_hitl.app_states.app_service import AppService


class Metrics:
    """
    Helper class for accessing experiment metrics.
    """

    def __init__(self, app_service: "AppService"):
        self._get_metrics = app_service.get_metrics

    def get_task_percent_complete(self) -> Optional[float]:
        """
        Get the current progression of the task, from 0.0 to 1.0.
        Returns None if the metric is unavailable.
        """
        metrics = self._get_metrics()
        return metrics.get("task_percent_complete", None)

    def get_task_explanation(self) -> Optional[str]:
        """
        Get an explanation of the task success in a human-readable format.
        Returns None if the metric is unavailable.
        """
        metrics = self._get_metrics()
        return metrics.get("task_explanation", None)

    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get a dictionary containing all registered metrics.
        """
        return self._get_metrics()
