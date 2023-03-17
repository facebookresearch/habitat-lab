# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
from typing import List, Optional, Tuple

import numpy as np
import torch


class Metric:
    def __init__(self, info=None, metric_names=None, log_json=None):
        self.info = info
        self.metric_names = sorted(metric_names) if metric_names else []

        self.metrics: List[List[Optional[float]]] = [
            [None, None, None] for _ in self.metric_names
        ]

        self.stats = []
        self.num_iters = 0

        self.log_json = log_json

    def update(self, values: List) -> None:
        assert isinstance(values, list)

        self.num_iters += 1
        current_stats = []

        for i in range(len(values)):
            if values[i] is None:
                continue

            if isinstance(values[i], list) is False:
                values[i] = [values[i]]

            if self.metrics[i][0] is None:
                self.metrics[i][0] = np.mean(values[i])
                self.metrics[i][1] = np.mean(values[i])
                self.metrics[i][2] = np.mean(values[i])
            else:
                self.metrics[i][0] = (
                    self.metrics[i][0] * (self.num_iters - 1)
                    + np.mean(values[i])
                ) / self.num_iters

                self.metrics[i][1] = 0.95 * self.metrics[i][
                    1
                ] + 0.05 * np.mean(values[i])

                self.metrics[i][2] = np.mean(values[i])

            self.metrics[i][0] = float(self.metrics[i][0])
            self.metrics[i][1] = float(self.metrics[i][1])
            self.metrics[i][2] = float(self.metrics[i][2])

            current_stats.append(self.metrics[i])

        self.stats.append(copy.deepcopy(current_stats))

    def get_stat_string(self, mode: int = 1) -> str:
        stat_string = ""

        for k, v in self.info.items():
            stat_string += "[{}:{}]".format(k, v)

        stat_string += "[iters:{}]\n".format(self.num_iters)
        for i in range(len(self.metric_names)):
            if self.metrics[i][mode] is not None:
                stat_string += "[{}:{:.3f}]".format(
                    self.metric_names[i],
                    self.metrics[i][mode],
                )

        return stat_string

    def get_stats(self, mode: int = 1) -> List[float]:
        stats = []
        for i in range(len(self.metric_names)):
            stats.append(self.metrics[i][mode])

        return stats

    def dump_log(self) -> bool:
        if self.log_json is None:
            return False

        dict_to_save = {"metric_names": self.metric_names, "stats": self.stats}

        with open(self.log_json, "w") as f:
            json.dump(dict_to_save, f)

        return True


class VqaMetric(Metric):
    def __init__(self, info=None, metric_names=None, log_json=None):
        super().__init__(info, metric_names, log_json)

    def compute_ranks(
        self, scores: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        accuracy = np.zeros(len(labels))
        ranks = np.full(len(labels), scores.shape[1])

        for i in range(scores.shape[0]):
            ranks[i] = scores[i].gt(scores[i][labels[i]]).sum() + 1
            if ranks[i] == 1:
                accuracy[i] = 1
        return accuracy, ranks


class NavMetric(Metric):
    def __init__(self, info=None, metric_names=None, log_json=None):
        super().__init__(info, metric_names, log_json)
