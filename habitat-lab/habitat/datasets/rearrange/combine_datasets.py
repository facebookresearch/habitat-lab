#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to combine multiple dataset files into one. For example:
```
python habitat/datasets/rearrange/combine_datasets.py data/datasets/replica_cad/rearrange/v2/train/rearrange_easy_1.json.gz data/datasets/replica_cad/rearrange/v2/train/rearrange_easy_2.json.gz data/datasets/replica_cad/rearrange/v2/train/rearrange_easy.json.gz
```
Or if you want to combine every file of the form `rearrange_easy_X.json.gz` you can use regex:
```
python habitat/datasets/rearrange/combine_datasets.py data/datasets/replica_cad/rearrange/v2/train/rearrange_easy_?.json.gz data/datasets/replica_cad/rearrange/v2/train/rearrange_easy.json.gz
```

"""

import gzip
import json
import sys


def combine_datasets(matches, write_path):
    all_episodes = []
    for path in matches:
        print("Merging ", path)
        with gzip.open(path, "rt") as f:
            dat = json.loads(f.read())
            all_episodes.extend(dat["episodes"])

    combined = {"episodes": all_episodes, "config": dat["config"]}
    with gzip.open(write_path, "wt") as f:
        f.write(json.dumps(combined))
    print("Write to ", write_path)


if __name__ == "__main__":
    combine_datasets(sys.argv[1:-1], sys.argv[-1])
