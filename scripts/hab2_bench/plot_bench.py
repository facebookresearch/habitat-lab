# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
import numpy as np


def plot(name_map, savename, set_title, base_name):
    names = list(name_map.keys())
    mean = []
    std = []
    for name in names:
        found_fps = None
        run_fps = []
        i = 1
        while True:
            fname = osp.join(f"data/profile/{base_name}{name}_{i}.txt")
            if not osp.exists(fname):
                break
            with open(fname, "r") as f:
                for l in f:
                    if "FPS" in l:
                        found_fps = float(l.split(": ")[1])
                        break
            if found_fps is None:
                raise ValueError()
            run_fps.append(found_fps)
            i += 1
        # assert len(run_fps) == 10, f"For {name}"
        mean.append(np.mean(run_fps))
        std.append(2.228 * np.std(run_fps) / np.sqrt(len(run_fps)))
    N = len(names)

    xpos = np.arange(0, 2 * N, 2)

    use_names = [name_map[k] for k in names]

    for n, m, s in zip(use_names, mean, std):
        print(f"{n}: {round(m)}&{{\\scriptsize$\\pm${round(s)}}}")
        print("")

    plt.barh(xpos, mean, xerr=std, align="center", ecolor="black", capsize=10)
    plt.yticks(xpos, use_names)
    plt.xlabel("FPS")
    plt.title(set_title)
    plt.grid(
        visible=True,
        which="major",
        color="lightgray",
        linestyle="--",
        axis="x",
    )
    plt.tight_layout()

    plt.savefig(
        osp.join("data/profile", savename + ".pdf"),
        format="pdf",
        bbox_inches="tight",
    )
    plt.clf()


# plot rearrange benchmark numbers
for i in [1, 16]:
    plot(
        {
            "idle_all": "[Idle]",
            "idle_noconcur": "[Idle No Concurrent Rendering]",
            "idle_nosleep": "[Idle No Auto-sleep]",
            # "idle_render": "[Idle Render Only]",
            "idle_single_camera_all": "[Idle (head-RGB)]",
            "idle_single_camera_noconcur": "[Idle (head-RGB) No Concurrent Rendering]",
            "idle_single_camera_nosleep": "[Idle (head-RGB) No Auto-sleep]",
            # "idle_single_camera_render": "[Idle (head-RGB) Render Only]",
            "interact_all": "[Interact]",
            "interact_noconcur": "[Interact No Concurrent Rendering]",
            "interact_nosleep": "[Interact No Auto-sleep]",
        },
        "opts_%i" % i,
        "ReplicaCAD: 200 Steps %i Processes" % i,
        "%i_200_-1_" % i,
    )
