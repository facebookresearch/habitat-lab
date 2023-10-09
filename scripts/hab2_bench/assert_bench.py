# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp

# The numbers obtained in NAME_MAP MINIMUM_PERFORMANCE_1_PROCESS MINIMUM_PERFORMANCE_16_PROCESS are
# obtained by running the CI at a point in time we were satisfied with the performance of habitat-lab.
# The numbers are obtained on the CI machine 'GPU Linux Medium'
# These numbers are not "maximum performance" and are only used to test for regression.
# DO NOT QUOTE THESE NUMBERS.

NAME_MAP = {
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
}

MINIMUM_PERFORMANCE_1_PROCESS = {
    "idle_all": 260,
    "idle_noconcur": 220,
    "idle_nosleep": 105,
    # "idle_render": "[Idle Render Only]",
    "idle_single_camera_all": 465,
    "idle_single_camera_noconcur": 430,
    "idle_single_camera_nosleep": 145,
    # "idle_single_camera_render": "[Idle (head-RGB) Render Only]",
    "interact_all": 40,
    "interact_noconcur": 39,
    "interact_nosleep": 40,
}

MINIMUM_PERFORMANCE_16_PROCESS = {
    "idle_all": 920,
    "idle_noconcur": 950,
    "idle_nosleep": 500,
    # "idle_render": "[Idle Render Only]",
    "idle_single_camera_all": 1450,
    "idle_single_camera_noconcur": 1550,
    "idle_single_camera_nosleep": 650,
    # "idle_single_camera_render": "[Idle (head-RGB) Render Only]",
    "interact_all": 920,
    "interact_noconcur": 950,
    "interact_nosleep": 500,
}


class RegressionError(AssertionError):
    pass


def check_benchmark_sps(name_map, minimum_performance_map, base_name):
    failed_runs = []
    for name in minimum_performance_map.keys():
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
        assert len(run_fps) > 0, f"Missing run for {fname}"
        mean_sps = sum(run_fps) / len(run_fps)
        if mean_sps < minimum_performance_map[name]:
            failed_runs.append(
                f"The performance of the setting {name} : {name_map.get(name, 'n/a')} has a SPS regression. Was expecting a minimum SPS of {minimum_performance_map[name]} but got an average of {mean_sps}"
            )

    return failed_runs


if __name__ == "__main__":
    failed_runs = []

    failed_runs.extend(
        check_benchmark_sps(
            NAME_MAP, MINIMUM_PERFORMANCE_1_PROCESS, "1_200_-1_"
        )
    )
    # TODO: understand CI multi-process issues before asserting on these bench results
    # failed_runs.extend(
    #    check_benchmark_sps(
    #        NAME_MAP, MINIMUM_PERFORMANCE_16_PROCESS, "16_200_-1_"
    #    )
    # )

    print(failed_runs)

    if len(failed_runs) == 0:
        print("No regression detected")
    else:
        raise RegressionError("\n" + "\n\n".join(failed_runs))
