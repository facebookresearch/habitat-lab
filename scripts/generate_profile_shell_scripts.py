#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""Reference python script for profiling DDPPO PointNav on the FAIR internal
cluster used by the Habitat team. This script can be modified locally to suit
your needs, for example, to profile a different program or to profile with
different settings.

For an overview of profiling and optimization in Habitat, see:
https://colab.research.google.com/gist/eundersander/b62bb497519b44cf4ceb10e2079525dc/faster-rl-training-profiling-and-optimization.ipynb

This script's intended usage is:
1. Review and locally edit the documented options at the top of this file.
2. Run this python script to generate profiling shell script(s).
3. See the printed instructions and run the shell script.
"""

import os

if __name__ == "__main__":
    # The Habitat-lab program to be profiled (the command you usually use to
    # invoke it).
    program_str = "python -u -m habitat_baselines.run --config-name=pointnav/ddppo_pointnav.yaml"

    # Path to Nsight Systems nsys command-line tool. This hard-coded path is
    # for the FAIR cluster.
    nsys_path = "/private/home/eundersander/nsight-systems-2020.3.1/bin/nsys"

    # You can either capture a step range or a time range. Capturing a step
    # range is generally a better workflow, but it requires integrating
    # profiling_utils.configure into your train program (beware,
    # profiling_utils.configure is not yet merged into Habitat-sim).
    do_capture_step_range = True

    if do_capture_step_range:
        # "Step" here refers to however you defined a train step in your train
        # program. See habitat-sim profiling_utils.configure. Prefer capturing a
        # range of steps that are representative of your entire train job, in
        # terms of the time spent in various parts of your program. Early train
        # steps may suffer from poor agent behavior, too-short episodes, etc. If
        # necessary, capture and inspect a very long-duration profile to
        # determine when your training FPS "settles".
        # DDPPO PointNav empirical test from Aug 2020, 8 nodes:
        #   FPS settled at ~190 steps
        # DDPPO PointNav empirical test from Oct 2020, 2 nodes:
        #   FPS settled at ~1200 steps
        capture_start_step = 1200

        # If you're focusing on optimizing the train loop body (work that
        # happens consistently every update), you don't need a large number
        # here. However, beware overlooking infrequent events like env resets,
        # scene loads, checkpointing, and eval. Beware profile storage
        # requirement. DDPPO PointNav empirical test from Aug 2020:
        #   qdrep: 3.3 MB per 100 steps
        #   sqlite: 12 MB per 100 steps
        # These figures are for a single task (see capture_all_tasks below).
        num_steps_to_capture = 100
    else:
        nsys_capture_delay_seconds = 120
        nsys_capture_duration_seconds = 120

    # Launch the program distributed, using slurm. See also slurm_submit_str
    # below for more slurm parameters like ntasks-per-node.
    do_slurm = True

    # Path can be absolute or relative to the working directory (where you
    # run the profiling shell script, which is probably the habitat-lab
    # root directory).
    profile_output_folder = "profiles"

    if do_slurm:
        # You must use ${SLURM_NODEID} and ${SLURM_LOCALID} if using
        # capture_all_tasks so that each profile gets a unique name. Use of
        # ${SLURM_JOB_ID} is optional.
        profile_output_filename_base = "profile_job${SLURM_JOB_ID}_node${SLURM_NODEID}_local${SLURM_LOCALID}"
    else:
        profile_output_filename_base = "local_profile"

    if do_slurm:
        # A job duration to provide to slurm. Provide a reasonable upper bound
        # here. It's not important to provide a tight bound. A too-short
        # duration will cause your slurm job to terminate before profiles are
        # saved. A much-too-large duration may result in a longer wait time
        # before slurm starts your job.
        # DDPPO PointNav empirical test from Aug 2020, 8 nodes:
        #   startup time is 2 minutes and 100 steps takes 12 minutes
        # DDPPO PointNav empirical test from Oct 2020, 2 nodes:
        #   startup time is 2 minutes and 100 steps takes 5.9 minutes
        buffered_start_minutes = 10
        buffered_minutes_per_100_steps = 8
        if do_capture_step_range:
            slurm_job_termination_minutes = buffered_start_minutes + int(
                (capture_start_step + num_steps_to_capture)
                * buffered_minutes_per_100_steps
                / 100
            )
        else:
            slurm_job_termination_minutes = (
                nsys_capture_delay_seconds + nsys_capture_duration_seconds
            ) * 60 + 5

        # If capture_all_tasks==True, we capture profiles for all tasks. Beware
        # large profile storage requirement in this case. If False, only one
        # task runs with profiling. The other tasks run without profiling. In
        # theory, all tasks behave similarly and so a single task's profile is
        # representative of all tasks. In my DDPPO PointNav empirical test from
        # Aug 2020, this was true.
        capture_all_tasks = False

    # Useful for understanding your program's CUDA usage on the GPU. Beware
    # large profile storage requirement.
    capture_cuda = False

    # Beware, support is poor on the FAIR cluster and Colab machines due to
    # older Nvidia drivers. For best OpenGL profiling, profile your desktop
    # linux machine using the Nsight Systems GUI, not the nsys command-line
    # tool.
    capture_opengl = False

    # nsys produces a .qdrep multithreaded trace file which can be viewed in the
    # Nsight GUI. Optionally, it can also export a .sqlite database file for use
    # with habitat-sim's compare_profiles.py helper script.
    export_sqlite = True

    # This is the end of the user-facing options, except see slurm_submit_str
    # below.
    # =========================================================================

    if do_capture_step_range:
        program_with_extra_args_str = (
            program_str
            + " profiling.capture_start_step "
            + str(capture_start_step)
            + " profiling.num_steps_to_capture "
            + str(num_steps_to_capture)
        )
    else:
        program_with_extra_args_str = program_str

    if do_capture_step_range:
        capture_range_args = '--capture-range=nvtx -p "habitat_capture_range" --stop-on-range-end=true'
    else:
        capture_range_args = (
            "--delay="
            + str(nsys_capture_delay_seconds)
            + " --duration="
            + str(nsys_capture_duration_seconds)
        )

    task_capture_str = (
        """export HABITAT_PROFILING=1
export NSYS_NVTX_PROFILER_REGISTER_ONLY=0
"""
        + nsys_path
        + " profile --sample=none --trace-fork-before-exec=true --force-overwrite=true --trace=nvtx"
        + (",cuda" if capture_cuda else "")
        + (",opengl" if capture_opengl else "")
        + " "
        + capture_range_args
        + ' --output="'
        + profile_output_folder
        + "/"
        + profile_output_filename_base
        + '" '
        + ("--export=sqlite" if export_sqlite else "")
        + " "
        + program_with_extra_args_str
    )

    if do_slurm:
        if capture_all_tasks:
            slurm_task_str = (
                """#!/bin/sh
"""
                + task_capture_str
                + """
"""
            )
        else:
            slurm_task_str = (
                """#!/bin/sh
if [ "${SLURM_NODEID}" = "0" ] && [ "${SLURM_LOCALID}" = "0" ]
then
"""
                + task_capture_str
                + """
else
"""
                + program_str
                + """
fi
"""
            )

        slurm_submit_str = (
            """#!/bin/bash
#SBATCH --job-name=capture_profile
#SBATCH --output=/checkpoint/%u/jobs/job.%j.out
#SBATCH --error=/checkpoint/%u/jobs/job.%j.err
#SBATCH --gpus-per-task 1
#SBATCH --nodes 2
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 8
#SBATCH --mem-per-cpu=5GB
#SBATCH --partition=dev
#SBATCH --time="""
            + str(slurm_job_termination_minutes)
            + """:00
#SBATCH --open-mode=append
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR
set -x
srun bash capture_profile_slurm_task.sh
"""
        )

    profile_output_filepath = (
        profile_output_folder + "/" + profile_output_filename_base + ".qdrep"
    )
    if not do_slurm and os.path.exists(profile_output_filepath):
        print(
            "warning: {} already exists and will be overwritten.".format(
                profile_output_filepath
            )
        )

    if not os.path.exists(profile_output_folder):
        os.makedirs(profile_output_folder)
        print("created directory: " + profile_output_folder)

    if do_slurm:
        with open("capture_profile_slurm_task.sh", "w") as f:
            f.write(slurm_task_str)
        print("wrote capture_profile_slurm_task.sh")

        with open("capture_profile_slurm.sh", "w") as f:
            f.write(slurm_submit_str)
        print("wrote capture_profile_slurm.sh")

        print(
            "\nTo start capture, do:\nchmod +x capture_profile_slurm_task.sh\nchmod +x capture_profile_slurm.sh\nsbatch capture_profile_slurm.sh"
        )

    else:
        with open("capture_profile.sh", "w") as f:
            f.write(task_capture_str)
        print("wrote capture_profile.sh")

        print(
            "\nTo start capture, do:\nchmod +x capture_profile.sh\n./capture_profile.sh"
        )
