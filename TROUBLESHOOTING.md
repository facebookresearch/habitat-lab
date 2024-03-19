# Troubleshooting

This is a list of common issues along with troubleshooting tips.

If filing a Github issue, please include:
* Information about your environment (OS, GPU, ...).
* Troubleshooting steps that you have already attempted.
* Screenshots, if applicable.

- [Common Issues](#common-issues)
  - [Habitat-Lab interactive\_play.py](#habitat-lab-interactive_playpy)
  - [Unable to create context](#unable-to-create-context)
  - [Black squares on NVIDIA A100](#black-squares-on-nvidia-a100)
- [Graphics Troubleshooting Tips](#graphics-troubleshooting-tips)
    - [General Tips](#general-tips)
    - [Linux](#linux)
    - [Windows](#windows)


## Common Issues

### Habitat-Lab interactive_play.py

On some systems, `examples/interactive_play.py` crashes due to the following error:

```
X Error of failed request:  BadAccess (attempt to access private resource denied)
```

This is an ongoing issue related to how the underlying `pygame` library interacts with Habitat. A replacement for the `interactive_play` script is [on the roadmap](habitat-hitl/README.md).

### Unable to create context

On some systems, a valid GPU cannot be found:

```
Platform::WindowlessEglApplication::tryCreateContext(): unable to find CUDA device 0 among 2 EGL devices in total
WindowlessContext: Unable to create windowless context
```

This is typically caused by:
* Outdated, missing or invalid graphics drivers.
* On Linux, this can be due to a missing or incomplete `libglvnd` installation.

Follow the graphics troubleshooting steps [below](#graphics-troubleshooting-tips). If the issue persists, feel free to file a Github issue.

### Black squares on NVIDIA A100

NVIDIA A100 GPUs may caused Habitat sensors to render black rectangular artifacts on some environments.

If this manifests on your setup, update your CUDA drivers to a recent version (at least 12.2).

See: https://github.com/facebookresearch/habitat-sim/issues/2310

## Graphics Troubleshooting Tips

These steps aim to narrow down your graphics-related issues.

#### General Tips

1. Increase logging verbosity.

    Launch the application with the following environment variables:
    * `HABITAT_SIM_LOG=Debug`
      * Possible values, most verbose first: `VeryVerbose`, `Verbose`/`Debug`, `Warning` *(default)*, `Error`.
    * `MAGNUM_LOG=verbose`
      * Possible values, most verbose first: `verbose`, `default`, `quiet`.
    * `MAGNUM_GPU_VALIDATION=ON`

2. Get test assets.

    * From the root of your `habitat-lab` or `habitat-sim` repository, run the following commands: `python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/`

3. Verify whether the problem occurs on a minimal setup by running the base viewer.

    * Using the `habitat-sim` Conda packages:
      * `MAGNUM_LOG=verbose MAGNUM_GPU_VALIDATION=ON habitat-viewer data/scene_datasets/habitat-test-scenes/skokloster-castle.glb`

    * Using `habitat-sim` built from source:
      * `MAGNUM_LOG=verbose MAGNUM_GPU_VALIDATION=ON {habitat-sim}/build/viewer data/scene_datasets/habitat-test-scenes/skokloster-castle.glb`

4. Create a new conda environment from scratch. After some time, the environment may diverge from baseline, causing conflicts to emerge.

#### Linux

1. *(NVIDIA)* Run `nvidia-smi`. The header of the output should indicate the expected driver and CUDA versions. For example:

    ```NVIDIA-SMI 535.129.03 Driver Version: 535.129.03 CUDA Version: 12.2```

    If the command returns an error or unexpected versions, reinstall your NVIDIA driver.

2. The command `eglinfo` should work without error.
    * *(NVIDIA, AMD)* The command should indicate that your hardware GPU is available.

3. *(NVIDIA)* Make sure that `libglvnd` is installed and properly configured. This is commonly incorrectly installed.
    1. Check that `libglvnd` is installed.
        * Debian-based: `apt list --installed | grep libglvnd`
        * RPM-based: `dnf list installed | grep libglvnd`

    2. Ensure that a valid entry exists for NVIDIA.
        * In `/usr/share/glvnd/egl_vendor.d/` (specific path may vary), make sure that a `10_nvidia.json` entry exists. If it doesn't exist, you may try creating it manually.
        * This file should have the following content:
            ```
            {
                "file_format_version" : "1.0.0",
                "ICD" : {
                    "library_path" : "libEGL_nvidia.so.0"
                }
            }
            ```
        * The `library_path` field should point at the correct NVIDIA EGL library. It can be found with the command `ldconfig -p | grep libEGL`. The library `libEGL_nvidia` is packaged with the NVIDIA driver - if it's missing, reinstall your drivers. If multiple instances exist, the driver may be incorrectly installed.
4. *(CPU Rendering)* If you don't have a GPU, make sure that your CPU graphics drivers are working.
5. If running on Wayland, try using X11, and vice-versa. While Habitat should work fine with either display protocol, this may shed light on the issue.

#### Windows

Windows is not officially supported. Habitat was reported work from within the linux subsystem (WSL) or virtual machines.
