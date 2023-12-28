# Troubleshooting and FAQ

This is a list of common issues, FAQs and troubleshooting steps.

If filing a Github issue, please include your environment and the troubleshooting steps that you have attempted.

## Common Issues

### Habitat-Lab interactive_play.py

On some systems, `examples/interactive_play.py` crashes due to the following error:

```
X Error of failed request:  BadAccess (attempt to access private resource denied)
```

This is an ongoing issue related to how the underlying `pygame` library interacts with Habitat. A replacement for the `interactive_play` script is on the roadmap.

### Unable to create context

Driver or graphics configuration issues commonly manifest as such:

```
Platform::WindowlessEglApplication::tryCreateContext(): unable to find CUDA device 0 among 2 EGL devices in total
WindowlessContext: Unable to create windowless context
```

On Linux, this is often caused by an invalid driver or `libglvnd` installation. Follow the troubleshooting steps [below](#troubleshooting-tips). If the issue persists, feel free to file a Github issue.

## Troubleshooting tips

These steps aim to narrow down your graphics-related issues.

#### First Steps:

1. Enable verbose logging.

    Launch the application with the following environment variables: `MAGNUM_LOG=verbose` and `MAGNUM_GPU_VALIDATION=ON`. This will print additional information to the logs that often indicate the problem.

2. Get test assets.

    * From the root of your `habitat-lab` or `habitat-sim` repository, run the following commands: `python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/`

3. Verify whether the problem occurs on a minimal setup by running the base viewer.

    * Using the `habitat-sim` Conda packages:
      * `MAGNUM_LOG=verbose MAGNUM_GPU_VALIDATION=ON habitat-viewer data/scene_datasets/habitat-test-scenes/skokloster-castle.glb`

    * Using `habitat-sim` built from source:
      * `MAGNUM_LOG=verbose MAGNUM_GPU_VALIDATION=ON {habitat-sim}/build/viewer data/scene_datasets/habitat-test-scenes/skokloster-castle.glb`

#### Linux:

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
6. Create a new conda environment from scratch. Occasionally, third party dependencies in the environment may interfere and cause errors.

#### Windows

Windows is not officially supported. Habitat was reported to be working from within the linux subsystem (WSL) or virtual machines.