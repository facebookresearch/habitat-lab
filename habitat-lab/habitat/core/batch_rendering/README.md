# Batch Rendering

This document provides instructions to train the Habitat 2.0 rearrangement tasks with batch rendering, with the goal to save GPU memory and enable more parallel environments.

You can also adapt these instructions to your own Habitat-lab training experiments, with [some caveats mentioned below](#limitations).

## Summary

Distributed training is done with a separate Python process per environment, each process having a Habitat simulator instance.

In the current (non-batched) paradigm, each simulator instance has its own renderer, which includes its own graphics context. Because they are independent, they each load their own copy of the resource-intensive assets. Therefore, as more concurrent simulators are added, GPU memory quickly fills up, limiting how wide training can scale.

Batch rendering is an *experimental* system that solves this issue by centralizing rendering. It changes the paradigm such as simulators do not have a renderer. Instead, a single renderer on the main training process aggregates simulation states and renders them simultaneously.

At the beginning of training, all graphics assets that will be used during a roll-out are pre-loaded exactly once. This leads to less GPU memory usage, reduced episode loading time and more efficient rendering (less drawcalls, more instancing, less context switching, better data locality, ...).

This feature is currently being developed. Refer to the [limitations](#limitations) section for support status.

## Setup:

### Composite files:

The batch renderer can load assets from a "composite file", which is a single *gltf* file that contains an entire dataset. Multiple composite files can be used. This is the preferred way to use the batch renderer due to increased performance.

We provide the composite file for ReplicaCAD. If your experiment uses different datasets (e.g. HM3D or HSSD), we encourage you to try out the batch renderer anyways. In this case it will load your 3D assets from individual files. The training speedup will be less than ideal because this use of individual model files limits batching efficiency, but you will still see GPU memory savings.

We don't yet offer a pipeline for creating your own composite GLTF files, but this is coming soon.

**Downloads**

The following composite files are available for download:

* [ReplicaCAD](https://drive.google.com/drive/folders/1zA6Bib_uNPPRgDOQqzQR_4uT460SzBkv)

**Configuration**

From the launch command:
```
habitat-baselines/habitat_baselines/run.py habitat.simulator.renderer.composite_files=[path/to/composite_1.gltf, path/to/composite_2.gltf]
```
From configuration:
```
habitat.simulator.renderer.composite_files:
    - path/to/composite_1.gltf
    - path/to/composite_2.gltf
```

### Training:

The following command launches training with the batch renderer. Refer to [this section](#composite-files) to set the composite file value.

```
habitat-baselines/habitat_baselines/run.py --config-name=rearrange/rl_skill.yaml habitat_baselines.trainer_name="ppo" habitat.simulator.renderer.enable_batch_renderer=True habitat.simulator.habitat_sim_v0.enable_gfx_replay_save=True habitat.simulator.create_renderer=False habitat.simulator.concur_render=False habitat.simulator.renderer.composite_files=[path/to/composite_file.gltf]
```

### Configuration:

To train with the batch renderer, the following configuration fields are required. The launch command above contains the same overrides.

```
habitat_baselines.trainer_name: "ppo"
habitat.simulator.renderer.enable_batch_renderer: True
habitat.simulator.habitat_sim_v0.enable_gfx_replay_save: True
habitat.simulator.create_renderer: False
habitat.simulator.concur_render: False
```

## Integration Notes:

The batch renderer is initialized with `VectorEnv.initialize_batch_renderer()`.

When `reset()` or `step()` are called, the simulators emplace gfx-replay keyframes into observations. See `HabitatSim.add_keyframe_to_observations()`. The simulators won't render visual sensors. Instead, they leave placeholder observations (e.g. `"uuid: None"`).

After all simulators have stepped, `VectorEnv.post_step()` has to be called to render the observations. Internally, this calls `EnvBatchRenderer.post_step()`. During this process, all keyframes are consumed, and placeholder observations are replaced by final renders.

See `test_rl_vectorized_envs_batch_renderer` ([link](https://github.com/facebookresearch/habitat-lab/blob/main/test/test_habitat_env.py#L298)) for an example.

## Limitations:

### Training:

* Only PPO is supported.

### Sensors:

* As of now, only **1 sensor** can be used.
* Semantic sensors are not supported.
* Sensor noise models are not supported.

### Features:

* PBR is not implemented.
* Skinned meshes do not work.
* Primitives are not rendered.
* Gfx-replay captures cannot be recorded while batch rendering.
* Debug rendering does not work (`habitat.simulator.debug_render`).

### Platform:

* MacOS is not yet supported.
* GPU-GPU (CUDA) is not supported.
