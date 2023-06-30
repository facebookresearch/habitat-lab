# Batch Rendering

To scale up training, multiple concurrent simulators are instantiated. Without batch rendering, each simulator renders observations independently. Therefore, they each require their own graphics context and memory allocations.

Batch rendering is an *experimental* system that renders all environments simultaneously using a single centralized renderer. Instead of loading assets independently, all graphics assets that will be used during a roll-out are pre-loaded exactly once at the beginning of training.

This leads to less GPU memory usage, reduced episode loading time and more efficient rendering (less drawcalls, more instancing, less context switching, better data locality, ...).

## Setup:

This feature is currently being developed. Refer to the [limitations](#limitations) section for support status.

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

### Composite files:

The batch renderer can optionally pre-load assets from a "composite file", which is a single *gltf* file that contains an entire dataset. It is possible to pre-load multiple composite files. This is the preferred way to use the batch renderer due to increased performance.

To use them, the following configuration field must be set:
```
habitat.simulator.renderer.composite_files:
    - path/to/composite_1.gltf
    - path/to/composite_2.gltf
```

The following composite files are available:
* [ReplicaCAD](https://drive.google.com/drive/folders/1zA6Bib_uNPPRgDOQqzQR_4uT460SzBkv)

If an asset is not found within the provided composite files, it will be loaded from disk as a fallback. A warning is emitted when this occurs.

## How it works:

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
