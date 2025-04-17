# Habitat Dataset Processing

Asset pipeline for processing Habitat datasets for external engines (e.g. Unity).

The processed datasets retain the directory structure of the input (e.g. `data/dataset/asset.obj`), enabling replay rendering in other engines.

- [Installation](#installation)
  - [Magnum Installation](#magnum-installation)
    - [MacOS](#macos)
    - [Linux](#linux)
- [Usage](#usage)
  - [Example usage:](#example-usage)
  - [Parameters](#parameters)
  - [Script Configuration](#script-configuration)
  - [Data Sources](#data-sources)
    - [AssetSource](#assetsource)
    - [HabitatDatasetSource](#habitatdatasetsource)
  - [metadata.json](#metadatajson)
    - [Group types](#group-types)


## Installation

1. Create a new conda environments for dataset processing.

```
conda create -n datasets python=3.10
conda activate datasets
```

2. Install the project.

```
cd scripts/habitat_dataset_processing
pip install -e .
```

3. Install Magnum. See the section below.

### Magnum Installation

[Magnum](https://github.com/mosra/magnum) python bindings are required to decimate and process assets. The Magnum version packaged with Habitat lacks necessary dependencies. Therefore a separate environment must be set up to process the datasets.

#### MacOS

1. Follow [magnum-bindings installation instructions](https://doc.magnum.graphics/python/building/#homebrew-formulas-for-macos).
   * In addition to `corrade`, `magnum`, and `magnum-bindings`, you may need `magnum-plugins`.
2. Test your install: `python -c "from magnum import math, meshtools, scenetools, trade"`.
   * Beware homebrew installs python packages like magnum to its own Python location, not your current conda environment's Python.
   * Depending on how Homebrew has installed Python, you may need to use `python3` instead of `python`.

#### Linux

It is recommended that you create a new `conda` environment so that it can be reused in the future without interfering with Habitat.

3. Download magnum binaries for Linux.
   * Navigate to the [Magnum CI](https://github.com/mosra/magnum-ci/actions/workflows/magnum-tools.yml).
   * Select the latest green workflow run.
   * Scroll down to "Artifacts".
   * Download your the binaries that match your system (e.g. On Linux: `magnum-tools-v2020.06-...-linux-x64`)
   * Extract to a convenient location.
4. Navigate to the `site-packages` of your new environment, e.g. `~/anaconda/envs/datasets/lib/python3.10/site-packages/`.
5. Create a `magnum.pth` file in this directory.
6. Add the absolute path to `magnum-tools`'s `python` folder to this file, e.g:
```
/home/USER/Documents/magnum-tools/linux-x64/python/
```
7. The Magnum libraries will now be included upon activating your `magnum` environment. Run these commands to validate:
   * `conda activate magnum`
   * `python -c "from magnum import math, meshtools, scenetools, trade"`

## Usage

Using this package, Python scripts can be used to configure data processing jobs. The process is as follows:

1. Create a Python script that configures the processing job.
   * Examples are provided in the `examples` subdirectory.
2. Invoke the script to initiate the dataset processing.

The result output directory retains the directory structure of the input `data/` folder, allowing the target engine to locate the assets from the same relative paths as Habitat. This allows for replay-rendering Habitat sessions from within another engine.

### Example usage:
```bash
python processing_script.py --input path/to/data
```

### Parameters

| Parameter | Description |
| -------- | ------- |
| --input | Path of the `data/` directory to process. |
| --output | Output directory. Will be created if it doesn't exist. |
| --verbose | Increase logging verbosity. |
| --debug | Disable multiprocessing to allow for debugging. |

### Script Configuration

Refer to the examples provided in the `examples` subdirectory.

### Data Sources

There are two ways to define a dataset to process: `AssetSource` and `HabitatDatasetSource`.

#### AssetSource

An `AssetSource` is used to locate a loose collection of assets.
A list of glob patterns is defined to local the assets.

This example shows how to import all `.glb` files in `humanoids`.
```
AssetSource(
   name="humanoids",
   assets=["humanoids/humanoid_**/*.glb"],
   settings=ProcessingSettings(
         operation=Operation.COPY,
         decimate=False,
   ),
)
```

#### HabitatDatasetSource

A `HabitatDatasetSource` is used to process a Habitat dataset (defined as a `scene_dataset_config.json`).
The path to the `scene_dataset_config.json` is used to locate the assets.

These datasets contain one or more of these subgroups:
* `stages`: Scene 3D models.
* `objects`: Object 3D models.
* `articulated_objects`: Articulated object 3D models (typically `.urdf`).

**Object Datasets**

Some datasets only contain objects and/or articulated objects.

Example (`ai2thorhab`):
```
HabitatDatasetSource(
   name="ai2thor_object_dataset",
   dataset_config="objects/ovmm_objects/train_val/ai2thorhab/ai2thor_object_dataset.scene_dataset_config.json",
   objects=ProcessingSettings(
      operation=Operation.PROCESS,
      decimate=False,
   ),
)
```

**Scene Datasets**

Other datasets contain scenes. It is possible to define a blacklist or whitelist to only include objects referenced by specific scenes.

Example (`hssd-hab`):

```
HabitatDatasetSource(
   name="hssd-hab-articulated",
   dataset_config="hssd-hab/hssd-hab-articulated.scene_dataset_config.json",
   stages=ProcessingSettings(
         operation=Operation.PROCESS,
         decimate=False,
         group=GroupType.GROUP_BY_SCENE,
   ),
   objects=ProcessingSettings(
         operation=Operation.PROCESS,
         decimate=True,
         group=GroupType.GROUP_BY_SCENE,
   ),
   articulated_objects=ProcessingSettings(
         operation=Operation.PROCESS,
         decimate=False,
         group=GroupType.GROUP_BY_SCENE,
   ),
   scene_whitelist=["102344250"],   # Only process `102344250.scene_instance.json`
   include_orphan_assets=False,     # Exclude objects that are not referenced by any scene.
)
```

### metadata.json

The output root contains a `metadata.json` file that includes hints on how to import and package the data in an external engine. This file is optional.
It lists how assets should be grouped, which can be used to generate downloadable bundles for WebGL builds, for example.


#### Group types

| Group Type | Description |
| -------- | ------- |
| `LOCAL` *(default)* | Hint that the assets should be included with the build. This increases the build size, which may be problematic on Android on WebGL targets. |
| `GROUP_BY_DATASET` | Hint that all assets within the data source should be grouped together. For example, because a robot is always loaded in its entirety, all robot assets should be grouped together. |
| `GROUP_BY_CHUNK` | Hint that assets should be grouped in chunks of N assets. For example, object datasets may have thousands of objects; grouping them by chunk balances the number of packages and wasted download bandwidth. |
| `GROUP_BY_SCENE` | Hint to group assets by scene. If the asset is in multiple scenes, it will be listed in multiple groups. Only works for `HabitatDatasetSource` with scenes. |
