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
  - [Processing Scripts](#processing-scripts)


## Installation

1. Create a new conda environments for dataset processing.

```
conda create -n unity python=3.10
conda activate unity
```

2. Install the project.

```
cd scripts/unity_dataset_processing
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
4. Navigate to the `site-packages` of your new environment, e.g. `~/anaconda/envs/unity/lib/python3.10/site-packages/`.
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

### Processing Scripts

Refer to the examples provided in the `examples` subdirectory.