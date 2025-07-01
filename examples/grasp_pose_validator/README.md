# Grasp Pose Validator

## Context

This tool generates offline datasets of valid grasp poses and corresponding images within fully furnished scene contexts. Given initial candidate grasp poses for target objects, it validates each pose using collision detection and other feasibility checks. Only grasps deemed feasible are included in the output, supporting robust benchmarking and training of grasping algorithms in realistic, context-rich environments.

## Installation

1. ### Create conda env
    ```bash
    # We recommend python=3.10 and cmake=3.31.6
    conda create -n habitat python=3.10 cmake=3.31.6
    conda activate habitat
    ```

1. ### Clone and install habitat-sim:
    From your project parent directory:
    ```bash
    git clone https://github.com/facebookresearch/habitat-sim
    cd habitat-sim
    pip install -r requirements.txt
    # build from source (adjust the number of parallel threads for your system to avoid overflowing RAM)
    python setup.py build_ext --parallel 6 install --bullet

    # verify install with
    python -c "import habitat_sim; print(habitat_sim)"

    ```

1. ### Install habitat-lab (this branch)
    From your project parent directory:
    ```
    git clone --branch alex_06-26_grasp_pose_validator https://github.com/facebookresearch/habitat-lab.git

    cd habitat-lab
    pip install -e habitat-lab

    #optionally symblink your data directory to habitat-sim/data
    ln -s ../habitat-sim/data data/

    #otherwise create a data directory
    mkdir data
    ```

## Download Data

Download the required datasets and place them in the `data/` directory. From the `habitat-lab/data/` directory:


1. [hab_murp](https://huggingface.co/datasets/ai-habitat/hab_murp) - the robot URDF and assets
    ```bash
    git clone https://huggingface.co/datasets/ai-habitat/hab_murp
    ```
2. [hssd-hab](https://huggingface.co/datasets/hssd/hssd-hab) - the HSSD scenes and assets (~12 GB)
    ```bash
    git clone https://huggingface.co/datasets/hssd/hssd-hab
    ```
3. (temp) [Habitat 3.0 HSSD RearrangeEpisode dataset](https://huggingface.co/datasets/ai-habitat/hab3_episodes) - TODO: we'll replace these with new targeted episodes.

    From your `habitat-lab/` root directory run:
    ```bash
    #with habitat-sim already installed use the dataset downloader
    python -m habitat_sim.utils.datasets_download --uids hab3-episodes ycb
    ```

## Run the Script

To run the prototype script from your `habitat-lab/` root directory:
```bash
python examples/grasp_pose_validator/grasp_pose_validator.py
```
For additional options, use:
```bash
python examples/grasp_pose_validator/grasp_pose_validator.py --help
```

## Generating New Episodes
Episodes can be generated from configuration in `hssd_single_obj.yaml`.

NOTE: we are explicitly targeting object `08437cca1420fadbceb39818cc68a000eb0cf37e` for this prototype.

TODO: we'll expand this to more scenes

Run the following from your `habitat-lab/` root directory to generate a dataset with one compatible RearrangeEpisode:

```bash
python habitat-lab/habitat/datasets/rearrange/run_episode_generator.py --config examples/grasp_pose_validator/hssd_single_obj.yaml --out data/datasets/grasp_pose_validator_episodes.json.gz --run
```


## TODOs:
- align on the correct hand model and DoF indexing
- load grasp poses from external datasets for the "star object"
- sampling camera positions with occlusion checking
- collision filtering and other feasibility checks
- depth and semantic mask images
- batching for large numbers of episodes
- bespoke episode generation logic
