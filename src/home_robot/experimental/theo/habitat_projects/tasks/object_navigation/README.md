## Table of contents
   1. [Environment setup](#environment-setup)
   2. [Dataset setup](#dataset-setup)
   3. [Demo setup](#demo-setup)
   4. [DETIC setup](#install-detic)
   5. [Run!](#run)

## Environment Setup

```

git clone git@github.com:cpaxton/home-robot-dev.git
cd home-robot-dev
git checkout lang-rearrange-baseline

conda create -n home-robot python=3.10 cmake pytorch -y
conda activate home-robot

git clone https://github.com/3dlg-hcvc/habitat-sim/tree/floorplanner
cd habitat-sim
git checkout floorplanner
pip install -r requirements.txt
python setup.py install --headless
# (if the above commands runs out of memory) 
# python setup.py build_ext --parallel 8 install --headless

cd ..
git clone --branch v0.2.2 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab 
pip install -r requirements.txt
python setup.py develop --all
pip install natsort scikit-image scikit-fmm pandas

cd ..
```

**[IMPORTANT]: Add habitat-lab path to PYTHONPATH**:

```
export PYTHONPATH=$PYTHONPATH:/path/to/home-robot-dev/habitat-lab/
```

[TEMPORARY]: Until we port to habitat v0.2.3.

> Comment out L36 in habitat-lab/habitat/tasks/rearrange/rearrange_sim.py

## Dataset Setup

### Scene dataset setup

```
wget --no-check-certificate https://aspis.cmpt.sfu.ca/projects/scenebuilder/fphab/v0.1.1/fphab.zip -O src/home_robot/experimental/theo/habitat_projects/datasets/scene_datasets/floorplanner.zip
unzip src/home_robot/experimental/theo/habitat_projects/datasets/scene_datasets/floorplanner.zip -d src/home_robot/experimental/theo/habitat_projects/datasets/scene_datasets/
mv src/home_robot/experimental/theo/habitat_projects/datasets/scene_datasets/fphab src/home_robot/experimental/theo/habitat_projects/datasets/scene_datasets/floorplanner
```

[TEMPORARY] Until the scene dataset is updated to reflect new semantic-id mapping:
```
wget https://www.dropbox.com/s/0p1bk1jpd2s7h7k/objects_updated_semantic_mapping.zip -O src/home_robot/experimental/theo/habitat_projects/datasets/scene_datasets/objects_updated_semantic_mapping.zip
unzip src/home_robot/experimental/theo/habitat_projects/datasets/scene_datasets/objects_updated_semantic_mapping.zip -d src/home_robot/experimental/theo/habitat_projects/datasets/scene_datasets/
rm -rf src/home_robot/experimental/theo/habitat_projects/datasets/scene_datasets/floorplanner/configs/objects
mv src/home_robot/experimental/theo/habitat_projects/datasets/scene_datasets/objects src/home_robot/experimental/theo/habitat_projects/datasets/scene_datasets/floorplanner/configs/objects
```


### Episode dataset setup

```
wget https://www.dropbox.com/s/hbpipa4bslussad/val_6_categories.zip -O src/home_robot/experimental/theo/habitat_projects/datasets/episode_datasets/val_6_categories.zip
unzip src/home_robot/experimental/theo/habitat_projects/datasets/episode_datasets/val_6_categories.zip -d src/home_robot/experimental/theo/habitat_projects/datasets/episode_datasets/val_6categories
wget https://www.dropbox.com/s/x2cid4m01a8glci/val_33_categories.zip -O src/home_robot/experimental/theo/habitat_projects/datasets/episode_datasets/val_33_categories.zip
unzip src/home_robot/experimental/theo/habitat_projects/datasets/episode_datasets/val_33_categories.zip -d src/home_robot/experimental/theo/habitat_projects/datasets/episode_datasets/objectgoal_floorplanner_33categories
```

[TEMPORARY] Floorplanner dataset episodes need to point to the right scene dataset config for scenes to load correctly:

> Add the below line after L93 of `habitat-lab/habitat/core/env.py`

```
self.current_episode.scene_dataset_config = "/path/to/home-robot-dev/src/home_robot/experimental/theo/habitat_projects/datasets/scene_datasets/floorplanner/hab-fp.scene_dataset_config.json"
```


## Demo setup

Update `GROUND_TRUTH_SEMANTICS:1` in `src/home_robot/experimental/theo/habitat_projects/tasks/object_navigation/configs/agent/floorplanner_eval.yaml` and run the following:

```
cd src
python home_robot/experimental/theo/habitat_projects/tasks/object_navigation/eval_scripts/eval_specific_episode.py
```

Results are saved to `src/home_robot/experimental/theo/habitat_projects/tasks/object_navigation/datadump/images/debug`.

## Install Detic
```
cd /path/to/home-robot-dev/src/home_robot/agent/perception/detection/detic
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install -r requirements.txt
mkdir models
wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth

# Test it with
wget https://web.eecs.umich.edu/~fouhey/fun/desk/desk.jpg
python demo.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml --input desk.jpg --output out.jpg --vocabulary lvis --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
```

[Optional]: If you face issues with torch's installation, uninstalling and re-installing using conda can help:

```
pip uninstall torch torch-vision
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
```

## Run

> Note: Ensure `GROUND_TRUTH_SEMANTICS:0` in `src/home_robot/experimental/theo/habitat_projects/tasks/object_navigation/configs/agent/floorplanner_eval.yaml` to test DETIC perception.

```
cd /path/to/home-robot-dev/src

# Single episode to debug (ensuring )
python home_robot/experimental/theo/habitat_projects/tasks/object_navigation/eval_scripts/eval_specific_episode.py

# Vectorized evaluation
sbatch home_robot/experimental/theo/habitat_projects/tasks/object_navigation/eval_scripts/eval_vectorized.sh --config_path home_robot/experimental/theo/habitat_projects/tasks/object_navigation/configs/agent/floorplanner_eval.yaml
```
