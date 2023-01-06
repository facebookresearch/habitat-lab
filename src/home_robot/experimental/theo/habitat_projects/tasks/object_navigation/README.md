## Dataset Setup

TODO

## Environment Setup

```
conda create -n home-robot python=3.7
conda install habitat-sim withbullet headless -c conda-forge -c aihabitat-nightly
git clone --branch v0.2.2 https://github.com/facebookresearch/habitat-lab.git; cd habitat-lab; pip install -e .
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch
pip install natsort scikit-image scikit-fmm pandas

# Install Detic
cd /path/to/home-robot-dev/src/home_robot/agent/perception/detection/detic
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install -r requirements.txt
mkdir models
wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth

# Test it with
wget https://web.eecs.umich.edu/~fouhey/fun/desk/desk.jpg
python demo.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml --input desk.jpg --output out.jpg --vocabulary lvis --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
```

## Run

```
cd /path/to/home-robot-dev/src

# Single episode to debug
python home_robot/experimental/theo/habitat_projects/tasks/object_navigation/eval_scripts/eval_specific_episode.py

# Vectorized evaluation
sbatch home_robot/experimental/theo/habitat_projects/tasks/object_navigation/eval_scripts/eval_vectorized.sh --config_path home_robot/experimental/theo/habitat_projects/tasks/object_navigation/configs/agent/floorplanner_eval.yaml
```
