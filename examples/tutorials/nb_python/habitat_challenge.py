# ---
# jupyter:
#   accelerator: GPU
#   colab:
#     collapsed_sections: []
#     name: habitat_challenge.ipynb
#     provenance: []
#   jupytext:
#     cell_metadata_filter: -all
#     formats: nb_python//py:percent,colabs//ipynb
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     name: python3
#   pycharm:
#     stem_cell:
#       cell_type: raw
#       metadata:
#         collapsed: false
#       source: []
# ---

# %% [markdown]
# # Habitat Challenge Tutorial
#
# ## Challenge page: https://aihabitat.org/challenge/2020/
#
# ## Challenge starter code: https://github.com/facebookresearch/habitat-challenge

# %% [markdown]
# # Install dependencies:
#
# *   git, wget, zip
# *   Nvidia drivers and CUDA
# *   Conda
# *   Docker
# *   Nvidia Docker v2
# *  EvalAI and auth token:
# ![set_token_screen](https://drive.google.com/uc?id=1LcJCIW6MNtvv52Gbs6VcqFWnJIGqvraI)

# %%
# Install dependencies
# !sudo apt-get update || true
# !sudo apt-get install -y --no-install-recommends \
#     build-essential \
#     git \
#     curl \
#     vim \
#     ca-certificates \
#     pkg-config \
#     wget \
#     zip \
#     unzip || true

# Install nvidia drivers and cuda
# !wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
# !sudo dpkg -i cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
# !sudo apt-get update || true
# !sudo apt-get --yes --force-yes install cuda
# !touch ./cuda_installed
# !nvidia-smi

# Install conda and dependencies
# !curl -o /content/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# !bash $HOME/miniconda.sh -b -p $HOME/miniconda
# !rm ~/miniconda.sh
# !export PATH=$HOME/miniconda/bin:/usr/local/cuda/bin:$PATH
# !conda create -y -n habitat python=3.6
# !conda activate habitat

# Install Docker
# !export PATH=$HOME/miniconda/bin:/usr/local/cuda/bin:$PATH
# !conda activate habitat;
# !curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
# !sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
# !sudo apt-get update
# !sudo apt-get install -y docker-ce
# !apt-cache policy docker-ce

# Install Nvidia Docker
# !curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
# !distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
# !curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
#   sudo tee /etc/apt/sources.list.d/nvidia-docker.list
# !sudo apt-get update
# !sudo apt-get install nvidia-docker2
# !sudo pkill -SIGHUP dockerd


# EvlaAI install
# !pip install "evalai>=1.2.3"
# Set EvalAI account token
# !evalai set_token $EVALAI_TOKEN

# %% [markdown]
# # Clone habitat-challenge repo and download required data:
#
# *   Clone https://github.com/facebookresearch/habitat-challenge
# *   Gibson scenes dataset from https://github.com/StanfordVL/GibsonEnv#database after signing an agreement.
# *   Task episodes dataset: PointNav v2 episodes for Gibson scenes
# *   DDPPO baseline pretrained checkpoint
#
#

# %%
# %cd ~
# !sudo rm -rf habitat-challenge
# !git clone https://github.com/facebookresearch/habitat-challenge
# %cd habitat-challenge

# Download Gibson scenes dataset from https://github.com/StanfordVL/GibsonEnv#database after signing an agreement
# !mkdir -p habitat-challenge-data/data/scene_datasets
# !cp -r $PATH_TO_SCENE_DATASETS habitat-challenge-data/data/

# Task episodes dataset: PointNav v2 episodes for Gibson
# !mkdir -p habitat-challenge-data/data/datasets/pointnav/gibson
# !wget -c https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v2/pointnav_gibson_v2.zip && unzip -o pointnav_gibson_v2.zip -d habitat-challenge-data/data/datasets/pointnav/gibson

# DDPPO baseline
# !wget https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo_pointnav_habitat2020_challenge_baseline_v1.pth


# %% [markdown]
# # Build Docker image Pointnav_DDPPO_baseline

# %%
# !cat Pointnav_DDPPO_baseline.Dockerfile
# !docker build . --file Pointnav_DDPPO_baseline.Dockerfile -t ddppo_pointnav_submission

# %% [markdown]
# # Run evaluation locally (takes 5 min)

# %%
# !bash /test_locally_pointnav_rgbd.sh --docker-name ddppo_pointnav_submission

# %% [markdown]
# # Push docker image to EvalAI Validation mini_val stage (50 episodes)
# Check results on [the PointGoalNav v2 Minival stage leaderboard](https://evalai.cloudcv.org/web/challenges/challenge-page/580/leaderboard/1630).
#
# ![leaderboard](https://drive.google.com/uc?id=1cvdEfAkNrTRA9GLtfizgIylwk_tgHigU)

# %%
# !evalai push ddppo_pointnav_submission:latest --phase habitat20-pointnav-minival


# %% [markdown]
# # Push docker image to EvalAI Test stage
# *Can take up to 36 hours to get result.*

# %%
# Push docker image to EvalAI docker registry
# !evalai push ddppo_pointnav_submission:latest --phase habitat20-pointnav-test-std

# %%
# !evalai submission 94203

# %% [markdown]
# # Happy hacking!

# %% [markdown]
#
# # Citing Habitat Challenge 2020
# Please cite [the following paper](https://arxiv.org/abs/1912.06321) for details about the 2020 PointNav challenge:
#
# ```
# @inproceedings{habitat2020sim2real,
#   title     =     {Are {W}e {M}aking {R}eal {P}rogress in {S}imulated {E}nvironments? {M}easuring the {S}im2{R}eal {G}ap in {E}mbodied {V}isual {N}avigation},
#   author    =     {{Abhishek Kadian*} and {Joanne Truong*} and Aaron Gokaslan and Alexander Clegg and Erik Wijmans and Stefan Lee and Manolis Savva and Sonia Chernova and Dhruv Batra},
#   booktitle =     {arXiv:1912.06321},
#   year      =     {2019}
# }
# ```
#
# Please cite [the following paper](https://arxiv.org/abs/2006.13171) for details about the 2020 ObjectNav challenge:
# ```
# @inproceedings{batra2020objectnav,
#   title     =     {Object{N}av {R}evisited: {O}n {E}valuation of {E}mbodied {A}gents {N}avigating to {O}bjects},
#   author    =     {Dhruv Batra and Aaron Gokaslan and Aniruddha Kembhavi and Oleksandr Maksymets and Roozbeh Mottaghi and Manolis Savva and Alexander Toshev and Erik Wijmans},
#   booktitle =     {arXiv:2006.13171},
#   year      =     {2020}
# }
# ```
