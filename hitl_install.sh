#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Conda environment
conda create -y -n siro-hitl-eval python=3.9
conda activate siro-hitl-eval

# Hab-lab
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout e7c17e49
pip install -e habitat-lab
pip install -e habitat-baselines
cd ..
echo "Step:1/7 Hab-lab SIRo install -- done"

# Hab-sim
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
git checkout d32d7510
python setup.py install --bullet
echo "Step:2/7 Hab-sim install -- done"

# humanoid data
cd ../habitat-lab
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1gSNq4oloeX7ZI8mJbr0Kn9sXB5GFDlJc' -O humanoids_skinned.zip
unzip humanoids_skinned.zip -d "data"
rm humanoids_skinned.zip
# Using humanoid skeleton instead of skinned human
cp data/humanoids/humanoid_data/female2_0.urdf data/humanoids/humanoid_data/female2_0_rigid.urdf
echo "Step:3/7 Get humanoid data -- done"

# robot and objects data
python -m habitat_sim.utils.datasets_download --uids ycb hab_fetch hab_spot_arm replica_cad_dataset rearrange_pick_dataset_v0 rearrange_dataset_v1 --data-path data/

# robot policy checkpoints
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1LGngFpIgDGuCrHItydkH10MQeDti78uy' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1LGngFpIgDGuCrHItydkH10MQeDti78uy" -O checkpoints.zip && rm -rf /tmp/cookies.txt
unzip checkpoints.zip -d "checkpoints"
rm checkpoints.zip
echo "Step:4/7 Get robot policies -- done"

# Floorplanner dataset
mkdir -p data/datasets/floorplanner/rearrange/scratch/train
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1e1AGXXcLisDI48oTRHnE3qobBidCpGnT' -O FPDatasets.zip
unzip FPDatasets.zip -d "data/datasets/floorplanner/rearrange/scratch/train"
rm FPDatasets.zip
echo "Step:5/7 Get floorplanner rearrange dataset -- done"

# more objects
cd habitat-lab/data/objects
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1k7GX1Xf94acJOhAGdyHTN9Dqcm3_D81y' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1k7GX1Xf94acJOhAGdyHTN9Dqcm3_D81y" -O google_object_dataset.tar.gz && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1E24Mc--E8rWNRUa4GDoH1WeJYVNSxK1a' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1E24Mc--E8rWNRUa4GDoH1WeJYVNSxK1a" -O amazon_berkeley.tar.gz && rm -rf /tmp/cookies.txt
tar -xvf google_object_dataset.tar.gz
tar -xvf amazon_berkeley.tar.gz
echo "Step:6/7 Get objects -- done"

# FPSS/Floorplanner scenes from hugging face
cd habitat-lab/data

case "$OSTYPE" in
  solaris*) echo "SOLARIS" ;;
  darwin*)  echo "OSX" ;; 
  linux*)   echo "LINUX" ;;
  bsd*)     echo "BSD" ;;
  msys*)    echo "WINDOWS" ;;
  *)        echo "unknown: $OSTYPE" ;;
esac

if [[ "$OSTYPE" =~ ^darwin ]]; then
    brew install git-lfs
fi

if [[ "$OSTYPE" =~ ^linux ]]; then
    sudo apt-get install git-lfs
fi

git lfs install
git clone https://huggingface.co/datasets/fpss/fphab
git checkout 6fb800903
mv fphab fpss

echo "Step:7/7 Get floorplanner scenes -- done"

