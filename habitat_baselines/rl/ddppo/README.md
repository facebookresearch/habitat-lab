# Decentralized Distributed PPO

Provides changes to the core baseline ppo algorithm and training script to implemented Decentralized Distributed PPO (DD-PPO).
DD-PPO leverages distributed data parallelism to seamlessly scale PPO to hundreds of GPUs with no centralized server.

See the [paper](https://arxiv.org/abs/1911.00357) for more detail.

## Running

There are two example scripts to run provided.  A single node script that leverages `torch.distributed.launch` to create multiple workers:
`single_node.sh`, and a multi-node script that leverages [SLURM](https://slurm.schedmd.com/documentation.html) to create all the works on multiple nodes: `multi_node_slurm.sh`.

The two recommended backends are GLOO and NCCL.  Use NCCL if your system has it, and GLOO if otherwise.

See [pytorch's distributed docs](https://pytorch.org/docs/stable/distributed.html#backends-that-come-with-pytorch)
and [pytorch's distributed tutorial](https://pytorch.org/tutorials/intermediate/dist_tuto.html) for more information.

## Pretrained Models (PointGoal Navigation with GPS+Compass)


All weights available as a zip [here](https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo/ddppo-models.zip).

### Depth models

| Architecture | Training Data | Val SPL | Test SPL | URL |
| ------------ | ------------- | ------- | -------- | --- |
| ResNet50 + LSTM512 | Gibson 4+ | 0.922 | 0.917 | [gibson-4plus-resnet50.pth](https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo/ddppo-models/gibson-4plus-resnet50.pth) |
| ResNet50 + LSTM512 | Gibson 4+ and MP3D(train/val/test)<br/> **Caution:** Trained on MP3D val and test | 0.956 | 0.941 | [gibson-4plus-mp3d-train-val-test-resnet50.pth](https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo/ddppo-models/gibson-4plus-mp3d-train-val-test-resnet50.pth) |
| ResNet50 + LSTM512 | Gibson 2+ | 0.956 | 0.944 | [gibson-2plus-resnet50.pth](https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo/ddppo-models/gibson-2plus-resnet50.pth)|
| SE-ResNeXt50 + LSTM512 | Gibson 2+ | 0.959 | 0.943 | [gibson-2plus-se-resneXt101.pth](https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo/ddppo-models/gibson-2plus-se-resneXt101.pth)|
| SE-ResNeXt101 + LSTM1024 | Gibson 2+ | 0.969 | 0.948 | [gibson-2plus-se-resneXt101-lstm1024.pth](https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo/ddppo-models/gibson-2plus-se-resneXt101-lstm1024.pth)|

### RGB models

| Architecture | Training Data | Val SPL | Test SPL | URL |
| ------------ | ------------- | ------- | -------- | --- |
| SE-ResNeXt50 + LSTM512 | Gibson 2+ and MP3D(train/val/test)<br/> **Caution:** Trained on MP3D val and test | 0.933 | 0.920 | [gibson-2plus-mp3d-train-val-test-se-resneXt50-rgb.pth](https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo/ddppo-models/gibson-2plus-mp3d-train-val-test-se-resneXt50-rgb.pth) |


### Blind Models

| Architecture | Training Data | Val SPL | Test SPL | URL |
| ------------ | ------------- | ------- | -------- | --- |
| LSTM512 | Gibson 0+ and MP3D(train/val/test)<br/> **Caution:** Trained on MP3D val and test | 0.729  |  0.676 | [gibson-0plus-mp3d-train-val-test-blind.pth](https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo/ddppo-models/gibson-0plus-mp3d-train-val-test-blind.pth) |




**Note:** Evaluation was done with *sampled* actions.

All model weights are subject to [Matterport3D Terms-of-Use](http://dovahkiin.stanford.edu/matterport/public/MP_TOS.pdf).


## Citing

If you use DD-PPO or the model-weights in your research, please cite the following [paper](https://arxiv.org/abs/1911.00357):

    @article{wijmans2020ddppo,
      title = {{DD-PPO}: {L}earning Near-Perfect PointGoal Navigators from 2.5 Billion Frames},
      author =  {Erik Wijmans and Abhishek Kadian and Ari Morcos and Stefan Lee and Irfan Essa and Devi Parikh and Manolis Savva and Dhruv Batra},
      journal = {International Conference on Learning Representations (ICLR)},
      year =    {2020}
    }
