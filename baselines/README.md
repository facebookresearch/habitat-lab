baselines
==============================


### Reinforcement Learning (RL)

**Proximal Policy Optimization (PPO)**

**paper**: [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)

**code**: majority of the PPO implementation is taken from 
[pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr).
 
**dependencies**: pytorch 1.0, for installing refer to [pytorch.org](https://pytorch.org/)

For training on sample data please follow steps in the repository README. You should download the sample [test scene data](http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip), extract it under the main repo (`habitat/`, extraction will create a data folder at `habitat/data`) and run the below training command.

**train**:
```bash
python -u baselines/train_ppo.py \
    --use-gae \
    --sim-gpu-id 0 \
    --pth-gpu-id 0 \
    --lr 2.5e-4 \
    --clip-param 0.1 \
    --value-loss-coef 0.5 \
    --num-processes 4 \
    --num-steps 128 \
    --num-mini-batch 4 \
    --num-updates 100000 \
    --use-linear-lr-decay \
    --use-linear-clip-decay \
    --entropy-coef 0.01 \
    --log-file "train.log" \
    --log-interval 5 \
    --checkpoint-folder "data/checkpoints" \
    --checkpoint-interval 50 \
    --task-config "tasks/pointnav.yaml" \


```

**test**:
```bash
python -u baselines/evaluate_ppo.py \
    --model-path "/path/to/checkpoint" \
    --sim-gpu-id 0 \
    --pth-gpu-id 0 \
    --num-processes 4 \
    --count-test-episodes 100 \
    --task-config "tasks/pointnav.yaml" \


```

Set argument `--task-config` to `tasks/pointnav_mp3d.yaml` for training on [MatterPort3D point goal navigation dataset](/README.md#task-datasets).

### Classic

**SLAM** (coming soon)
