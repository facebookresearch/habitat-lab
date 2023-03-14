Variable Experience Rollout (VER)
=================================


Implementation of variable experience rollout (ver). VER applies the sampling techniques for async to sync and collects a variable amount of experience for each environments rollout to mitigate the straggler effect.


The system has 5 components. 3 of these are the key components for learning:

* Environment workers receive actions to simulate. They write the output of their environment into shared memory.
* Inference workers receive batches of environment steps and perform inference. They write the next action to take into shared memory. They then write the batch of experience for learning into GPU shared memory.
* The Learner takes batches of experience and updates the policy.

How these pieces are connected together is best seen via this diagram:

![ver System Diagram](images/ver-system.svg)

The Shared CPU Memory block is implemented via the `transfer_buffers` and the Shared GPU Memory block is implemented via the `rollouts`.

There are two components that serve auxiliary functions:

* The Report worker is responsible for tracking the progress of training. It writes metrics to tensorboard (or wandb). This lives in a separate process for maximum training speed.
* The Preemption decider decides when to preempt stragglers when combining VER and DD-PPO

## Usage

To use VER, simply change the `trainer_name` to `"ver"` in either the config or via the command line:

```bash
python -u -m habitat_baselines.run \
  --config-name=pointnav/ppo_pointnav_example.yaml \
  habitat_baselines.trainer_name=ver
```

By default, this will configure the VER with the recommended settings: variable experience rollouts, 2 inference workers, and non-overlapped experience collection and learning. Depending on your workload, you may get better performance by changing these values. Here are some guidelines for when
to change these values.

If you have environments with extreme differences in simulation time (i.e. the fastest environment is more than 100x faster than the slowest), consider disabling variable experience rollouts.

```bash
python .... \
  habitat_baselines.rl.ver.variable_experience=False
```

If you have a very small policy, consider reducing the number of inference workers. If you have a very large model, consider increasing the number of inference workers.

```bash
python .... \
  habitat_baselines.rl.ver.num_inference_workers=<number of inference workers>
```

If your environment is largely dominated by CPU time, consider overlapping experience collection and learning. This will harm sample efficiency and increase memory usage but can be worthwhile in certain cases.

```bash
python .... \
  habitat_baselines.rl.ver.overlap_rollouts_and_learn=True
```

## Citing

If you use VER in your research, please cite the following [paper](https://arxiv.org/abs/2210.05064):

    @inproceedings{wijmans2022ver,
      title = {{VER}: {S}caling On-Policy RL Leads to the Emergence of Navigation in Embodied Rearrangement},
      author =  {Erik Wijmans and Irfan Essa and Dhruv Batra},
      booktitle = {Conference on Neural Information Processing Systems (NeurIPS)},
      year =    {2022}
    }

If you use VER on 1 more than 1 GPU, please also cite the [DD-PPO paper](https://arxiv.org/abs/1911.00357):

    @inproceedings{wijmans2020ddppo,
      title = {{DD-PPO}: {L}earning Near-Perfect PointGoal Navigators from 2.5 Billion Frames},
      author =  {Erik Wijmans and Abhishek Kadian and Ari Morcos and Stefan Lee and Irfan Essa and Devi Parikh and Manolis Savva and Dhruv Batra},
      booktitle = {International Conference on Learning Representations (ICLR)},
      year =    {2020}
    }
