Variable Experience Rollout (VER)
=================================


Implementation of variable experience rollout (VER). VER applies the sampling techniques for async to sync and collects a variable amount of experience for each environments rollout to mitigate the straggler effect.


The system has 5 components. 3 of these are the key components for learning:

* Environment workers receive actions to simulate. They write the output of their environment into shared memory.
* Inference workers receive batches of environment steps and perform inference. They write the next action to take into shared memory. They then write the batch of experience for learning into GPU shared memory.
* The Learner takes batches of experience and updates the policy.

How these pieces are connected together is best seen via this diagram:

![VER System Diagram](/habitat_baselines/rl/ver/images/ver-system.svg)

The Shared CPU Memory block is implemented via the `transfer_buffers` and the Shared GPU Memory block is implemented via the `rollouts`.

There are two components that serve auxiliary functions:

* The Report worker is responsible for tracking the progress of training. It writes metrics to tensorboard (or wandb). This lives in a separate process for maximum training speed.
* The Preemption decider decides when to preempt stragglers when combining VER and DD-PPO

## Citing

If you use VER or the model-weights in your research, please cite the following [paper](https://tbd):

    @inproceedings{wijmans2022ver,
      title = {{VER}: {S}caling On-Policy RL Leads to the Emergence of Navigation in Embodied Rearrangement},
      author =  {Erik Wijmans and Irfan Essa and Dhruv Batra},
      booktitle = {arXiv},
      year =    {2022}
    }
