Habitat 2.0 Benchmarking
============================

The utilities in this directory are intended to assist with user system benchmarking for comparison to results reported in Table 2 with the release of Habitat 2.0:

[Habitat 2.0: Training Home Assistants to Rearrange their Habitat](https://arxiv.org/abs/2106.14405) Andrew Szot, Alex Clegg, Eric Undersander, Erik Wijmans, Yili Zhao, John Turner, Noah Maestre, Mustafa Mukadam, Devendra Chaplot, Oleksandr Maksymets, Aaron Gokaslan, Vladimir Vondrus, Sameer Dharur, Franziska Meier, Wojciech Galuba, Angel Chang, Zsolt Kira, Vladlen Koltun, Jitendra Malik, Manolis Savva, Dhruv Batra. Advances in Neural Information Processing Systems (NeurIPS), 2021.


### Running the benchmark
First install habitat-sim and habitat-lab with support for Bullet physics as described in the [installation section](https://github.com/facebookresearch/habitat-lab#installation) of Habitat-lab.

- Download the benchmark assets: python -m habitat_sim.utils.datasets_download --uids hab2_bench_assets
- Setup the benchmark episode:
   - copy the pre-generated episode from the dataset cp data/hab2_bench_assets/bench_scene.json.gz data/ep_datasets/
   - or generate a new episode: python habitat/datasets/rearrange/run_episode_generator.py --run --config habitat/datasets/rearrange/configs/bench_config.yaml --num-episodes 1 --out data/ep_datasets/bench_scene.json.gz
- Run the benchmark: bash scripts/hab2_bench/bench_runner.sh
- Plot the results: python scripts/hab2_bench/plot_bench.py

Alternatively run customized benchmarks with scripts/hab2_bench/hab2_benchmark.py

### Interpreting the results
User systems will likely not match exactly the reported benchmark which was conducted under the following conditions (quote):

> Benchmarking was done on machines with dual Intel Xeon Gold 6226R CPUs – 32 cores/64 threads
(32C/64T) total – and 8 NVIDIA GeForce 2080 Ti GPUs. For single-GPU benchmarking processes
are confined to 8C/16T of one CPU, simulating an 8C/16T single GPU workstation. For single-GPU
multi-process benchmarking, 16 processes were used. For multi-GPU benchmarking, 64 processes
were used with 8 processes assigned to each GPU. We used python-3.8 and gcc-9.3 for compiling
H2.0. We report average SPS over 10 runs and a 95% confidence-interval computed via standard error
of the mean. Note that 8 processes do not fully utilize a 2080 Ti and thus multi-process multi-GPU
performance may be better on machines with more CPU cores.

TODO: add some example results on
