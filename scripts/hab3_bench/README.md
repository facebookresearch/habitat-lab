Habitat 3.0 Benchmarking
============================

The utilities in this directory are intended to assist with user system benchmarking for comparison to results reported with the release of Habitat 3.0:

INSERT CITATION HERE


### Running the benchmark
First install habitat-sim and habitat-lab with support for Bullet physics as described in the [installation section](https://github.com/facebookresearch/habitat-lab#installation) of Habitat-lab.

- Download the benchmark assets from huggingface:
Using habitat-sim:
```
# with conda install
python -m habitat_sim.utils.datasets_download --uids hab3_bench_assets

# with source
python /path/to/habitat_sim/src_python/habitat_sim/utils/datasets_download.py --uids hab3_bench_assets
```

Or manually:
 ```
 # put the dataset wherever you want
 git clone https://huggingface.co/datasets/ai-habitat/hab3_bench_assets --username <your HF username> --password <your HF password>
 # create a symblink to the dataset
 ls -s /path/to/hab_3_bench_assets/ data/hab3_bench_assets
 ```

- Run the benchmark: bash scripts/hab3_bench/bench_runner.sh
- Plot the results: python scripts/hab3_bench/h3_plot_bench.py

Alternatively run customized benchmarks with scripts/hab3_bench/hab3_benchmark.py

### Interpreting the results
User systems will likely not match exactly the reported benchmark which was conducted under.
TODO: addtional notes on the different settings
