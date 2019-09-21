# Human Benchmarking setup

1. Install habitat-sim from https://github.com/facebookresearch/habitat-sim
1. Clone habitat-api from https://github.com/facebookresearch/habitat-api and enter that directory
1. Run `git checkout human-benchmarking` to checkout the correct branch!
1. Download `human_benchmarking_data.zip` from https://drive.google.com/open?id=1cXwGpCiky5sKiRaw9TtCIQSDgysJvCwp
    1. Put the zip into the habitat-api directory
    1. Unzip via `unzip human_benchmarking_data.zip`
1. Run `python human_benchmarking.py`, follow the instructions and do as many episodes of pointnav as you want!
    * Make sure to only issue the stop command once you are within 0.2 meters of the target!
1. There will be a file produced ending with `-results.json`.  Send that to Erik.  This file will be updated after ever episode and will persist through different runs, so your data won't be lost!
