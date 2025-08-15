

# Building for an existing conda env
```
cd /path/to/mochi

# pick a build folder for our python version
PYVER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
BUILD_DIR="build-py$PYVER"

echo "Building Mochi python extension to $BUILD_DIR..."
mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
cmake .. -DMOCHI_BUILD_ALL=ON
cmake --build . -- -j$(nproc)

pip install -e mochi_gym/

# configure MOCHI_HOME and PYTHONPATH
export MOCHI_HOME="/home/eric/projects/mochi"
export PYTHONPATH="$MOCHI_HOME/mochi_agents_api/py_extension:$PYTHONPATH"
export PYTHONPATH="$MOCHI_HOME/$BUILD_DIR/bin:$PYTHONPATH"
```


# Building and testing Mochi in isolation

```
# Clone Mochi repo. Don't clone inside habitat-lab!
cd ~
git clone https://github.com/facebookresearch/mochi 
cd mochi

# set up a python env just for Mochi
mamba env create -f ./mochi_gym/mochi_gym.yml
mamba activate mochi_gym

# build
mkdir -p build && cd build
cmake .. -DMOCHI_BUILD_ALL=ON
cmake --build . -- -j$(nproc)

# optional: build debug in case you want to step through C++ code
mkdir -p build_debug && cd build_debug
cmake .. -DMOCHI_BUILD_ALL=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build . -- -j$(nproc)

# run hello world
cd ../
./build/bin/mochi_hello_world

# Get assets folder from fbsource and set MOCHI_ASSETS_PATH. Below instructions only work for a Meta-provisioned machine. We should find a clean way to make these assets available on non-Meta machines.
fbclone fbsource --eden
# beware fbclone is doing lazy cloning (?); make sure assets folder exists on your local machine
ls ~/fbsource/arvr/libraries/mochi/common/assets
export MOCHI_ASSETS_PATH="$HOME/fbsource/arvr/libraries/mochi/common/assets"

# run sample browser and play with some samples
./build/bin/mochi_samples_app

pip install -e mochi_gym/

# configure MOCHI_HOME and PYTHONPATH
export MOCHI_HOME="/home/eric/projects/mochi"
export PYTHONPATH="$MOCHI_HOME/mochi_agents_api/py_extension:$PYTHONPATH"
export PYTHONPATH="$MOCHI_HOME/build/bin:$PYTHONPATH"

# run various python programs
python ./mochi_gym/apps/demos/demo_benchmarks.py cartpole --save_to_disk

python ./mochi_gym/apps/rllib/train_benchmarks.py --num_env_runners 16 --pattern cartpole

python ./mochi_gym/mochi_gym/envs/allegro_in_hand_env.py
```