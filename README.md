habitat-api
==============================


A suite to train embodied agents across a variety of tasks, 
environments, simulators. habitat-api defines an API through which users can 
train agents on existing embodied tasks as well as add new simulators, 
tasks. It also aims to make benchmarking of agents across datasets, 
environments and tasks easier. 

<img src='docs/img/habitat-api_structure.png' alt="teaser results" width="100%"/>

### Setup


1. Clone the github repository and install using:
```bash
cd habitat-api
pip install -e .
```

2. Install `habitat-sim` from: [github repo](https://github.com/facebookresearch/habitat-sim).

3. Add `habitat-sim` to `PYTHONPATH`: 
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/habitat-sim/:/path/to/habitat-sim/build/esp/bindings"
```

4. Download test data from [dropbox share](https://www.dropbox.com/sh/dl/h02865ucoh3ix07/AABkVrHCfPI0BAmSeHCytrsya) and place it in folder:`data/habitat-sim/test/`


### Example

```python
import habitat
from habitat.config.default import cfg
from habitat.tasks.nav.nav_task import NavigationEpisode

config = cfg()
config.freeze()
env = habitat.Env(config=config)
env.episodes = [NavigationEpisode(episode_id='0',
                                  scene_id=config.SIMULATOR.SCENE,
                                  start_position=None,
                                  start_rotation=None,
                                  goals=[])]

observations = env.reset()

for i in range(100):
    # randomly move around inside the environment
    observations = env.step(env.action_space.sample())

    # observations by default contains rgb, semantic and depth modalities
    if env.episode_over:
        observations = env.reset()     
```

### Data
To set up data folder with symlink to relevant data for devfair you can run:

 `bash habitat/internal/data/download_data_fair.sh`

**Episode datasets**

The episode datasets will be loaded to S3, while you can access it here: 
1. PointNav splits for Gibson and Matterport3D: `/private/home/maksymets/data/habitat_datasets/pointnav`
2. EQA splits for Matterport3D: `/private/home/maksymets/data/habitat_datasets/eqa`

## License
habitat-api is MIT licensed. See the LICENSE file for details.
