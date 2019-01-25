habitat-api
==============================


A suite to train embodied agents across a variety of tasks, 
environments, simulators. habitat-api defines an API through which users can 
train agents on existing embodied tasks as well as add new simulators, 
tasks. It also aims to make benchmarking of agents across datasets, 
environments and tasks easier. 


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

4. Download test data from [dropbox share](https://www.dropbox.com/sh/dl/h02865ucoh3ix07/AABkVrHCfPI0BAmSeHCytrsya) and place it in folder:`data/esp/test/`


### Example

```python
import habitat
from habitat.config.experiments.nav import sim_nav_cfg
from habitat.tasks.nav.nav_task import NavigationEpisode

config = sim_nav_cfg()
config.scene = 'data/esp/test/test.glb'
config.task_name = 'Nav-v0'
env = habitat.RLEnv(config=config)
env.episodes = [NavigationEpisode(episode_id='0', scene_id=config.scene, 
                                  start_position=None, start_rotation=None, 
                                  goals=[])]

observation, reward, done, info = env.reset()

# randomly move around inside the environment
while not done:
    observation, reward, done, info = env.step(env.action_space.sample())
```

### Data

**Matterport3D**

1. PointNav val split: `/private/home/akadian/habitat-api-data/mp3d-splits`
