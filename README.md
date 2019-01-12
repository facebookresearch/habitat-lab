habitat-api
==============================


A suite to train embodied agents across a variety of tasks, 
environments, simulators. habitat-api defines an API through which users can 
train agents on existing embodied tasks as well as add new simulators, 
tasks. It also aims to make benchmarking of agents across datasets, 
environments and tasks easier. 


### Install
Clone the github repository and then run:
```bash
cd habitat-api
pip install -e .
```


### Example

**TODO(akadian)**: Adapt below example according to any changes in API 

```python
import habitat
from habitat.config.experiments.sim_nav import sim_nav_cfg
from habitat.tasks.nav.nav_task import NavigationEpisode
import numpy as np

config = sim_nav_cfg()
config.task_name = 'Nav-v0'
env = habitat.Env(config=config)
env.episodes = [NavigationEpisode(episode_id='0', scene_id=config.scene, 
                                  start_position=None, start_rotation=None, 
                                  goals=[])]

observation, reward, done, info = env.reset()

# randomly move around inside the environment
while not done:
    observation, reward, done, info = env.step(env.action_space.sample())
```

### Data

**TODO(akadian)**: Update the URLs and description for below data resources

**Matterport3D**

1. PointNav val split: `/private/home/akadian/habitat-api-data/mp3d-splits`
2. ObjectNav val split: **TODO(akadian)**
3. EQA val split: **TODO(maksymets)**