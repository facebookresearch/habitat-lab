Habitat Lab Demo
################

.. contents::
    :class: m-block m-default

.. code:: py

    import random
    import matplotlib.pyplot as plt

    import habitat
    from habitat.config import read_write

All the boilerplate code in the habitat-sim to set sensor config and agent
config is abstracted out in the Habitat Lab config system. Default habitat structured configs are at
:gh:`habitat-lab/habitat/config/default_structured_configs.py <facebookresearch/habitat-lab/blob/main/habitat-lab/habitat/config/default_structured_configs.py>`.
You can override defaults by specifying them in a separate file and pass it to
the :ref:`habitat.config.get_config()` function or use `read_write` to edit
the config object.

.. code-figure::

    .. code:: py

        config = habitat.get_config(config_paths="benchmark/nav/pointnav/pointnav_mp3d.yaml")
        with read_write(config):
            config.habitat.dataset.split = "val"

        env = habitat.Env(config=config)

    .. code:: shell-session
        :class: m-nopad

        2019-06-06 16:11:35,200 initializing sim Sim-v0
        2019-06-06 16:11:46,171 initializing task Nav-v0

`Scene semantic annotations`_
=============================

.. code-figure::

    .. code:: py

        def print_scene_recur(scene, limit_output=10):
            count = 0
            for level in scene.levels:
                print(
                    f"Level id:{level.id}, center:{level.aabb.center},"
                    f" dims:{level.aabb.sizes}"
                )
                for region in level.regions:
                    print(
                        f"Region id:{region.id}, category:{region.category.name()},"
                        f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
                    )
                    for obj in region.objects:
                        print(
                            f"Object id:{obj.id}, category:{obj.category.name()},"
                            f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                        )
                        count += 1
                        if count >= limit_output:
                            return None

        # Print semantic annotation information (id, category, bounding box details)
        # for the current scene in a hierarchical fashion
        scene = env.sim.semantic_annotations()
        print_scene_recur(scene, limit_output=15)

        env.close()
        # Note: Since only one OpenGL is allowed per process,
        # you have to close the current env before instantiating a new one.

    .. code:: shell-session
        :class: m-nopad m-console-wrap

        Level id:0, center:[11.0210495  3.996935   3.3452997], dims:[ 43.0625    8.19569 -30.1122 ]
        Region id:0_0, category:rec/game, center:[16.61225    2.7802274 11.577564 ], dims:[10.364299   5.5838847 -4.14447  ]
        Object id:0_0_0, category:ceiling, center:[16.5905   4.54488 11.269  ], dims:[9.984315  4.0917997 2.1377602]
        Object id:0_0_1, category:wall, center:[16.5865     2.6818905 13.4147   ], dims:[9.69278   0.5280709 5.4398193]
        Object id:0_0_2, category:wall, center:[21.6013     1.7400599 11.3493   ], dims:[3.5423203  0.41668844 3.921341  ]
        Object id:0_0_3, category:door, center:[11.5374     1.2431393 10.386599 ], dims:[1.2573967  2.5311599  0.41445923]
        Object id:0_0_4, category:door, center:[20.6332     1.2136002 13.5958   ], dims:[0.15834427 2.4860601  1.1674671 ]
        Object id:0_0_5, category:wall, center:[16.5946    2.66614   9.331001], dims:[9.72554    0.23693037 5.3787804 ]
        Object id:0_0_6, category:window, center:[16.5822    2.852209 13.596898], dims:[1.5934639  0.16375065 1.2588081 ]
        Object id:0_0_7, category:beam, center:[16.6094    5.32839  11.348299], dims:[0.5116577  0.35226822 3.8936386 ]
        Object id:0_0_8, category:floor, center:[16.586       0.07907867 11.406     ], dims:[10.48608    4.3792195  0.2833004]
        Object id:0_0_9, category:lighting, center:[11.798      1.9214487 11.313999 ], dims:[0.25683594 0.5076561  0.15560722]
        Object id:0_0_10, category:wall, center:[11.57       1.7476702 11.3347   ], dims:[3.54352    0.41701245 3.9231815 ]
        Object id:0_0_11, category:misc, center:[16.5943   2.29591 11.4341 ], dims:[10.428299  4.48172   4.676901]
        Object id:0_0_12, category:door, center:[11.5234     1.2489185 12.228199 ], dims:[1.2521439  2.5423803  0.46386147]
        Object id:0_0_13, category:door, center:[16.5833     1.1790485 13.490699 ], dims:[5.45306   0.3474083 2.4161606]
        Object id:0_0_14, category:window, center:[21.6362     1.2518396 12.2613   ], dims:[1.1998444  2.5486398  0.37800598]

`Actions and sensors`_
======================

.. code:: py
    :class: m-console-wrap

    import numpy as np
    from PIL import Image
    from habitat_sim.utils.common import d3_40_colors_rgb
    from habitat.config.default import get_agent_config
    from habitat.config.default_structured_configs import HabitatSimSemanticSensorConfig

    def display_sample(rgb_obs, semantic_obs, depth_obs):
        rgb_img = Image.fromarray(rgb_obs, mode="RGB")

        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")

        depth_img = Image.fromarray((depth_obs * 255).astype(np.uint8), mode="L")

        arr = [rgb_img, semantic_img, depth_img]

        titles = ['rgb', 'semantic', 'depth']
        plt.figure(figsize=(12 ,8))
        for i, data in enumerate(arr):
            ax = plt.subplot(1, 3, i+1)
            ax.axis('off')
            ax.set_title(titles[i])
            plt.imshow(data)
        plt.show()

    config = habitat.get_config(config_paths="benchmark/nav/pointnav/pointnav_mp3d.yaml")
    with read_write(config):
        config.habitat.dataset.split = "val"
        agent_config = get_agent_config(sim_config=config.habitat.simulator)
        agent_config.sim_sensors.update(
            {"semantic_sensor": HabitatSimSemanticSensorConfig(height=256, width=256)}
        )
        config.habitat.simulator.turn_angle = 30

    env = habitat.Env(config=config)
    env.episodes = random.sample(env.episodes, 2)

    max_steps = 4

    action_mapping = {
        0: 'stop',
        1: 'move_forward',
        2: 'turn left',
        3: 'turn right'
    }

    for i in range(len(env.episodes)):
        observations = env.reset()

        display_sample(observations['rgb'], observations['semantic'], np.squeeze(observations['depth']))

        count_steps = 0
        while count_steps < max_steps:
            action = random.choice(list(action_mapping.keys()))
            print(action_mapping[action])
            observations = env.step(action)
            display_sample(observations['rgb'], observations['semantic'], np.squeeze(observations['depth']))

            count_steps += 1
            if env.episode_over:
                break

    env.close()

.. image:: ../images/habitat-lab-demo-images/habitat-lab-demo.png
    :alt: Actions and sensors
