Habitat 2.0 Overview
#############################################

:summary: An overview of Habitat 2.0 with documentation, quickstart code, and reproducing the benchmark results.

*An overview of Habitat 2.0 with documentation, quickstart code, and reproducing the benchmark results.*

`Quick Start`_
========================
To get started with Habitat 2.0, see the `quick start Colab tutorial <https://colab.research.google.com/github/facebookresearch/habitat-lab/blob/main/examples/tutorials/colabs/Habitat2_Quickstart.ipynb>`__ and the `Gym API tutorial <https://colab.research.google.com/github/facebookresearch/habitat-lab/blob/main/examples/tutorials/colabs/habitat2_gym_tutorial.ipynb>`__.

`Local Installation`_
======================

See the `Habitat Lab README <https://github.com/facebookresearch/habitat-lab/tree/main#installation>`_ for steps to install Habitat.

`Interactive Play Script`_
==========================
Test the Habitat environments using your keyboard and mouse to control the robot. On your local machine with a display connected run the following:

.. code:: sh

    python examples/interactive_play.py --never-end

You may be asked to first install a specific version of PyGame. This script will work on Linux or MacOS. For more information about the interactive play script, see the `documentation string at the top of the file <https://github.com/facebookresearch/habitat-lab/blob/main/examples/interactive_play.py>`__.

`RL Training with Habitat Baselines`_
=====================================
Habitat includes an implementation of DD-PPO. As an example, start training a pick skill policy with:

.. code:: sh

    python -u habitat_baselines/run.py \
        --exp-config habitat_baselines/config/rearrange/ddppo_pick.yaml \
        --run-type train

Find the `complete list of RL configurations here <https://github.com/facebookresearch/habitat-lab/tree/main/habitat-baselines/habitat_baselines/config/rearrange>`__, any config starting with "ddppo" can be substituted into :code:`--exp-config`. See `here <https://github.com/facebookresearch/habitat-lab/tree/main/habitat-baselines/habitat_baselines#baselines>`__  for more information on how to run with Habitat Baselines.

`Home Assistant Benchmark (HAB) Tasks`_
=======================================

To run the HAB tasks, use any of the training configurations `here <https://github.com/facebookresearch/habitat-lab/tree/main/habitat-baselines/habitat_baselines/config/rearrange/hab>`__. For example, to run monolithic RL training on the Tidy House task run:

.. code:: sh

    python -u habitat_baselines/run.py \
        --exp-config habitat_baselines/config/rearrange/hab/ddppo_tidy_house.yaml \
        --run-type train

`Task-Planning with Skills RL Baseline`_
========================================
Here we will detail how to run the Task-Planning with Skills trained via reinforcement learning (TP-SRL) baseline from `the Habitat 2.0 paper <https://arxiv.org/abs/2106.14405>`__. This method utilizes a task-planner and a `Planning Domain Definition Language <https://en.wikipedia.org/wiki/Planning_Domain_Definition_Language>`__ to sequence together low-level skill policies trained independently with reinforcement learning (RL).

1. First, train the skills via reinforcement learning. For example, to train the Place policy run

.. code:: sh

    python -u habitat_baselines/run.py \
        --exp-config habitat_baselines/config/rearrange/ddppo_place.yaml \
        --run-type train \
        checkpoint_folder=./place_checkpoints/

2. To work on HAB tasks, you must also train a :code:`pick`, :code:`nav_to_obj`, :code:`open_cab`, :code:`close_cab`, :code:`open_fridge`, and :code:`close_fridge` policy. To do so, substitute the name of the other skill for :code:`place` in the above command.

3. By default, the TP-SRL baseline will look for the skill checkpoints as :code:`data/models/[skill name].pth` in the Habitat Lab directory as configured `here for each skill <https://github.com/facebookresearch/habitat-lab/blob/710beab2a5500074793b0c8047e3835fdb8f7b7e/habitat_baselines/config/rearrange/hab/tp_srl.yaml#L94>`__. The :code:`tp-srl.yaml` file can be changed to point to the skills you would like to evaluate, or you can copy the model checkpoints in :code:`data/models/`.

4. Evaluate the TP-SRL baseline on the :code:`tidy_house` HAB task via:

.. code:: sh

    python -u habitat_baselines/run.py \
        --exp-config habitat_baselines/config/rearrange/hab/tp_srl.yaml \
        --run-type eval \
        benchmark/rearrange=tidy_house

Evaluate on different HAB tasks by overriding the :code:`benchmark/rearrange`. The TP-SRL baseline only runs in evaluation mode.


`Running the Benchmark`_
========================
To reproduce the benchmark table from `the Habitat 2.0 paper <https://arxiv.org/abs/2106.14405>`__ follow these steps:

1. Download the benchmark assets:

.. code:: sh

    python -m habitat_sim.utils.datasets_download --uids hab2_bench_assets

2. Copy the benchmark episodes into the data folder.

.. code:: sh

  cp data/hab2_bench_assets/bench_scene.json.gz data/ep_datasets/

3. Run the benchmark.

.. code:: sh

   bash scripts/hab2_bench/bench_runner.sh

4. Generate the results table.

.. code:: sh

   python scripts/hab2_bench/plot_bench.py
