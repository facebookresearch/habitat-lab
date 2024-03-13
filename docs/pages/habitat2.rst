Habitat 2.0 Overview
#############################################

:summary: An overview of Habitat 2.0 with documentation, quickstart code, and reproducing the benchmark results.

`Quick Start`_
========================
To get started with Habitat 2.0, see the `quick start Jupyter notebook tutorial <https://github.com/facebookresearch/habitat-lab/blob/main/examples/tutorials/notebooks/Habitat2_Quickstart.ipynb>`__ and the `Gym API tutorial <https://github.com/facebookresearch/habitat-lab/blob/main/examples/tutorials/notebooks/habitat2_gym_tutorial.ipynb>`__.

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

    python -u -m habitat_baselines.run \
        --config-name=rearrange/rl_skill.yaml

This trains the Pick skill by default. To train the other skills, specify: :code:`benchmark/rearrange=skill_name` where :code:`skill_name` can be :code:`close_cab`, :code:`close_fridge`, :code:`open_fridge`, :code:`pick`, :code:`place`, or :code:`nav_to_obj`. See `here <https://github.com/facebookresearch/habitat-lab/tree/main/habitat-baselines/habitat_baselines#baselines>`__  for more information on how to run with Habitat Baselines.

`Task-Planning with Skills RL Baseline`_
========================================
Here we will detail how to run the Task-Planning with Skills trained via reinforcement learning (TP-SRL) baseline from `the Habitat 2.0 paper <https://arxiv.org/abs/2106.14405>`__. This method utilizes a task-planner and a `Planning Domain Definition Language <https://en.wikipedia.org/wiki/Planning_Domain_Definition_Language>`__ to sequence together low-level skill policies trained independently with reinforcement learning (RL).

1. First, train the skills via reinforcement learning. For example, to train the Place policy run

.. code:: sh

    python -u -m habitat_baselines.run \
        --config-name=rearrange/rl_skill.yaml \
        checkpoint_folder=./place_checkpoints/ \
        benchmark/rearrange=place

2. To work on HAB tasks, you must also train a :code:`pick`, :code:`nav_to_obj`, :code:`open_cab`, :code:`close_cab`, :code:`open_fridge`, and :code:`close_fridge` policy. To do so, substitute the name of the other skill for :code:`place` in the above command.

3. By default, the TP-SRL baseline will look for the skill checkpoints as :code:`data/models/[skill name].pth`. The :code:`tp-srl.yaml` file can be changed to point to the skills you would like to evaluate, or you can copy the model checkpoints in :code:`data/models/`.

4. Evaluate the TP-SRL baseline on the :code:`tidy_house` HAB task via:

.. code:: sh

    python -u -m habitat_baselines.run \
        --config-name=rearrange/tp_srl.yaml \
        habitat_baselines.evaluate=True \
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
