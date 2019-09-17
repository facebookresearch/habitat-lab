Quickstart
##########

In this quickstart we will briefly introduce the habitat stack using which we
will setup the pointnav task and step around in the environment.

.. role:: sh(code)
    :language: sh

`Habitat`_
==========

Habitat is a platform for embodied AI research that consists of:

1.  **Habitat-Sim**: A flexible, high-performance 3D simulator with
    configurable agents, multiple sensors, and generic 3D dataset handling
    (with built-in support for
    `MatterPort3D <https://niessner.github.io/Matterport/>`_,
    `Gibson <http://gibsonenv.stanford.edu/database/>`_ and other datasets).
    :gh:`[github-repo] <facebookresearch/habitat-sim>`

2.  **Habitat-API**: A modular high-level library for end-to-end development in
    embodied AI --- defining embodied AI tasks (e.g. navigation, instruction
    following, question answering), configuring embodied agents (physical form,
    sensors, capabilities), training these agents (via imitation or
    reinforcement learning, or no learning at all as in classical SLAM), and
    benchmarking their performance on the defined tasks using standard metrics.
    :gh:`[github-repo] <facebookresearch/habitat-api>`

For installing Habitat-Sim and Habitat-API follow instructions
:gh:`here <facebookresearch/habitat-api#installation>`.

`Example`_
==========

In this example we will setup a PointNav task in which the agent is tasked to
go from a source location to a target location. For this example the agent will
be you (the user). You will be able to step around in an environment using
keys.

For running this example both Habitat-Sim and Habitat-API should be installed
successfully. The data for scene should also be downloaded (steps to do this
are provided in the :gh:`installation instructions <facebookresearch/habitat-api#installation>`
of Habitat-API). Running the code below also requires installation of cv2 which
you can install using: :sh:`pip install opencv-python`.

.. code:: py

    import habitat
    import cv2


    FORWARD_KEY="w"
    LEFT_KEY="a"
    RIGHT_KEY="d"
    FINISH="f"


    def transform_rgb_bgr(image):
        return image[:, :, [2, 1, 0]]


    def example():
        env = habitat.Env(
            config=habitat.get_config("configs/tasks/pointnav.yaml")
        )

        print("Environment creation successful")
        observations = env.reset()
        print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
            observations["pointgoal"][0], observations["pointgoal"][1]))
        cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

        print("Agent stepping around inside environment.")

        count_steps = 0
        while not env.episode_over:
            keystroke = cv2.waitKey(0)

            if keystroke == ord(FORWARD_KEY):
                action = habitat.SimulatorActions.MOVE_FORWARD
                print("action: FORWARD")
            elif keystroke == ord(LEFT_KEY):
                action = habitat.SimulatorActions.TURN_LEFT
                print("action: LEFT")
            elif keystroke == ord(RIGHT_KEY):
                action = habitat.SimulatorActions.TURN_RIGHT
                print("action: RIGHT")
            elif keystroke == ord(FINISH):
                action = habitat.SimulatorActions.STOP
                print("action: FINISH")
            else:
                print("INVALID KEY")
                continue

            observations = env.step(action)
            count_steps += 1

            print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
                observations["pointgoal"][0], observations["pointgoal"][1]))
            cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

        print("Episode finished after {} steps.".format(count_steps))

        if action == habitat.SimulatorActions.STOP and observations["pointgoal"][0] < 0.2:
            print("you successfully navigated to destination point")
        else:
            print("your navigation was unsuccessful")


    if __name__ == "__main__":
        example()

Running the above code will initialize an agent inside an environment, you can
move around in the environment using :label-default:`W`, :label-default:`A`,
:label-default:`D`, :label-default:`F` keys. On the terminal a destination
vector in polar format will be printed with distance to goal and angle to goal.
Once you are withing 0.2m of goal you can press the :label-default:`F` key to
``STOP`` and finish the episode successfully. If your finishing distance to
goal is :math:`> 0.2m` or if you spend more than 500 steps in the environment
your episode will be unsuccessful.

Below is a demo of what the example output will look like:

.. image:: quickstart.png

For more examples refer to
:gh:`Habitat-API examples <facebookresearch/habitat-sim/tree/master/examples>`
and :gh:`Habitat-Sim examples <facebookresearch/habitat-sim/tree/master/examples>`.

`Citation`_
===========

If you use habitat in your work, please cite:

.. code:: bibtex
    :class: m-console-wrap

    @article{habitat19arxiv,
      title =   {Habitat: A Platform for Embodied AI Research},
      author =  {Manolis Savva, Abhishek Kadian, Oleksandr Maksymets, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, Devi Parikh and Dhruv Batra},
      journal = {arXiv preprint arXiv:1904.01201},
      year =    {2019}
    }
