Humanoid Design
==============================

Habitat also supports humanoid avatars. These avatars are represented as articulated objects connected via spherical or fixed joints. The current version only includes a kinematic humanoid, which can be controlled by specifying the joint rotations and object transformations.

You can download the humanoid avatar by running:

```
python -m habitat_sim.utils.datasets_download --uids habitat_humanoids  --data-path data/
```

From the home directory.

---

## Humanoid Component Design

1. **Humanoid**
    - `kinematic_humanoid.py` is built upon `mobile_manipulator.py` and can be controlled by setting the joint and object transforms, via the function `set_joint_transform`.

1. **Mobile Manipulator Parameters**
    - Parameters such as the camera's transformation, end-effector position, control gain, and more are specified in each humanoid class (e.g., here, `kinematic_humanoid.py`).

1. **Humanoid Articulated Object**

The humanoid is represented as an articulated object made out of capsules and spheres. It contains 20 links and 19 joints, and is designed to match the morphology of the SMPL skeleton, as referenced [here](https://files.is.tue.mpg.de/black/talks/SMPL-made-simple-FAQs.pdf). Note that the current model does not include the hand or foot joint, hence the 20 links instead of 24. Furthermore, the root joint is represented via the object transform, and the wrist joints are fixed. Resulting in 17 spherical joints.

## Testing the Humanoid

You can test a simple demo where the humanoid moves their arms at random using:

```
python examples/interactive_play.py --never-end --disable-inverse-kinematics --control-humanoid --cfg benchmark/rearrange/play_human.yaml
```

## Controlling the humanoid

While you can control the humanoid avatars by directly setting joint rotations, you may be interested in easily generating realistic humanoid motion. We provide a set of controllers that convert high level commands into low-level controls. We currently provide two controllers, as described below:

### HumanoidRearrangeController

The [HumanoidRearrangeController](../../articulated_agent_controllers/humanoid_rearrange_controller.py) controller, designed to drive the humanoid to navigate around a scene, or pick and place objects. This controller is used for the Social Navigation and Social Rearrangement tasks. It allows to generate walking motions, that drive the agent to a particular object position, and reaching motions, which change the agent pose so that the right or left hand reach a specific coordinate. You can test the controller by running:

```
python -m pytest test/test_humanoid.py:test_humanoid_controller
```
Which will generate a sequence of a human walking in an empty plane.

### SequentialPoseController

The [SequentialPoseController](../../articulated_agent_controllers/seq_pose_controller.py), designed to replay a pre-saved motion data file either coming from motion capture or a motion generation model. You can test the controller by running:


```
python -m pytest test/test_humanoid.py:test_humanoid_seqpose_controller
```

We also provide [a script]() to convert motion from a SMPL-X format to a file that can be played in our controller.

## Creating new humanoids
