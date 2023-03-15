Humanoid Design
==============================

Habitat also supports humanoid avatars. These avatars are represented as articulated objects connected via spherical or fixed joints. The current version only includes a kinematic humanoid, which can be controlled by specifying the joint rotations and object transformation.

You can download the humanoid avatar by running:

```
python -m habitat_sim.utils.datasets_download --uids humanoid_data  --data-path data/
```

From the home directory.

---

## Humanoid Component Design

1. **Humanoid**
    - `kinematic_humanoid.py` is built upon `mobile_manipulator.py` and can be controlled by setting the joint and object transforms, via the function `set_joint_transform`.

1. **Mobile Manipulator Parameters**
    - Parameters such as the camera's transformation, end-effector position, control gain, and more are specified in each robot class (e.g., here, `kinematic_humanoid.py`).

1. **Humanoid Articulated Object**

The humanoid is represented as an articulated object made out of capsules and spheres. It contains 20 links and 19 joints, and is designed to match the morphology of the SMPL skeleton, as referenced [here](https://files.is.tue.mpg.de/black/talks/SMPL-made-simple-FAQs.pdf). Note that the current model does not include the hand or foot joint, hence the 20 links instead of 24. Furthermore, the root joint is represented via the object transform, and the wrist joints are fixed. Resulting in 17 spherical joints.

## Testing the Humanoid

You can test a simple demo where the humanoid moves their arms at random using:

```
python examples/interactive_play.py --never-end --disable-inverse-kinematics --control-humanoid --cfg benchmark/rearrange/play_human.yaml
```
