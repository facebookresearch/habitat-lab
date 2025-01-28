# Rearrange HITL application

The user-controlled human and policy-controlled robot must accomplish a collaborative rearrangement task in HSSD scenes. See on-screen help for keyboard/mouse controls. If you complete an episode or exit the app, the app will save session-data files like "my_session.0.json.gz" and "my_session.gfx_replay.json.gz". See `hitl_rearrange.yaml habitat_hitl.data_collection` for details. See also [test_episode_save_files.py](../../../habitat-hitl/habitat_hitl/scripts/test_episode_save_files.py) as an example of consuming these data files.

This app is similar to the human-in-the-loop tool used in the [Habitat 3.0 paper](https://arxiv.org/abs/2310.13724) to evaluate policies against real human collaborators.

![rearrange_screenshot](https://github.com/facebookresearch/habitat-lab/assets/6557808/acc5ae02-fb62-4fa3-9e0a-b2df6a735983)

## Installing and using HITL apps
See [habitat-hitl/README.md](../../../habitat-hitl/README.md).

## Example launch command

```bash
python examples/hitl/rearrange/rearrange.py
```

## Configuration
See `config/hitl_rearrange.yaml`.
