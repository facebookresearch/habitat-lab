# Rearrange HITL application

The user-controlled human and policy-controlled robot must accomplish a collaborative rearrangement task in HSSD scenes. See on-screen help for keyboard/mouse controls. If you complete an episode or exit the app, the app will save session-data files like "my_session.0.json.gz" and "my_session.gfx_replay.json.gz". See `hitl_rearrange.yaml habitat_hitl.data_collection` for details.

This app is similar to the human-in-the-loop tool used in the [Habitat 3.0 paper](https://arxiv.org/abs/2310.13724) to evaluate policies against real human collaborators.

## Installing and using HITL apps
See [siro_sandbox/README.md](../../siro_sandbox/README.md).

## Example launch command

```bash
HABITAT_SIM_LOG=warning MAGNUM_LOG=warning \
python examples/hitl/rearrange/rearrange.py
```

## Configuration
See `config/hitl_rearrange.yaml`.
