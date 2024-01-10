# Rearrange HITL application

The user-controlled human and policy-controlled robot must accomplish a collaborative rearrangement task in HSSD scenes. See on-screen help for keyboard/mouse controls. This is similar to the human-in-the-loop tool used in the [Habitat 3.0 paper](https://arxiv.org/abs/2310.13724) to evaluate policies against real human collaborators.

## Installing and using HITL apps
See [siro_sandbox/README.md](../../siro_sandbox/README.md).

## Example launch command

```bash
HABITAT_SIM_LOG=warning MAGNUM_LOG=warning \
python examples/hitl/rearrange/rearrange.py \
--disable-inverse-kinematics \
--never-end \
--gui-controlled-agent-index 1 \
--cfg examples/hitl/rearrange/config/hitl_rearrange.yaml
```

## Configuration
See `config/hitl_rearrange.yaml`.
