# Basic_viewer HITL application

Inspect Habitat-lab policy evaluation in real time with a user-controlled free camera.

## Installing and using HITL apps
See [siro_sandbox/README.md](../../siro_sandbox/README.md).

## Example launch command

```bash
HABITAT_SIM_LOG=warning MAGNUM_LOG=warning \
python examples/hitl/basic_viewer/basic_viewer.py \
--disable-inverse-kinematics \
--never-end \
--cfg examples/hitl/basic_viewer/config/basic_viewer.yaml
```

## Configuration
See `config/basic_viewer.yaml`.
