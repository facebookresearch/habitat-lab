# Basic_viewer HITL application

Inspect Habitat environments, episodes, and agent policy evaluation in real time with a user-controlled free camera.

## Installing and using HITL apps
See [siro_sandbox/README.md](../../siro_sandbox/README.md).

## Example launch command

```bash
HABITAT_SIM_LOG=warning MAGNUM_LOG=warning \
python examples/hitl/basic_viewer/basic_viewer.py habitat_hitl.episodes_filter='0 2 4 10:15 1000:4000:500'
```

## Configuration
See `config/basic_viewer.yaml`.
