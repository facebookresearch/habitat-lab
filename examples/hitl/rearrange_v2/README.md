# Rearrange_v2 HITL application

SIRo's 2024H1 data-collection app. Work in progress.

## Installing and using HITL apps
See [habitat-hitl/README.md](../../../habitat-hitl/README.md).

## Example launch commands

Local testing (local display and keyboard control):
```bash
python examples/hitl/rearrange_v2/rearrange_v2.py
```

Headed server (includes local display and keyboard control):
```bash
python examples/hitl/rearrange_v2/rearrange_v2.py habitat_hitl.networking.enable=True
```

Headless server:
```bash
python examples/hitl/rearrange_v2/rearrange_v2.py +experiment=headless_server
```


## Configuration
See `config/rearrange_v2.yaml` and `config/experiment/headless_server.yaml`.
