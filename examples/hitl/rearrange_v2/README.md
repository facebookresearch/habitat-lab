# Rearrange_v2 HITL application

SIRo's 2024H1 data-collection app. Work in progress.

## Installing and using HITL apps
See [habitat-hitl/README.md](../../../habitat-hitl/README.md).

Episodes and Unity data:
* Download `demo.json.gz` and `data.zip` from: https://drive.google.com/drive/folders/18I07T8o7cL9ZAtWV68EkgQynpzftlhKa
* Copy episodes to `data/demo.json.gz`.
* Extract `data.zip` to Unity client's `Assets/Resources/data`.

Articulated scenes:
* Clone `fphab` to somewhere convenient: https://huggingface.co/datasets/fpss/fphab
* Checkout the `articulated-scenes` branch.
* Create a symlink to the root of the repository (where dataset config files are found) here: `data/fpss`

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
