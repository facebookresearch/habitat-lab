# Rearrange_v2 HITL application

SIRo's 2024H1 data-collection app. Work in progress.

## Installing and using HITL apps
See [habitat-hitl/README.md](../../../habitat-hitl/README.md).

Unity data:
* Download Unity data folder (`data.zip`) from: https://drive.google.com/drive/folders/18I07T8o7cL9ZAtWV68EkgQynpzftlhKa
* Extract `data.zip` to Unity client's `Assets/Resources/data`.

Articulated scenes:
* Clone [fphab](https://huggingface.co/datasets/fpss/fphab) to `data/`
* Checkout the `articulated-scenes` branch.
* Rename `data/fphab` to `data/fpss`

## Controls
See on-screen help text. Press `1` or `2` to select an episode.

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
