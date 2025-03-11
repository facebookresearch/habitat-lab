# Rearrange_v2 HITL application

SIRo's 2024H1 data-collection app. Work in progress.

## Installation
See [habitat-hitl/README.md](../../../habitat-hitl/README.md).

In addition to the core HITL data above, you need the private [fphab](https://huggingface.co/datasets/fpss/fphab) repo. Reach out to the SIRo team for access.
```
cd path/to/habitat-lab/data
git clone --branch articulated-scenes --single-branch --depth 1 https://huggingface.co/datasets/fpss/fphab
mv fphab fpss
```

To test the Habitat-LLM episodes in `rearrange_v2` you'll need to download and unzip the following [episode dataset](https://drive.google.com/file/d/1zFCBiWE_XFY0Ry9CZOV_NF_rfxBw1y-F/view?usp=sharing) in Habitat-Lab root directory. In addition, you'll need YCB, GSO, AI2THOR, and ABO object assets. To download these assets use the following commands:

```
cd data
git clone https://huggingface.co/datasets/ai-habitat/OVMM_objects objects --recursive
cd objects
git checkout 3893a735352b92d46505f35d759553f5fc82a39b
```

## Data directory

Run `rearrange_v2` from the Habitat-lab root directory. It will expect `data/` for Habitat-lab data, and it will also look for `examples/hitl/rearrange_v2/app_data/demo.json.gz` (included alongside source files in our git repo).

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

To test Habitat-LLM episodes using a user-controlled humanoid use:
```bash
python examples/hitl/rearrange_v2/rearrange_v2.py --config-name lang_rearrange_humanoid_only
```

## Controls
See on-screen help text. In addition, press `1` or `2` to select an episode.

## Configuration
See `config/rearrange_v2.yaml` and `config/experiment/headless_server.yaml`.

## Browser client

`rearrange_v2` has additional requirements for the [Unity VR client](../pick_throw_vr/README.md#vr).
* Beware these instructions are a work in progress!
* It is designed for use with a desktop browser-based (not VR) version of the Unity client.
* Use the [`webgl-demo`](https://github.com/eundersander/siro_hitl_unity_client/tree/webgl-demo) branch.
* Download Unity data folder (`data.zip`) from: https://drive.google.com/drive/folders/12VLJGf5_ntr5nztZn1KjzyzBG_UKDPme
* Extract `data.zip` to Unity client's `Assets/Resources/data`.
* Open and run the `PlayerMouseKeyboard` scene.
