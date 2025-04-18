# Skills-in-the-loop

- [Setup](#setup)
  - [Structure](#structure)
  - [Packages](#packages)
- [Installation - Server](#installation---server)
- [Installation - Client](#installation---client)
- [Running HITL](#running-hitl)

## Setup

### Structure

This application needs to be run from `habitat-lab`, not `habitat-llm`.
It assumes that `habitat-lab` and `habitat-llm` are cloned side-by-side.
You may create a symlink to your data folder in `habitat-lab` to simplify development.

```
./habitat-lab
./habitat-lab/data
./habitat-llm
```

### Packages

`pip list | grep habitat` should result in:

```
habitat-baselines     /path/habitat-lab/habitat-baselines
habitat-hitl          /path/habitat-lab/habitat-hitl
habitat-lab           /path/habitat-lab/habitat-lab
habitat_llm           /path/habitat-llm
habitat-sim           <Any>
```

## Installation - Server

1. Checkout `habitat-lab` : `0mdc/skills_eval`
2. Download an episode set.
    1. Download https://drive.google.com/file/d/1Z6tJpLwZ2yzUIy1HjevQ_BhFWCGIjYOi/view?usp=drive_link
    2. Move the episode set to `data/episodes`

## Installation - Client

1. Download the client:
    1. Download: https://drive.google.com/file/d/1DIcmC7Ls6S6LBCoFlykYR1Do2EWoTLnc/view?usp=drive_link
    2. Extract.
        1. Recommended to place near the other repositories for simplicity.

## Running HITL

1. Launch the client.
    1. Locate the client extracted above.
    2. Run `serve.sh`.
2. Launch the server.
    1. From the root of `habitat-lab`, run the launch command:
    2. `python examples/hitl/rearrange_v2/main.py --config-name hitl_single.yaml hydra.searchpath=[file://../habitat-llm/habitat_llm/conf] +experiment=habitat_llm_base habitat.dataset.data_path="data/episodes/2024_07_29_val_hitl_stacked.json.gz"`
3. Open the client.
    1. Navigate to: http://localhost:3333/?server_hostname=localhost&server_port=18000&asset_hostname=localhost&asset_port=9999&asset_path=ServerData&episodes=1,2,3
