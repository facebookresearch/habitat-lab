# Basic_viewer HITL application

Inspect Habitat environments, episodes, and agent policy evaluation in real time with a user-controlled free camera. See on-screen help text for keyboard/mouse controls.

![basic_viewer_screenshot](https://github.com/facebookresearch/habitat-lab/assets/6557808/5f1b47db-8988-4eef-8275-46cf58f1d6ec)

Note on this screenshot: the debug images on the right (`debug_third_person_viewport` and `agent_1_head_depth`) are disabled by default. You can enable them in `basic_viewer.yaml`.

## Installing and using HITL apps
See [habitat-hitl/README.md](../../../habitat-hitl/README.md)

## Example launch command

```bash
HABITAT_SIM_LOG=warning MAGNUM_LOG=warning \
python examples/hitl/basic_viewer/basic_viewer.py habitat_hitl.episodes_filter='0 2 4 10:15 1000:4000:500'
```

## Configuration
As an example, the app is configured to inspect the `pop_play` Habitat baseline on the Habitat-lab `hssd_spot_human` multi-agent benchmark. See [`config/basic_viewer.yaml`](./config/basic_viewer.yaml). See also [`habitat-baselines/README.md`](../../../habitat-baselines/README.md).
