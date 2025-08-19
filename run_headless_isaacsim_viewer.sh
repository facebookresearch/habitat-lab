#!/usr/bin/env bash
export MOCHI_HOME=/Users/eundersander/projects/mochi
export PYTHONPATH=/Users/eundersander/projects/mochi/mochi_agents_api/py_extension:/Users/eundersander/projects/mochi/build/bin:$PYTHONPATH

python examples/hitl/isaacsim_viewer/isaacsim_viewer.py \
    habitat_hitl.networking.enable=True \
    habitat_hitl.experimental.headless.do_headless=true \
    habitat_hitl.window=null
