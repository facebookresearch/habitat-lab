#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.rearrange.sub_tasks.pick_task import RearrangePickTaskV1


@registry.register_task(name="RearrangePlaceTask-v0")
class RearrangePlaceTaskV1(RearrangePickTaskV1):
    def reset(self, episode: Episode):
        sim = self._sim
        super().reset(episode)

        abs_obj_idx = sim.scene_obj_ids[self._targ_idx]

        sim.grasp_mgr.snap_to_obj(abs_obj_idx, force=True)
        sim.internal_step(-1)

        self.was_prev_holding = self.targ_idx

        sim.internal_step(-1)
        return super(RearrangePickTaskV1, self).reset(episode)
