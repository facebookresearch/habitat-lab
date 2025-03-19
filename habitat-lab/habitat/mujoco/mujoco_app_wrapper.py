# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import mujoco

from habitat.mujoco.mujoco_visualizer import MuJoCoVisualizer

class MuJoCoAppWrapper:
    def __init__(self, hab_sim):
        self._model = None
        self._data = None
        self._hab_sim = hab_sim

    def load_model_from_xml(self, xml_filepath):
        self._model = mujoco.MjModel.from_xml_path(xml_filepath)
        self._data = mujoco.MjData(self._model)

        self._visualizer = MuJoCoVisualizer(self._hab_sim, self._model, self._data)

    def add_render_map(self, render_map_filepath):
        self._visualizer.add_render_map(render_map_filepath)

    def step(self, num_steps=1):
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model_from_xml() first.")
        
        for _ in range(num_steps):
            mujoco.mj_step(self._model, self._data)
    
    def pre_render(self):

        self._visualizer.flush_to_hab_sim()