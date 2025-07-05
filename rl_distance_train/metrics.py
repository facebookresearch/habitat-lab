"""metrics.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Reward and diagnostic measures for Habitat navigation tasks **using the built‑in
`Success` measure** instead of checking `task.is_stop_called` + distance.

This file defines three registry measures:

* **OrientationToGoal** – diagnostic heading‑error measure.
* **ZERReward** – reward from *Zero‑Experience Required* (Al‑Halah et al., 2022).
* **OVRLReward** – reward from *OVRL‑V2* (Yadav et al., 2023).

### Update (2025‑07‑04)
* Rewards now depend on the stock `success` metric. No manual distance / stop
  checks needed.
* `reset_metric` declares a hard dependency on both `distance_to_goal` **and**
  `success`.

Minimal YAML:
```yaml
measurements:
  - type: ZERReward  # or OVRLReward, OrientationToGoal
    # (all reward hyper‑parameters are optional)
```
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from hydra.core.config_store import ConfigStore
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.config.default_structured_configs import MeasurementConfig
from omegaconf import DictConfig

# ----------------------------------------------------------------------------
# Helper ---------------------------------------------------------------------

def _heading_error(agent_pos: np.ndarray, agent_rot, goal_pos: np.ndarray) -> float:
    """Unsigned heading error (radians) between the agent forward (−Z) and the
    goal vector, computed with Habitat's `quaternion_rotate_vector` utility."""
    forward_world = quaternion_rotate_vector(agent_rot, np.array([0.0, 0.0, -1.0]))
    goal_vec = goal_pos - agent_pos

    # Project both to horizontal plane
    forward_world[1] = 0.0
    goal_vec[1] = 0.0
    if np.linalg.norm(goal_vec) < 1e-6:
        return 0.0

    forward_world /= np.linalg.norm(forward_world) + 1e-9
    goal_vec /= np.linalg.norm(goal_vec) + 1e-9

    cosang = float(np.clip(np.dot(forward_world, goal_vec), -1.0, 1.0))
    return float(np.arccos(cosang))

# ----------------------------------------------------------------------------
# Orientation diagnostic ------------------------------------------------------

@registry.register_measure
class OrientationToGoal(Measure):
    """Returns the current heading error (radians) to the first navigation goal."""

    cls_uuid: str = "orientation_to_goal"

    def __init__(self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any):
        self._sim = sim
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self.update_metric(episode)

    def update_metric(self, episode, *args: Any, **kwargs: Any):
        agent_state = self._sim.get_agent_state()
        self._metric = _heading_error(
            agent_state.position,
            agent_state.rotation,
            np.array(episode.goals[0].position, dtype=np.float32),
        )

@dataclass
class OrientationToGoalMeasurementConfig(MeasurementConfig):
    """
    Configuration for the OrientationToGoal diagnostic measure.
    """
    type: str = "OrientationToGoal"


# ----------------------------------------------------------------------------
# Base reward mixin -----------------------------------------------------------

class _ProgressReward(Measure):
    """Shared machinery for ZER and OVRL rewards (success‑aware version)."""

    def __init__(self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any):
        self._sim = sim
        # Optional hyper‑parameters ------------------------------------------------
        self.cs = config.get("success_weight", 5.0)           # bonus when Success==1
        self.ca = config.get("angle_success_weight", 5.0)     # extra bonus if within angle
        self.rg = config.get("goal_radius", 1.0)              # radius already baked into Success but kept for clarity
        self.theta_g = np.deg2rad(config.get("angle_success", 25.0))
        self.gamma = config.get("slack_penalty", 0.01)        # step cost
        self.prev_dist: Optional[float] = None
        super().__init__()

    # ---------------------------------------------------------------------
    # Mandatory Habitat overrides
    # ---------------------------------------------------------------------

    def _get_dist(self, task: EmbodiedTask) -> float:
        return task.measurements.measures["distance_to_goal"].get_metric()

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        # Depend on both goal distance and success ------------------------
        task.measurements.check_measure_dependencies(self.uuid, ["distance_to_goal", "success"])
        self.prev_dist = None
        self._update(episode, task)  # first call

    def update_metric(self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any):
        self._update(episode, task)

    # ---------------------------------------------------------------------
    # Internal helper ------------------------------------------------------
    # ---------------------------------------------------------------------

    def _update(self, episode, task: EmbodiedTask):
        dist = self._get_dist(task)
        agent_state = self._sim.get_agent_state()
        core = self._reward_core(dist, agent_state, episode)
        reward = core - self.gamma

        success = task.measurements.measures["success"].get_metric() > 0.0
        if success:
            theta = _heading_error(agent_state.position, agent_state.rotation, np.array(episode.goals[0].position, dtype=np.float32))
            reward += self.cs
            if theta < self.theta_g:
                reward += self.ca

        self._metric = reward

    # Child classes must implement this ----------------------------------
    def _reward_core(self, dist: float, agent_state, episode):
        raise NotImplementedError

# ----------------------------------------------------------------------------
# ZER reward ------------------------------------------------------------------

@registry.register_measure
class ZERReward(_ProgressReward):
    """Reward from *Zero‑Experience Required* (Eq. 4)."""

    cls_uuid: str = "zer_reward"

    def __init__(self, *args: Any, **kwargs: Any):
        self.prev_theta: Optional[float] = None
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _reward_core(self, dist: float, agent_state, episode):
        theta = _heading_error(agent_state.position, agent_state.rotation, np.array(episode.goals[0].position, dtype=np.float32))
        dd = 0.0 if self.prev_dist is None else (self.prev_dist - dist)
        dt = 0.0 if self.prev_theta is None else (self.prev_theta - theta)
        reward = dd + (dt if dist < self.rg else 0.0)
        self.prev_dist, self.prev_theta = dist, theta
        return reward

@dataclass
class ZERRewardMeasurementConfig(MeasurementConfig):
    """
    Configuration for ZERReward (Zero-Experience Required) reward measure.
    """
    type: str = "ZERReward"
    success_weight: float = 5.0        # bonus when Success == 1
    angle_success_weight: float = 5.0  # extra bonus if within angle threshold
    goal_radius: float = 1.0           # goal radius
    angle_success: float = 25.0        # degrees threshold for angle bonus
    slack_penalty: float = 0.01        # step cost penalty

# ----------------------------------------------------------------------------
# OVRL reward -----------------------------------------------------------------

class OVRLReward(_ProgressReward):
    """Reward from *OVRL‑V2* (Eq. 3)."""

    cls_uuid: str = "ovrl_reward"

    def __init__(self, *args: Any, **kwargs: Any):
        self.prev_theta_hat: Optional[float] = None
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _reward_core(self, dist: float, agent_state, episode):
        theta = _heading_error(agent_state.position, agent_state.rotation, np.array(episode.goals[0].position, dtype=np.float32))
        theta_hat = theta if dist < self.rg else np.pi
        dd = 0.0 if self.prev_dist is None else (self.prev_dist - dist)
        dtheta_hat = 0.0 if self.prev_theta_hat is None else (self.prev_theta_hat - theta_hat)
        reward = dd + dtheta_hat
        self.prev_dist, self.prev_theta_hat = dist, theta_hat
        return reward

@dataclass
class OVRLRewardMeasurementConfig(MeasurementConfig):
    """
    Configuration for OVRLReward reward measure.
    """
    type: str = "OVRLReward"
    success_weight: float = 5.0        # bonus when Success == 1
    angle_success_weight: float = 5.0  # extra bonus if within angle threshold
    goal_radius: float = 1.0           # goal radius
    angle_success: float = 25.0        # degrees threshold for angle bonus
    slack_penalty: float = 0.01        # step cost penalty

cs = ConfigStore.instance()
cs.store(
    package="habitat.task.measurements",
    name="orientation_to_goal",
    node=OrientationToGoalMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements",
    name="zer_reward",
    node=ZERRewardMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements",
    name="ovrl_reward",
    node=OVRLRewardMeasurementConfig,
)
