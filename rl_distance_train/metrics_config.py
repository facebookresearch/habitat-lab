from dataclasses import dataclass
from typing import Any, Dict
from hydra.core.config_store import ConfigStore
from habitat.config.default_structured_configs import MeasurementConfig

# ----------------------------------------------------------------------------
# Custom measure configs for OrientationToGoal, ZERReward, OVRLReward
# ----------------------------------------------------------------------------

@dataclass
class OrientationToGoalMeasurementConfig(MeasurementConfig):
    """
    Configuration for the OrientationToGoal diagnostic measure.
    """
    type: str = "orientation_to_goal"

@dataclass
class ZERRewardMeasurementConfig(MeasurementConfig):
    """
    Configuration for ZERReward (Zero-Experience Required) reward measure.
    """
    type: str = "zer_reward"
    success_weight: float = 5.0        # bonus when Success == 1
    angle_success_weight: float = 5.0  # extra bonus if within angle threshold
    goal_radius: float = 1.0           # goal radius
    angle_success: float = 25.0        # degrees threshold for angle bonus
    slack_penalty: float = 0.01        # step cost penalty

@dataclass
class OVRLRewardMeasurementConfig(MeasurementConfig):
    """
    Configuration for OVRLReward reward measure.
    """
    type: str = "ovrl_reward"
    success_weight: float = 5.0        # bonus when Success == 1
    angle_success_weight: float = 5.0  # extra bonus if within angle threshold
    goal_radius: float = 1.0           # goal radius
    angle_success: float = 25.0        # degrees threshold for angle bonus
    slack_penalty: float = 0.01        # step cost penalty

# Register configs in Hydra
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
