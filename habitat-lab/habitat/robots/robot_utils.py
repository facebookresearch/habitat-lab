# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import magnum as mn
import numpy as np

from habitat.robots.robot_interface import RobotInterface
from habitat_sim.physics import JointMotorSettings
from habitat_sim.simulator import Simulator
