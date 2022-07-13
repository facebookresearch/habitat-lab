# ---
# jupyter:
#   accelerator: GPU
#   jupytext:
#     cell_metadata_filter: -all
#     formats: nb_python//py:percent,colabs//ipynb
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.8.0
# ---

# %% [markdown]
# ## Using Polymetis with Habitat
#
# Official instructions on installing polymetis [here](https://facebookresearch.github.io/fairo/polymetis/installation.html)
#
# If jupyter was not launched from the polymetis environment, set up the polymetis environment as a kernel:
# 1. Activate the polymetis environment
# 2. Run the following command: `ipython kernel install --user --name=POLYMETIS_ENV_NAME`
# 3. Reload this page
# 4. Open Kernel > Change Kernel... and select the polymetis environment name

# %% [markdown]
# The client configuration for habitat is included in polymetis. To use it, specify `robot_client=habitat_sim` and set the `habitat_scene_path` as an **absolute path** to your scene file.

# %% [markdown]
# ```
# launch_robot.py robot_client=habitat_sim habitat_scene_path=/PATH/TO/SCENE use_real_time=false gui=true
# ```

# %% [markdown]
# To change the robot model, specify the `robot_model`:

# %% [markdown]
# ```
# launch_robot.py robot_client=habitat_sim robot_model=ROBOT_MODEL habitat_scene_path=/PATH/TO/SCENE use_real_time=false gui=true
# ```

# %% [markdown]
# ### Defining a controller

# %%
from typing import Dict

import numpy as np
import torch
import torchcontrol as toco
from polymetis import RobotInterface

# %%
class MySinePolicy(toco.PolicyModule):
    """
    Custom policy that executes a sine trajectory on joint 6
    (magnitude = 0.5 radian, frequency = 1 second)
    """

    def __init__(self, time_horizon, hz, magnitude, period, kq, kqd, **kwargs):
        """
        Args:
            time_horizon (int):         Number of steps policy should execute
            hz (double):                Frequency of controller
            kq, kqd (torch.Tensor):     PD gains (1d array)
        """
        super().__init__(**kwargs)

        self.hz = hz
        self.time_horizon = time_horizon
        self.m = magnitude
        self.T = period

        # Initialize modules
        self.feedback = toco.modules.JointSpacePD(kq, kqd)

        # Initialize variables
        self.steps = 0
        self.q_initial = torch.zeros_like(kq)

    def forward(self, state_dict: Dict[str, torch.Tensor]):
        # Parse states
        q_current = state_dict["joint_positions"]
        qd_current = state_dict["joint_velocities"]

        # Initialize
        if self.steps == 0:
            self.q_initial = q_current.clone()

        # Compute reference position and velocity
        q_desired = self.q_initial.clone()
        q_desired[5] = self.q_initial[5] + self.m * torch.sin(
            np.pi * self.steps / (self.hz * self.T)
        )
        qd_desired = torch.zeros_like(qd_current)

        # Execute PD control
        output = self.feedback(
            q_current, qd_current, q_desired, torch.zeros_like(qd_current)
        )

        # Check termination
        if self.steps > self.time_horizon:
            self.set_terminated()
        self.steps += 1

        return {"joint_torques": output}


# %%
# Initialize robot interface
robot = RobotInterface(
    ip_address="localhost",
)


# Reset
robot.go_home()

# Create policy instance
hz = robot.metadata.hz
default_kq = torch.Tensor(robot.metadata.default_Kq)
default_kqd = torch.Tensor(robot.metadata.default_Kqd)
policy = MySinePolicy(
    time_horizon=5 * hz,
    hz=hz,
    magnitude=0.5,
    period=2.0,
    kq=default_kq,
    kqd=default_kqd,
)

# Run policy
print("\nRunning custom sine policy ...\n")
state_log = robot.send_torch_policy(policy)
