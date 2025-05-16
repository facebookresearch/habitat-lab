import pymomentum.geometry as pym_geometry
import pymomentum.skel_state as pym_skel_state
import pymomentum.solver as pym_solver

import torch

from pymomentum.solver import ErrorFunctionType
import os

class DifferentialInverseKinematics:
    def __init__(self):

        self.robot = pym_geometry.Character.load_urdf(
                os.getcwd() + "/data/hab_murp/fr3/fr3_no_gripper.urdf",
            )


    def inverse_kinematics(self, pose, q):


        #pymomentum uses centimeters for positions.
        position_cons_targets = [ [pose.x*100, pose.y*100, pose.z*100] ]

        _orientation = pose.UnitQuaternion().A
        orientation_cons_targets = [ [_orientation[1], _orientation[2], _orientation[3], _orientation[0]] ]
        return solve_one_ik_problem(self.robot, position_cons_targets, orientation_cons_targets, q).detach().numpy()[0]
        


def solve_one_ik_problem(character, _position_cons_targets, _orientation_cons_targets, initial_guess) -> torch.Tensor:
    
    n_joints = character.skeleton.size
    n_params = character.parameter_transform.size

    batch_size = 1

    # Ensure repeatability in the rng:
    torch.manual_seed(0)
    model_params_init = torch.zeros(batch_size, n_params, dtype=torch.float64)
    model_params_init[0] = torch.tensor(initial_guess, dtype=torch.float64)

    active_error_functions = [
        ErrorFunctionType.Limit,
        ErrorFunctionType.Position,
        ErrorFunctionType.Orientation,
        ErrorFunctionType.Collision,
    ]
    error_function_weights = torch.ones(
        batch_size,
        len(active_error_functions),
        requires_grad=True,
        dtype=torch.float64,
    )

    pos_cons_targets = torch.tensor(
        _position_cons_targets, dtype=torch.float64
    )  

    pos_cons_parents = torch.arange(n_joints-1, n_joints)

    rot_cons_targets = torch.tensor(
        _orientation_cons_targets, dtype=torch.float64)



    return pym_solver.solve_ik(
        character=character,
        active_parameters=character.parameter_transform.all_parameters,
        model_parameters_init=model_params_init,
        active_error_functions=active_error_functions,
        error_function_weights=error_function_weights,
        position_cons_parents= pos_cons_parents,
        position_cons_targets=pos_cons_targets,
        orientation_cons_targets=rot_cons_targets,
        orientation_cons_parents=pos_cons_parents,
        
    )


