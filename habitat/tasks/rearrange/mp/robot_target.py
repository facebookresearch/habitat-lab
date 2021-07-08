import attr
import magnum as mn
import numpy as np


@attr.s(auto_attribs=True, slots=True)
class RobotTarget:
    """
    Data class to define the input for the motion planner
    """

    # End-effector in world coordinate frame.
    ee_targ: np.ndarray = None
    obj_targ: int = None
    js_targ: np.ndarray = None
    is_guess: bool = False


@attr.s(auto_attribs=True, slots=True)
class ObjPlanningData:
    """
    Data class to define the input to the grasp planner.
    """

    bb: mn.Range3D
    trans: mn.Matrix4
