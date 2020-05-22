# File: util
# File Created: Wednesday, 20th November 2019 3:56:01 pm
# Author: Steven Atkinson (212726320@ge.com)

from typing import Callable

import deap.gp
import torch

from ..data.experiment import Experiment
from ..function import Function


def get_residual_function(
    op: Callable, experiment: Experiment, pset: deap.gp.PrimitiveSet
) -> Function:
    """
    Create the parametric residual function r(x; Theta)

    :param op: Operator over functions, aka a graph as a compiled function

    :return: Callable with signature r(x, theta_0=val, ...theta_m-1=val)
    """

    # TODO would like to make it easier to see how many parameters "op" expects
    def residual(x, **parameters):
        # First, evaluate the operator over functions and parameters:
        func = op(
            *[experiment.left_hand_side[key] for key in pset.arguments], **parameters
        )
        # Then, subtract the inhomogeneous function
        if experiment.inhomogeneous is not None:
            func = func - experiment.inhomogeneous

        return func(x)

    return residual


def tensor_to_parameter_dict(x: torch.Tensor) -> dict:
    """
    Take an array of parameter values and restructure it as a dict that's a 
    valid input for the **parameters kwarg for a residual function returned by 
    `get_residual_function()`

    :param x: 1D array of parameters

    :return: (dict) parameter specification
    """

    return {"theta_%i" % i: val for i, val in enumerate(x)}
