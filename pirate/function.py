# File: function.py
# File Created: Tuesday, 19th November 2019 10:11:41 am
# Author: Steven Atkinson (212726320@ge.com)

"""
Dependent variable functions in PIRATE.

All functions are supposed to be of the form
"""


# File: data.py
# File Created: Thursday, 14th March 2019 3:17:21 pm
# Author: Steven Atkinson (212726320@ge.com)

"""
Classes for handling data
"""

import pandas as pd
import operator
import types
import typing
import numpy as np
import torch


class DivergenceError(ValueError):
    """
    Error type to catch when a divergence won't work (e.g. requiest)
    """

    pass


class GradientError(ValueError):
    """
    Catch errors where the gradient won't work.
    """

    pass


class CUDAError(ValueError):
    """
    Some diagreement about whether we should be using CUDA
    """

    pass


def _meta_function(
    metafunc: types.FunctionType, *funcs: types.FunctionType
) -> types.FunctionType:
    """
    Define a function composed of the outputs of some other functions.
    Examples:
    1) add(f1, f2) -> f1(x) + f2(x).
        "add" is the metafunction.
        f1 and f2 are the functions (arguments to the metafunction)
    2) exp(f1) -> exp(f1(x))
        exp is the metafunction
        f1 is the function

    :type metafunc: function
    :type funcs: function(s)
    :return: f(x) = metaop(f1, f2)(x)
    """

    def func(x, **kwargs):
        return metafunc(*[func(x, **kwargs) for func in funcs])

    return func


class Function(object):
    """
    Special callable with signature f(x, **kwargs), where x is a torch.Tensor of 
    shape [N x DX].

    kwargs might include...
    * parameters
    * parameterization for random functions

    We also track whether we expect CUDA
    """

    def __init__(self, func: types.FunctionType):
        """
        :param data: the independent variables on which the dataset is defined.
        :param func: A function that takes a torch.Tensor as input and
            returns a torch.Tensor as output,.
            Usually, you'll use the .forward() method of a torch.nn.Module 
            instance.
        :type func: function
        :param cuda: Whether we use cude to evaluate
        :param _output_dimension: How many output dimensions the func returns.
        :param dtype: The data type expected by the function's inputs
        """

        self._func = func

    def __call__(self, x: torch.Tensor, **kwargs):
        return self.func(x, **kwargs)

    @property
    def func(self):
        """
        Get the function that's under the hood.
        If you want to *evaluate* the function, just use __call__
        """

        return self._func

    # Operators:

    def __add__(self, other):
        """
        Creates a Function where the function is the sum of the outputs
        of self and other's functions (whatever those are)
        """
        return Function(
            _meta_function(operator.add, self.func, other.func),
            **self.pass_on_properties(other=other)
        )

    def __sub__(self, other):
        return Function(
            _meta_function(operator.sub, self.func, other.func),
            **self.pass_on_properties(other=other)
        )

    def __mul__(self, other):
        return Function(
            _meta_function(operator.mul, self.func, other.func),
            **self.pass_on_properties(other=other)
        )

    def __neg__(self):
        return Function(
            _meta_function(operator.neg, self.func), **self.pass_on_properties()
        )

    def __truediv__(self, other):
        return Function(
            _meta_function(operator.truediv, self.func, other.func),
            **self.pass_on_properties(other=other)
        )

    def cos(self):
        return Function(
            _meta_function(torch.cos, self.func), **self.pass_on_properties()
        )

    def divergence(self, dims):
        """
        Divergence operator.
        """

        if isinstance(dims, int):
            dims = [dims]

        def op(x, **kwargs):
            # Split-and-join for faster forward pass.
            # Here, we do a full flatten to get x_sequence as a list of length
            # n*dx, where each entry is a scalar.
            # We do this because we need to take dy_ij / dx_ij for all i and j.
            n, dx = x.shape
            x_sequence = [xi for xi in x]
            x_rejoined = torch.stack(x_sequence).reshape((n, dx))
            # Use PyTorch's autograd sequence handling to do one by one, then
            # stack them back together.
            y = self.func(x_rejoined)
            if y.shape[1] < max(dims) + 1:
                raise DivergenceError(
                    "Divergence requires %i dimensions but function only has %i."
                    % (max(dims) + 1, x.shape[1])
                )

            # Inner stack: computes gradient and keeps the dimension we want
            # Outer concat: repeats for every dimension
            dyi_dxi = torch.cat(
                [  # Dimensions
                    torch.stack(  # Data
                        torch.autograd.grad(
                            [yi[in_j] for yi in y], x_sequence, create_graph=True
                        )
                    )[:, [out_j]]
                    for in_j, out_j in enumerate(dims)
                ],
                dim=1,
            )
            # Finally, sum down over the output dimensions
            div = dyi_dxi.sum(dim=1, keepdim=True)
            return div

        return Function(op, **self.pass_on_properties())

    def exp(self):
        return Function(
            _meta_function(torch.exp, self.func), **self.pass_on_properties()
        )

    def gradient(self, dims):
        """
        Get the gradient of this modeled dataset.
        This is accomplished by replacing the model with a model of its gradient

        :param dims: If specified, retain only those dimensions.  NOTE: taking a
            divergence clears this.
        """

        if isinstance(dims, int):
            # i.e. an ordinary derivative.
            dims = [dims]

        def op(x, **kwargs):
            # Break out x so that autograd can track each one.
            x_sequence = [xi for xi in x]
            x2 = torch.stack(x_sequence)
            # Use PyTorch's autograd sequence handling to do one by one, then
            # stack them back together.
            y = self.func(x2)
            if y.shape[1] > 1:
                raise GradientError(
                    "Tried to take the gradient of a vector-valued function.\n"
                    + "Jacobians not supported."
                )

            dydx = torch.stack(
                torch.autograd.grad([yi[0] for yi in y], x_sequence, create_graph=True)
            )

            dydx = dydx[:, dims]
            # Hack to fix gradient flow in certain cases (e.g. second derivative
            # of a constant): add input (times zero) to result.
            dydx = dydx + 0.0 * x[:, dims]
            return dydx

        return Function(op, **self.pass_on_properties())

    def sin(self):
        return Function(
            _meta_function(torch.sin, self.func), **self.pass_on_properties()
        )

    # Other functionality:

    def pass_on_properties(self, other=None, **extra_kwargs):
        """
        Creates a dict of properties that we want to pass on through operators.

        Optionally pass "other" Function (from binary operators).  
        If provided, make sure to merge properties correctly:
        1) Ensure a valid output dimension. Determine it, minding potential 
            broadcasting.
        * (Other rules as needed).

        :type other: Function
        """

        self_kwargs = {}
        self_kwargs.update(extra_kwargs)

        return self_kwargs


class Constant(Function):
    """
    A constant function (to be used as a parameter)
    """

    def __init__(self, val: torch.Tensor):

        if not isinstance(val, torch.Tensor):
            raise TypeError("Must provide torch.Tensor as val for Constant function")

        def func(x, **kwargs):
            return val + 0.0 * x[:, [0]]  # Ensures requires_grad = True

        super().__init__(func)


def wrap(f) -> np.ndarray:
    """
    Wrapper to handle torch/numpy types, CUDA...

    As needed:
    x -> PyTorch -> CUDA -> y=f(x) -> CPU -> numpy
    """

    def wrapped(x, **kwargs):
        from_numpy = isinstance(x, np.ndarray)
        if from_numpy:
            x = torch.Tensor(x)
        x.requires_grad_(True)

        if x.ndimension() == 1:
            x = x[:, None]
        y: torch.Tensor = f(x, **kwargs)

        if from_numpy:
            y: np.ndarray = y.detach().numpy()

        return y

    return wrapped
