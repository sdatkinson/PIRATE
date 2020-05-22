# File: operator.py
# File Created: Thursday, 14th March 2019 7:09:59 pm
# Author: Steven Atkinson (212726320@ge.com)

"""
Operators.
"""

from .function import Function


def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def neg(a):
    return -a


def mul(a, b):
    return a * b


def cos(x: Function) -> Function:
    return x.cos()


def exp(x: Function) -> Function:
    return x.exp()


def safe_divide(left, right):
    """
    A "safe" division that defines division-by-zero to have a quotient of 0.
    """
    try:
        return left / right
    except ZeroDivisionError:
        return 0.0 * left  # Ensure type match


def sin(x: Function) -> Function:
    return x.sin()


class Gradient(object):
    """
    Callable for taking a gradient of a function
    """

    def __init__(self, dims):
        self._dims = [dims] if isinstance(dims, int) else dims

    def __call__(self, f: Function) -> Function:
        return f.gradient(self._dims)

    @property
    def __name__(self):
        return "grad"


class ScalarGradient(Gradient):
    """
    Callable for computing the gradient of a scalar function w.r.t. one of its
    input dimensions
    """

    def __init__(self, dims):
        assert isinstance(dims, int)
        super().__init__(dims)

    @property
    def __name__(self):
        return "ddx%i" % self._dims[0]


class Divergence(object):
    def __init__(self, dims):
        self._dims = [dims] if isinstance(dims, int) else dims

    def __call__(self, f: Function) -> Function:
        return f.divergence(self._dims)

    @property
    def __name__(self):
        return "div"
