# File: mass_spring_damper.py
# File Created: Thursday, 14th March 2019 4:02:43 pm
# Author: Steven Atkinson (212726320@ge.com)

"""
Some ordinary differential equations
"""

import abc
from warnings import warn

import numpy as np
import pandas as pd
from scipy.integrate import odeint

from .base import SupervisedSystem, Parameterizable


class _ScalarODE(SupervisedSystem, Parameterizable):
    """
    Base class for scalar ordinary differential equations

    The system_parameters for the class are just the initial conditions.
    """

    def __init__(self, t_min: float, t_max: float):
        SupervisedSystem.__init__(self, 1, 1)
        Parameterizable.__init__(self)

        self._input_dimensions = ("t",)
        self._output_dimensions = ("x",)
        self.t_min, self.t_max = t_min, t_max

        # x(t) Data from the ODE solver
        self._t = np.linspace(self.t_min, self.t_max, 1000)
        self._x = None

    @property
    def observables(self):
        return self.output_dimensions

    @property
    def t_range(self) -> float:
        return self.t_max - self.t_min

    def sample(self, n: int, seed=None) -> pd.DataFrame:
        """
        Draw samples from n randomly-selected timepoints.
        """
        if self._x is None:
            self._x = self._integrate()
        if seed is not None:
            np.random.seed(seed)
        i = np.random.permutation(self._t.size)[:n]
        t, x = self._t[i], self._x[i]
        return pd.DataFrame(
            np.stack((t, x)).T, columns=self.input_dimensions + self.output_dimensions
        )

    @abc.abstractmethod
    def _dxdt(self, x, t):
        """
        Computes the derivative.
        """
        raise NotImplementedError("Implement derivative")

    def _integrate(self) -> np.ndarray:
        """
        Run the ODE solver.  Fill in self._t and self._x
        """
        return odeint(self._dxdt, self.system_parameters, self._t)[:, 0]


class FirstOrder(_ScalarODE):
    """
    dx(t)/dt + cx(t) = 0

    c is a constant called the "time constant".
    """

    def __init__(
        self, time_constant: float = 1.0, t_min: float = 0.0, t_max: float = 5.0
    ):
        super().__init__(t_min, t_max)
        self.time_constant = time_constant

    def randomize_system_parameters(self):
        self._system_parameters = np.random.randn(1)
        self._x = None

    def _default_system_parameters(self):
        return np.array([1.0])

    def _dxdt(self, x, t):
        return -self.time_constant * x


class SecondOrder(_ScalarODE):
    """
    ax'' + bx' + cx = 0

    Taking the analogy of a damped linear oscillator:
    * a is the mass
    * b is the damping coefficient
    * c is the spring coefficient

    system_parameters are [x(0), dx/dt(t=0)]

    As a system of equations,
    x1 = x(t)
    x2 = dx/dt
    dx1/dt = x2
    dx2/dt = -x2 - x1
    """

    def __init__(
        self,
        mass: float = 1.0,
        damping: float = 1.0,
        spring: float = 1.0,
        t_min: float = 0.0,
        t_max: float = 5.0,
    ):
        super().__init__(t_min, t_max)
        self.mass = mass
        self.damping = damping
        self.spring = spring

    def randomize_system_parameters(self):
        self._system_parameters = np.random.randn(2)
        self._x = None

    def _default_system_parameters(self):
        return np.array([1.0, 0.0])

    def _dxdt(self, x, t):
        """
        As a system of equations,
        x1 = x(t)
        x2 = dx/dt

        So,
        dx1/dt = x2
        dx2/dt = -b/a x2 - c/a x1
        """
        return np.array([x[1], -(self.damping * x[1] + self.spring * x[0]) / self.mass])


class NonlinearSecondOrder(_ScalarODE):
    """
    ax'' + b(x' + x'^3) + cx = 0

    Taking the analogy of a damped linear oscillator:
    * a is the mass
    * b is the damping coefficient
    * c is the spring coefficient

    system_parameters are [x(0), dx/dt(t=0)]

    As a system of equations,
    x1 = x(t)
    x2 = dx/dt
    dx1/dt = x2
    dx2/dt = -x2 - x1
    """

    def __init__(
        self,
        mass: float = 1.0,
        damping: float = 1.0,
        spring: float = 1.0,
        t_min: float = 0.0,
        t_max: float = 5.0,
    ):
        super().__init__(t_min, t_max)
        self.mass = mass
        self.damping = damping
        self.spring = spring

    def randomize_system_parameters(self):
        self._system_parameters = np.random.randn(2)
        self._x = None

    def _default_system_parameters(self):
        return np.array([1.0, 0.0])

    def _dxdt(self, x, t):
        """
        As a system of equations,
        x1 = x(t)
        x2 = dx/dt

        So,
        dx1/dt = x2
        dx2/dt = -b/a x2 - c/a x1
        """
        return np.array(
            [x[1], -(self.damping * (x[1] ** 3) + self.spring * x[0]) / self.mass]
        )
