# File: elliptic.py
# File Created: Thursday, 14th March 2019 4:02:43 pm
# Author: Steven Atkinson (212726320@ge.com)

"""
Elliptic differential equations
"""

import abc
import numpy as np
import os
from typing import Sequence, Tuple

try:
    import fenics

    FENICS = True
except ModuleNotFoundError as e:
    FENICS = False
import pandas as pd
from warnings import warn

from .base import SupervisedSystem, Parameterizable


class _Basic(SupervisedSystem, Parameterizable):
    """
    Base class for elliptic PDE systems:
    div(-a*grad(u)) = f.
    a is conductivity
    u is solution
    f is the forcing function

    System parameters for Elliptic classes refer to coefficients for a linear
    combination of solutions.
    For simplicity, we're restricting oru implementation to two componenents.
    """

    def __init__(
        self,
        nelem: int = None,
        input_points: np.ndarray = None,
        use_fenics=None,
        system_parameters=None,
    ):
        """
        :param nelem: How many elements in the FEM mesh
        :param input_points: ...or provide the inputs where you want the 
            solution explicitly.
        (but not both of these)
        :param use_fenics: Whether we use FEM or manufactured solutions
        """
        SupervisedSystem.__init__(self, 1, 1)
        Parameterizable.__init__(self, system_parameters=system_parameters)

        self._input_dimensions = ("x",)
        self._output_dimensions = ("conductivity", "solution", "force")

        if not (nelem is not None) ^ (input_points is not None):
            raise ValueError("Must provide either nelem or input_points")
        self._nelem = nelem
        self._input_points = input_points

        self._use_fenics = False  # FENICS if use_fenics is None else use_fenics
        if self._use_fenics and not FENICS:
            raise ValueError("Tried to use FEniCS, but not supported.")

        # Functions of x
        self._conductivity = None
        self._solution = None
        self._force = None

        self._check_init()

    @property
    def nelem(self):
        return self._nelem

    @property
    def observables(self):
        return self.output_dimensions

    def conductivity(self, x):
        """
        The conductivity parameter for the PDE
        """
        return self._pde_set_quantity("_conductivity", x)

    def force(self, x):
        """
        Obtain the forcing function f(x) to the PDE
        """
        return self._pde_set_quantity("_force", x)

    def randomize_system_parameters(self):
        self._system_parameters = np.random.rand(
            *self._default_system_parameters().shape
        )

    def sample(self, n=None, seed=None):
        if seed is not None:
            np.random.seed(seed)
        if self._input_points is not None and n is not None:
            raise ValueError("input_points is set; cannot provide n.")
        if self._input_points is None and n is None:
            raise ValueError("Must provide n")

        x = (
            self._input_points
            if self._input_points is not None
            else np.sort(np.random.rand(n))
        )
        a = self.conductivity(x)
        u = self.solution(x)
        f = self.force(x)
        return pd.DataFrame(
            np.stack((x, a, u, f)).T,
            columns=self.input_dimensions + self.output_dimensions,
        )

    def solution(self, x):
        """
        Obtain the solution u(x) to the PDE
        """
        return self._pde_set_quantity("_solution", x)

    def _check_init(self):
        if self._input_points is not None and (not self._input_points.ndim == 1):
            raise ValueError("Provided input_points must be 1D")

    def _default_system_parameters(self):
        return np.array([1.0, 0.0])

    def _pde_set(self):
        """
        Generate the inputs and particular solution to the PDE and cache the 
        data in the object.
        """
        if self._solution is None:
            assert (
                self._force is None
            ), "Expect force function to be unset if solution is unset."
            assert (
                self._conductivity is None
            ), "Expect conductivity function to be unset if solution is unset."
            if self._use_fenics:
                self._pde_set_fenics()
            else:
                # warn("FEniCS is not available.  Using ._pde_set_manual()")
                self._pde_set_manual()

    @abc.abstractmethod
    def _pde_set_fenics(self):
        """
        Use FEniCS to solve a FEM problem
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _pde_set_manual(self):
        """
        Implement a manufactured solution in case FEniCS is not available 
        (e.g. on Widnows...)
        """
        raise NotImplementedError("Implement a manufactured solution")

    def _pde_set_quantity(self, func_str, x):
        """
        Any quantitity that depends on the PDE solution.

        We reference the function by string name instead of passing the function
        itself into this function because we can't pass by reference (I think).
        This is how we ensure that the function is set after calling 
        ._pde_set().

        :param func_str: string name of the member function.
        :type func_str: str
        :return: (float or np.ndarray) function evaluated at x.
        """
        self._pde_set()
        func = getattr(self, func_str)
        assert func is not None, "Function {} is None".format(func_str)

        y = (
            func(x)
            if not isinstance(x, np.ndarray)
            else np.array([func(xi) for xi in x])
        )
        return y


class Elliptic1DNonlinear(_Basic):
    """
    Elliptic problem with nonlinear conductivity a(u(x))
    d(-a*du) = f
    a(u) = exp(u)
    u(0) = 0
    u(1) = 1
    """

    def _pde_set_fenics(self):
        raise NotImplementedError("Waad please implement")

    def _pde_set_manual(self):
        omega = 2.5
        c = self.system_parameters

        self._solution = lambda x: c[0] * np.sin(omega * x) + c[1] * x ** 2
        self._conductivity = lambda x: np.exp(self._solution(x))

        du = lambda x: c[0] * omega * np.cos(omega * x) + c[1] * 2.0 * x
        ddu = lambda x: -c[0] * omega * omega * np.sin(omega * x) + 2.0 * c[1]
        # da(u(x))/dx = du/dx * da/du.
        # a(u) = exp(u) -> da/du = exp(u)
        da = lambda x: du(x) * self._conductivity(x)

        self._force = lambda x: -da(x) * du(x) - self._conductivity(x) * ddu(x)


class Elliptic1DLinearHeterogeneous(_Basic):
    """
    Linear problem with heterogeneous conductivity a(x)
    """

    def _pde_set_fenics(self):
        raise NotImplementedError("Waad please implement")

    def _pde_set_manual(self):
        alpha = 2.5
        c = self.system_parameters
        self._solution = lambda x: c[0] * np.exp(alpha * x) + c[1] * x ** 2
        self._conductivity = lambda x: 2.0 + np.cos(x)
        self._force = (
            lambda x: -2.0 * c[0] * alpha ** 2 * np.exp(alpha * x)
            - 4.0 * c[1]
            - c[0]
            * alpha
            * (-np.sin(x) * np.exp(alpha * x) + alpha * np.cos(x) * np.exp(alpha * x))
            - 2.0 * c[1] * (-np.sin(x) * x + np.cos(x))
        )


class Elliptic2DAtkinsonZabaras(SupervisedSystem, Parameterizable):
    """
    This is a base class for 2D elliptic problems.

    In the future, this should be joined with the elliptic 1D, but due to lack
    of FEniCS on Windows, this will wrap the dataset in:

    S. Atkinson and N. Zabaras, 
    "Structured Bayesian Gaussian process latent variable model: applications to
    data-driven dimensionality reduction and high-dimensional inversion"
    (J. Comp Phys 2019)

    Data taken from: https://github.com/cics-nd/sgplvm-inverse

    For simplicity and lightweightness, this class uses the first 16 training 
    examples only.
    """

    def __init__(
        self,
        kl: int = 32,
        x1_limit: Sequence[float] = None,
        x2_limit: Sequence[float] = None,
    ):
        """
        :param kl: number of terms in the truncated KL expansion for the second
            layer of the warped GP inputs. ("KL 16-32" or "KL 16-128" in the 
            original dataset)
        """
        SupervisedSystem.__init__(self, 2, 2)  # 2D input, 1D output
        Parameterizable.__init__(self)

        self._input_dimensions = ("x1", "x2")
        self._output_dimensions = ("log_conductivity", "solution")
        self._dataframe = None  # Where loaded data is stored
        self._n_solutions = 16
        self._x1_limit, self._x2_limit = x1_limit, x2_limit

        assert kl == 32 or kl == 128, "kl=32 or 128 only"
        self._kl = kl
        self._system_parameters_setter_hooks.append(self._load_data)
        self._system_parameters_setter_hooks.append(self._apply_limits)
        self.system_parameters = self._default_system_parameters()

    @property
    def num_data(self) -> int:
        return self._dataframe.shape[0]

    @property
    def input_shape(self) -> Tuple[int, int]:
        """
        If you want to reshape the data back to make surface plots, what should
        the shape be?
        """
        return tuple([len(set(self._dataframe[col])) for col in self.input_dimensions])

    def get_dataframe(self) -> pd.DataFrame:
        return self._dataframe.copy()

    def randomize_system_parameters(self):
        val = np.array([np.floor(self._n_solutions * np.random.rand())], dtype=int)
        self.system_parameters = val

    def sample(self, n: int = 1) -> pd.DataFrame:
        assert n <= self.num_data, "Only have %i data available" % self.num_data
        i = np.random.permutation(self.num_data)[:n]

        return pd.DataFrame(self._dataframe.loc[i, :].copy())

    def _apply_limits(self):
        """
        Subselects the data for a smaller domain in u and t
        """

        def _apply_limit(lim, var):
            if lim is not None:
                self._dataframe = self._dataframe.loc[
                    (self._dataframe[var] >= lim[0]) & (self._dataframe[var] < lim[1]),
                    :,
                ].reset_index(drop=True)

        _apply_limit(self._x1_limit, "x1")
        _apply_limit(self._x2_limit, "x2")

    def _default_system_parameters(self):
        """
        The system parameter in this class is just an index for a solution, so 
        it lives in the space of integers from 0 to 1023
        """
        return np.array([0], dtype=int)

    def _load_data(self):
        """
        Read the data files corresponding to the system parameter currently set.
        """
        i = self.system_parameters[0]
        dirname = os.path.join(
            os.path.dirname(__file__),
            "elliptic_2d_atkinson_zabaras",
            "kl_16_%i" % self._kl,
        )
        a_path = os.path.join(dirname, "%i_log_a.npy" % i)
        u_path = os.path.join(dirname, "%i_u.npy" % i)

        log_a, u = np.log(np.load(a_path)), np.load(u_path)
        n_per_side = log_a.shape[0]

        xi = np.linspace(0, 1, n_per_side)
        x1 = np.tile(xi[:, np.newaxis], (1, n_per_side)).flatten()
        x2 = np.tile(xi[np.newaxis, :], (n_per_side, 1)).flatten()

        log_a_flat, u_flat = log_a.flatten(), u.flatten()

        self._dataframe = pd.DataFrame(
            {"x1": x1, "x2": x2, "log_conductivity": log_a_flat, "solution": u_flat}
        )
