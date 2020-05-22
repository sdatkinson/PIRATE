# File: base.py
# File Created: Wednesday, 2nd January 2019 2:36:33 pm
# Author: Steven Atkinson (212726320@ge.com)

"""
Base classes for data
"""

import abc
from typing import List

import numpy as np
from pandas import DataFrame


class System(object):
    pass


class UnsupervisedSystem(System, abc.ABC):
    """
    Base class for supervised data.

    We adopt the "generative" convention, in which the unlabeled data are 
    regarded as "outputs"
    """

    def __init__(self, dy: int):
        """
        :param dy: output dimension
        """
        self._output_dimensions = tuple(["y" + str(i + 1) for i in range(dy)])

    @property
    def dy(self) -> int:
        """
        Output dimensionality of the system.
        """
        return len(self.output_dimensions)

    @property
    def observables(self):
        raise NotImplementedError("Define observable dimensions for your system")

    @property
    def output_dimensions(self):
        """
        Get the list of names of the output dimensions

        :return: Iterable whose elements are str instances
        """
        return self._output_dimensions

    @abc.abstractclassmethod
    def sample(self, n: int = 1) -> DataFrame:
        """
        Produce data randomly from the system.

        To generate data, we (1) draw random samples from a distribution over 
        the inputs, then (2) compute the corresponding outputs.

        :param n: How many data to produce
        :return: (DataFrame, [n x dx+dy]), the data.  Column names should
            match the input & output dimension names.
        """
        raise NotImplementedError("")


class SupervisedSystem(UnsupervisedSystem):
    """
    Base class for supervised data
    """

    def __init__(self, dx: int, dy: int):
        """
        :param dx: input dimension
        :param dy: output dimension
        """
        super().__init__(dy)
        self._input_dimensions = tuple(["x" + str(i + 1) for i in range(dx)])

    @property
    def dx(self) -> int:
        """
        Input dimensionality of the system.

        :return: int
        """
        return len(self.input_dimensions)

    @property
    def input_dimensions(self):
        """
        Get the list of names of the input dimensions

        :return: Iterable whose elements are str instances
        """
        return self._input_dimensions

    @abc.abstractclassmethod
    def sample(self, n: int = 1) -> DataFrame:
        """
        Produce data randomly from the system.

        To generate data, we (1) draw random samples from a distribution over 
        the inputs, then (2) compute the corresponding outputs.

        :param n: How many data to produce
        :return: (DataFrame, [n x dx+dy]), the data.  Column names should
            match the input & output dimension names.
        """
        raise NotImplementedError("")


class Parameterizable(abc.ABC):
    """
    A system with a parameterization
    """

    def __init__(self, system_parameters=None):
        # Just for declaration within __init__:
        self._system_parameters = None
        # Store callables in here that will be invoked at the end of setting
        # self.system_parameters
        self._system_parameters_setter_hooks = []

        # The actual setter:
        # Note: this won't have child classes' hooks yet, so you'll probably
        # need to invoke it again after the child classes' super().__init__()...
        self.system_parameters = (
            self._default_system_parameters()
            if system_parameters is None
            else system_parameters
        )

    @property
    def system_parameters(self):
        return self._system_parameters

    @system_parameters.setter
    def system_parameters(self, val: np.ndarray):
        if not np.all(val.shape == self._default_system_parameters().shape):
            raise ValueError(
                "System parameters shape {} ".format(val.shape)
                + "doesn't match expected {}".format(
                    self._default_system_parameters().shape
                )
            )
        self._system_parameters = val
        for hook in self._system_parameters_setter_hooks:
            hook()

    @abc.abstractmethod
    def randomize_system_parameters(self) -> None:
        """
        Pick a new random value for the system parameters and set it to 
        self._system_parameters.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _default_system_parameters(self):
        raise NotImplementedError("")


class Ensemble(object):
    """
    Class for an ensemble of systems, where each System is e.g. a particular 
    solution, and this Ensemble shares the same underlying governing equations.

    One example of this is the elliptic PDE with different a(x) and f(x)'es, 
    where we could talk about an ensemble of particular solutions that are all
    from the same governing PDE.
    """

    def __init__(
        self,
        system_type: type(UnsupervisedSystem),
        system_init_args=None,
        system_init_kwargs=None,
        common_inputs: np.ndarray = None,
    ):
        """
        :param system_type: The System type that we'll build an ensemble out of.
        :param system_init_args: 
        :param system init_kwargs:
        :param common_inputs:
        """
        self._system_type = system_type
        self._system_init_args = (
            system_init_args if system_init_args is not None else ()
        )
        self._system_init_kwargs = (
            system_init_kwargs if system_init_kwargs is not None else {}
        )

        print("kwargs = {}".format(self._system_init_kwargs))

        # Hold a list of systems that have been generated
        self._systems = []
        # So that we can sample each System instance at the same inputs.
        self._common_inputs = common_inputs

        # self._check_init()

    def new_system(self, n: int = 1, system_parameters: List[np.ndarray] = None):
        """
        Initialize new systems in this ensemble

        :param n: How many enw systems to instantiate.
        :param system_parameters: Can use this to get a specific ensemble of 
            systems
        """
        system_parameters = (
            [None] * n if system_parameters is None else system_parameters
        )
        assert (
            len(system_parameters) == n
        ), "Number of provided system_parameters must match n"

        kwargs = self._system_init_kwargs
        if self._common_inputs is not None:
            kwargs.update({"input_points": self._common_inputs})

        for _, sp in zip(range(n), system_parameters):
            new_system = self._system_type(*self._system_init_args, **kwargs)
            if sp is None:
                new_system.randomize_system_parameters()
            else:
                new_system.system_parameters = sp
            self._systems.append(new_system)

    def sample(self, *args, **kwargs):
        return [system.sample(*args, **kwargs) for system in self.systems()]

    def systems(self):
        """
        Generator to yield the systems in this ensemble
        """
        for s in self._systems:
            yield s

    def systems(self):
        """
        Generator to yield the systems in this ensemble
        """
        for s in self._systems:
            yield s

    def _check_init(self):
        pass
