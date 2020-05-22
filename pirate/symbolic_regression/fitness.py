# File: fitness.py
# File Created: Thursday, 16th May 2019 4:23:50 pm
# Author: Steven Atkinson (212726320@ge.com)

"""
Fitness functions
"""

import abc
from copy import deepcopy
import types
from typing import Tuple, Sequence, Callable
from typing import List
from warnings import warn

import deap
import numpy as np
from pandas import DataFrame
import torch

from ..data.experiment import Experiment
from ..function import Function, wrap
from ..util import rms

from . import calibration
from .deap_extensions import PrimitiveSet, PrimitiveTree, compile
from .util import get_residual_function, tensor_to_parameter_dict


def simplify_expression(expression):
    # warn("Implement!")
    return expression


def add_memoization(cls):
    """
    Adds memoization to a fitness class's fitness function evaluations
    """

    class Memoized(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._expression_cache = {}
            self.verbose_cache_updates = False

        @property
        def expression_cache(self):
            return self._expression_cache

        def _call(self, individual: PrimitiveTree, **kwargs) -> tuple:
            simplified = simplify_expression(individual)
            hashed_expression = str(simplified)
            if not hashed_expression in self._expression_cache:
                self._expression_cache[hashed_expression] = cls._call(
                    self, simplified, **kwargs
                )
                if self.verbose_cache_updates:
                    print("(Cache size={})".format(len(self._expression_cache)))
            return self._expression_cache[hashed_expression]

    return Memoized


class Base(abc.ABC):
    """
    Basic fitness function
    """

    def __init__(self, expected_exceptions: Sequence = None):
        """
        :param expected_exceptions: A sequence of exception types that we might
            expect to encounter while trying various individuals.  If one of 
            these happen, don't worry!
        """
        self._expected_exceptions = (
            () if expected_exceptions is None else expected_exceptions
        )

    def __call__(self, individual: PrimitiveTree, **kwargs) -> tuple:
        """
        Take an expression, and evaluate its fitness

        :return: (tuple of float objects)
        """
        try:
            return self._call(individual, **kwargs)
        except self._expected_exceptions as e:
            return (np.inf,)

    @property
    def primitive_set(self):
        return self._primitive_set

    @abc.abstractmethod
    def _call(self, individual, **kwargs) -> Tuple:
        raise NotImplementedError("Implement fitness function evalutation")


class Compiling(Base):
    """
    Fitness functions that require compiling the individual.
    For this, we need a primitive set.
    """

    def __init__(
        self, primitive_set: PrimitiveSet, expected_exceptions: Sequence = None
    ):
        super().__init__(expected_exceptions=expected_exceptions)
        self._primitive_set = primitive_set


class Metric(Compiling):
    """
    Fitnesses based some metric notion
    """

    def __init__(
        self,
        primitive_set: PrimitiveSet,
        metric: Callable[[np.ndarray], float],
        expected_exceptions: Sequence = None,
    ):
        super().__init__(primitive_set, expected_exceptions=expected_exceptions)
        self._metric = metric


class Regression(Metric):
    """
    Fitness function for regression problems that seek to find a function

    f: reals^dx -> reals^dy.

    We use the (unary) metric on the residual.
    """

    def __init__(
        self,
        dataset: DataFrame,
        inputs: tuple,
        targets: tuple,
        primitive_set: PrimitiveSet,
        expected_exceptions: Sequence = None,
        metric: Callable[[np.ndarray], float] = rms,
    ):
        """
        :type inputs: tuple of str objects
        :type outputs: tuple of str objects
        """
        super().__init__(primitive_set, metric, expected_exceptions=expected_exceptions)
        self._inputs, self._outputs = None, None
        self._unpack_dataset(dataset, inputs, targets)

    def _call(self, individual: PrimitiveTree, **kwargs) -> tuple:
        """
        Compile the expression encoded by the individual, evalutate it on the
        available input data, and compare it against the targets using our 
        metric.

        Because deap allows for multiple objectives, we return the fitness as a
        size-1 tuple holding a float.

        :return: (tuple size 1 of floats) the fitness
        """
        func = compile(expr=individual, pset=self.primitive_set)
        outputs = func(*self._inputs)
        residual = outputs - self._targets
        return (self._metric(residual),)

    def _unpack_dataset(self, dataset: DataFrame, inputs: tuple, targets: tuple):
        self._inputs = [dataset[i].values for i in inputs]

        if len(targets) > 1:
            raise ValueError("Only one output allowed right now.")
            # We can extend this later.
        self._targets = dataset[targets[0]].values


def differential_residual(
    func: Callable,
    experiment: Experiment,
    pset: PrimitiveSet,
    inputs: np.ndarray,
    parameters=None,
    **kwargs
) -> np.ndarray:
    """
    Compute the residual of the compositional operator defined by func, a 
    compiled deap individual

    F[f1(x), ..., fn(x); theta1, ...thetam] = r(x; Theta)

    :param func: output of compile(individual, primitive_set).  A 
    composed operator that takes in functions (and possibly parameters) and 
    outputs a function from spatiotemporal inputs to (hopefully) a scalar.
    :param experiment: Holds all Functions and data that are required by func.
    :param pset: primitive set.  Its .arguments tell us what to get from 
    experiment.
    :param inputs: Set of spatiotemporal locations at which the composed 
    function should be evaluated.
    :param kwargs: provided directly to the residual function.
    """

    residual = Function(get_residual_function(func, experiment, pset))

    return wrap(residual)(inputs, **kwargs)


class DifferentialResidual(Metric):
    """
    Fitness function for the residual of a differential operator e.g. L[u] - f
    """

    def __init__(
        self,
        experiments: List[Experiment],
        primitive_set: PrimitiveSet,
        differential_operators: List[str] = None,
        expected_exceptions: Sequence = None,
        metric: Callable[[np.ndarray], float] = rms,
        threshold: float = 1.0e-6,
        ensemble_metric: Callable[[np.ndarray], float] = np.mean,
        require_function: bool = True,
    ):
        """
        Evaluate how well a differential operator L[u]=f describes things.

        :param experiments: the experiments we're evaluating 
        :param primitive_set: the operators we're prepared to work with
        :param differential_operators: A list of names (as strings) that the 
            differential operators go by.  We requier at least one of them to be
            present in an individual to consider it a valid differential 
            equation.
        :param require_function: If true, we regard any individual without at 
        least one terminal that is a (dependent variable) Function to be 
        invalid.
        """
        super().__init__(primitive_set, metric, expected_exceptions=expected_exceptions)
        self.differential_operators = (
            ["gradient"] if differential_operators is None else differential_operators
        )
        self._experiments = (
            [experiments] if isinstance(experiments, Experiment) else experiments
        )
        self._threshold = threshold
        self._ensemble_metric = ensemble_metric
        self.require_function = require_function

    @property
    def experiments(self):
        return self._experiments

    def residual(
        self,
        func: Callable,
        experiment: Experiment,
        inputs: np.ndarray = None,
        **kwargs
    ) -> np.ndarray:
        """
        Compute the residual of a function 
        """

        inputs = experiment.data.values if inputs is None else inputs
        return differential_residual(
            func, experiment, self.primitive_set, inputs, **kwargs
        )

    def _bad_fitness(self, fitness, experiment):
        """
        Check the provided fitness for signs of pathological behavior
        """
        # Catch NaNs:
        if np.any(np.isnan(fitness)):
            return True

        # Catch trivial equations numerically:
        if experiment.inhomogeneous is None and fitness < self._threshold:
            return True

        # Other bad things here...

        return False

    def _bad_individual(self, individual):
        """
        Check the provided individual for signs of pathological behavior
        """
        # Make sure it's a DIFFERENTIAL operator!
        if self.differential_operators and not np.any(
            [d in str(individual) for d in self.differential_operators]
        ):
            return True

        # Ensure that we have a dependent variable function in the graph (not
        # just parameters)
        if self.require_function and not np.any(
            [f in str(individual) for f in self.experiments[0].left_hand_side.keys()]
        ):
            return True

        # If the operator describes a function whose output is non-scalar, then
        # it can't lead to the residual of a *single* differential equation,
        # which is all that we consider at the moment.
        if not self._individual_maps_to_scalar(individual):
            return True

        # Other bad things here.

        return False

    def _bad_residual(self, residual: Tuple[float]) -> bool:
        """
        Check the provided residual for signs of pathological behavior
        """
        # Other things that are bad go here

        return False

    def _call(self, individual: PrimitiveTree, inputs: np.ndarray = None) -> tuple:
        """
        Reduce the fitnesses over the ensemble of experiments
        """
        # Catch bad individuals
        if self._bad_individual(individual):
            return (np.inf,)

        if individual.has_parameters:
            # This should be easy to lift...
            assert len(self.experiments) == 1, "Only one experiment for now"
            calibration_results = calibration.functional.calibrate(
                individual, self.primitive_set, self.experiments[0]
            )
            fitness = calibration_results.loss
            individual.calibration_results = calibration_results
        else:
            func = compile(expr=individual, pset=self.primitive_set)
            fitnesses = np.array(
                [
                    self._call_single_experiment(func, ex, inputs=inputs)
                    for ex in self._experiments
                ]
            )
            fitness = self._ensemble_metric(fitnesses)

        return (fitness,)

    def _call_single_experiment(
        self,
        func: Callable,
        experiment: Experiment,
        inputs: np.ndarray = None,
        **kwargs
    ) -> Tuple[float]:
        """
        Evaluate the residual function on the provided experiment at the
        available input data, and reduce it using our metric.

        Because deap allows for multiple objectives, we return the fitness as a
        size-1 tuple holding a float.

        :return: (tuple size 1 of floats) the fitness
        """
        residual = self.residual(func, experiment, inputs=inputs)

        # Catch bad residuals and metrics:
        if self._bad_residual(residual):
            return (np.inf,)
        fitness = self._metric(residual)
        if self._bad_fitness(fitness, experiment):
            return (np.inf,)

        return (fitness,)

    def _individual_maps_to_scalar(self, individual: PrimitiveTree) -> bool:
        """
        Determine whether the compositional operator encoded by the individual
        results in a function that maps to a scalar output.

        :param individual: The graph representation of the operator
        """

        func = compile(expr=individual, pset=self.primitive_set)

        # Get a datum from an experiment
        experiment = self.experiments[0]
        datum = experiment.data.loc[experiment.data.index[[0]], :].values
        parameters = tensor_to_parameter_dict(torch.zeros(individual.num_parameters))
        # And compute
        residual = self.residual(func, experiment, inputs=datum, **parameters)

        return residual.shape[1] == 1
