# File: symbolic_regression.py
# File Created: Wednesday, 6th February 2019 9:33:12 am
# Author: Steven Atkinson (212726320@ge.com)

"""
Symbolic regression capabilities, built on DEAP

TODO:
* LaTeX-ify expressions
* Simplify expressions
"""

from functools import partial
import operator
import math
from warnings import warn
import inspect
import types
from typing import Callable, Tuple

from deap import base, creator, tools, gp
import numpy as np
import matplotlib.pyplot as plt

from . import algorithms
from .deap_extensions import Parameter, PrimitiveSet, PrimitiveTree, compile
from .fitness import Base as BaseFitness


def get_primitive_set(
    primitives,
    num_variables: int = None,
    variable_names=None,
    parameters=False,
    name="MAIN",
) -> PrimitiveSet:
    """
    The primitive set is what DEAP uses to construct individuals 
    (i.e. PrimitiveTree objects).
    This funciton simplifies the creation of that primitive set; just provide 
    the operators that you want to use and either the number of variables or
    the variable names.

    :param primitives: The operators on functions that the genetic programming 
    algorithm will utilize.
    :type primitives: tuple of callables
    :param num_variables: How many variables (i.e. dependent variable functions)
    there are that will act as leaves on the graphs.
    :type num_variables: int
    :param variable_names: Alternatively to providing num_variables, you can 
    provide the names of the variables directly.
    :type variable_names: tuple of strings
    :param parameters: Whether or not we'll include aprameters (i.e. constant 
    functions) as leaves.
    :type parameters: bool

    :return: (PrimitiveSet)
    """
    assert (num_variables is not None) ^ (
        variable_names is not None
    ), "Must provide num_variables xor variable_names"
    if variable_names is not None:
        num_variables = len(variable_names)
    primitive_set = PrimitiveSet(name, num_variables)
    for p in primitives:
        primitive_set.addPrimitive(p, len(inspect.signature(p).parameters))

    # Rename arguments to something sensible for us:
    if variable_names is not None:
        renames = {}
        for i, variable_name in enumerate(variable_names):
            renames["ARG{}".format(i)] = variable_name
        primitive_set.renameArguments(**renames)

    # Add parameters:
    if parameters:
        primitive_set.add_parameter("theta")

    return primitive_set


class SymbolicRegression(object):
    """
    Main class for doing symbolic regression
    """

    def __init__(
        self,
        primitive_set: PrimitiveSet,
        fitness_function: BaseFitness,
        population: int = 100,
        mating_probability: float = 0.5,
        mutation_probability: float = 0.1,
        hall_of_fame_size: int = 10,
    ):
        """
        :param primitive_set: see get_primitive_set()
        :param fitness_function: Evaluates how good an individual is.
        """
        self.primitive_set = primitive_set
        self.fitness_function = fitness_function

        self._define_deap_fitness()

        self.toolbox = self._initialize_toolbox()

        self.population = self.toolbox.population(n=population)
        self.hall_of_fame = self._initialize_hall_of_fame(hall_of_fame_size)

        # Population statistics
        self.statistics = self._initialize_statistics()
        self.logbook = None

        self.mating_probability = mating_probability
        self.mutation_probability = mutation_probability

    def get_best_expression(self):
        return compile(self.hall_of_fame[0], self.primitive_set)

    def plot_fitness(
        self,
        annotate: bool = False,
        ground_truth: float = None,
        show: bool = True,
        save_as: str = None,
    ):
        """
        Look at the fitness of the population as optimization goes.

        :param ground_truth: If you want to plot one, then here you go.
        """
        # Plot curves
        median_fitness = np.array([g["Median"] for g in self.logbook])
        min_fitness = np.array([g["Min"] for g in self.logbook])
        generation = np.arange(len(min_fitness))
        plt.figure()
        plt.step(generation, median_fitness, where="post", label="Median")
        plt.step(generation, min_fitness, where="post", label="Best")
        if ground_truth is not None:
            plt.gca().axhline(
                y=ground_truth, color="r", linestyle="--", label="Ground truth"
            )
        plt.legend()
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.yscale("log")

        # Annotate new bests:
        if annotate:
            print("Best expressions:")
            for i, expression, val in zip(*self._new_best_individuals()):
                print(
                    " Generation {}:\n  Expression = {}\n  Fitness = {}".format(
                        i, expression, val
                    )
                )

        if save_as is not None:
            plt.savefig(save_as)
        if show:
            plt.show()
        else:
            plt.close()

    def run(self, iterations=100, verbose=True, fitness_threshold: Tuple[float] = None):
        """
        Carry out the symbolic regression optimization.
        Return the best function.
        """

        pop, self.logbook = algorithms.eaSimple(
            self.population,
            self.toolbox,
            self.mating_probability,
            self.mutation_probability,
            iterations,
            stats=self.statistics,
            halloffame=self.hall_of_fame,
            verbose=verbose,
            fitness_threshold=fitness_threshold,
        )
        # TODO: a .latex() member function for these expressions
        if verbose:
            print(
                "Best individual is:\n{}\nfitness = {}".format(
                    self.hall_of_fame[0], self.hall_of_fame[0].fitness.values[0]
                )
            )

    def _define_deap_fitness(self):
        """
        Define the elements for individuals and their fitness in deap's creator
        """

        # Honestly, I'm not confident about where this gets used. Perhaps in the
        # hall of fame?
        creator.create("FitnessMin", base.Fitness, weights=self._fitness_weights())
        creator.create(
            "Individual",
            PrimitiveTree,
            fitness=creator.FitnessMin,
            pset=self.primitive_set,
        )

    def _fitness_weights(self):
        return (-1.0,)

    def _initialize_hall_of_fame(self, n):
        return tools.HallOfFame(n)

    def _initialize_statistics(self):
        statistics = tools.Statistics(lambda ind: ind.fitness.values)
        statistics.register("Min", np.min)
        statistics.register("Median", np.median)
        statistics.register("Max", np.max)
        return statistics

    def _initialize_toolbox(self):
        toolbox = base.Toolbox()
        toolbox.register(
            "expr", gp.genHalfAndHalf, pset=self.primitive_set, min_=1, max_=2
        )
        toolbox.register(
            "individual", tools.initIterate, creator.Individual, toolbox.expr
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("lambdify", compile, pset=self.primitive_set)
        toolbox.register("evaluate", self.fitness_function)
        toolbox.register("select", self._get_selection_function())
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register(
            "mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=self.primitive_set
        )
        return toolbox

    def _new_best_individuals(self):
        """
        For the generations where the best individual changed, return the 
        generation index, the new best individual, and its fitness.
        """
        if not hasattr(self.fitness_function, "expression_cache"):
            raise RuntimeError(
                "Use a fitness function with an expression cache to "
                + "annotate the best fitnesses"
            )

        min_fitness = np.array([g["Min"] for g in self.logbook])
        d_best = np.concatenate((np.array([-np.inf]), np.diff(min_fitness)))

        generation = np.where(np.logical_not(d_best == 0.0))[0].tolist()
        fitness = min_fitness[generation]
        expression = []
        for i, val in zip(generation, fitness):
            for cache_key, cache_val in self.fitness_function.expression_cache.items():
                if cache_val[0] == val:
                    expression.append(cache_key)
                    break

        return generation, expression, fitness

    def _get_selection_function(self):
        return partial(tools.selTournament, tournsize=3)
