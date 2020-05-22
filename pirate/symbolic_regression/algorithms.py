# File: algorithms.py
# File Created: Saturday, 20th July 2019 7:40:04 pm
# Author: Steven Atkinson (212726320@ge.com)

"""
Custom extensions on the deap algorithms.
"""

import abc
from typing import Tuple
from warnings import warn

from deap.algorithms import varAnd, varOr
from deap import tools


class AlgorithmState(object):
    """
    Class handling the state of the algorithm
    """

    def __init__(self, ngen: int):
        self.ngen = ngen
        self.complete_iterations = 0
        # How many new fitness evaluations we do in the current iteration
        self.nevals_this_iteration = None

        # Whether something has happened that means we need to terminate the
        # algorithm:
        self.flag_break = False

        # Ensure we're fresh:
        self.reset()

    def proceed(self) -> bool:
        """
        Whether we are good going with another generation of the algorithm
        """
        if self.flag_break:
            return False

        # Other criteria...
        if self.ngen is None:
            raise RuntimeError("Must set ngen")

        return self.complete_iterations < self.ngen

    def reset(self, ngen: int = None):
        """
        Reset before starting iteration through generations

        The values we reset to are:

        ngen -> what was provided (if applicable)
        complete_iterations -> 0
        flag_break -> False
        """
        if ngen is not None:
            self.ngen = ngen
        if self.ngen is None:
            raise ValueError("Must provide number of generations!")
        self.complete_iterations = 0
        self.flag_break = False


class Base(abc.ABC):
    """
    Base class for the algorithms
    """

    _gen_end_hooks = []

    def __init__(
        self,
        population,
        toolbox,
        cxpb,
        mutpb,
        ngen,
        stats=None,
        halloffame=None,
        verbose=__debug__,
        fitness_threshold: Tuple[float] = None,
    ):
        """
        :param fitness_threshold: If the fittest individual beats this 
            threshold, then signal to terminate the loop.
        """
        self.population = population
        self.toolbox = toolbox
        self.cxpb = cxpb
        self.mutpb = mutpb
        self._algorithm_state = AlgorithmState(ngen)
        self.stats = stats
        self.halloffame = halloffame
        self.verbose = verbose
        self.fitness_threshold = fitness_threshold

        # Hooks for the end of each iteration:
        self._gen_end_hooks = [
            self._increment_completed_generations,
            self._update_logbook,
            self._check_threshold,
        ]

        # Initialized in self._pre_run():
        self.logbook = None

    def run(self, ngen: int = None):
        self._pre_run()
        self._algorithm_state.reset(ngen=ngen)
        while self._algorithm_state.proceed():
            self._algorithm_state.nevals_this_iteration = self._generation()
            for hook in self._gen_end_hooks:
                hook()
        self._post_run()

    # Post-iteration hooks:

    def _update_logbook(self):
        """
        Append the current generation statistics to the logbook
        """
        record = self.stats.compile(self.population) if self.stats else {}
        self.logbook.record(
            gen=self._algorithm_state.complete_iterations,
            nevals=self._algorithm_state.nevals_this_iteration,
            **record
        )
        if self.verbose:
            print(self.logbook.stream)

    def _increment_completed_generations(self):
        """
        Hook for end of an iteration
        
        Increment the number of completed iterations
        """
        self._algorithm_state.complete_iterations += 1

    def _check_threshold(self):
        """
        Check to see if the fittest individual beats the provided threshold.
        If so, then set the algorithm state to flag for termination.
        """
        if self.fitness_threshold is None:
            return

        for ind in self.population:
            if all(
                [
                    fit_i < thresh_i
                    for fit_i, thresh_i in zip(
                        ind.fitness.values, self.fitness_threshold
                    )
                ]
            ):
                self._algorithm_state.flag_break = True
                break

    # Methods implementing before, during, and after the loop:

    def _generation(self) -> int:
        """
        This function shall implement a single iteration (generation) of the
        evolutionary algorithm

        :return: number of function evaluations this iteration
        """
        warn("Base class does not evolve the population.")

    def _pre_run(self):
        """
        Tasks to be run before the main loop of the algorithm
        """
        self.logbook = tools.Logbook()
        self.logbook.header = ["gen", "nevals"] + (
            self.stats.fields if self.stats else []
        )

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in self.population if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if self.halloffame is not None:
            self.halloffame.update(self.population)

        record = self.stats.compile(self.population) if self.stats else {}
        self.logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if self.verbose:
            print(self.logbook.stream)

    def _post_run(self):
        """
        Any post-hooks we might want after the loop terminates
        """
        pass


class EvolutionaryAlgorithmSimple(Base):
    """
    cf. deap.algorithms.eaSimple
    """

    def _generation(self) -> int:
        # Select the next generation individuals
        offspring = self.toolbox.select(self.population, len(self.population))

        # Vary the pool of individuals
        offspring = varAnd(offspring, self.toolbox, self.cxpb, self.mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if self.halloffame is not None:
            self.halloffame.update(offspring)

        # Replace the current population by the offspring
        self.population[:] = offspring

        return len(invalid_ind)


def eaSimple(
    population,
    toolbox,
    cxpb,
    mutpb,
    ngen,
    stats=None,
    halloffame=None,
    verbose=__debug__,
    fitness_threshold: Tuple[float] = None,
):
    """
    Wrapper to conform to the API for DEAP

    Additional kwargs:
    :param fitness_threshold: See Base
    """
    alg = EvolutionaryAlgorithmSimple(
        population,
        toolbox,
        cxpb,
        mutpb,
        ngen,
        stats=stats,
        halloffame=halloffame,
        verbose=verbose,
        fitness_threshold=fitness_threshold,
    )
    alg.run()

    return alg.population, alg.logbook
