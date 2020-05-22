# File: experiment_util.py
# File Created: Saturday, 20th July 2019 12:40:03 pm
# Author: Steven Atkinson (212726320@ge.com)

"""
Utilities for the experiments
"""

import argparse
from collections import namedtuple
import math

import numpy as np
import deap.gp
import matplotlib.pyplot as plt

from pirate.function import wrap
from pirate.util import rms


def parse_args_non_adaptive(testing):
    if testing:
        Args = namedtuple("Args", ("n_train", "seed", "overwrite", "no_write"))
        return Args(5, 0, False, True)
    parser = argparse.ArgumentParser()
    parser.add_argument("n_train", type=int, help="Number of training data")
    parser.add_argument("seed", type=int, help="RNG seed")
    parser.add_argument(
        "--overwrite", "-o", action="store_true", help="Overwrite existing results"
    )
    parser.add_argument("--no-write", action="store_true", help="Don't write results")
    return parser.parse_args()


def parse_args_adaptive(testing):
    if testing:
        Args = namedtuple("Args", ("seed", "threshold", "overwrite", "no_write"))
        return Args(0, 0.1, False, True)
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int, help="RNG seed")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Termination threshold for adaptive data acquisition",
    )
    parser.add_argument(
        "--overwrite", "-o", action="store_true", help="Overwrite existing results"
    )
    parser.add_argument("--no-write", action="store_true", help="Don't write results")
    return parser.parse_args()


def check_answer(
    x_test,
    symbolic_regression,
    ground_truth_model,
    rel_tol=1.0e-4,
    abs_tol=1.0e-4,
    show=False,
) -> int:
    """
    Does the fittest individual from the SR match the ground truth?

    We use a comparison of the residual over a more extensive test set for a 
    more demanding criterion of being a match.

    If there is no match on the test set, then we use the training set (that 
    guided the genetic programming algorithm) to decide how we went wrong.

    Possible returns:
        0: Yes, they match!
        1: No match: best answer has higher fitness than ground truth 
           (didn't search hard enough?)
        2: No match: best answer has lower fitness than the ground truth
           (Models aren't good enough to rule out wrong answers?)
        -1: Some other problem?
    """
    # Take one experiment to use
    experiment = symbolic_regression.fitness_function.experiments[0]
    x_train = experiment.data.values

    # Compute residual on truth and best, train and test:
    gtm = wrap(ground_truth_model(experiment))
    res_true_train = gtm(x_train)
    res_true_test = gtm(x_test)

    # get the best residual
    f_best = deap.gp.compile(
        expr=symbolic_regression.hall_of_fame[0], pset=symbolic_regression.primitive_set
    )
    res_best_train = symbolic_regression.fitness_function.residual(
        f_best, experiment, x_train
    )
    res_best_test = symbolic_regression.fitness_function.residual(
        f_best, experiment, inputs=x_test
    )

    # Sometimes when there's no inhomogeneous term we can be off by a sign flip
    if experiment.inhomogeneous is None:
        res_best_test *= np.sign(res_best_test) * np.sign(res_true_test)

    rms_best, rms_true = rms(res_best_train), rms(res_true_train)
    print("RMS:\n Best : %e\n True : %e" % (rms_best, rms_true))

    if show:
        # Extra evaluations to be extra-sure!
        res_true_test2 = ground_truth_model(experiment).eval(
            x_test, no_sample_state=True
        )
        res_best_test2 = symbolic_regression.fitness_function.residual(
            f_best, experiment, inputs=x_test
        )

        plt.figure()
        plt.plot(x_test, res_best_test, label="Best")
        plt.plot(x_test, res_best_test2, "--", color="C0", label="Best2")
        plt.plot(x_test, res_true_test, color="C1", label="Truth")
        plt.plot(x_test, res_true_test2, "--", color="C1", label="Truth2")
        plt.legend()
        plt.show()

    # Check for a match over a test set (much harder to pass than accidentally
    # getting a close fitness on the train set)
    if all(
        [
            math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
            for a, b in zip(res_true_test, res_best_test)
        ]
    ):
        return 0
    # Otherwise, who was better?
    elif rms_best > rms_true:
        return 1
    elif rms_best < rms_true:
        return 2
    return -1


def ground_truth_fitness(experiments, ground_truth_model) -> float:
    fitnesses = []
    for experiment in experiments:
        # if experiment.is_random(): experiment.sample_state()
        residual_model = wrap(ground_truth_model(experiment))
        residual = residual_model(experiment.data.values)
        fitnesses.append(rms(residual))

    mean_fitness = np.mean(fitnesses)
    fitness_threshold = mean_fitness  # min(fitnesses)
    print("  Mean ground truth fitness = %e" % mean_fitness)
    print("  Fitness threshold         = %e" % fitness_threshold)
    return fitness_threshold
