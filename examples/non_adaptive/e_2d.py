# File: e_het.py
# File Created: Saturday, 20th July 2019 7:22:37 am
# Author: Steven Atkinson (212726320@ge.com)

"""
2D elliptic problem with heterogeneous linear conductivity
"""

import os
import operator
import random
import sys
from time import time

import deap.gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pirate import operator as pirate_operator
from pirate.data.experiment import Experiment
from pirate.function import Function, GradientError, DivergenceError
from pirate.models.gaussian_process import GP
from pirate.symbolic_regression.base import get_primitive_set
from pirate.symbolic_regression.base import SymbolicRegression
from pirate.symbolic_regression.fitness import DifferentialResidual
from pirate.symbolic_regression.fitness import add_memoization

from pirate.systems.elliptic import Elliptic2DAtkinsonZabaras as Elliptic

from pirate.util import rms
from pirate.experiment_util import (
    check_answer,
    ground_truth_fitness,
    parse_args_non_adaptive,
)

torch.set_default_dtype(torch.double)


def grid_2d(n1, n2, epsilon=0.0) -> np.ndarray:
    """
    Make a 2D grid of points
    :param epsilon: scale of randn to perturb the points
    """
    xlin1 = np.linspace(0.0, 1.0, n1)[np.newaxis, :]
    x1 = np.tile(xlin1, (n2, 1)).flatten()
    xlin2 = np.linspace(0.0, 1.0, n2)[:, np.newaxis]
    x2 = np.tile(xlin2, (1, n1)).flatten()

    x = np.stack((x1, x2)).T
    return x + epsilon * np.random.randn(*x.shape)


def make_experiments(system, dataset):
    x = dataset[["x1", "x2"]].values
    funcs = {}
    for col, depvar in zip(("log_conductivity", "solution"), ("log_a", "u")):
        y = dataset[col].values

        gp = GP(x, y[:, np.newaxis])
        gp.train(num_epochs=1000, learning_rate=0.1)
        gp.predict_mode()

        # Extract function
        # Sorry for the a-kludge :(
        f = Function(gp.model.get_mean_function())
        if depvar == "log_a":
            # Convert to normal conductivity
            # TODO make this remain a random function
            f = f.exp()
            depvar = "a"

        funcs[depvar] = f

    experiments = [
        Experiment({"u": funcs["u"], "a": funcs["a"]}, dataset[["x1", "x2"]])
    ]
    fitness_threshold = ground_truth_fitness(experiments, ground_truth_model)

    return experiments, fitness_threshold


def ground_truth_model(experiment) -> Function:
    a = experiment.left_hand_side["a"]
    u = experiment.left_hand_side["u"]
    return (-a * u.gradient([0, 1])).divergence([0, 1])


def main(testing=False):
    args = parse_args_non_adaptive(testing)

    if not args.no_write:
        result_file = os.path.join(
            os.path.dirname(__file__),
            "..",
            "output",
            "e_2d",
            "results",
            "n_%i_s_%i.txt" % (args.n_train, args.seed),
        )
        result_dir = os.path.dirname(result_file)
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        if os.path.isfile(result_file) and not args.overwrite:
            print("Already found %s; exit." % result_file)
            exit()

    # RNG seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Dataset & experiment
    system = Elliptic(x1_limit=[0.1, 0.5], x2_limit=[0.25, 0.75])
    dataset = system.sample(args.n_train ** 2)
    dataset["solution"] += 1.0
    experiments, fitness_threshold = make_experiments(system, dataset)

    # Do symbolic regression
    t_start = time()
    operators = (
        operator.add,
        operator.neg,
        operator.mul,
        pirate_operator.Gradient([0, 1]),
        pirate_operator.Divergence([0, 1]),
    )
    primitive_set = get_primitive_set(operators, variable_names=("a", "u"))
    expected_exceptions = (GradientError, DivergenceError)
    fitness_function = add_memoization(DifferentialResidual)(
        experiments,
        primitive_set,
        expected_exceptions=expected_exceptions,
        differential_operators=["grad"],
    )
    symbolic_regression = SymbolicRegression(
        primitive_set,
        fitness_function,
        population=512,
        mating_probability=0.2,
        mutation_probability=0.8,
    )
    symbolic_regression.run(
        iterations=2 if testing else 50,
        verbose=True,
        fitness_threshold=(0.99 * fitness_threshold,),
    )
    t_elapsed = time() - t_start
    print("Elapsed time = {}".format(t_elapsed))

    # Check if we got it right
    x_test = grid_2d(4, 4, epsilon=1.0e-3)
    res = check_answer(x_test, symbolic_regression, ground_truth_model)
    print("Result = %i" % res)
    if not args.no_write:
        np.savetxt(result_file, [res], fmt="%d")


if __name__ == "__main__":
    main()
