# File: e_nonlinear_adaptive.py
# File Created: Saturday, 20th July 2019 7:22:37 am
# Author: Steven Atkinson (212726320@ge.com)

"""
1D elliptic problem with nonlinear conductivity.
Adaptive acquisition of data.
"""

import random
import sys
import os
import operator
from time import time
import argparse
import pdb
from typing import Callable

import deap.gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pirate.data.experiment import Experiment

from pirate.experiment_util import (
    parse_args_adaptive,
    check_answer,
    ground_truth_fitness,
)
from pirate.function import Function

from pirate.models.gaussian_process import GP

import pirate.operator as pirate_operator

from pirate.symbolic_regression.base import get_primitive_set
from pirate.symbolic_regression.base import SymbolicRegression
from pirate.symbolic_regression.fitness import DifferentialResidual
from pirate.symbolic_regression.fitness import add_memoization

from pirate.systems.elliptic import Elliptic1DNonlinear as Elliptic

from pirate.util import rms

torch.set_default_dtype(torch.double)

FitnessFunction = add_memoization(DifferentialResidual)


def make_experiments(system, dataset):
    x = dataset["x"].values
    funcs = {}
    for col, depvar in zip(("solution", "force"), ("u", "f")):
        y = dataset[col].values

        gp = GP(x[:, np.newaxis], y[:, np.newaxis])
        gp.train(num_epochs=1000, learning_rate=0.1)  # , show_loss=True)
        gp.predict_mode()

        funcs[depvar] = Function(gp.model.get_mean_function())

    experiments = [
        Experiment({"u": funcs["u"]}, pd.DataFrame({"x": x}), inhomogeneous=funcs["f"])
    ]
    fitness_threshold = ground_truth_fitness(experiments, ground_truth_model)
    return experiments, fitness_threshold


def ground_truth_model(experiment) -> Function:
    u = experiment.left_hand_side["u"]
    f = experiment.inhomogeneous
    return (-u.exp() * u.gradient(0)).gradient(0) - f


def get_best_individual_residual(symbolic_regression) -> Callable:
    """
    Take the SR object and build a model (function) out of its fittest 
    individual.
    """
    f_best = deap.gp.compile(
        expr=symbolic_regression.hall_of_fame[0], pset=symbolic_regression.primitive_set
    )

    def func(experiment, x):
        # Takes care of the inhomogeneous term automatically!
        residual = symbolic_regression.fitness_function.residual(
            f_best, experiment, inputs=x
        )
        assert residual.shape[1] == 1  # Need 1D output!

        return residual.flatten()

    return func


def find_next_x(symbolic_regression):
    x_test = np.random.rand(1000)  # Where to try
    residuals = []  # Acquisition function samples
    residual_function = get_best_individual_residual(symbolic_regression)
    for experiment in symbolic_regression.fitness_function.experiments:
        residuals.append(residual_function(experiment, x_test[:, np.newaxis]))
    abs_residual = np.abs(np.array(residuals).mean(axis=0))

    i_new = np.argmax(abs_residual)
    max_residual = abs_residual[i_new]
    x_new = x_test[i_new]

    return x_new, max_residual


def iteration(system, dataset, testing):
    experiments, fitness_threshold = make_experiments(system, dataset)

    # Do symbolic regression
    t_start = time()
    operators = (
        operator.add,
        operator.neg,
        operator.mul,
        pirate_operator.ScalarGradient(0),
        pirate_operator.exp,
    )
    primitive_set = get_primitive_set(operators, variable_names=("u",))
    fitness_function = FitnessFunction(
        experiments, primitive_set, differential_operators=["ddx0"]
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

    return symbolic_regression


def append_datum(system, dataset, x):
    u, f = system.solution(x), system.force(x)
    dataset_new = dataset.append(
        pd.DataFrame({"x": [x], "solution": [u], "force": [f]})
    ).reset_index(drop=True)

    print("New dataset:\n", dataset_new)

    return dataset_new


def main(testing=False):
    args = parse_args_adaptive(testing)
    if not args.no_write:
        result_file = os.path.join(
            os.path.dirname(__file__),
            "..",
            "output",
            "e_nonlinear_adaptive",
            "results",
            "s_%i.txt" % args.seed,
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
    system = Elliptic(nelem=100)
    dataset = system.sample(2)  # initial design

    n, residuals = [], []
    for step in range(2 if testing else 256):
        symbolic_regression = iteration(system, dataset, testing)

        x_new, residual = find_next_x(symbolic_regression)
        n.append(dataset.shape[0])
        residuals.append(residual)

        print("This iteration:\n", n[-1], residuals[-1])

        if residual < args.threshold:
            break
        else:
            dataset = append_datum(system, dataset, x_new)

    # Check if we got it right
    x_test = np.linspace(0.0, 1.0, 1000)[:, np.newaxis]
    res = check_answer(x_test, symbolic_regression, ground_truth_model)
    print("Result = %i" % res)
    if not args.no_write:
        np.savetxt(result_file, [res], fmt="%d")

    print("N:\n", n)
    print("Residuals:\n", residuals)


if __name__ == "__main__":
    main()
