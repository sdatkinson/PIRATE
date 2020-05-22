# File: e_het.py
# File Created: Saturday, 20th July 2019 7:22:37 am
# Author: Steven Atkinson (212726320@ge.com)

"""
1D elliptic problem with heterogeneous linear conductivity.
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

from pirate import operator as pirate_operator
from pirate.data.experiment import Experiment

from pirate.experiment_util import (
    parse_args_adaptive,
    check_answer,
    ground_truth_fitness,
)

from pirate.function import Function

from pirate.models.gaussian_process import GP

from pirate.symbolic_regression.base import get_primitive_set
from pirate.symbolic_regression.base import SymbolicRegression
from pirate.symbolic_regression.fitness import DifferentialResidual
from pirate.symbolic_regression.fitness import add_memoization

from pirate.systems.elliptic import Elliptic1DLinearHeterogeneous as Elliptic

from pirate.util import rms

torch.set_default_dtype(torch.double)
torch.set_num_threads(1)

FitnessFunction = add_memoization(DifferentialResidual)


def _show_samples(system, dataset, experiments):
    x_train = dataset["x"].values
    x_test = np.linspace(0.0, 1.0, 200)[:, None]
    alpha = 0.5 if len(experiments) > 1 else 1.0

    # Left hand side variables
    for i, col_depvar in enumerate(
        [("conductivity", "a"), ("solution", "u"), ("force", "f")]
    ):
        col, depvar = col_depvar
        y_train = dataset[col].values

        plt.figure()
        for j, experiment in enumerate(experiments):
            if j == 5:
                break
            md = (
                experiment.left_hand_side[depvar]
                if not depvar == "f"
                else experiment.inhomogeneous
            )
            y_test = md.eval(x_test)
            plt.plot(x_test, y_test, color="C%i" % i, alpha=alpha)
            if depvar == "f":
                u = experiment.left_hand_side["u"]
                a = experiment.left_hand_side["a"]
                rhs = (a * u.gradient()).gradient()  # Negative RHS, actually...
                plt.plot(x_test, rhs.eval(x_test), color="C3", alpha=alpha)
        plt.scatter(x_train, y_train, color=[0] * 3)
        ylim = plt.ylim()
        plt.plot(x_test, getattr(system, col)(x_test), color="k")
        plt.ylim(ylim)
        plt.xlabel("x", fontsize=18)
        plt.ylabel("%s(x)" % depvar, fontsize=18)
        plt.show()


def make_experiments(system, dataset):
    x = dataset["x"].values
    funcs = {}
    for col, depvar in zip(("conductivity", "solution", "force"), ("a", "u", "f")):
        y = dataset[col].values

        gp = GP(x[:, np.newaxis], y[:, np.newaxis])
        gp.train(num_epochs=1000, learning_rate=0.1)  # , show_loss=True)
        gp.predict_mode()

        funcs[depvar] = Function(gp.model.get_mean_function())

    experiments = [
        Experiment(
            {"u": funcs["u"], "a": funcs["a"]},
            pd.DataFrame({"x": x}),
            inhomogeneous=funcs["f"],
        )
    ]
    fitness_threshold = ground_truth_fitness(experiments, ground_truth_model)
    return experiments, fitness_threshold


def ground_truth_model(experiment) -> Function:
    a = experiment.left_hand_side["a"]
    u = experiment.left_hand_side["u"]
    f = experiment.inhomogeneous
    return (-a * u.gradient(0)).gradient(0) - f


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
    )
    primitive_set = get_primitive_set(operators, variable_names=("a", "u"))
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
    a, u, f = system.conductivity(x), system.solution(x), system.force(x)
    dataset_new = dataset.append(
        pd.DataFrame({"x": [x], "conductivity": [a], "solution": [u], "force": [f]})
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
            "e_het_adaptive",
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
