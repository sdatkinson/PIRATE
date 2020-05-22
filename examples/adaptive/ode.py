# File: ode2_adaptive.py
# File Created: Thursday, 5th September 2019 3:37:03 pm
# Author: Steven Atkinson (212726320@ge.com)

"""
2nd order ODE
Adaptive acquisition of data.
"""

import operator
import os
import random
import sys
from time import time
from typing import Callable

import deap.gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pirate.function import Function
import pirate.operator as pirate_operator
from pirate.data.experiment import Experiment
from pirate.experiment_util import (
    parse_args_adaptive,
    check_answer,
    ground_truth_fitness,
)

from pirate.models.gaussian_process import GP

from pirate.symbolic_regression.base import get_primitive_set
from pirate.symbolic_regression.base import SymbolicRegression
from pirate.symbolic_regression.fitness import DifferentialResidual
from pirate.symbolic_regression.fitness import add_memoization

from pirate.systems.ode import SecondOrder

from pirate.util import rms

torch.set_default_dtype(torch.double)

FitnessFunction = add_memoization(DifferentialResidual)


def make_experiments(system, dataset):
    t = dataset["t"].values
    lhs = {}
    x = dataset["x"].values

    gp = GP(t[:, np.newaxis], x[:, np.newaxis])
    gp.train(num_epochs=1000, learning_rate=0.1)  # , show_loss=True)
    gp.predict_mode()

    # Extract function for modeled dataset
    fx = Function(gp.model.get_mean_function())
    experiments = [Experiment({"x": fx}, pd.DataFrame({"t": t}))]

    # Extras
    fitness_threshold = ground_truth_fitness(experiments, ground_truth_model)

    return experiments, fitness_threshold


def ground_truth_model(experiment) -> Function:
    x = experiment.left_hand_side["x"]
    return x.gradient(0).gradient(0) + x.gradient(0) + x


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


def find_next_x(symbolic_regression, t_test):
    residuals = []  # Acquisition function samples
    residual_function = get_best_individual_residual(symbolic_regression)
    for experiment in symbolic_regression.fitness_function.experiments:
        residuals.append(residual_function(experiment, t_test[:, np.newaxis]))
    abs_residual = np.abs(np.array(residuals).mean(axis=0))

    # Find the next point that isn't already being used:
    i_list = np.argsort(abs_residual)[::-1]
    for i in i_list:
        t_new = t_test[i]
        if (
            t_new
            in symbolic_regression.fitness_function.experiments[0].data["t"].values
        ):
            print("Already added %f, use the next" % t_new)
            continue

        max_residual = abs_residual[i]
        return t_new, max_residual

    raise RuntimeError("Ran out of data!")


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
    primitive_set = get_primitive_set(operators, variable_names=("x",))
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
        iterations=2 if testing else 200,
        verbose=True,
        fitness_threshold=(0.99 * fitness_threshold,),
    )
    t_elapsed = time() - t_start
    print("Elapsed time = {}".format(t_elapsed))

    return symbolic_regression


def append_datum(system, dataset, t):
    i = np.where(system._t == t)[0][0]
    x = system._x[i]
    dataset_new = dataset.append(pd.DataFrame({"t": [t], "x": [x]})).reset_index(
        drop=True
    )

    print("New dataset:\n", dataset_new)

    return dataset_new


def main(testing=False):
    args = parse_args_adaptive(testing)
    if not args.no_write:
        result_file = os.path.join(
            os.path.dirname(__file__),
            "..",
            "output",
            "ode_adaptive",
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
    system = SecondOrder()
    system.system_parameters = np.array([10.0, 5.0])
    dataset = system.sample(4)  # initial design

    n, residuals = [], []
    for step in range(2 if testing else 256):
        symbolic_regression = iteration(system, dataset, testing)

        t_new, residual = find_next_x(symbolic_regression, system._t)
        n.append(dataset.shape[0])
        residuals.append(residual)

        print("This iteration:\n", n[-1], residuals[-1])

        if residual < args.threshold:
            break
        else:
            dataset = append_datum(system, dataset, t_new)

    # Check if we got it right
    t_test = system._t.copy()
    res = check_answer(t_test, symbolic_regression, ground_truth_model)
    print("Result = %i" % res)
    if not args.no_write:
        np.savetxt(result_file, [res], fmt="%d")

    print("N:\n", n)
    print("Residuals:\n", residuals)


if __name__ == "__main__":
    main()
