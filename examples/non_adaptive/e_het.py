# File: e_het.py
# File Created: Saturday, 20th July 2019 7:22:37 am
# Author: Steven Atkinson (212726320@ge.com)

"""
1D elliptic problem with heterogeneous linear conductivity
"""

import argparse
import operator
import os
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
from pirate.function import Function
from pirate.models.gaussian_process import GP
from pirate.symbolic_regression.base import get_primitive_set
from pirate.symbolic_regression.base import SymbolicRegression
from pirate.symbolic_regression.fitness import DifferentialResidual
from pirate.symbolic_regression.fitness import add_memoization
from pirate.systems.elliptic import Elliptic1DLinearHeterogeneous as Elliptic

from pirate.util import rms
from pirate.experiment_util import (
    check_answer,
    ground_truth_fitness,
    parse_args_non_adaptive,
)

torch.set_default_dtype(torch.double)


def make_experiments(system, dataset: pd.DataFrame):
    x = dataset["x"].values
    funcs = {}
    for col, depvar in zip(("conductivity", "solution", "force"), ("a", "u", "f")):
        y = dataset[col].values

        gp = GP(x[:, np.newaxis], y[:, np.newaxis])
        gp.train(num_epochs=1000, learning_rate=0.1)
        gp.predict_mode()
        funcs[depvar] = Function(gp.model.get_mean_function())

    experiments = [
        Experiment(
            {"u": funcs["u"], "a": funcs["a"]}, dataset[["x"]], inhomogeneous=funcs["f"]
        )
    ]

    fitness_threshold = ground_truth_fitness(experiments, ground_truth_model)

    return experiments, fitness_threshold


def ground_truth_model(experiment) -> Function:
    a = experiment.left_hand_side["a"]
    u = experiment.left_hand_side["u"]
    f = experiment.inhomogeneous
    return (-a * u.gradient(0)).gradient(0) - f


def main(testing=False):
    args = parse_args_non_adaptive(testing)
    if not args.no_write:
        result_file = os.path.join(
            os.path.dirname(__file__),
            "..",
            "output",
            "e_het",
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
    system = Elliptic(nelem=100)
    dataset = system.sample(args.n_train)
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
    fitness_function = add_memoization(DifferentialResidual)(
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

    # Check if we got it right
    x_test = np.linspace(0.0, 1.0, 1000)[:, np.newaxis]
    res = check_answer(x_test, symbolic_regression, ground_truth_model)
    print("Result = %i" % res)
    if not args.no_write:
        np.savetxt(result_file, [res], fmt="%d")


if __name__ == "__main__":
    main()
