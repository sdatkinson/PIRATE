# File: calibration.py
# File Created: Friday, 15th November 2019 3:47:07 pm
# Author: Steven Atkinson (212726320@ge.com)

"""
Tools for parameter calibration.
"""

import deap.gp
import matplotlib.pyplot as plt
import pyro
from pyro.distributions import Distribution, MultivariateNormal, Normal
from pyro.distributions.transforms import LowerCholeskyTransform
import torch
from torch.distributions import Distribution, Normal
from tqdm import tqdm

from ...data.experiment import Experiment
from ...util import rms

from ..deap_extensions import Parameter, PrimitiveTree, compile
from ..util import get_residual_function, tensor_to_parameter_dict

from .results import CalibrationResults


class Flat(Distribution):
    """
    (Improper) flat distribution
    """

    def __init__(self, shape):
        super().__init__()
        self._shape = shape

    def sample(self, sample_shape=()):
        return torch.zeros(*sample_shape, *self._shape)

    def log_prob(self, x):
        return 0.0 * x.sum(dim=[-(i + 1) for i in range(len(self._shape))])


class Calibrator(torch.nn.Module):
    """
    Calibrates parameters in a provided individual:
    * Ephemeral constants (if any)
    * Observation likelihood (i.e. variance of a Gaussian likelihood)

    All updating is done in-place.
    """

    def __init__(
        self,
        individual: PrimitiveTree,
        pset: deap.gp.PrimitiveSet,
        experiment: Experiment,
    ):
        super().__init__()

        pyro.clear_param_store()

        self.individual = individual
        self.pset = pset
        self.experiment = experiment

        self.f_residual = get_residual_function(
            compile(individual, pset), experiment, pset
        )

        # Variational posterior variational parameters (multivariate Gaussian)
        # self.q_mu = torch.zeros(self.num_parameters)
        # self.raw_q_sqrt = torch.zeros(self.num_parameters, self.num_parameters)

        # Input points at which we evaluate the residual function
        self._inputs = self._inputs = torch.Tensor(self.experiment.data.values)
        self._inputs.requires_grad_(True)

        self._scheduler = None
        self._svi = None

    @property
    def num_parameters(self):
        return self.individual.num_parameters

    @property
    def q_mu(self) -> torch.Tensor:
        """
        Variational posterior mean
        """

        return pyro.param("q_mu")

    @property
    def q_sqrt(self) -> torch.Tensor:
        """
        Parameters' posterior covariance
        """

        return pyro.param("q_sqrt")

    def calibrate(self, iters=200, verbose=True, show=False):
        """
        Perform calibration
        """

        lr_initial = 0.1
        lr_final = 0.001
        final_factor = lr_final / lr_initial
        self._scheduler = pyro.optim.ExponentialLR(
            {
                "optimizer": torch.optim.Adam,  # NOT pyro!
                "optim_args": {"lr": lr_initial},
                "gamma": final_factor ** (1 / iters),
            }
        )
        self._svi = pyro.infer.SVI(
            self._model, self._guide, self._scheduler, loss=pyro.infer.Trace_ELBO()
        )

        losses = []
        iterator = tqdm(range(iters)) if verbose else range(iters)
        for i in iterator:
            losses.append(self._svi.step())
            self._scheduler.step()

        if verbose:
            print("Calibrated:")
            print(" expr:  %s" % str(self.individual))
            print(" q_mu:  %s" % str(self.q_mu))
            print(" q_var: %s" % str((self.q_sqrt @ self.q_sqrt.t()).diag().sqrt()))
        if show:
            plt.figure()
            plt.plot(losses)
            plt.xlabel("SVI iteration")
            plt.ylabel("Loss")
            plt.title("expr = %s" % str(self.individual))
            plt.show()

    def get_loss(self, num_particles=128, verbose=True):
        """
        Get an estimate of the negative evidence lower bound based on the 
        current variational posterior.

        :param num_particles: Number of particles used to estimate the ELBO
        """

        svi = pyro.infer.SVI(
            self._model,
            self._guide,
            self._scheduler,
            loss=pyro.infer.Trace_ELBO(num_particles=num_particles),
        )

        # .evaluate_loss() doesn't seem to work if there are grads inside the
        # model?
        # NB: loss = -ELBO
        loss = svi.evaluate_loss()
        if verbose:
            print("Loss = %f" % loss)
        return loss

    def get_results(self, detach_posterior=True) -> CalibrationResults:
        """
        Distill the results of the calibration

        :param detach: If true, make sure that the posterior's parameters are
        detached (can cause issues if you try deepcopying)
        """

        loc, scale_tril = self.q_mu, self.q_sqrt
        if detach_posterior:
            loc, scale_tril = loc.detach(), scale_tril.detach()

        return CalibrationResults(
            self.experiment,
            self.get_loss(),
            MultivariateNormal(loc, scale_tril=scale_tril),
        )

    def _model(self):
        """
        Prior + posterior for the calibration model
        """

        with torch.enable_grad():  # Because pyro tries to disable it...
            theta = pyro.sample("theta", Flat((self.num_parameters,)))

            parameters = tensor_to_parameter_dict(theta)
            # Could also calibrate random functions...

            r = self.f_residual(self._inputs, **parameters)
            sigma = torch.sqrt(torch.mean(r ** 2))
            with pyro.plate("targets"):
                pyro.sample("residual", Normal(torch.zeros(1), sigma), obs=r)

    def _guide(self):
        """
        Pyro variational posterior ("guide")
        """

        q_mu = pyro.param("q_mu", torch.zeros(self.num_parameters))
        q_sqrt = pyro.param(
            "q_sqrt",
            torch.eye(self.num_parameters),
            constraint=torch.distributions.constraints.lower_cholesky,
        )

        pyro.sample("theta", MultivariateNormal(q_mu, scale_tril=q_sqrt))


def calibrate(
    individual: PrimitiveTree,
    pset: deap.gp.PrimitiveSet,
    experiment: Experiment,
    verbose=True,
    show=False,
) -> None:
    """
    Calibrate the parameters of the provided individual.
    
    :return: (CalibrationResults) Information for calculating the fitness.
    """

    c = Calibrator(individual, pset, experiment)
    c.calibrate(verbose=verbose, show=show)

    return c.get_results()
