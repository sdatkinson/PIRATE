# File: results
# File Created: Tuesday, 31st March 2020 1:37:03 pm
# Author: Steven Atkinson (212726320@ge.com)

from torch.distributions import Distribution

from ...data.experiment import Experiment


class CalibrationResults(object):
    """
    Declares the structure of the calibration results
    """

    def __init__(self, experiment: Experiment, loss: float, posterior: Distribution):
        """
        :param experiment: The experimental data that we performed this calibration on.
        :param loss: The loss (i.e. negative ELBO) associated with the calibration.
        :param posterior: Sample the posterior from this.
        """

        self.experiment = experiment
        self.loss = loss
        self.posterior = posterior
