# File: gaussian_process.py
# File Created: Monday, 18th March 2019 4:42:27 pm
# Author: Steven Atkinson (212726320@ge.com)

"""
Gaussian process models
"""

import torch
from gptorch.util import TensorType
import numpy as np
import gptorch
from warnings import warn

from .base import Model


class _GPR(gptorch.models.gpr.GPR):
    """
    Gaussian process regression model.
    
    The default implementation of .forward() is the posterior mean, but you can 
    use .get_draw_function() to extract a RandomFunction to draw samples from 
    the GP posterior over functions.
    """

    def __init__(self, *args, **kwargs):
        # The gptorch init:
        super().__init__(*args, **kwargs)
        self.dtype = gptorch.util.torch_dtype

    def forward(self, x):
        return self.get_mean_function()(x)

    def get_mean_function(self):
        return lambda x, **kwargs: self.predict_f(x)[0]


class _SVGP(gptorch.models.SVGP):
    pass  # TODO


class _GPTorchModels(Model):
    """
    Basic functionality common to wrapping any gptorch model.
    """

    def __init__(self, x, y, model_type, **kwargs):
        super().__init__(x, y, **kwargs)
        self._model = model_type(y, x, gptorch.kernels.Rbf(x.shape[1], ARD=True))
        # Initial guess: likelihood variance is 1% of the data variance
        self._model.likelihood.variance.data = TensorType([np.log(np.var(y) * 1.0e-2)])

    def loss(self, x, y):
        warn("gptorch models ignore args to .loss()")
        return self.model.compute_loss()

    def train_gptorch(self, num_epochs, optimizer="L-BFGS-B", show_loss=False):
        # Need different output!
        opt_out = self._model.optimize(method=optimizer, max_iter=num_epochs)

        if show_loss:
            print("Can't show loss for GPs; skip")
            # self._show_loss(losses)


class GP(_GPTorchModels):
    """
    Gaussian process model.
    
    This model can use either the posterior mean or random samples from the 
    posterior--see _GPR.
    """

    def __init__(self, x, y, **kwargs):
        super().__init__(x, y, _GPR, **kwargs)


class SVGP(_GPTorchModels):
    def __init__(self, x, y, **kwargs):
        super().__init__(x, y, _SVGP, **kwargs)
