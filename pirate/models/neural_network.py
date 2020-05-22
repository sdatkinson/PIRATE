# File: neural_network.py
# File Created: Wednesday, 13th March 2019 3:01:11 pm
# Author: Steven Atkinson (212726320@ge.com)

"""
Neural networks!
"""

import torch

from .base import Model


class Sine(torch.nn.Module):
    """
    Sine nonlinearity module
    """
    def forward(self, x):
        return torch.sin(x)


class FCNN(Model):
    def __init__(self, x, y, model=None, **kwargs):
        super().__init__(x, y, **kwargs)
        self._model = model if model is not None else \
            FCNN.default_architecture(self.input_dimension, 
            self.output_dimension)
        if self.cuda:
            self._model.cuda()

    def loss(self, x, y):
        return torch.nn.MSELoss()(self._model(x), y)

    def predict_mode(self):
        super().predict_mode()
        self._model.eval()  # Turn off BatchNorm, dropout, etc...
    
    @staticmethod
    def default_architecture(d_in, d_out, layers=2, units=256, 
            Nonlinearity=Sine):
        
        def _block():
            """
            Linear-batchnorm-nonlinearity block of layers
            """
            return torch.nn.BatchNorm1d(units), torch.nn.Linear(units, units), \
                Nonlinearity()

        return torch.nn.Sequential(
            torch.nn.Linear(d_in, units),
            Nonlinearity(),
            *(_block() * layers), 
            torch.nn.BatchNorm1d(units),
            torch.nn.Linear(units, d_out)
        )
        