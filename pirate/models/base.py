# File: base.py
# File Created: Thursday, 14th March 2019 4:08:16 pm
# Author: Steven Atkinson (212726320@ge.com)

import abc
from time import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader


def _optimizer_requires_closure(opt):
    return opt in {torch.optim.LBFGS}


class Model(object):
    """
    This is the basic template for a (PyTorch-based) model.
    """

    def __init__(self, x, y, **kwargs):
        self.x, self.y = x, y
        self._model = None
        if kwargs:
            print("pirate.models.base.Model: dropping provided kwargs:")
            for key, val in kwargs.items():
                print(" {key}: {val}".format(key=key, val=val))

    @property
    def input_dimension(self):
        return self.x.shape[1]

    @property
    def model(self) -> torch.nn.Module:
        """
        Access the inner PyTorch module object
        """
        return self._model

    @property
    def output_dimension(self):
        return self.y.shape[1]

    def __call__(self, x):
        # TODO decorators to do the PyTorch wrapping
        with torch.no_grad():
            x = torch.Tensor(x)
            y = self._model(x)
            y = y.numpy()
        return y

    @abc.abstractmethod
    def loss(self, x, y):
        pass

    def predict_mode(self):
        """
        Turn off gradient following on the model.
        Should speed up predictions somewhat...
        """
        for param in self._model.parameters():
            param.requires_grad_(False)
        print("Done")

    def train(
        self,
        num_epochs=100,
        show_loss=False,
        optimizer=torch.optim.Adam,
        learning_rate=0.01,
        validation_size=None,
        show_loss_log_y=False,
    ):

        # Prepare the data
        x_all, y_all = self.x, self.y
        if validation_size is None:
            x_train, x_valid, y_train, y_valid = x_all, None, y_all, None
        else:
            x_train, x_valid, y_train, y_valid = train_test_split(
                x_all, y_all, test_size=validation_size, random_state=42
            )
            x_valid, y_valid = torch.Tensor(x_valid), torch.Tensor(y_valid)
            if self.cuda:
                x_valid, y_valid = x_valid.cuda(), y_valid.cuda()
        data_loader = self._get_data_loader(x_train, y_train)

        # Prepare the optimizer
        opt = optimizer(self._model.parameters(), lr=learning_rate)

        # Optimization loop
        t_start = time()
        epoch_losses = []
        validation_losses = []
        for epoch in tqdm(range(num_epochs)):
            intra_epoch_losses = []
            for x, y in data_loader:
                loss = self.loss(x, y)
                intra_epoch_losses.append(loss.item())
                opt.zero_grad()
                loss.backward()
                opt_kwargs = (
                    {"closure": lambda: self._loss_no_grad(x, y)}
                    if _optimizer_requires_closure(optimizer)
                    else {}
                )
                opt.step(**opt_kwargs)
            if validation_size is not None:
                validation_losses.append(self.loss(x_valid, y_valid).item())
            t_elapsed = time() - t_start
            epoch_losses.append(np.array(intra_epoch_losses).mean())

        if show_loss:
            self._show_loss(
                epoch_losses, validation_losses=validation_losses, log_y=show_loss_log_y
            )

    def _get_data_loader(self, x, y):
        """
        Prepare a torch DataLoader instance based on the current data.
        """

        kwargs = {"batch_size": min(x.shape[0], 2 ** 14), "shuffle": True}
        dataset = TensorDataset(torch.Tensor(x), torch.Tensor(y))
        data_loader = DataLoader(dataset, **kwargs)

        return data_loader

    def _loss_no_grad(self, x, y):
        """
        Non-grad enabled loss for closure during optimization
        """

        with torch.no_grad():
            return self.loss(x, y)

    def _show_loss(self, losses, validation_losses=None, log_y=False):
        plt.plot(losses)
        if validation_losses:
            plt.plot(validation_losses, label="Validation")
            plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        if log_y:
            plt.yscale("log")
        plt.show()
