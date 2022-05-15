from __future__ import annotations

from typing import Optional, Union

import torch
from abc import ABC, abstractmethod
from botorch.utils.sampling import draw_sobol_normal_samples
from torch import Tensor
from torch.nn import Module


class Utility(Module, ABC):
    r"""Abstract base class for utility functions."""

    @abstractmethod
    def forward(self, Y: Tensor) -> Tensor:
        r"""Evaluate the utility function on the candidate set Y.
        Args:
            Y: A `(b) x q`-dim Tensor of `(b)` t-batches with `q` objective values each.
        Returns:
            A `(b)`-dim Tensor of utility function values at the given outcome values `Y`.
        """
        pass


class MaxObjectiveValue(Utility):
    r"""."""

    def __init__(self) -> None:
        r"""Constructor for the MaxObjectiveValue utility function class."""
        super().__init__()

    def forward(self, Y: Tensor) -> Tensor:
        r"""Evaluate the utility function on the candidate set Y.
        Args:
            Y: A `(b) x q`-dim Tensor of `(b)` t-batches with `q` objective values each.
        Returns:
            A `(b)`-dim Tensor of utility function values at the given outcome values `Y`.
        """
        return Y.max(dim=-1).values


class Probit(Utility):
    r"""."""

    def __init__(self, noise_std: Optional[Union[float, Tensor]] = 1.0) -> None:
        r"""Constructor for the Probit utility function class.
        Args:
            num_samples: .
        """
        super().__init__()
        self.noise_std = torch.tensor(noise_std)
        self.std_norm = torch.distributions.normal.Normal(torch.zeros(1), torch.ones(1))
        self.register_buffer("sqrt2", torch.sqrt(torch.tensor(2.0)) * self.noise_std)
        print("TEST BEGINS")
        print(torch.sqrt(torch.tensor(2.0)))
        print(self.sqrt2)
        print("TEST ENDS")

    def forward(self, Y: Tensor) -> Tensor:
        r"""Evaluate the utility function on the candidate set Y.
        Args:
            Y: A `(b) x 2`-dim Tensor of `(b)` t-batches with `2` objective values each.
        Returns:
            A `(b)`-dim Tensor of utility function values at the given outcome values `Y`.
        """
        prob0 = self.std_norm.cdf((Y[..., 0] - Y[..., 1]) / self.sqrt2)
        prob1 = 1.0 - prob0
        utility = (prob0 * Y[..., 0]) + (prob1 * Y[..., 1])
        return utility


class Logit(Utility):
    r"""."""

    def __init__(self) -> None:
        r"""Constructor for the Probit utility function class.
        Args:
            num_samples: .
        """
        super().__init__()
        self.soft_max = torch.nn.Softmax(dim=-1)

    def forward(self, Y: Tensor) -> Tensor:
        r"""Evaluate the utility function on the candidate set Y.
        Args:
            Y: A `(b) x 2`-dim Tensor of `(b)` t-batches with `2` objective values each.
        Returns:
            A `(b)`-dim Tensor of utility function values at the given outcome values `Y`.
        """
        probs = self.soft_max(Y)
        utility = (probs * Y).sum(dim=-1)
        return utility
