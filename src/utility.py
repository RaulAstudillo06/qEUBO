from __future__ import annotations

from typing import Union

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

    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        noise_level: Union[float, Tensor],
    ) -> None:
        r"""Constructor for the Probit utility function class.
        Args:
            num_samples: .
        """
        super().__init__()
        self.num_samples = num_samples
        self.num_samples = batch_size
        noise_samples = draw_sobol_normal_samples(
            d=batch_size,
            n=num_samples,
        )
        noise_samples = torch.as_tensor(noise_level) * noise_samples
        noise_samples = noise_samples.unsqueeze(1)
        noise_samples = noise_samples.unsqueeze(1)
        self.register_buffer("noise_samples", noise_samples)

    def forward(self, Y: Tensor) -> Tensor:
        r"""Evaluate the utility function on the candidate set Y.
        Args:
            Y: A `(b) x q`-dim Tensor of `(b)` t-batches with `q` objective values each.
        Returns:
            A `(b)`-dim Tensor of utility function values at the given outcome values `Y`.
        """
        corrupted_Y_samples = Y.unsqueeze(0) + self.noise_samples
        corrupted_max_Y_samples = corrupted_Y_samples.max(dim=-1).values
        return corrupted_max_Y_samples.mean(0)
