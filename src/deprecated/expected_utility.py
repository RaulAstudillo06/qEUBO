from __future__ import annotations

from typing import Optional

from botorch.acquisition import MCAcquisitionFunction
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.models.model import Model
from botorch.sampling.samplers import MCSampler
from botorch.utils.transforms import (
    concatenate_pending_points,
    t_batch_mode_transform,
)
from torch import Tensor

from src.utility import Utility


class qExpectedUtility(MCAcquisitionFunction):
    r""" """

    def __init__(
        self,
        model: Model,
        utility: Utility,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        r"""q-Expected Utility.

        Args:
            model (Model): .
            utility (Utility): .
            sampler (Optional[MCSampler], optional): . Defaults to None.
            objective (Optional[MCAcquisitionObjective], optional): . Defaults to None.
            X_pending (Optional[Tensor], optional): . Defaults to None.
        """
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            X_pending=X_pending,
        )
        self.utility = utility

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qExpectedUtility on the candidate set `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.

        Returns:
            A `batch_shape'`-dim Tensor of Expected Utility values at the
            given design points `X`, where `batch_shape'` is the broadcasted batch shape
            of model and input `X`.
        """
        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)
        obj_samples = self.objective(samples)
        utility_samples = self.utility(obj_samples)
        expected_utility = utility_samples.mean(dim=0)
        return expected_utility
