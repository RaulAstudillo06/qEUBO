from __future__ import annotations

from typing import Optional

import torch
from botorch.acquisition import AcquisitionFunction, MCAcquisitionFunction
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.models.model import Model
from botorch.sampling import MCSampler
from botorch.utils.transforms import (
    concatenate_pending_points,
    t_batch_mode_transform,
)
from torch import Tensor


class ExpectedMaxObjectiveValue(AcquisitionFunction):
    r""""""

    def __init__(
        self,
        model: Model,
    ) -> None:
        r"""Analytic Expected Max Objective Value.

        Args:
            model (Model): .
        """
        super().__init__(model=model)
        self.standard_normal = torch.distributions.normal.Normal(
            torch.zeros(1),
            torch.ones(1),
        )

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate PreferentialOneStepLookahead on the candidate set X.
        Args:
            X: A `batch_shape x 2 x d`-dim Tensor.
        Returns:
            The acquisition value for each batch as a tensor of shape `batch_shape`.
        """
        posterior = self.model(X)
        mean = posterior.mean
        cov = posterior.covariance_matrix
        delta = mean[..., 0] - mean[..., 1]
        sigma = torch.sqrt(
            cov[..., 0, 0] + cov[..., 1, 1] - cov[..., 0, 1] - cov[..., 1, 0]
        )
        u = delta / sigma

        ucdf = self.standard_normal.cdf(u)
        updf = torch.exp(self.standard_normal.log_prob(u))
        acqf_val = sigma * (updf + u * ucdf)
        acqf_val += mean[..., 1]
        return acqf_val


class qExpectedMaxObjectiveValue(MCAcquisitionFunction):
    r""" """

    def __init__(
        self,
        model: Model,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        r"""q-Preferential Noisy Expected Improvement.

        Args:
            outcome_model (Model): .
            pref_model (Model): .
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

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qNoisyExpectedImprovement on the candidate set `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.

        Returns:
            A `batch_shape'`-dim Tensor of Noisy Expected Improvement values at the
            given design points `X`, where `batch_shape'` is the broadcasted batch shape
            of model and input `X`.
        """
        posterior_X = self.model.posterior(X)
        Y_samples = self.sampler(posterior_X)
        obj_val_samples = self.objective(Y_samples)
        max_obj_val_samples = obj_val_samples.max(dim=-1).values
        exp_max_obj_val = max_obj_val_samples.mean(dim=0)
        return exp_max_obj_val
