#!/usr/bin/env python3

import torch

from gpytorch.distributions import base_distributions
from gpytorch.likelihoods import _OneDimensionalLikelihood, Likelihood


class PreferentialSoftmaxLikelihood(Likelihood):
    r"""
    Implements the Softmax likelihood used for GP preference learning.

    .. math::
        p(\mathbf y \mid \mathbf f) = \text{Softmax} \left( \mathbf f \right)

    :param int num_points: Number of points.
    """

    def __init__(self, num_points):
        super().__init__()
        self.num_points = num_points
        self.noise = torch.tensor(1e-4)

    def forward(self, function_samples, *params, **kwargs):
        # print(function_samples[1, ...])
        function_samples = function_samples.reshape(
            function_samples.shape[:-1]
            + torch.Size(
                (int(function_samples.shape[-1] / self.num_points), self.num_points)
            )
        )
        # print(function_samples[1, ...])
        num_points = function_samples.shape[-1]
        # exp_logits = torch.exp(function_samples)
        # probs = exp_logits / exp_logits.sum(dim=-1, keepdim=True)
        if num_points != self.num_points:
            raise RuntimeError("There should be %d points" % self.num_points)

        res = base_distributions.Categorical(logits=function_samples)
        return res
