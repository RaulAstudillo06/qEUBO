#!/usr/bin/env python3

import warnings

import torch

from gpytorch.distributions import Distribution, MultitaskMultivariateNormal, base_distributions
from gpytorch.likelihoods import _OneDimensionalLikelihood


class PreferentialSoftmaxLikelihood(_OneDimensionalLikelihood):
    r"""
    Implements the Softmax likelihood used for GP preference learning.

    .. math::
        p(\mathbf y \mid \mathbf f) = \text{Softmax} \left( \mathbf f \right)

    :param int num_points: Number of points.
    """

    def __init__(self, num_points):
        super().__init__()
        self.num_points = num_points

    def forward(self, function_samples, *params, **kwargs):
        function_samples = function_samples.reshape(function_samples.shape[:-1] + torch.Size((int(function_samples.shape[-1] / self.num_points), self.num_points)))
        num_data, num_points = function_samples.shape[-2:]

        if num_points != self.num_points:
            raise RuntimeError("There should be %d points" % self.num_points)

        res = base_distributions.Categorical(logits=function_samples)
        return res
