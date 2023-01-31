#!/usr/bin/env python3

import warnings

import torch

from gpytorch.distributions import Distribution, MultitaskMultivariateNormal, base_distributions
from gpytorch.likelihoods import Likelihood


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

    def forward(self, function_samples, *params, **kwargs):
        function_samples = function_samples.reshape(function_samples.shape[:-1] + torch.Size((int(function_samples.shape[-1] / self.num_points), self.num_points)))
        num_data, num_points = function_samples.shape[-2:]

        if num_points != self.num_points:
            raise RuntimeError("There should be %d points" % self.num_points)

        res = base_distributions.Categorical(logits=function_samples)
        return res

    def __call__(self, function, *params, **kwargs):
        if isinstance(function, Distribution) and not isinstance(function, MultitaskMultivariateNormal):
            print(e)
            warnings.warn(
                "The input to SoftmaxLikelihood should be a MultitaskMultivariateNormal (num_data x num_tasks). "
                "Batch MultivariateNormal inputs (num_tasks x num_data) will be deprectated.",
                DeprecationWarning,
            )
            function = MultitaskMultivariateNormal.from_batch_mvn(function)
        return super().__call__(function, *params, **kwargs)
