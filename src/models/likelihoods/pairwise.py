#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Tuple, Any

import torch
from gpytorch.likelihoods import Likelihood
from torch import Tensor
from torch.distributions import Bernoulli


class PairwiseLikelihood(Likelihood):
    """Pairwise likelihood base class for Laplace approximation-based PairwiseGP class"""

    def forward(self, utility: Tensor, D: Tensor, **kwargs: Any) -> Bernoulli:
        """Given the difference in (estimated) utility util_diff = f(v) - f(u),
        return a Bernoulli distribution object representing the likelihood of
        the user prefer v over u."""
        return Bernoulli(probs=self.p(utility=utility, D=D, log=False))

    def p(self, utility: Tensor, D: Tensor, log: bool = False) -> Tensor:
        """Given the difference in (estimated) utility util_diff = f(v) - f(u),
        return the probability of the user prefer v over u.
        Args:
            utility: A Tensor of shape `(batch_size) x n`, the utility at MAP point
            D: D is `(batch_size x) m x n` matrix with all elements being zero in last
                dimension except at two positions D[..., i] = 1 and D[..., j] = -1
                respectively, representing item i is preferred over item j.
            log: if true, return log probability
        """
        raise NotImplementedError

    def negative_log_gradient_sum(self, utility: Tensor, D: Tensor) -> Tensor:
        """Calculate the sum of negative log gradient with respect to each item's latent
            utility values
        Args:
            utility: A Tensor of shape `(batch_size x) n`, the utility at MAP point
            D: D is `(batch_size x) m x n` matrix with all elements being zero in last
                dimension except at two positions D[..., i] = 1 and D[..., j] = -1
                respectively, representing item i is preferred over item j.
        Returns:
            A `(batch_size x) n` Tensor representing the sum of negative log gradient
            values of the likelihood over all comparisons (i.e., the m dimension)
            with respect to each item.
        """
        raise NotImplementedError

    def negative_log_hessian_sum(self, utility: Tensor, D: Tensor) -> Tensor:
        """Calculate the sum of negative log hessian with respect to each item's latent
            utility values
        Args:
            utility: A Tensor of shape `(batch_size) x n`, the utility at MAP point
            D: D is `(batch_size x) m x n` matrix with all elements being zero in last
                dimension except at two positions D[..., i] = 1 and D[..., j] = -1
                respectively, representing item i is preferred over item j.
        Returns:
            A `(batch_size x) n x n` Tensor representing the sum of negative log hessian
            values of the likelihood over all comparisons (i.e., the m dimension) with
            respect to each item.
        """
        raise NotImplementedError


class PairwiseProbitLikelihood(PairwiseLikelihood):
    # Clamping z values for better numerical stability. See self._calc_z for detail
    # norm_cdf(z=3) ~= 0.999, top 0.1% percent
    _zlim = 3

    def _calc_z(self, utility: Tensor, D: Tensor) -> Tensor:
        scaled_util = (utility / math.sqrt(2)).to(D)
        z = D @ scaled_util
        z = z.clamp(-self._zlim, self._zlim)
        return z

    def _calc_z_derived(self, z: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        std_norm = torch.distributions.normal.Normal(
            torch.zeros(1, dtype=z.dtype, device=z.device),
            torch.ones(1, dtype=z.dtype, device=z.device),
        )
        z_logpdf = std_norm.log_prob(z)
        z_cdf = std_norm.cdf(z)
        z_logcdf = torch.log(z_cdf)
        hazard = torch.exp(z_logpdf - z_logcdf)
        return z_logpdf, z_logcdf, hazard

    def p(self, utility: Tensor, D: Tensor, log: bool = False) -> Tensor:
        z = self._calc_z(utility=utility, D=D)
        std_norm = torch.distributions.normal.Normal(
            torch.zeros(1, dtype=z.dtype, device=z.device),
            torch.ones(1, dtype=z.dtype, device=z.device),
        )
        z_cdf = std_norm.cdf(z)
        return torch.log(z_cdf) if log else z_cdf

    def negative_log_gradient_sum(self, utility: Tensor, D: Tensor) -> Tensor:
        # Compute the sum over of grad. of negative Log-LH wrt utility f.
        # Original grad should be of dimension m x n, as in (6) from [Chu2005preference]_.
        # The sum over the m dimension of grad. of negative log likelihood
        # with respect to the utility
        z = self._calc_z(utility, D)
        _, _, h = self._calc_z_derived(z)
        h_factor = h / math.sqrt(2)
        grad = h_factor @ (-D)

        return grad

    def negative_log_hessian_sum(self, utility: Tensor, D: Tensor) -> Tensor:
        # Original hess should be of dimension m x n x n, as in (7) from
        # [Chu2005preference]_ Sum over the first dimension and return a tensor of
        # shape n x n.
        # The sum over the m dimension of hessian of negative log likelihood
        # with respect to the utility
        DT = D.T
        z = self._calc_z(utility, D)
        _, _, h = self._calc_z_derived(z)
        mul_factor = h * (h + z) / 2
        weighted_DT = DT * mul_factor.unsqueeze(-2).expand(*DT.size())
        hess = weighted_DT @ D

        return hess


class PairwiseLogitLikelihood(PairwiseLikelihood):
    # Clamping logit values for better numerical stability. See self._calc_logit for detail
    # logistic(8) ~= 0.9997, top 0.03% percent
    _logit_lim = 8

    def _calc_logit(self, utility: Tensor, D: Tensor) -> Tensor:
        logit = D @ utility.to(D)
        logit = logit.clamp(-self._logit_lim, self._logit_lim)
        return logit

    def p(self, utility: Tensor, D: Tensor, log: bool = False) -> Tensor:
        logit = self._calc_logit(utility=utility, D=D)
        probs = torch.sigmoid(logit)
        return torch.log(probs) if log else probs

    def negative_log_gradient_sum(self, utility: Tensor, D: Tensor) -> Tensor:
        winner_indices = (D == 1).nonzero(as_tuple=True)[-1]
        loser_indices = (D == -1).nonzero(as_tuple=True)[-1]
        ex, ey = torch.exp(utility[winner_indices]), torch.exp(utility[loser_indices])
        unsigned_grad = ey / (ex + ey)
        grad = unsigned_grad @ (-D)
        return grad

    def negative_log_hessian_sum(self, utility: Tensor, D: Tensor) -> Tensor:
        DT = D.T
        neg_logit = -(D @ utility)
        term = torch.sigmoid(neg_logit)
        unsigned_hess = term - (term) ** 2
        weighted_DT = DT * unsigned_hess.unsqueeze(-2).expand(*DT.size())
        hess = weighted_DT @ D

        return hess
