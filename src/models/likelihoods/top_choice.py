from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch
from gpytorch.likelihoods import Likelihood
from torch import Tensor
from torch.distributions import Bernoulli


def log_softmax(x, winner_idx=None):
    """
    Args:
        x: (batch_shape x m x) k_choice, the utility values
    Returns:
        log softmax: (m x) k_choice
    """
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    ret = log_softmax(x)
    return ret if winner_idx is None else ret[..., winner_idx]


def jac_neg_log_softmax(x, winner_idx=None):
    """
    Args:
        x: (batch_shape x m x) k_choice, the utility values
    Returns:
        jacobian of negative log_softmax, shape = (m x) k_choice x k_choice if winner_idx is None
        otherwise, only keep the winner index of the 2nd to last dimension
    """
    softmax = torch.nn.Softmax(dim=-1)
    softmax_x = softmax(x)
    jac_mat = softmax_x.unsqueeze(-2).expand(*x.shape[:-1], x.shape[-1], x.shape[-1])
    eye = torch.eye(jac_mat.shape[-1]).expand(jac_mat.shape)
    jac_mat = jac_mat - eye
    return jac_mat if winner_idx is None else jac_mat[..., winner_idx, :]


def hess_neg_log_softmax(x, winner_idx=None):
    """
    Args:
        x: (batch_shape x m x) k_choice, the utility values
    Returns:
        hessian of negative log_softmax, shape = (m x) k_choice x k_choice x k_choice if dim is None
        otherwise, only keep the winner index of the 3rd to last dimension
    """
    softmax = torch.nn.Softmax(dim=-1)
    softmax_x = softmax(x)
    sjk = softmax_x.unsqueeze(-1) @ softmax_x.unsqueeze(-2)
    hess = torch.diag_embed(softmax_x) - sjk
    hess = hess.unsqueeze(-3).expand(
        *x.shape[:-1], x.shape[-1], x.shape[-1], x.shape[-1]
    )
    return hess if winner_idx is None else hess[..., winner_idx, :, :]


class TopChoiceLikelihood(Likelihood, ABC):
    """
    Pairwise likelihood base class for pairwise preference GP (e.g., PairwiseGP).

    :meta private:
    """

    def __init__(self, max_plate_nesting: int = 1):
        """
        Initialized like a `gpytorch.likelihoods.Likelihood`.

        Args:
            max_plate_nesting: Defaults to 1.
        """
        super().__init__(max_plate_nesting)

    def forward(self, utility: Tensor, D: Tensor, **kwargs: Any) -> Bernoulli:
        return NotImplementedError

    @abstractmethod
    def p(self, utility: Tensor, D: Tensor) -> Tensor:
        """Given the difference in (estimated) utility util_diff = f(v) - f(u),
        return the probability of the user prefer v over u.

        Args:
            utility: A Tensor of shape `(batch_size) x n`, the utility at MAP point
            D: D is `(batch_size x) m x n` matrix with all elements being zero in last
                dimension except at two positions D[..., i] = 1 and D[..., j] = -1
                respectively, representing item i is preferred over item j.
            log: if true, return log probability
        """

    def log_p(self, utility: Tensor, D: Tensor) -> Tensor:
        """return the log of p"""
        return torch.log(self.p(utility=utility, D=D))

    def negative_log_gradient_sum(self, utility: Tensor, D: Tensor) -> Tensor:
        """Calculate the sum of negative log gradient with respect to each item's latent
            utility values. Useful for models using laplace approximation.

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
            utility values. Useful for models using laplace approximation.

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


class TopChoiceLogitLikelihood(TopChoiceLikelihood):
    """Pairwise likelihood using logistic (i.e., softmax) function"""

    def __init__(self, k_choice, max_plate_nesting: int = 1):
        """
        Initialized like a `gpytorch.likelihoods.Likelihood`.

        Args:
            max_plate_nesting: Defaults to 1.
        """
        super().__init__(max_plate_nesting)
        self.k_choice = k_choice

    def _reconstruct_choice_indices(self, D: Tensor) -> Tensor:
        winner_indices = (D == 1).nonzero(as_tuple=True)[-1]
        winner_indices = winner_indices.reshape(D.shape[:-1] + (1,))
        loser_indices = (D == -1).nonzero(as_tuple=True)[-1]
        loser_indices = loser_indices.reshape(D.shape[:-1] + (self.k_choice - 1,))
        # cannot use original `choices` because loser indices are changed
        choice_indices = torch.cat((winner_indices, loser_indices), dim=-1)
        return choice_indices

    def _get_choice_util(self, utility: Tensor, D: Tensor) -> Tensor:
        reshaped_util = utility.unsqueeze(-2).expand(D.shape)
        winner_util = reshaped_util[(D == 1)].reshape(*D.shape[:-1], 1)
        loser_util = reshaped_util[(D == -1)].reshape(
            *D.shape[:-1], self.k_choice - 1
        )  # loser order is potentially changed here
        choice_util = torch.cat((winner_util, loser_util), dim=-1)
        return choice_util

    def log_p(self, utility: Tensor, D: Tensor) -> Tensor:
        choice_util = self._get_choice_util(utility=utility, D=D)
        return log_softmax(x=choice_util, winner_idx=0)

    def p(self, utility: Tensor, D: Tensor) -> Tensor:
        return self.log_p(utility=utility, D=D).exp()

    def negative_log_gradient_sum(self, utility: Tensor, D: Tensor) -> Tensor:
        m, n = D.shape[-2:]
        choice_util = self._get_choice_util(utility=utility, D=D)

        jac = jac_neg_log_softmax(x=choice_util, winner_idx=0)
        choice_indices = self._reconstruct_choice_indices(D=D)

        scattered_jac = torch.zeros(
            utility.shape[:-1] + (m, n), dtype=utility.dtype, device=utility.device
        )
        scattered_jac.scatter_(dim=-1, index=choice_indices, src=jac)

        return scattered_jac.sum(dim=-2)

    def negative_log_hessian_sum(self, utility: Tensor, D: Tensor) -> Tensor:
        m, n = D.shape[-2:]
        choice_util = self._get_choice_util(utility=utility, D=D)
        hess = hess_neg_log_softmax(x=choice_util, winner_idx=0)
        choice_indices = self._reconstruct_choice_indices(D=D)

        scattered_hess = torch.zeros(
            utility.shape[:-1] + (m, n, n), dtype=utility.dtype, device=utility.device
        )
        flat_choice_indices = choice_indices.view(-1, self.k_choice)
        flat_scattered_hess = scattered_hess.view(-1, n, n)
        for i, single_hess in enumerate(
            hess.view(-1, self.k_choice, self.k_choice).unbind()
        ):
            cart_prod_idx = torch.cartesian_prod(
                flat_choice_indices[i, :], flat_choice_indices[i, :]
            )
            flat_scattered_hess[
                i, cart_prod_idx[:, 0], cart_prod_idx[:, 1]
            ] = single_hess.flatten()

        return scattered_hess.sum(dim=-3)
