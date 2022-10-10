r"""
TopChoice likelihood for top choice preference model (e.g., TopChoiceGP).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from gpytorch.likelihoods import Likelihood
from torch import Tensor
from torch.distributions import Bernoulli


def jac_neg_log_softmax(x, winner_idx=None):
    """
    Args:
        x: (batch_shape x m x) k_choice, the utility values
    Returns:
        jacobian of neg_log_softmax, shape = (m x) k_choice x k_choice if dim is None
    """
    softmax_x = x.softmax(dim=-1)
    jac_mat = softmax_x.unsqueeze(-2).expand(*x.shape[:-1], x.shape[-1], x.shape[-1])
    eye = torch.eye(jac_mat.shape[-1]).expand(jac_mat.shape)
    jac_mat = jac_mat - eye
    return jac_mat if winner_idx is None else jac_mat[..., winner_idx, :]


def hess_neg_log_softmax(x, winner_idx=None):
    """
    Args:
        x: (batch_shape x m x) k_choice, the utility values
    Returns:
        hessian of neg_log_softmax, shape = (m x) k_choice x k_choice x k_choice if dim is None
    """
    softmax_x = x.softmax(dim=-1)
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
            choices: choices as in TopChoiceGP


        Returns:
            A `(batch_size x) n` Tensor representing the sum of negative log gradient
            values of the likelihood over all comparisons (i.e., the m dimension)
            with respect to each item.
        """
        raise NotImplementedError

    def negative_log_hessian_sum(self, utility: Tensor, choices: Tensor) -> Tensor:
        """Calculate the sum of negative log hessian with respect to each item's latent
            utility values. Useful for models using laplace approximation.

        Args:
            utility: A Tensor of shape `(batch_size) x n`, the utility at MAP point
            choices: choices as in TopChoiceGP

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
        return choice_util.log_softmax(dim=-1)[..., 0]

    def p(self, utility: Tensor, D: Tensor) -> Tensor:
        return self.log_p(utility=utility, D=D).exp()

    def negative_log_gradient_sum(self, utility: Tensor, choices: Tensor) -> Tensor:
        batch_shape = utility.shape[:-1]
        m, n = choices.shape[-2], utility.shape[-1]
        choice_util = utility[choices]

        jac = jac_neg_log_softmax(x=choice_util, winner_idx=0)
        choice_indices = choices

        scattered_jac = torch.zeros(
            batch_shape + (n,), dtype=utility.dtype, device=utility.device
        )
        scattered_jac.scatter_add_(
            dim=-1,
            index=choice_indices.flatten(start_dim=-2),
            src=jac.flatten(start_dim=-2),
        )
        return scattered_jac

    def negative_log_hessian_sum(self, utility: Tensor, choices: Tensor) -> Tensor:
        batch_shape = utility.shape[:-1]
        k_choice, n = choices.shape[-1], utility.shape[-1]
        choice_util = utility[choices]
        hess = hess_neg_log_softmax(x=choice_util, winner_idx=0)
        choice_indices = choices

        scattered_hess = torch.zeros(
            batch_shape + (n * n,), dtype=utility.dtype, device=utility.device
        )
        flat_cart_indices = (
            choice_indices.unsqueeze(-1).expand(choice_indices.shape + (k_choice,)) * n
            + choice_indices.unsqueeze(-2)
        ).flatten(start_dim=-3)
        flat_hess = hess.flatten(start_dim=-3)
        scattered_hess.scatter_add_(dim=-1, index=flat_cart_indices, src=flat_hess)
        scattered_hess = scattered_hess.reshape(batch_shape + (n, n))
        return scattered_hess
