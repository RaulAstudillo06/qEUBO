from typing import Optional

import torch
from botorch.acquisition import OneShotAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.samplers import MCSampler, SobolQMCNormalSampler
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor


class PreferentialKnowledgeGradient(OneShotAcquisitionFunction):
    r"""Preferential Knowledge Gradient."""

    def __init__(
        self,
        model: Model,
        X_baseline: Optional[Tensor] = None,
        sampler: Optional[MCSampler] = None,
    ) -> None:
        r"""Preferential Knowledge Gradient (one-shot optimization).
        Performs a one-step lookahead obtained by conditioning on a fantasy query asked
        to the DM. The reward is the maximum of the posterior mean conditioned on such
        fantasy query.
        Args:
            model: A fitted model.
            X_baseline: .
        """
        super(OneShotAcquisitionFunction, self).__init__(model=model)
        self.model = model
        self.model.eval()  # make sure model is in eval mode
        if X_baseline is not None:
            self.register_buffer("X_baseline", X_baseline)
        else:
            self.X_baseline = X_baseline

        self.augmented_q_batch_size = 4

        self.register_buffer(
            "preference_scenario_1",
            torch.tensor([0, 1]).unsqueeze(0),
        )
        self.register_buffer(
            "preference_scenario_2",
            torch.tensor([1, 0]).unsqueeze(0),
        )
        self.preference_scenarios = [
            self.preference_scenario_1,
            self.preference_scenario_2,
        ]
        self.std_norm = torch.distributions.normal.Normal(
            torch.zeros(
                1, dtype=model.datapoints.dtype, device=model.datapoints.device
            ),
            torch.ones(1, dtype=model.datapoints.dtype, device=model.datapoints.device),
        )

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate PreferentialKnowledgeGradient on the candidate set X.
        Args:
            X: A `batch_shape x augmented_q_batch_size x d`-dim Tensor.
        Returns:
            The acquisition value for each batch as a tensor of shape `batch_shape`.
        """
        assert X.shape[-2] == self.augmented_q_batch_size
        Xs_query = X[..., :2, :]
        Xs_reward = [X[..., [2], :], X[..., [3], :]]

        ## This part of the code computes the probabilities of each of the two
        ## scenarios: {y_1 > y_2} and {y_1 < y_2}
        ## (This is the same code as the one used in for PreferentialOneStepLookahead)
        std_norm = torch.distributions.normal.Normal(
            torch.zeros(1, dtype=X.dtype, device=X.device),
            torch.ones(1, dtype=X.dtype, device=X.device),
        )
        posterior = self.model(Xs_query)
        mean = posterior.mean
        cov = posterior.covariance_matrix

        mean_util_diff = mean[..., 0] - mean[..., 1]
        sigma_star = torch.sqrt(
            2.0 + cov[..., 0, 0] + cov[..., 1, 1] - cov[..., 0, 1] - cov[..., 1, 0]
        )
        probs = [self.std_norm.cdf(mean_util_diff / sigma_star)]
        probs.append(1.0 - probs[0])
        ##
        acqf_val = 0.0
        for i, preference_scenario in enumerate(self.preference_scenarios):
            fantasy_model = self.model.condition_on_observations(
                X=Xs_query,
                Y=preference_scenario.expand(X.shape[:-2] + torch.Size([-1, -1])),
            )
            # Fantasy preference posterior mean evaluated at x'_i
            reward = fantasy_model(Xs_reward[i]).mean.squeeze(-1)
            acqf_val = acqf_val + probs[i] * reward
        return acqf_val

    def get_augmented_q_batch_size(self) -> int:
        r"""Get augmented q batch size for one-shot optimzation.
        Returns:
            The augmented size for one-shot optimzation (including variables
            parameterizing the fantasy solutions).
        """
        return 4

    def extract_candidates(self, X_full: Tensor) -> Tensor:
        r"""We only return X as the set of candidates post-optimization.
        Args:
            X_full: A `batch_shape x augmented_q_batch_size x d`-dim Tensor.
        Returns:
            A `batch_shape x (augmented_q_batch_size - 2) x d`-dim Tensor.
        """
        return X_full[..., : self.augmented_q_batch_size - 2, :]
