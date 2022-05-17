#!/usr/bin/env python3

from re import T
from copy import deepcopy
from typing import Union

import torch
from botorch.fit import fit_gpytorch_model
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.mlls.variational_elbo import VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.utils.broadcasting import _mul_broadcast_shape
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from torch import Tensor

from src.models.kernels.pairwise_kernel import PairwiseKernel


class PairwiseKernelVariationalGPAux(ApproximateGP, GPyTorchModel):
    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        inducing_points: Tensor,
        scales: Union[Tensor, float] = 1.0,
    ) -> None:
        # Construct variational dist/strat
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(-2)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=False,
        )
        super(PairwiseKernelVariationalGPAux, self).__init__(variational_strategy)

        self.likelihood = BernoulliLikelihood()

        # Mean and cov
        self.mean_module = ConstantMean()

        ls_prior = GammaPrior(3.0, 6.0 / scales)

        self.covar_module = PairwiseKernel(
            latent_kernel=ScaleKernel(
                RBFKernel(
                    ard_num_dims=train_x.shape[-1] // 2, lengthscale_prior=ls_prior
                ),
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
        )
        self._num_outputs = 1
        self.train_inputs = (train_x,)
        self.train_targets = train_y

    def forward(self, X: Tensor) -> MultivariateNormal:
        mean_X = self.mean_module(X)
        covar_X = self.covar_module(X)
        return MultivariateNormal(mean_X, covar_X)

    # COPIED FROM ExactGP from gpytorch/models/exact_gp.py
    # parts related to prediction_strategy and likelihood are commented out
    def get_fantasy_model(self, inputs, targets, **kwargs):
        """
        Returns a new GP model that incorporates the specified inputs and targets as new training data.

        Using this method is more efficient than updating with `set_train_data` when the number of inputs is relatively
        small, because any computed test-time caches will be updated in linear time rather than computed from scratch.

        .. note::
            If `targets` is a batch (e.g. `b x m`), then the GP returned from this method will be a batch mode GP.
            If `inputs` is of the same (or lesser) dimension as `targets`, then it is assumed that the fantasy points
            are the same for each target batch.

        :param Tensor inputs: (`b1 x ... x bk x m x d` or `f x b1 x ... x bk x m x d`) Locations of fantasy
            observations.
        :param Tensor targets: (`b1 x ... x bk x m` or `f x b1 x ... x bk x m`) Labels of fantasy observations.
        :return: An `ExactGP` model with `n + m` training examples, where the `m` fantasy examples have been added
            and all test-time caches have been updated.
        :rtype: ~gpytorch.models.ExactGP
        """
        # if self.prediction_strategy is None:
        #     raise RuntimeError(
        #         "Fantasy observations can only be added after making predictions with a model so that "
        #         "all test independent caches exist. Call the model on some data first!"
        #     )

        model_batch_shape = self.train_inputs[0].shape[:-2]

        if self.train_targets.dim() > len(model_batch_shape) + 1:
            raise RuntimeError(
                "Cannot yet add fantasy observations to multitask GPs, but this is coming soon!"
            )

        if not isinstance(inputs, list):
            inputs = [inputs]

        inputs = [i.unsqueeze(-1) if i.ndimension() == 1 else i for i in inputs]

        target_batch_shape = targets.shape[:-1]
        input_batch_shape = inputs[0].shape[:-2]
        tbdim, ibdim = len(target_batch_shape), len(input_batch_shape)

        if not (tbdim == ibdim + 1 or tbdim == ibdim):
            raise RuntimeError(
                f"Unsupported batch shapes: The target batch shape ({target_batch_shape}) must have either the "
                f"same dimension as or one more dimension than the input batch shape ({input_batch_shape})"
            )

        # Check whether we can properly broadcast batch dimensions
        err_msg = (
            f"Model batch shape ({model_batch_shape}) and target batch shape "
            f"({target_batch_shape}) are not broadcastable."
        )
        _mul_broadcast_shape(model_batch_shape, target_batch_shape, error_msg=err_msg)

        if len(model_batch_shape) > len(input_batch_shape):
            input_batch_shape = model_batch_shape
        if len(model_batch_shape) > len(target_batch_shape):
            target_batch_shape = model_batch_shape

        # If input has no fantasy batch dimension but target does, we can save memory and computation by not
        # computing the covariance for each element of the batch. Therefore we don't expand the inputs to the
        # size of the fantasy model here - this is done below, after the evaluation and fast fantasy update
        train_inputs = [
            tin.expand(input_batch_shape + tin.shape[-2:]) for tin in self.train_inputs
        ]
        train_targets = self.train_targets.expand(
            target_batch_shape + self.train_targets.shape[-1:]
        )

        full_inputs = [
            torch.cat(
                [train_input, input.expand(input_batch_shape + input.shape[-2:])],
                dim=-2,
            )
            for train_input, input in zip(train_inputs, inputs)
        ]
        full_targets = torch.cat(
            [train_targets, targets.expand(target_batch_shape + targets.shape[-1:])],
            dim=-1,
        )

        try:
            fantasy_kwargs = {"noise": kwargs.pop("noise")}
        except KeyError:
            fantasy_kwargs = {}

        # full_output = super(PairwiseKernelGP, self).__call__(*full_inputs, **kwargs)

        # Copy model without copying training data or prediction strategy (since we'll overwrite those)
        # old_pred_strat = self.prediction_strategy
        old_train_inputs = self.train_inputs
        old_train_targets = self.train_targets

        old_likelihood = self.likelihood
        # self.prediction_strategy = None
        self.train_inputs = None
        self.train_targets = None
        self.likelihood = None
        new_model = deepcopy(self)
        # self.prediction_strategy = old_pred_strat
        self.train_inputs = old_train_inputs
        self.train_targets = old_train_targets
        self.likelihood = old_likelihood

        new_model.likelihood = old_likelihood.get_fantasy_likelihood(**fantasy_kwargs)
        # new_model.prediction_strategy = old_pred_strat.get_fantasy_strategy(
        #     inputs, targets, full_inputs, full_targets, full_output, **fantasy_kwargs
        # )

        # if the fantasies are at the same points, we need to expand the inputs for the new model
        if tbdim == ibdim + 1:
            new_model.train_inputs = [
                fi.expand(target_batch_shape + fi.shape[-2:]) for fi in full_inputs
            ]
        else:
            new_model.train_inputs = full_inputs
        new_model.train_targets = full_targets

        return new_model


class PairwiseKernelVariationalGP(Model):
    def __init__(
        self,
        queries: Tensor,
        responses: Tensor,
    ) -> None:
        super().__init__()
        self.queries = queries
        self.responses = responses
        self.train_inputs = None
        self.train_targets = None
        self.covar_module = None
        self.input_dim = queries.shape[-1]
        train_x = queries.flatten(start_dim=-2, end_dim=-1)
        train_y = 1.0 - responses.squeeze(-1)
        bounds = torch.tensor(
            [[0, 1] for _ in range(self.input_dim)], dtype=torch.double
        ).T
        bounds_aug = torch.cat((bounds, bounds), dim=1)
        inducing_points = draw_sobol_samples(
            bounds=bounds_aug, n=2 ** (self.input_dim + 2), q=1
        ).squeeze(1)
        inducing_points = torch.cat([inducing_points, train_x], dim=0)
        scales = bounds[1, :] - bounds[0, :]
        aux_model = PairwiseKernelVariationalGPAux(
            train_x, train_y, inducing_points, scales
        )
        mll = VariationalELBO(
            likelihood=aux_model.likelihood,
            model=aux_model,
            num_data=train_y.numel(),
        )
        mll = fit_gpytorch_model(mll)
        self.aux_model = aux_model

    def posterior(self, X: Tensor, posterior_transform=None) -> MultivariateNormal:
        X0 = torch.zeros(size=X.shape, requires_grad=False)
        X_aug = torch.cat([X, X0], dim=-1)
        return self.aux_model.posterior(X_aug)

    def forward(self, X: Tensor) -> MultivariateNormal:
        return self.posterior(X)

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        return 1
