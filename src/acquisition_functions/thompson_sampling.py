from __future__ import annotations

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.multitask import MultiTaskGP
from botorch.utils.gp_sampling import (
    batched_to_model_list,
    get_deterministic_model_multi_samples,
    get_weights_posterior,
    RandomFourierFeatures,
)
from botorch.utils.transforms import (
    t_batch_mode_transform,
)
from torch import Tensor

from src.utils import (
    optimize_acqf_and_get_suggested_query,
)


def gen_thompson_sampling_query(model, batch_size, bounds):
    query = []
    print(model.likelihood)
    for i in range(batch_size):
        acquisition_function = GaussianProcessSample(model=model)
        new_x = optimize_acqf_and_get_suggested_query(
            acq_func=acquisition_function,
            bounds=bounds,
            batch_size=1,
        )
        query.append(new_x.clone())

    query = torch.cat(query, dim=0)
    print(query.shape)
    return query


class GaussianProcessSample(AcquisitionFunction):
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
        self.gp_sample = get_gp_samples(model=model, num_outputs=1, n_samples=1)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate PreferentialOneStepLookahead on the candidate set X.
        Args:
            X: A `batch_shape x q x d`-dim Tensor.
        Returns:
            The acquisition value for each batch as a tensor of shape `batch_shape`.
        """
        # batch_q_shape = X.shape[:-1]
        # X_dim = X.shape[-1]
        fX = self.gp_sample.posterior(X).mean
        print(X.shape)
        print(fX.shape)
        return fX


def get_gp_samples(
    model: Model, num_outputs: int, n_samples: int, num_rff_features: int = 500
) -> GenericDeterministicModel:
    r"""Sample functions from GP posterior using RFFs. The returned
    `GenericDeterministicModel` effectively wraps `num_outputs` models,
    each of which has a batch shape of `n_samples`. Refer
    `get_deterministic_model_multi_samples` for more details.

    Args:
        model: The model.
        num_outputs: The number of outputs.
        n_samples: The number of functions to be sampled IID.
        num_rff_features: The number of random Fourier features.

    Returns:
        A batched `GenericDeterministicModel` that batch evaluates `n_samples`
        sampled functions.
    """
    if num_outputs > 1:
        if not isinstance(model, ModelListGP):
            models = batched_to_model_list(model).models
        else:
            models = model.models
    else:
        models = [model]
    if isinstance(models[0], MultiTaskGP):
        raise NotImplementedError

    weights = []
    bases = []
    for m in range(num_outputs):
        train_X = models[m].train_inputs[0]
        train_targets = models[m].train_targets
        # get random fourier features
        # sample_shape controls the number of iid functions.
        basis = RandomFourierFeatures(
            kernel=models[m].covar_module,
            input_dim=train_X.shape[-1],
            num_rff_features=num_rff_features,
            sample_shape=torch.Size([n_samples]),
        )
        bases.append(basis)
        # TODO: when batched kernels are supported in RandomFourierFeatures,
        # the following code can be uncommented.
        # if train_X.ndim > 2:
        #    batch_shape_train_X = train_X.shape[:-2]
        #    dataset_shape = train_X.shape[-2:]
        #    train_X = train_X.unsqueeze(-3).expand(
        #        *batch_shape_train_X, n_samples, *dataset_shape
        #    )
        #    train_targets = train_targets.unsqueeze(-2).expand(
        #        *batch_shape_train_X, n_samples, dataset_shape[0]
        #    )
        phi_X = basis(train_X)
        # Sample weights from bayesian linear model
        # 1. When inputs are not batched, train_X.shape == (n, d)
        # weights.sample().shape == (n_samples, num_rff_features)
        # 2. When inputs are batched, train_X.shape == (batch_shape_input, n, d)
        # This is expanded to (batch_shape_input, n_samples, n, d)
        # to maintain compatibility with RFF forward semantics
        # weights.sample().shape == (batch_shape_input, n_samples, num_rff_features)
        mvn = get_weights_posterior(
            X=phi_X,
            y=train_targets,
            sigma_sq=torch.tensor(0.0),
        )
        weights.append(mvn.sample())

    # TODO: Ideally support RFFs for multi-outputs instead of having to
    # generate a basis for each output serially.
    return get_deterministic_model_multi_samples(weights=weights, bases=bases)
