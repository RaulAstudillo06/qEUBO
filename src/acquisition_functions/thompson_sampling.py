import torch
from botorch.acquisition.analytic import PosteriorMean
from botorch.utils.gp_sampling import get_gp_samples
from copy import deepcopy

from src.models.pairwise_gp import PairwiseGP
from src.models.pairwise_kernel_variational_gp import PairwiseKernelVariationalGP
from src.utils import (
    optimize_acqf_and_get_suggested_query,
)


def gen_thompson_sampling_query(model, batch_size, bounds):
    query = []
    for _ in range(batch_size):
        model_rff_sample = get_pairwise_gp_rff_sample(model=model, n_samples=1)
        acquisition_function = PosteriorMean(model=model_rff_sample)
        new_x = optimize_acqf_and_get_suggested_query(
            acq_func=acquisition_function,
            bounds=bounds,
            batch_size=1,
            batch_limit=1,
            init_batch_limit=1,
        )
        query.append(new_x.clone())

    query = torch.cat(query, dim=-2)
    query = query.unsqueeze(0)
    return query


def get_pairwise_gp_rff_sample(model, n_samples):
    model = model.eval()
    if isinstance(model, PairwiseGP):
        # force the model to infer utility
        model.posterior(model.datapoints)
        modified_model = deepcopy(model)
        modified_model.likelihood.noise = torch.tensor(1.0).double()
        modified_model.train_targets = model.utility
    elif isinstance(model, PairwiseKernelVariationalGP):
        modified_model = deepcopy(model)
        queries = modified_model.queries.clone()
        queries_items = queries.view(
            (queries.shape[0] * queries.shape[1], queries.shape[2])
        )
        modified_model.train_inputs = [queries_items]
        sample_at_queries_items = modified_model.posterior(queries_items).sample()
        modified_model.train_targets = sample_at_queries_items.view(
            (queries_items.shape[0],)
        )
        modified_model.covar_module = (
            modified_model.aux_model.covar_module.latent_kernel
        )

        class LikelihoodForRFF:
            noise = torch.tensor(1.0).double()

        modified_model.likelihood = LikelihoodForRFF()

    gp_samples = get_gp_samples(
        model=modified_model,
        num_outputs=1,
        n_samples=n_samples,
        num_rff_features=1000,
    )

    return gp_samples
