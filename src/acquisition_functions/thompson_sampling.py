import torch
from botorch.acquisition.analytic import PosteriorMean
from botorch.utils.gp_sampling import get_gp_samples
from copy import deepcopy

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
        )
        query.append(new_x.clone())

    query = torch.cat(query, dim=-2)
    query = query.unsqueeze(0)
    return query


def get_pairwise_gp_rff_sample(model, n_samples):
    model = model.eval()
    # force the model to infer utility
    model.posterior(model.datapoints)
    modified_model = deepcopy(model)

    modified_model.likelihood.noise = torch.tensor(1.0).double()

    modified_model.train_targets = model.utility

    gp_samples = get_gp_samples(
        model=modified_model,
        num_outputs=1,
        n_samples=n_samples,
        num_rff_features=500,
    )

    return gp_samples
