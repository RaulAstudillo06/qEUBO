from typing import Dict

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.generation.gen import get_best_candidates
from botorch.fit import fit_gpytorch_model
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.optim.optimize import optimize_acqf
from torch import Tensor
from torch.distributions import Normal


def fit_model(
    datapoints: Tensor,
    comparisons: Tensor,
    jitter=1e-4,
):
    model = PairwiseGP(
        datapoints,
        comparisons,
        jitter=jitter,
    )

    mll = PairwiseLaplaceMarginalLogLikelihood(model)
    fit_gpytorch_model(mll)
    model = model.to(device=datapoints.device, dtype=datapoints.dtype)

    return model


def generate_initial_data(
    num_queries: int,
    batch_size: int,
    input_dim: int,
    obj_func,
    comp_noise_type,
    comp_noise,
    seed: int = None,
):
    # generate initial data

    queries = generate_random_queries(num_queries, batch_size, input_dim, seed)
    obj_vals = get_obj_vals(queries, obj_func)
    responses = generate_responses(obj_vals, comp_noise_type, comp_noise)
    return queries, obj_vals, responses


def generate_random_queries(
    num_queries: int, batch_size: int, input_dim: int, seed: int = None
):
    # generate `num_queries` queries each constituted by `batch_size` points chosen uniformly at random
    if seed is not None:
        old_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        queries = torch.rand([num_queries, batch_size, input_dim])
        torch.random.set_rng_state(old_state)
    else:
        queries = torch.rand([num_queries, batch_size, input_dim])
    return queries


def get_obj_vals(queries, obj_func):
    queries_2d = queries.reshape(
        torch.Size([queries.shape[0] * queries.shape[1], queries.shape[2]])
    )
    obj_vals = obj_func(queries_2d)
    obj_vals = obj_vals.reshape(torch.Size([queries.shape[0], queries.shape[1]]))
    return obj_vals


def generate_responses(obj_vals, noise_type, noise_level):
    # generate simulated comparisons based on true underlying objective
    corrupted_obj_vals = corrupt_obj_vals(obj_vals, noise_type, noise_level)
    responses = torch.argmax(corrupted_obj_vals, dim=-1)
    return responses


def corrupt_obj_vals(obj_vals, noise_type, noise_level):

    if noise_type == "noiseless":
        corrupted_obj_vals = obj_vals
    elif noise_type == "probit":
        normal = Normal(torch.tensor(0.0), torch.tensor(noise_level))
        noise = normal.sample(sample_shape=obj_vals.shape)
        corrupted_obj_vals = obj_vals + noise

    return corrupted_obj_vals


def training_data_for_pairwise_gp(queries, responses):
    num_queries = queries.shape[0]
    batch_size = queries.shape[1]
    datapoints = []
    comparisons = []
    for i in range(num_queries):
        best_item_id = batch_size * i + responses[i]
        for j in range(batch_size):
            datapoints.append(queries[i, j, :].unsqueeze(0))
            if j != responses[i]:
                comparisons.append(
                    torch.tensor([best_item_id, batch_size * i + j]).unsqueeze(0)
                )

    datapoints = torch.cat(datapoints, dim=0)
    comparisons = torch.cat(comparisons, dim=0)
    return datapoints, comparisons


def optimize_acqf_and_get_suggested_query(
    acq_func: AcquisitionFunction,
    bounds: Tensor,
    batch_size: int,
) -> Tensor:
    """Optimizes the acquisition function, and returns the candidate solution."""
    input_dim = bounds.shape[1]
    q = batch_size
    raw_samples = 100 * input_dim * batch_size
    num_restarts = 10 * input_dim * batch_size

    candidates, acq_values = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={
            "batch_limit": 2,
            "maxiter": 100,
            "nonnegative": True,
            "method": "L-BFGS-B",
        },
        return_best_only=False,
    )

    candidates = candidates.detach()
    acq_values_sorted, indices = torch.sort(acq_values.squeeze(), descending=True)
    print("Acquisition values:")
    print(acq_values_sorted)
    print("Candidates:")
    print(candidates[indices].squeeze())
    print(candidates.squeeze())
    new_x = get_best_candidates(batch_candidates=candidates, batch_values=acq_values)
    return new_x
