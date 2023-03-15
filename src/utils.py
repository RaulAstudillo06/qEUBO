#!/usr/bin/env python3

from typing import Optional

import torch
from botorch.acquisition import AcquisitionFunction, PosteriorMean
from botorch.generation.gen import get_best_candidates
from botorch.fit import fit_gpytorch_mll
from botorch.models.likelihoods.pairwise import (
    PairwiseLogitLikelihood,
    PairwiseProbitLikelihood,
)
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.optim.optimize import optimize_acqf
from gpytorch.mlls.variational_elbo import VariationalELBO
from torch import Tensor
from torch.distributions import Bernoulli, Normal, Gumbel


from src.acquisition_functions.eubo import ExpectedUtilityOfBestOption
from src.models.pairwise_kernel_variational_gp import PairwiseKernelVariationalGP
from src.models.variational_preferential_gp import VariationalPreferentialGP
from src.models.top_choice_gp import (
    TopChoiceGP,
    TopChoiceLaplaceMarginalLogLikelihood,
)


def fit_model(
    queries: Tensor,
    responses: Tensor,
    model_type: str,
    likelihood: Optional[str] = "logit",
):
    if model_type == "pairwise_gp":
        datapoints, comparisons = training_data_for_pairwise_gp(queries, responses)

        if queries.shape[1] == 2:
            if likelihood == "logit":
                likelihood_func = PairwiseLogitLikelihood()
            elif likelihood == "probit":
                likelihood_func = PairwiseProbitLikelihood()
            model = PairwiseGP(
                datapoints,
                comparisons,
                likelihood=likelihood_func,
                jitter=1e-4,
            )

            mll = PairwiseLaplaceMarginalLogLikelihood(
                likelihood=model.likelihood, model=model
            )
        else:
            model = TopChoiceGP(datapoints=datapoints, choices=comparisons)
            mll = TopChoiceLaplaceMarginalLogLikelihood(model.likelihood, model)

        fit_gpytorch_mll(mll)
        model = model.to(device=queries.device, dtype=queries.dtype)
    elif model_type == "pairwise_kernel_variational_gp":
        model = PairwiseKernelVariationalGP(queries, responses)
        model.eval()
    elif model_type == "preferential_variational_gp":
        model = VariationalPreferentialGP(queries, responses)
        model.train()
        model.likelihood.train()
        if False:
            #######################
            training_iterations = 400
            # Use the adam optimizer
            train_x = queries.reshape(
                queries.shape[0] * queries.shape[1], queries.shape[2]
            )
            train_y = responses.squeeze(-1)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.2)

            # "Loss" for GPs - the marginal log likelihood
            # num_data refers to the number of training datapoints
            mll = VariationalELBO(model.likelihood, model, 2 * model.num_data)

            for i in range(training_iterations):
                # Zero backpropped gradients from previous iteration
                optimizer.zero_grad()
                # Get predictive output
                output = model(train_x)
                print(model.covar_module.raw_outputscale)
                # print(train_y)
                # print(output.mean)
                # Calc loss and backprop gradients
                loss = -mll(output, train_y)
                loss.backward()
                if True:
                    print(
                        "Iter %d/%d - Loss: %.3f"
                        % (i + 1, training_iterations, loss.item())
                    )
                optimizer.step()
        #############################
        else:
            mll = VariationalELBO(
                likelihood=model.likelihood,
                model=model,
                num_data=2 * model.num_data,
            )
            mll = fit_gpytorch_mll(mll)
            # print(model.covar_module.raw_outputscale)
        # Make sure model and likelihood are in eval mode
        model.eval()
        model.likelihood.eval()
    # print(model.state_dict())
    # train_y = responses.squeeze(-1)
    # print(train_y)
    # print(model.variational_strategy.inducing_points.shape)
    # mean = model.posterior(queries).mean.squeeze()
    # train_x = queries.reshape(queries.shape[0] * queries.shape[1], queries.shape[2])
    # mean = model.posterior(train_x + 1.0).mean.squeeze()
    # mean_diff = mean[..., 0] - mean[..., 1]
    # predicted_pref = torch.where(mean_diff > 0.0, 0, 1)
    # print(train_y - predicted_pref)
    return model


def generate_initial_data(
    num_queries: int,
    num_alternatives: int,
    input_dim: int,
    obj_func,
    noise_type,
    noise_level,
    add_baseline_point: bool,
    seed: int = None,
):
    queries = generate_random_queries(num_queries, num_alternatives, input_dim, seed)
    if add_baseline_point:  # If true, this adds 30 queries including a
        # "high-quality baseline point". The baseline point is hardcoded in generate_queries_against_baseline
        queries_against_baseline = generate_queries_against_baseline(
            30, num_alternatives, input_dim, obj_func, seed
        )
        queries = torch.cat([queries, queries_against_baseline], dim=0)
    obj_vals = get_obj_vals(queries, obj_func)
    responses = generate_responses(obj_vals, noise_type, noise_level)
    return queries, obj_vals, responses


def generate_random_queries(
    num_queries: int, num_alternatives: int, input_dim: int, seed: int = None
):
    # Generate `num_queries` queries each constituted by `num_alternatives` points chosen uniformly at random
    if seed is not None:
        old_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        queries = torch.rand([num_queries, num_alternatives, input_dim])
        torch.random.set_rng_state(old_state)
    else:
        queries = torch.rand([num_queries, num_alternatives, input_dim])
    return queries


def generate_queries_against_baseline(
    num_queries: int, num_alternatives: int, input_dim: int, obj_func, seed: int = None
):
    baseline_point = torch.tensor([0.51] * input_dim)  # This baseline point was meant
    # to be used with the Alpine1 function (with normalized input space) exclusively
    queries = generate_random_queries(
        num_queries, num_alternatives - 1, input_dim, seed + 2
    )
    queries = torch.cat([baseline_point.expand_as(queries), queries], dim=1)
    return queries


def get_obj_vals(queries, obj_func):
    queries_2d = queries.reshape(
        torch.Size([queries.shape[0] * queries.shape[1], queries.shape[2]])
    )
    obj_vals = obj_func(queries_2d)
    obj_vals = obj_vals.reshape(torch.Size([queries.shape[0], queries.shape[1]]))
    return obj_vals


def generate_responses(obj_vals, noise_type, noise_level):
    # Generate simulated comparisons based on true underlying objective
    corrupted_obj_vals = corrupt_obj_vals(obj_vals, noise_type, noise_level)
    responses = torch.argmax(corrupted_obj_vals, dim=-1)
    return responses


def corrupt_obj_vals(obj_vals, noise_type, noise_level):
    # Noise in the decision-maker's responses is simulated by corrupting the objective values
    if noise_type == "noiseless":
        corrupted_obj_vals = obj_vals
    elif noise_type == "probit":
        normal = Normal(torch.tensor(0.0), torch.tensor(noise_level))
        noise = normal.sample(sample_shape=obj_vals.shape)
        corrupted_obj_vals = obj_vals + noise
    elif noise_type == "logit":
        gumbel = Gumbel(torch.tensor(0.0), torch.tensor(noise_level))
        noise = gumbel.sample(sample_shape=obj_vals.shape)
        corrupted_obj_vals = obj_vals + noise
    elif noise_type == "constant":
        corrupted_obj_vals = obj_vals.clone()
        n = obj_vals.shape[0]
        for i in range(n):
            coin_toss = Bernoulli(noise_level).sample().item()
            if coin_toss == 1.0:
                corrupted_obj_vals[i, 0] = obj_vals[i, 1]
                corrupted_obj_vals[i, 1] = obj_vals[i, 0]
    return corrupted_obj_vals


def training_data_for_pairwise_gp(queries, responses):
    num_queries = queries.shape[0]
    num_alternatives = queries.shape[1]
    datapoints = []
    comparisons = []
    for i in range(num_queries):
        best_item_id = num_alternatives * i + responses[i]
        comparison = [best_item_id]
        for j in range(num_alternatives):
            datapoints.append(queries[i, j, :].unsqueeze(0))
            if j != responses[i]:
                comparison.append(num_alternatives * i + j)
        comparisons.append(torch.tensor(comparison).unsqueeze(0))

    datapoints = torch.cat(datapoints, dim=0)
    comparisons = torch.cat(comparisons, dim=0)
    return datapoints, comparisons


def optimize_acqf_and_get_suggested_query(
    acq_func: AcquisitionFunction,
    bounds: Tensor,
    batch_size: int,
    num_restarts: int,
    raw_samples: int,
    batch_initial_conditions: Optional[Tensor] = None,
    batch_limit: Optional[int] = 2,
    init_batch_limit: Optional[int] = 30,
) -> Tensor:
    """Optimizes the acquisition function and returns the (approximate) optimum."""

    candidates, acq_values = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        batch_initial_conditions=batch_initial_conditions,
        options={
            "batch_limit": batch_limit,
            "init_batch_limit": init_batch_limit,
            "maxiter": 100,
            "nonnegative": False,
            "method": "L-BFGS-B",
        },
        return_best_only=False,
    )

    candidates = candidates.detach()
    # acq_values_sorted, indices = torch.sort(acq_values.squeeze(), descending=True)
    # print("Acquisition values:")
    # print(acq_values_sorted)
    # print("Candidates:")
    # print(candidates[indices].squeeze())
    # print(candidates.squeeze())
    # print(candidates.shape)
    # print(acq_values.shape)
    new_x = get_best_candidates(batch_candidates=candidates, batch_values=acq_values)
    # print(new_x)
    return new_x


def get_eubo_init_for_pkg(model, pkg_acqf, bounds, num_restarts, raw_samples):
    aux_num_restarts = int(num_restarts / 4)
    post_mean_func = PosteriorMean(model=model)
    max_post_mean_func = optimize_acqf_and_get_suggested_query(
        acq_func=post_mean_func,
        bounds=bounds,
        batch_size=1,
        num_restarts=aux_num_restarts,
        raw_samples=raw_samples,
        batch_limit=aux_num_restarts,
    )
    max_post_mean_func = max_post_mean_func.squeeze(0)

    emov_acqf = ExpectedUtilityOfBestOption(model=model)

    max_emov_acqf = optimize_acqf_and_get_suggested_query(
        acq_func=emov_acqf,
        bounds=bounds,
        batch_size=2,
        num_restarts=aux_num_restarts,
        raw_samples=raw_samples,
        batch_limit=aux_num_restarts,
    )

    pkg_initial_conditions = gen_batch_initial_conditions(
        acq_function=pkg_acqf,
        bounds=bounds,
        q=4,
        raw_samples=raw_samples,
        num_restarts=aux_num_restarts * 2,
    )

    for i in range(aux_num_restarts):
        pkg_initial_conditions[i, 2, :] = max_post_mean_func.clone()
        pkg_initial_conditions[i, 3, :] = max_post_mean_func.clone()

    pkg_initial_conditions[0, :2, :] = max_emov_acqf
    return pkg_initial_conditions
