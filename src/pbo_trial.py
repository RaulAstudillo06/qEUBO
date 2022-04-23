#!/usr/bin/env python3

from typing import Callable, Dict, Optional

import numpy as np
import os
import sys
import time
import torch
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from torch import Tensor

from src.acquisition_functions.expected_utility import qExpectedUtility
from src.acquisition_functions.thompson_sampling import gen_thompson_sampling_query
from src.utility import MaxObjectiveValue, Probit
from src.utils import (
    fit_model,
    generate_initial_data,
    generate_random_queries,
    get_obj_vals,
    generate_responses,
    optimize_acqf_and_get_suggested_query,
    training_data_for_pairwise_gp,
)


def pbo_trial(
    problem: str,
    obj_func: Callable,
    input_dim: int,
    comp_noise_type: str,
    comp_noise: float,
    algo: str,
    algo_params: Optional[Dict],
    batch_size: int,
    num_init_queries: int,
    num_max_iter: int,
    trial: int,
    restart: bool,
    ignore_failures: bool = False,
) -> None:

    algo_id = algo

    # Get script directory
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    project_path = script_dir[:-11]
    results_folder = (
        project_path + "/experiments/results/" + problem + "/" + algo_id + "/"
    )

    if restart:
        # Check if training data is already available
        try:
            # Current available evaluations
            queries_reshaped = np.loadtxt(
                results_folder + "queries/queries_" + str(trial) + ".txt"
            )
            queries = queries.reshape(
                queries_reshaped[0], batch_size, queries_reshaped[1] / batch_size
            )
            queries = torch.tensor(queries)
            obj_vals = torch.tensor(
                np.loadtxt(results_folder + "obj_vals/obj_vals_" + str(trial) + ".txt")
            )
            responses = torch.tensor(
                np.loadtxt(
                    results_folder + "responses/responses_" + str(trial) + ".txt"
                )
            )
            # Historical best latent objective values and running times
            best_latent_obj_vals = list(
                np.loadtxt(
                    results_folder + "best_latent_obj_vals_" + str(trial) + ".txt"
                )
            )
            runtimes = list(
                np.loadtxt(results_folder + "runtimes/runtimes_" + str(trial) + ".txt")
            )

            # Current best latent objective value and cumulative cost
            best_latent_obj_val = torch.tensor(best_latent_obj_vals[-1])

            iteration = len(best_latent_obj_vals) - 1
            print("Restarting experiment from available data.")

        except:

            # Initial evaluations
            queries, obj_vals, responses = generate_initial_data(
                num_queries=num_init_queries,
                batch_size=batch_size,
                input_dim=input_dim,
                obj_func=obj_func,
                comp_noise_type=comp_noise_type,
                comp_noise=comp_noise,
                seed=trial,
            )

            # Current best latent objective value
            best_latent_obj_val = obj_vals.max().item()

            # Historical best latent objective values and running times
            best_latent_obj_vals = [best_latent_obj_val]
            runtimes = []

            iteration = 0
    else:
        # Initial evaluations
        queries, obj_vals, responses = generate_initial_data(
            num_queries=num_init_queries,
            batch_size=batch_size,
            input_dim=input_dim,
            obj_func=obj_func,
            comp_noise_type=comp_noise_type,
            comp_noise=comp_noise,
            seed=trial,
        )

        # Current best latent objective value and cumulative cost
        best_latent_obj_val = obj_vals.max().item()

        # Historical best latent objective values and runtimes
        best_latent_obj_vals = [best_latent_obj_val]
        runtimes = []

        iteration = 0

    while iteration < num_max_iter:
        iteration += 1
        print("Problem: " + problem)
        print("Sampling policy: " + algo_id)
        print("Trial: " + str(trial))
        print("Iteration: " + str(iteration))

        # New suggested query
        t0 = time.time()
        new_query = get_new_suggested_query(
            algo=algo,
            queries=queries,
            responses=responses,
            batch_size=batch_size,
            input_dim=input_dim,
            algo_params=algo_params,
        )

        t1 = time.time()
        runtimes.append(t1 - t0)

        # Get response at new query
        new_obj_vals = get_obj_vals(new_query, obj_func)
        new_response = generate_responses(
            new_obj_vals, noise_type="noiseless", noise_level=0.01
        )

        # Update training data
        queries = torch.cat((queries, new_query))
        obj_vals = torch.cat([obj_vals, new_obj_vals], 0)
        responses = torch.cat((responses, new_response))

        # Update historical best latent objective values and cumulative cost
        best_latent_obj_val = obj_vals.max().item()
        best_latent_obj_vals.append(best_latent_obj_val)
        print("Best value found so far: " + str(best_latent_obj_val))

        # Save data
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        if not os.path.exists(results_folder + "queries/"):
            os.makedirs(results_folder + "queries/")
        if not os.path.exists(results_folder + "responses/"):
            os.makedirs(results_folder + "responses/")
        if not os.path.exists(results_folder + "runtimes/"):
            os.makedirs(results_folder + "runtimes/")

        queries_reshaped = queries.numpy().reshape(responses.shape[0], -1)
        np.savetxt(
            results_folder + "queries/queries_" + str(trial) + ".txt", queries_reshaped
        )
        np.savetxt(
            results_folder + "responses/responses_" + str(trial) + ".txt",
            responses.numpy(),
        )
        np.savetxt(
            results_folder + "runtimes/runtimes_" + str(trial) + ".txt",
            np.atleast_1d(runtimes),
        )
        np.savetxt(
            results_folder + "best_latent_obj_vals_" + str(trial) + ".txt",
            np.atleast_1d(best_latent_obj_vals),
        )


def get_new_suggested_query(
    algo: str,
    queries: Tensor,
    responses: Tensor,
    batch_size,
    input_dim: int,
    algo_params: Optional[Dict] = None,
) -> Tensor:

    if algo == "Random":
        return generate_random_queries(
            num_queries=1, batch_size=batch_size, input_dim=input_dim
        )
    elif algo == "EMOV":
        datapoints, comparisons = training_data_for_pairwise_gp(queries, responses)
        model = fit_model(datapoints, comparisons)
        sampler = SobolQMCNormalSampler(num_samples=64, collapse_batch_dims=True)
        utility = MaxObjectiveValue()
        acquisition_function = qExpectedUtility(
            model=model, utility=utility, sampler=sampler
        )
    elif algo == "EPOV":
        datapoints, comparisons = training_data_for_pairwise_gp(queries, responses)
        model = fit_model(datapoints, comparisons)
        sampler = SobolQMCNormalSampler(num_samples=64, collapse_batch_dims=True)
        utility = Probit(num_samples=64, batch_size=batch_size, noise_level=1.0)
        acquisition_function = qExpectedUtility(
            model=model, utility=utility, sampler=sampler
        )
    elif algo == "NEI":
        datapoints, comparisons = training_data_for_pairwise_gp(queries, responses)
        model = fit_model(datapoints, comparisons)
        sampler = SobolQMCNormalSampler(num_samples=64, collapse_batch_dims=True)
        acquisition_function = qNoisyExpectedImprovement(
            model=model, X_baseline=datapoints, sampler=sampler
        )
    elif algo == "TS":
        datapoints, comparisons = training_data_for_pairwise_gp(queries, responses)
        model = fit_model(datapoints, comparisons)
        standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])
        return gen_thompson_sampling_query(model, batch_size, standard_bounds)

    standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])

    new_query = optimize_acqf_and_get_suggested_query(
        acq_func=acquisition_function,
        bounds=standard_bounds,
        batch_size=batch_size,
        algo_params=algo_params,
    )

    new_query = new_query.unsqueeze(0)

    return new_query
