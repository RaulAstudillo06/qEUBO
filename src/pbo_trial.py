#!/usr/bin/env python3

from typing import Callable, Dict, Optional

import numpy as np
import os
import sys
import time
import torch
from botorch.acquisition import (
    PosteriorMean,
    qExpectedImprovement,
    qNoisyExpectedImprovement,
)
from botorch.models.model import Model
from botorch.sampling.samplers import SobolQMCNormalSampler
from torch import Tensor

from src.acquisition_functions.emov import (
    ExpectedMaxObjectiveValue,
    qExpectedMaxObjectiveValue,
)
from src.acquisition_functions.preferential_knowledge_gradient import (
    PreferentialKnowledgeGradient,
)
from src.acquisition_functions.thompson_sampling import gen_thompson_sampling_query
from src.utils import (
    fit_model,
    generate_initial_data,
    generate_random_queries,
    get_eubo_init_for_pkg,
    get_obj_vals,
    generate_responses,
    optimize_acqf_and_get_suggested_query,
)


def pbo_trial(
    problem: str,
    obj_func: Callable,
    input_dim: int,
    comp_noise_type: str,
    comp_noise: float,
    algo: str,
    batch_size: int,
    num_init_queries: int,
    num_algo_queries: int,
    trial: int,
    restart: bool,
    model_type: str,
    add_baseline_point: bool,
    ignore_failures: bool,
    algo_params: Optional[Dict] = None,
) -> None:

    algo_id = algo + "_" + str(batch_size)

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
            queries = np.loadtxt(
                results_folder + "queries/queries_" + str(trial) + ".txt"
            )
            queries = queries.reshape(
                queries.shape[0], batch_size, int(queries.shape[1] / batch_size)
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
            # Historical max objective within queries
            max_obj_vals_within_queries = list(
                np.loadtxt(
                    results_folder
                    + "max_obj_vals_within_queries_"
                    + str(trial)
                    + ".txt"
                )
            )
            # Historical objective values at the maximum of the posterior mean
            obj_vals_at_max_post_mean = list(
                np.loadtxt(
                    results_folder + "obj_vals_at_max_post_mean_" + str(trial) + ".txt"
                )
            )
            # Historical acquisition runtimes
            runtimes = list(
                np.atleast_1d(
                    np.loadtxt(
                        results_folder + "runtimes/runtimes_" + str(trial) + ".txt"
                    )
                )
            )

            # Fit GP model
            t0 = time.time()
            model = fit_model(
                queries,
                responses,
                model_type=model_type,
                likelihood=comp_noise_type,
            )
            t1 = time.time()
            model_training_time = t1 - t0

            iteration = len(max_obj_vals_within_queries) - 1
            print("Restarting experiment from available data.")

        except:
            # Initial data
            queries, obj_vals, responses = generate_initial_data(
                num_queries=num_init_queries,
                batch_size=batch_size,
                input_dim=input_dim,
                obj_func=obj_func,
                comp_noise_type=comp_noise_type,
                comp_noise=comp_noise,
                add_baseline_point=add_baseline_point,
                seed=trial,
            )

            # Fit GP model
            t0 = time.time()
            model = fit_model(
                queries,
                responses,
                model_type=model_type,
                likelihood=comp_noise_type,
            )
            t1 = time.time()
            model_training_time = t1 - t0

            # Historical objective values at the maximum of the posterior mean
            obj_val_at_max_post_mean = compute_obj_val_at_max_post_mean(
                obj_func, model, input_dim
            )
            obj_vals_at_max_post_mean = [obj_val_at_max_post_mean]

            # Historical max objective values within queries and runtimes
            max_obj_val_within_queries = obj_vals.max().item()
            max_obj_vals_within_queries = [max_obj_val_within_queries]

            # Historical acquisition runtimes
            runtimes = []

            iteration = 0
    else:
        # Initial data
        queries, obj_vals, responses = generate_initial_data(
            num_queries=num_init_queries,
            batch_size=batch_size,
            input_dim=input_dim,
            obj_func=obj_func,
            comp_noise_type=comp_noise_type,
            comp_noise=comp_noise,
            add_baseline_point=add_baseline_point,
            seed=trial,
        )

        # Fit GP model
        t0 = time.time()
        model = fit_model(
            queries,
            responses,
            model_type=model_type,
            likelihood=comp_noise_type,
        )
        t1 = time.time()
        model_training_time = t1 - t0

        # Historical objective values at the maximum of the posterior mean
        obj_val_at_max_post_mean = compute_obj_val_at_max_post_mean(
            obj_func, model, input_dim
        )
        obj_vals_at_max_post_mean = [obj_val_at_max_post_mean]

        # Historical max objective values within queries and runtimes
        max_obj_val_within_queries = obj_vals.max().item()
        max_obj_vals_within_queries = [max_obj_val_within_queries]

        # Historical acquisition runtimes
        runtimes = []

        iteration = 0

    while iteration < num_algo_queries:
        iteration += 1
        print("Problem: " + problem)
        print("Sampling policy: " + algo_id)
        print("Trial: " + str(trial))
        print("Iteration: " + str(iteration))

        # New suggested query
        t0 = time.time()
        new_query = get_new_suggested_query(
            algo=algo,
            model=model,
            batch_size=batch_size,
            input_dim=input_dim,
            algo_params=algo_params,
            comp_noise=comp_noise,
            model_type=model_type,
        )
        t1 = time.time()
        acquisition_time = t1 - t0
        runtimes.append(acquisition_time + model_training_time)

        # Get response at new query
        new_obj_vals = get_obj_vals(new_query, obj_func)
        new_response = generate_responses(
            new_obj_vals, noise_type=comp_noise_type, noise_level=comp_noise
        )

        # Update training data
        queries = torch.cat((queries, new_query))
        obj_vals = torch.cat([obj_vals, new_obj_vals], 0)
        responses = torch.cat((responses, new_response))

        # Fit GP model
        t0 = time.time()
        model = fit_model(
            queries,
            responses,
            model_type=model_type,
            likelihood=comp_noise_type,
        )
        # lambd = 1.0 / model.covar_module.outputscale.item()
        # print("Current estimate of lambda: " + str(lambd))
        t1 = time.time()
        model_training_time = t1 - t0

        # Append current objective value at the maximum of the posterior mean
        obj_val_at_max_post_mean = compute_obj_val_at_max_post_mean(
            obj_func, model, input_dim
        )
        obj_vals_at_max_post_mean.append(obj_val_at_max_post_mean)
        print(
            "Objective value at the maximum of the posterior mean: "
            + str(obj_val_at_max_post_mean)
        )

        # Append current max objective val within queries
        max_obj_val_within_queries = obj_vals.max().item()
        max_obj_vals_within_queries.append(max_obj_val_within_queries)
        print("Max objecive value within queries: " + str(max_obj_val_within_queries))

        # Save data
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        if not os.path.exists(results_folder + "queries/"):
            os.makedirs(results_folder + "queries/")
        if not os.path.exists(results_folder + "obj_vals/"):
            os.makedirs(results_folder + "obj_vals/")
        if not os.path.exists(results_folder + "responses/"):
            os.makedirs(results_folder + "responses/")
        if not os.path.exists(results_folder + "runtimes/"):
            os.makedirs(results_folder + "runtimes/")

        queries_reshaped = queries.numpy().reshape(queries.shape[0], -1)
        np.savetxt(
            results_folder + "queries/queries_" + str(trial) + ".txt", queries_reshaped
        )
        np.savetxt(
            results_folder + "obj_vals/obj_vals_" + str(trial) + ".txt",
            obj_vals.numpy(),
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
            results_folder + "obj_vals_at_max_post_mean_" + str(trial) + ".txt",
            np.atleast_1d(obj_vals_at_max_post_mean),
        )
        np.savetxt(
            results_folder + "max_obj_vals_within_queries_" + str(trial) + ".txt",
            np.atleast_1d(max_obj_vals_within_queries),
        )


def get_new_suggested_query(
    algo: str,
    model: Model,
    batch_size,
    input_dim: int,
    comp_noise: float,
    model_type: str,
    algo_params: Optional[Dict] = None,
) -> Tensor:

    standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])
    num_restarts = 2 * input_dim * batch_size
    raw_samples = 60 * input_dim * batch_size
    batch_initial_conditions = None

    if algo == "Random":
        return generate_random_queries(
            num_queries=1, batch_size=batch_size, input_dim=input_dim
        )
    elif algo == "ANALYT_EMOV":
        acquisition_function = ExpectedMaxObjectiveValue(model=model)
    elif algo == "EMOV":
        acquisition_function = qExpectedMaxObjectiveValue(model=model)
    elif algo == "PKG":
        acquisition_function = PreferentialKnowledgeGradient(model=model)
    elif algo == "PKG_EUBO_INIT":
        acquisition_function = PreferentialKnowledgeGradient(model=model)
        batch_initial_conditions = get_eubo_init_for_pkg(
            model=model,
            pkg_acqf=acquisition_function,
            bounds=standard_bounds,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
    elif algo == "EI":
        sampler = SobolQMCNormalSampler(num_samples=64, collapse_batch_dims=True)
        if model_type == "pairwise_gp":
            X_baseline = model.datapoints.clone()
        elif model_type == "pairwise_kernel_variational_gp":
            X_baseline = model.queries.clone()
            X_baseline = X_baseline.view(
                (X_baseline.shape[0] * X_baseline.shape[1], X_baseline.shape[2])
            )
        posterior = model.posterior(X_baseline)
        mean = posterior.mean

        acquisition_function = qExpectedImprovement(
            model=model,
            best_f=mean.max().item(),
            sampler=sampler,
        )
    elif algo == "NEI":
        sampler = SobolQMCNormalSampler(num_samples=64, collapse_batch_dims=True)
        if model_type == "pairwise_gp":
            X_baseline = model.datapoints.clone()
        elif model_type == "pairwise_kernel_variational_gp":
            X_baseline = model.queries.clone()
            X_baseline = X_baseline.view(
                (X_baseline.shape[0] * X_baseline.shape[1], X_baseline.shape[2])
            )
        acquisition_function = qNoisyExpectedImprovement(
            model=model,
            X_baseline=X_baseline,
            sampler=sampler,
            prune_baseline=True,
        )
    elif algo == "TS":
        standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])
        return gen_thompson_sampling_query(
            model, batch_size, standard_bounds, 2 * input_dim, 60 * input_dim
        )

    new_query = optimize_acqf_and_get_suggested_query(
        acq_func=acquisition_function,
        bounds=standard_bounds,
        batch_size=4 if algo == "PKG" else batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        batch_initial_conditions=batch_initial_conditions,
    )

    new_query = new_query.unsqueeze(0)
    return new_query


def compute_obj_val_at_max_post_mean(
    obj_func: Callable,
    model: Model,
    input_dim: int,
) -> Tensor:

    standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])
    num_restarts = 6 * input_dim
    raw_samples = 180 * input_dim

    post_mean_func = PosteriorMean(model=model)
    max_post_mean_func = optimize_acqf_and_get_suggested_query(
        acq_func=post_mean_func,
        bounds=standard_bounds,
        batch_size=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )

    obj_val_at_max_post_mean_func = obj_func(max_post_mean_func).item()
    return obj_val_at_max_post_mean_func
