from copy import copy
from matplotlib import pyplot as plt
import numpy as np
import os
import seaborn as sns
import sys

sns.set_style("darkgrid")

problem = "ackley"

if problem == "dropwave":
    budget = 50.0
    delta = 5.0
    n_trials = 30
    algos = ["EI", "B-MS-EI_111"]
elif problem == "alpine1":
    budget = 100.0
    delta = 10.0
    n_trials = 30
    algos = ["EI", "B-MS-EI_111"]
    opt_val = 0.0
    log_regret = False
elif problem == "ackley":
    budget = 100.0
    delta = 10.0
    n_trials = 30
    algos = ["EI", "B-MS-EI_111"]
    opt_val = 0.0
    log_regret = False

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
problem_results_dir = script_dir + "/results/" + problem + "/"

print(problem)
cumulative_costs = np.arange(0.0, budget, delta)

for a, algo in enumerate(algos):
    print(algo)
    if "B-MS-EI" in algo:
        algo_id = algo + "_" + str(int(budget))
    else:
        algo_id = algo

    algo_results_dir = problem_results_dir + algo_id + "/"

    X = np.loadtxt(algo_results_dir + "X/X_1.txt")
    if X.ndim == 1:
        X = np.expand_dims(X, axis=-1)
    input_dim = X.shape[1]
    n_init_evals = 2 * (input_dim + 1)

    best_obs_vals_all_trials = []
    for trial in range(1, n_trials + 1):
        obj_vals = np.loadtxt(algo_results_dir + "Y/Y_" + str(trial) + ".txt")
        costs = np.loadtxt(algo_results_dir + "costs/costs_" + str(trial) + ".txt")
        best_obs_vals = []

        cumulative_cost = 0.0
        i = 0

        for cost in cumulative_costs:
            while cumulative_cost < cost:
                cumulative_cost += costs[i]
                i += 1
            best_obs_vals.append(max(obj_vals[: i + n_init_evals]))

        best_obs_vals_all_trials.append(copy(best_obs_vals))

    best_obs_vals_all_trials = np.asarray(best_obs_vals_all_trials)
    if log_regret:
        best_obs_vals_all_trials = np.log10(opt_val - best_obs_vals_all_trials)

    best_obs_vals_mean = best_obs_vals_all_trials.mean(axis=0)
    best_obs_vals_err = 1.96 * best_obs_vals_all_trials.std(axis=0) / np.sqrt(n_trials)
    plt.errorbar(cumulative_costs, best_obs_vals_mean, best_obs_vals_err, label=algo)

plt.legend()
plt.xlabel("cumulative cost")
if log_regret:
    plt.ylabel("log-regret")
else:
    plt.ylabel("best objective value found")
plt.savefig(problem_results_dir + problem + ".pdf")
