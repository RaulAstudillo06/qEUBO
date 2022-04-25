import numpy as np
import os
import sys

problem = "levy"
show_incomplete_trials = False

if problem == "levy":
    n_iter = 50
    n_trials = 30
    algos = ["EPOV", "NEI"]
elif problem == "hartmann":
    n_iter = 100
    n_trials = 30
    algos = ["EPOV", "NEI"]

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
problem_results_dir = script_dir + "/results/" + problem + "/"

for a, algo in enumerate(algos):
    print(algo)
    algo_results_dir = problem_results_dir + algo + "/runtimes/"

    for trial in range(1, n_trials + 1):
        try:
            runtimes = np.loadtxt(algo_results_dir + "runtimes_" + str(trial) + ".txt")
            n_completed_iter = len(runtimes)
            if n_completed_iter < n_iter and show_incomplete_trials:
                print("Trial {} is not complete yet.".format(trial))
                print("Number of completed iterations is: {}".format(n_completed_iter))
        except:
            print("Trial {} was not found.".format(trial))
