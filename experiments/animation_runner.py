#!/usr/bin/env python3

import numpy as np
from sklearn import svm
from sklearn.gaussian_process.kernels import RBF
import os
import sys
import torch
from botorch.settings import debug
from torch import Tensor

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
debug._set_state(False)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
project_path = script_dir[:-12]
sys.path.append(project_path)
data_folder = project_path + "/experiments/animation/data/"

from src.experiment_manager import experiment_manager


# Objective function
input_dim = 5
datapoints = np.loadtxt(data_folder + "datapoints_norm.txt")
comparisons = np.loadtxt(data_folder + "responses.txt")
n_queries = comparisons.shape[0]

K = RBF(length_scale=0.2)


def my_kernel(x, y):
    a = x[:5].reshape(1, -1)
    b = x[5:].reshape(1, -1)
    c = y[:5].reshape(1, -1)
    d = y[5:].reshape(1, -1)
    return K(a, c) - K(a, d) - K(b, c) + K(b, d)


def proxy_kernel(X, Y):
    """Another function to return the gram_matrix,
    which is needed in SVC's kernel or fit
    """
    gram_matrix = np.zeros((len(X), len(Y)))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            gram_matrix[i, j] = my_kernel(x, y)
    return gram_matrix


inputs = []
labels = []
for i in range(n_queries):
    inputs.append(datapoints[2 * i : 2 * (i + 1), :].flatten())
    labels.append(0 if comparisons[i, 0] < comparisons[i, 1] else 1)

aux_model = svm.SVC(kernel=proxy_kernel)
aux_model.fit(inputs, labels)

print(
    aux_model.decision_function(np.asarray([0.1 * i for i in range(10)]).reshape(1, -1))
)

N = 1000


def obj_func(X: Tensor) -> Tensor:
    reference_point = torch.zeros(size=X.size()) + 0.5
    X_aux = torch.cat([X, reference_point], dim=-1)
    if X.shape[0] > N:
        n_batches = int(X.shape[0] / N)
        objective_X = []
        for i in range(n_batches):
            objective_X.append(
                torch.tensor(
                    aux_model.decision_function(X_aux[i * N : (i + 1) * N, ...].numpy())
                )
            )
        objective_X = torch.cat(objective_X, dim=0)
    else:
        objective_X = torch.tensor(aux_model.decision_function(X_aux.numpy()))
    objective_X = 10 * objective_X
    return objective_X


# Algos
# algo = "random"
# algo = "analytic_eubo"
algo = "qeubo"
# algo = "qei"
# algo = "qnei"
# algo = "qts"
# algo = "mpes"

# Noise level
noise_type = "logit"
noise_level_id = 2

if noise_type == "logit":
    noise_levels = [0.1916, 0.3051, 0.9254]

noise_level = noise_levels[noise_level_id - 1]

# Run experiment
if len(sys.argv) == 3:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[2])
elif len(sys.argv) == 2:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[1])

experiment_manager(
    problem="animation",
    obj_func=obj_func,
    input_dim=input_dim,
    noise_type=noise_type,
    noise_level=noise_level,
    algo=algo,
    num_alternatives=2,
    num_init_queries=4 * input_dim,
    num_algo_queries=150,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=False,
)
