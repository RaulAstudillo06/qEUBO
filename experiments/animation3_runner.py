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
data_folder = project_path + "/experiments/animation_data/"

from src.experiment_manager import experiment_manager
from src.get_noise_level import get_noise_level


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
# algo = "Random"
algo = "EMOV"
# algo = "EI"
# algo = "NEI"
# algo = "TS"
# algo = "PKG"

# estimate noise level
comp_noise_type = "logit"
noise_level_id = 2

if False:
    noise_level = get_noise_level(
        obj_func,
        input_dim,
        target_error=0.1 * float(noise_level_id),
        top_proportion=0.01,
        num_samples=10000,
        comp_noise_type=comp_noise_type,
    )
    print(noise_level)

if comp_noise_type == "probit":
    noise_level = 0.0
elif comp_noise_type == "logit":
    noise_level = 0.3051

# Run experiment
if len(sys.argv) == 3:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[2])
elif len(sys.argv) == 2:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[1])

experiment_manager(
    problem="animation3",
    obj_func=obj_func,
    input_dim=input_dim,
    comp_noise_type=comp_noise_type,
    comp_noise=noise_level,
    algo=algo,
    batch_size=2,
    num_init_queries=5 * input_dim,
    num_algo_queries=250,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=False,
)
