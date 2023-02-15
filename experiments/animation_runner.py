import numpy as np
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
from src.get_noise_level import get_noise_level
from src.models.preferential_variational_gp import PreferentialVariationalGP


# Objective function
input_dim = 5

datapoints = np.loadtxt(data_folder + "datapoints_norm.txt")
comparisons = np.loadtxt(data_folder + "responses.txt")
n_queries = comparisons.shape[0]
queries = []
responses = []

for i in range(n_queries):
    queries.append(datapoints[2 * i : 2 * (i + 1), :])
    responses.append(0 if comparisons[i, 0] < comparisons[i, 1] else 1)

queries = torch.tensor(np.array(queries))
responses = torch.tensor(np.array(responses))

aux_model = PreferentialVariationalGP(queries, responses)
animation_surrogate_state_dict = torch.load("animation_surrogate_state_dict")
aux_model.load_state_dict(animation_surrogate_state_dict, strict=False)
print(aux_model(torch.tensor(datapoints)).mean)
aux_model.eval()
print(aux_model(torch.tensor(datapoints)).mean)

N = 1000


def obj_func(X: Tensor) -> Tensor:
    if X.shape[0] > N:
        n_batches = int(X.shape[0] / N)
        objective_X = []
        for i in range(n_batches):
            objective_X.append(aux_model(X[i * N : (i + 1) * N, ...]).mean.detach())
        objective_X = torch.cat(objective_X, dim=0)
    else:
        objective_X = aux_model(X).mean.detach()
    return objective_X


# Algos
# algo = "random"
# algo = "analytic_eubo"
algo = "eubo"
# algo = "ei"
# algo = "nei"
# algo = "ts"

# estimate noise level
comp_noise_type = "logit"
noise_level_id = 2

if True:
    noise_level = get_noise_level(
        obj_func,
        input_dim,
        target_error=0.1 * float(noise_level_id),
        top_proportion=0.01,
        num_samples=10000,
        comp_noise_type=comp_noise_type,
    )
    print(noise_level)

if comp_noise_type == "logit":
    # noise_levels = #[0.1916, 0.3051, 0.9254]
    # noise_level = noise_levels[noise_level_id - 1]
    noise_level = 0.1449  # c2 0.0863  #c0 0.0529

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
    comp_noise_type=comp_noise_type,
    comp_noise=noise_level,
    algo=algo,
    batch_size=2,
    num_init_queries=4 * input_dim,
    num_algo_queries=200,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=True,
)
