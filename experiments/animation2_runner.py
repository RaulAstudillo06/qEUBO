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
from load_animation_surrogate import load_animation_surrogate


# Objective function
input_dim = 5
surrogate_model = load_animation_surrogate("pairwisegp")
N = 1000


def obj_func(X: Tensor) -> Tensor:
    if X.shape[0] > N:
        n_batches = int(X.shape[0] / N)
        objective_X = []
        for i in range(n_batches):
            objective_X.append(
                surrogate_model(X[i * N : (i + 1) * N, ...]).mean.detach()
            )
        objective_X = torch.cat(objective_X, dim=0)
    else:
        objective_X = surrogate_model(X).mean.detach()
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
    noise_level = 0.0901

# Run experiment
if len(sys.argv) == 3:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[2])
elif len(sys.argv) == 2:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[1])

experiment_manager(
    problem="animation2",
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
