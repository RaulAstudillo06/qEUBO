import numpy as np
import os
import sys
import torch

from botorch.settings import debug

from torch import Tensor

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)
debug._set_state(False)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
project_path = script_dir[:-12]
sys.path.append(project_path)
data_folder = project_path + "/experiments/animation_data/"

from src.experiment_manager import experiment_manager
from src.get_noise_level import get_noise_level
from src.models.likelihoods.pairwise import PairwiseLogitLikelihood
from src.models.pairwise_gp import PairwiseGP


# Objective function
input_dim = 5

datapoints = torch.tensor(np.loadtxt(data_folder + "datapoints_norm.txt"))
comparisons = torch.tensor(np.loadtxt(data_folder + "responses.txt"))
likelihood_func = PairwiseLogitLikelihood()
aux_model = PairwiseGP(
    datapoints,
    comparisons,
    likelihood=likelihood_func,
    jitter=1e-4,
)
animation_surrogate_state_dict = torch.load("animation_surrogate_state_dict")
aux_model.load_state_dict(animation_surrogate_state_dict)
aux_model(datapoints)
aux_model.eval()

N = 10000


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
# algo = "Random"
# algo = "EMOV"
algo = "EI"
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
        num_samples=100000,
        comp_noise_type=comp_noise_type,
    )
    print(noise_level)

if comp_noise_type == "probit":
    noise_level = 0.0
elif comp_noise_type == "logit":
    noise_level = 0.0440

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
    num_init_queries=5 * input_dim,
    num_algo_queries=250,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=False,
)
