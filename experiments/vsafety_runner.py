import os
import sys
import torch

from botorch.settings import debug

from torch import Tensor

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
debug._set_state(False)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
print(script_dir[:-12])
sys.path.append(script_dir[:-12])

from experiments.problems import vehicle_safety_obj_func, NegativeVehicleSafety
from src.experiment_manager import experiment_manager
from src.get_noise_level import get_noise_level


# Objective function
input_dim = NegativeVehicleSafety().dim
bounds = torch.tensor(NegativeVehicleSafety()._bounds)


def obj_func(X: Tensor) -> Tensor:
    X_unnorm = (bounds[:, 1] - bounds[:, 0]) * X + bounds[:, 0]
    objective_X = vehicle_safety_obj_func(X_unnorm)
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

if False:
    noise_level = get_noise_level(
        obj_func,
        input_dim,
        target_error=0.1 * float(noise_level_id),
        top_proportion=0.01,
        num_samples=10000000,
        comp_noise_type=comp_noise_type,
    )
    print(noise_level)

if comp_noise_type == "logit":
    noise_level = 0.1318

# Run experiment
if len(sys.argv) == 3:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[2])
elif len(sys.argv) == 2:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[1])

experiment_manager(
    problem="vsafety",
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
