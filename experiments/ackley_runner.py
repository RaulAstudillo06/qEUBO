import os
import sys
import torch

from botorch.settings import debug
from botorch.test_functions.synthetic import Ackley

from torch import Tensor

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)
debug._set_state(False)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
print(script_dir[:-12])
sys.path.append(script_dir[:-12])

from src.experiment_manager import experiment_manager
from src.get_noise_level import get_noise_level


# Objective function
input_dim = 5


def obj_func(X: Tensor) -> Tensor:
    X_unnorm = (4.0 * X) - 2.0
    ackley = Ackley(dim=input_dim)
    objective_X = -ackley.evaluate_true(X_unnorm)
    return objective_X


# Algos
# algo = "Random"
algo = "EMOV"
# algo = "EI"
# algo = "NEI"
# algo = "TS"
# algo = "PKG"

# estimate noise level
comp_noise_type = "probit"

if False:
    noise_level = get_noise_level(
        obj_func,
        input_dim,
        target_error=0.2,
        top_proportion=0.01,
        num_samples=10000000,
        comp_noise_type=comp_noise_type,
    )
    print(noise_level)

if comp_noise_type == "probit":
    noise_level = 0.1872
elif comp_noise_type == "logit":
    noise_level = 0.1569

# Run experiment
if len(sys.argv) == 3:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[2])
elif len(sys.argv) == 2:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[1])

experiment_manager(
    problem="ackley",
    obj_func=obj_func,
    input_dim=input_dim,
    comp_noise_type=comp_noise_type,
    comp_noise=noise_level,
    algo=algo,
    batch_size=2,
    num_init_queries=2 * (input_dim + 1),
    num_max_iter=150,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=True,
)
