import os
import sys
import torch

from botorch.settings import debug
from botorch.test_functions.synthetic import Levy

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
def obj_func(X: Tensor) -> Tensor:
    X_unnorm = (20.0 * X) - 10.0
    levy = Levy(dim=2)
    objective_X = -levy.evaluate_true(X_unnorm)
    return objective_X


input_dim = 2

# Algos
# algo = "Random"
# algo = "EMOV"
algo = "EMOV"
# algo = "NEI"
# algo = "TS"

# estimate noise level
comp_noise_type = "logit"
noise_level_id = 2

if False:
    noise_level = get_noise_level(
        obj_func,
        input_dim,
        target_error=0.1 * float(noise_level_id),
        top_proportion=0.01,
        num_samples=1000000,
        comp_noise_type=comp_noise_type,
    )
    print(noise_level)

if comp_noise_type == "probit":
    noise_levels = [0.0318, 0.0746, 0.1385]
elif comp_noise_type == "logit":
    noise_levels = [0.0259, 0.0607, 0.1216]

noise_level = noise_levels[noise_level_id - 1]

# Run experiment
if len(sys.argv) == 3:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[2])
elif len(sys.argv) == 2:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[1])

experiment_manager(
    problem="levy",
    obj_func=obj_func,
    input_dim=input_dim,
    comp_noise_type=comp_noise_type,
    comp_noise=noise_level,
    algo=algo,
    batch_size=2,
    num_init_queries=2 * (input_dim + 1),
    num_max_iter=200,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=False,
)
