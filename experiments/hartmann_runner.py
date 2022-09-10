import os
import sys
import torch

from botorch.settings import debug
from botorch.test_functions.synthetic import Hartmann

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
    hartmann = Hartmann()
    objective_X = -hartmann.evaluate_true(X)
    return objective_X


input_dim = 6

# Algos
# algo = "Random"
# algo = "EMOV"
algo = "EI"
# algo = "NEI"
# algo = "TS"

# estimate noise level
comp_noise_type = "probit"
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

if comp_noise_type == "probit":
    noise_levels = [0.0817, 0.1928, 0.3768]
elif comp_noise_type == "logit":
    noise_levels = [0.0673, 0.1619, 0.3242]

noise_level = noise_levels[noise_level_id - 1]

# Run experiment
if len(sys.argv) == 3:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[2])
elif len(sys.argv) == 2:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[1])

experiment_manager(
    problem="hartmann",
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
