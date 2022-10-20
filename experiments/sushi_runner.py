import os
import sys
import torch

from botorch.settings import debug

from torch import Tensor

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)
debug._set_state(False)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
print(script_dir[:-12])
sys.path.append(script_dir[:-12])

from experiments.evalset.test_funcs import Sushi
from src.experiment_manager import experiment_manager
from src.get_noise_level import get_noise_level


# Objective function
input_dim = 4


def obj_func(X: Tensor) -> Tensor:
    sushi = Sushi()
    objective_X = -torch.tensor(sushi.do_evaluate(X)).squeeze(-1)
    return objective_X


# Algos
algo = "Random"
# algo = "EMOV"
# algo = "EI"
# algo = "NEI"
# algo = "TS"
# algo = "PKG"

# estimate noise level
comp_noise_type = "logit"
noise_level_id = 3

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
    noise_level = 0.0153
elif comp_noise_type == "logit":
    noise_levels = [0.0053, 0.0128, 0.0255]
    noise_level = noise_levels[noise_level_id - 1]


# Run experiment
if len(sys.argv) == 3:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[2])
elif len(sys.argv) == 2:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[1])

experiment_manager(
    problem="sushi",
    obj_func=obj_func,
    input_dim=input_dim,
    comp_noise_type=comp_noise_type,
    comp_noise=noise_level,
    algo=algo,
    batch_size=2,
    num_init_queries=5 * input_dim,
    num_algo_queries=200,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=False,
)
