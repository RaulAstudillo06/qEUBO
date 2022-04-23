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
from src.get_probit_noise_level import get_probit_noise_level


# Objective function
def obj_func(X: Tensor) -> Tensor:
    hartmann = Hartmann()
    objective_X = -hartmann.evaluate_true(X)
    return objective_X


input_dim = 6

# Algos
# algo = "Random"
# algo = "EMOV"
# algo = "NEI"
# algo = "TS"
algo = "EPOV"

# estimate noise level
if False:
    probit_noise_level = get_probit_noise_level(
        obj_func,
        input_dim,
        target_error=0.1,
        top_proportion=0.1,
        num_samples=1000000,
    )
    print(probit_noise_level)

probit_noise_level = 0.15

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
    comp_noise_type="probit",
    comp_noise=probit_noise_level,
    algo=algo,
    batch_size=2,
    num_init_queries=2 * (input_dim + 1),
    num_max_iter=100,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=False,
)
