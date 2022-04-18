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


# Objective and cost functions
def obj_func(X: Tensor) -> Tensor:
    hartmann = Hartmann()
    objective_X = -hartmann.evaluate_true(X)
    return objective_X


# Algos
# algo = "Random"
algo = "EMOV"
# algo = "NEI"
# algo = "TS"

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
    input_dim=6,
    algo=algo,
    batch_size=2,
    num_init_queries=14,
    num_max_iter=100,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=False,
)
