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


# Objective and cost functions
def obj_func(X: Tensor) -> Tensor:
    X_unnorm = (20.0 * X) - 10.0
    levy = Levy(dim=2)
    objective_X = -levy.evaluate_true(X_unnorm)
    return objective_X


# Algos
# algo = "Random"
# algo = "EMOV"
# algo = "NEI"
algo = "TS"

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
    input_dim=2,
    algo=algo,
    batch_size=2,
    num_init_queries=6,
    num_max_iter=50,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=False,
)
