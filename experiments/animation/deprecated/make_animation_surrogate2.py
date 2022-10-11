import numpy as np
import os
import sys
import torch

from botorch.acquisition import PosteriorMean
from botorch import fit_gpytorch_model

torch.set_default_dtype(torch.float64)


script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
project_path = script_dir[:-12]
sys.path.append(project_path)
data_folder = project_path + "/experiments/animation_data/"

from src.models.pairwise_kernel_variational_gp import PairwiseKernelVariationalGP
from src.utils import optimize_acqf_and_get_suggested_query

datapoints = np.loadtxt(data_folder + "datapoints.txt")
comparisons = np.loadtxt(data_folder + "responses.txt")
datapoints[:, 1] = (datapoints[:, 1] + 2.0) / 4.0
datapoints[:, 2] = datapoints[:, 2] / 4.0
datapoints[:, 3] = (datapoints[:, 3] - 5.0) / 20.0
datapoints[:, 4] = datapoints[:, 4] - 1.0
np.savetxt(data_folder + "datapoints_norm.txt", datapoints)
n_queries = comparisons.shape[0]
queries = []
responses = []

for i in range(n_queries):
    queries.append(datapoints[2 * i : 2 * (i + 1), :])
    responses.append(0 if comparisons[i, 0] < comparisons[i, 1] else 1)

queries = torch.tensor(queries)
responses = torch.tensor(responses)
datapoints = torch.tensor(datapoints)
test_random_points = torch.rand(size=datapoints.size())

save_surrogate = True
if save_surrogate:
    model = PairwiseKernelVariationalGP(queries, responses)
    print(queries[:5, ...])
    print(responses[:5, ...])
    print(model(datapoints).mean[:10, ...])
    print(model(test_random_points).mean[:5, ...])

    torch.save(model.aux_model.state_dict(), "animation_surrogate_state_dict2")

print(model.aux_model.state_dict())
print(e)
aux_model = PairwiseKernelVariationalGP(queries, responses, fit_aux_model_flag=False)
animation_surrogate_state_dict = torch.load("animation_surrogate_state_dict2")
aux_model.aux_model.load_state_dict(animation_surrogate_state_dict)
aux_model.aux_model(aux_model.aux_model.train_inputs[0])
aux_model.aux_model.eval()
print(aux_model(test_random_points).mean[:5, ...])

input_dim = 5
standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])
num_restarts = 6 * input_dim
raw_samples = 180 * input_dim

post_mean_func = PosteriorMean(model=aux_model)
max_post_mean_func = optimize_acqf_and_get_suggested_query(
    acq_func=post_mean_func,
    bounds=standard_bounds,
    batch_size=1,
    num_restarts=num_restarts,
    raw_samples=raw_samples,
)
print(max_post_mean_func)
print(post_mean_func(max_post_mean_func).item())
