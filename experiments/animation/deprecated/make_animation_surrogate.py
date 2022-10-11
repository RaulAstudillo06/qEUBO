import numpy as np
import os
import sys
import torch

from botorch.acquisition import PosteriorMean
from botorch import fit_gpytorch_model

# from scipy.optimize import minimize

torch.set_default_dtype(torch.float64)


script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
project_path = script_dir[:-12]
sys.path.append(project_path)
data_folder = project_path + "/experiments/animation_data/"

from src.models.likelihoods.pairwise import PairwiseLogitLikelihood
from src.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from src.utils import optimize_acqf_and_get_suggested_query

datapoints = np.loadtxt(data_folder + "datapoints.txt")
comparisons = torch.tensor(np.loadtxt(data_folder + "responses.txt"))
datapoints[:, 1] = (datapoints[:, 1] + 2.0) / 4.0
datapoints[:, 2] = datapoints[:, 2] / 4.0
datapoints[:, 3] = (datapoints[:, 3] - 5.0) / 20.0
datapoints[:, 4] = datapoints[:, 4] - 1.0
np.savetxt(data_folder + "datapoints_norm.txt", datapoints)
datapoints = torch.tensor(datapoints)

save_model = False

if save_model:
    likelihood_func = PairwiseLogitLikelihood()
    model = PairwiseGP(
        datapoints,
        comparisons,
        likelihood=likelihood_func,
        jitter=1e-4,
    )
    mll = PairwiseLaplaceMarginalLogLikelihood(likelihood=model.likelihood, model=model)
    fit_gpytorch_model(mll)
    print(datapoints[:5, ...])
    print(comparisons[:5, ...])
    print(model(datapoints + 1.0).mean[:5, ...])

    torch.save(model.state_dict(), "animation_surrogate_state_dict")

likelihood_func = PairwiseLogitLikelihood()
aux_model = PairwiseGP(
    datapoints,
    comparisons,
    likelihood=likelihood_func,
    jitter=1e-4,
)

animation_surrogate_state_dict = torch.load("animation_surrogate_state_dict")
print(animation_surrogate_state_dict)
aux_model.load_state_dict(animation_surrogate_state_dict)
aux_model(datapoints)
aux_model.eval()
print(aux_model(torch.rand(size=datapoints.size())).mean[:5, ...])

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
