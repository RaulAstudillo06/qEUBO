import numpy as np
import os
import sys
import torch

from botorch.models.likelihoods.pairwise import PairwiseLogitLikelihood
from botorch.models.pairwise_gp import PairwiseGP

torch.set_default_dtype(torch.float64)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
project_path = script_dir[:-12]
print(project_path)
sys.path.append(project_path)
data_folder = project_path + "/experiments/animation/data/"

from src.models.preferential_variational_gp import PreferentialVariationalGP


def load_animation_surrogate(surrogate_id):
    datapoints = torch.tensor(np.loadtxt(data_folder + "datapoints_norm.txt"))
    comparisons = torch.tensor(np.loadtxt(data_folder + "responses.txt"))
    n_queries = comparisons.shape[0]
    if surrogate_id == "pvgp":
        queries = []
        responses = []

        for i in range(n_queries):
            queries.append(datapoints[2 * i : 2 * (i + 1), :])
            responses.append(0 if comparisons[i, 0] < comparisons[i, 1] else 1)

        queries = torch.tensor(np.array(queries))
        responses = torch.tensor(np.array(responses))

        surrogate_aux_model = PreferentialVariationalGP(queries, responses)
        animation_surrogate_state_dict = torch.load(
            project_path + "/experiments/animation/animation_surrogate_state_dict_pvgp"
        )
        surrogate_aux_model.load_state_dict(
            animation_surrogate_state_dict, strict=False
        )
        print(surrogate_aux_model(torch.tensor(datapoints)).mean)
        surrogate_aux_model.eval()
        print(surrogate_aux_model(torch.tensor(datapoints)).mean)

    elif surrogate_id == "pairwisegp":
        likelihood_func = PairwiseLogitLikelihood()
        surrogate_aux_model = PairwiseGP(
            datapoints,
            comparisons,
            likelihood=likelihood_func,
            jitter=1e-4,
        )
        animation_surrogate_state_dict = torch.load(
            project_path
            + "/experiments/animation/animation_surrogate_state_dict_pairwisegp"
        )
        # print(animation_surrogate_state_dict)
        surrogate_aux_model.load_state_dict(
            animation_surrogate_state_dict, strict=False
        )
        surrogate_aux_model.load_state_dict(animation_surrogate_state_dict)
        surrogate_aux_model(datapoints)
        surrogate_aux_model.eval()
        print(surrogate_aux_model(torch.rand(size=datapoints.size())).mean[:5, ...])
    return surrogate_aux_model
