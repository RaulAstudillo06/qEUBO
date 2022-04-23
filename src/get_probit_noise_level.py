from distutils.log import error
import torch

from scipy.optimize import minimize


def get_probit_noise_level(
    obj_func, input_dim, target_error, top_proportion, num_samples
):
    X = torch.rand([num_samples, input_dim])
    Y = obj_func(X)
    target_Y = Y.sort().values[-int(num_samples * top_proportion) :]
    target_Y = target_Y[torch.randperm(target_Y.shape[0])]
    target_Y = target_Y.reshape(-1, 2)

    print(target_Y.min())
    print(target_Y.max())

    # estimate probit error
    true_comps = target_Y[:, 0] >= target_Y[:, 1]

    res = minimize(error_rate_loss, x0=0.01, args=(target_Y, true_comps, target_error))
    print(res)

    probit_noise = res.x[0]
    probit_noise = round(probit_noise, 4)

    error_rate = estimate_error_rate(probit_noise, target_Y, true_comps)
    print(error_rate)
    return probit_noise


def estimate_error_rate(x, obj_vals, true_comps):
    std_norm = torch.distributions.normal.Normal(0, 1)
    choose_0_prob = std_norm.cdf((obj_vals[:, 0] - obj_vals[:, 1]) / x)
    correct_prob = torch.cat(
        (choose_0_prob[true_comps], 1 - choose_0_prob[~true_comps])
    )
    error_rate = 1 - correct_prob.mean()
    return error_rate.item()


def error_rate_loss(x, obj_vals, true_comps, target_error):
    return abs(estimate_error_rate(x, obj_vals, true_comps) - target_error)
