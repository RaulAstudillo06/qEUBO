import torch

from botorch.models.gpytorch import GPyTorchModel
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import SoftmaxLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from torch import Tensor

#from src.models.kernels.scale_kernel import ScaleKernel

class PreferentialVariationalGP(GPyTorchModel, ApproximateGP):
    def __init__(
        self, 
        queries: Tensor,
        responses: Tensor,
    ) -> None:
        self.queries = queries
        self.responses = responses
        self.input_dim = queries.shape[-1]
        self.q = queries.shape[-2]
        self.num_data = queries.shape[-3]
        train_x = queries.flatten(start_dim=-3, end_dim=-2)
        train_y = responses.squeeze(-1)
        bounds = torch.tensor(
            [[0, 1] for _ in range(self.input_dim)], dtype=torch.double
        ).T
        inducing_points = draw_sobol_samples(
            bounds=bounds, n=2 ** (self.input_dim + 2), q=1, seed=0
        ).squeeze(1)
        inducing_points = torch.cat([inducing_points, train_x], dim=0)
        # Construct variational dist/strat
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(-2)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=False,
        )
        super().__init__(variational_strategy)
        self.likelihood = SoftmaxLikelihood(num_features=self.q, num_classes=self.q, mixing_weights=False)
        # Mean and cov
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=self.input_dim, lengthscale_prior=GammaPrior(3.0, 6.0), outputscale_prior=GammaPrior(2.0, 0.15)))
        self.train_inputs = (queries,)
        self.train_targets = train_y

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        return 1