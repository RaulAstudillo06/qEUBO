import torch
from botorch.test_functions.base import MultiObjectiveTestProblem
from botorch.test_functions.multi_objective import VehicleSafety
from torch import Tensor


class CarCabDesign(MultiObjectiveTestProblem):
    r"""RE9-7-1 car cab design from Tanabe & Ishibuchi (2020)"""

    dim = 7
    num_objectives = 9
    _bounds = [
        (0.5, 1.5),
        (0.45, 1.35),
        (0.5, 1.5),
        (0.5, 1.5),
        (0.875, 2.625),
        (0.4, 1.2),
        (0.4, 1.2),
    ]
    _ref_point = [0.0, 0.0]  # TODO: Determine proper reference point

    def evaluate_true(self, X: Tensor) -> Tensor:
        f = torch.empty(
            X.shape[:-1] + (self.num_objectives,), dtype=X.dtype, device=X.device
        )

        X1 = X[..., 0]
        X2 = X[..., 1]
        X3 = X[..., 2]
        X4 = X[..., 3]
        X5 = X[..., 4]
        X6 = X[..., 5]
        X7 = X[..., 6]
        # # stochastic variables
        # X8 = 0.006 * (torch.randn_like(X1)) + 0.345
        # X9 = 0.006 * (torch.randn_like(X1)) + 0.192
        # X10 = 10 * (torch.randn_like(X1)) + 0.0
        # X11 = 10 * (torch.randn_like(X1)) + 0.0

        # not using stochastic variables for the real function
        X8 = torch.zeros_like(X1)
        X9 = torch.zeros_like(X1)
        X10 = torch.zeros_like(X1)
        X11 = torch.zeros_like(X1)

        # First function
        # negate the first function as we want minimize car weight
        f[..., 0] = -(
            1.98
            + 4.9 * X1
            + 6.67 * X2
            + 6.98 * X3
            + 4.01 * X4
            + 1.75 * X5
            + 0.00001 * X6
            + 2.73 * X7
        )
        # Second function
        f[..., 1] = 1 - (
            1.16
            - 0.3717 * X2 * X4
            - 0.00931 * X2 * X10
            - 0.484 * X3 * X9
            + 0.01343 * X6 * X10
        )
        # Third function
        f[..., 2] = 0.32 - (
            0.261
            - 0.0159 * X1 * X2
            - 0.188 * X1 * X8
            - 0.019 * X2 * X7
            + 0.0144 * X3 * X5
            + 0.87570001 * X5 * X10
            + 0.08045 * X6 * X9
            + 0.00139 * X8 * X11
            + 0.00001575 * X10 * X11
        )
        # Fourth function
        f[..., 3] = 0.32 - (
            0.214
            + 0.00817 * X5
            - 0.131 * X1 * X8
            - 0.0704 * X1 * X9
            + 0.03099 * X2 * X6
            - 0.018 * X2 * X7
            + 0.0208 * X3 * X8
            + 0.121 * X3 * X9
            - 0.00364 * X5 * X6
            + 0.0007715 * X5 * X10
            - 0.0005354 * X6 * X10
            + 0.00121 * X8 * X11
            + 0.00184 * X9 * X10
            - 0.018 * X2 * X2
        )
        # Fifth function
        f[..., 4] = 0.32 - (
            0.74
            - 0.61 * X2
            - 0.163 * X3 * X8
            + 0.001232 * X3 * X10
            - 0.166 * X7 * X9
            + 0.227 * X2 * X2
        )
        # SiXth function
        tmp = (
            (
                28.98
                + 3.818 * X3
                - 4.2 * X1 * X2
                + 0.0207 * X5 * X10
                + 6.63 * X6 * X9
                - 7.77 * X7 * X8
                + 0.32 * X9 * X10
            )
            + (
                33.86
                + 2.95 * X3
                + 0.1792 * X10
                - 5.057 * X1 * X2
                - 11 * X2 * X8
                - 0.0215 * X5 * X10
                - 9.98 * X7 * X8
                + 22 * X8 * X9
            )
            + (46.36 - 9.9 * X2 - 12.9 * X1 * X8 + 0.1107 * X3 * X10)
        ) / 3
        f[..., 5] = 32 - tmp
        # Seventh function
        f[..., 6] = 32 - (
            4.72
            - 0.5 * X4
            - 0.19 * X2 * X3
            - 0.0122 * X4 * X10
            + 0.009325 * X6 * X10
            + 0.000191 * X11 * X11
        )
        # EighthEighth function
        f[..., 7] = 4 - (
            10.58
            - 0.674 * X1 * X2
            - 1.95 * X2 * X8
            + 0.02054 * X3 * X10
            - 0.0198 * X4 * X10
            + 0.028 * X6 * X10
        )
        # Ninth function
        f[..., 8] = 9.9 - (
            16.45
            - 0.489 * X3 * X7
            - 0.843 * X5 * X6
            + 0.0432 * X9 * X10
            - 0.0556 * X9 * X11
            - 0.000786 * X11 * X11
        )

        Y_bounds = torch.tensor(
            [
                [
                    -4.2150e01,
                    -4.7829e-01,
                    -1.1563e02,
                    -7.2040e-03,
                    -1.8255e-01,
                    -1.0168e01,
                    2.7023e01,
                    -8.0731e00,
                    -6.4556e00,
                ],
                # Old upper bound from 1e8 points
                # [-16.0992,   0.9511, 112.7138,   0.2750,   0.1909,  14.4804,  28.9855, -2.4875, -0.8270],
                # make upper bounds of constraints to be something > 0 so that it's possible to not violate the constraints
                [
                    -16.0992,
                    0.9511,
                    112.7138,
                    0.2750,
                    0.1909,
                    14.4804,
                    28.9855,
                    0.5,
                    0.5,
                ],
            ]
        ).to(f)
        f = (f - Y_bounds[0, :]) / (Y_bounds[1, :] - Y_bounds[0, :])

        # normalize f to between 0 and 1 roughly so that we won't disadvantage ParEGO
        return f


class PiecewiseLinear(torch.nn.Module):
    def __init__(self, beta1, beta2, thresholds):
        super().__init__()
        self.register_buffer("beta1", beta1)
        self.register_buffer("beta2", beta2)
        self.register_buffer("thresholds", thresholds)

    def calc_raw_util_per_dim(self, Y):
        # below thresholds
        bt = Y < self.thresholds
        b1 = self.beta1.expand(Y.shape)
        b2 = self.beta2.expand(Y.shape)
        shift = (b2 - b1) * self.thresholds
        util_val = torch.empty_like(Y)

        # util_val[bt] = Y[bt] * b1[bt]
        util_val[bt] = Y[bt] * b1[bt] + shift[bt]
        util_val[~bt] = Y[~bt] * b2[~bt]

        return util_val

    def forward(self, Y, X=None):
        util_val = self.calc_raw_util_per_dim(Y)
        util_val = util_val.sum(dim=-1)
        return util_val


def car_cab_obj_func(X: Tensor) -> Tensor:
    beta1 = torch.tensor([7.0, 6.75, 6.5, 6.25, 6.0, 5.75, 5.5, 5.25, 5.0])
    beta2 = torch.tensor([0.5, 0.4, 0.375, 0.35, 0.325, 0.3, 0.275, 0.25, 0.225])
    thresholds = torch.tensor([0.55, 0.54, 0.53, 0.52, 0.51, 0.5, 0.49, 0.48, 0.47])

    car_cab = CarCabDesign()
    utility_func = PiecewiseLinear(beta1=beta1, beta2=beta2, thresholds=thresholds)

    return utility_func(car_cab(X))


class NegativeVehicleSafety(VehicleSafety):
    def evaluate_true(self, X: Tensor) -> Tensor:
        f = -super().evaluate_true(X)
        Y_bounds = torch.tensor(
            [
                [-1.7040e03, -1.1708e01, -2.6192e-01],
                [-1.6619e03, -6.2136e00, -4.2879e-02],
            ]
        ).to(X)
        f = (f - Y_bounds[0, :]) / (Y_bounds[1, :] - Y_bounds[0, :])
        return f


def vehicle_safety_obj_func(X: Tensor) -> Tensor:
    beta1 = torch.tensor([2, 6, 8])
    beta2 = torch.tensor([1, 2, 2])
    thresholds = torch.tensor([0.5, 0.8, 0.8])

    vehicle_safety = NegativeVehicleSafety()
    utility_func = PiecewiseLinear(beta1=beta1, beta2=beta2, thresholds=thresholds)

    return utility_func(vehicle_safety(X))
