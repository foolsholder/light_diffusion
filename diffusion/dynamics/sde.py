import torch

from typing import Dict, List, Tuple, Union


class SDE:
    def __init__(self, N: int, ode_sampling: bool = False):
        self.N = N
        self.ode_sampling = ode_sampling

    def sde(self, x, t) -> Dict[str, torch.Tensor]:
        raise "Not impl yet"

    @property
    def T(self):
        return 1

    def prior_sampling(self, shape) -> torch.Tensor:
        raise "Not impl yet"

    def marginal_forward(self, x, t) -> Dict[str, torch.Tensor]:
        raise "Not impl yet"

    def calc_score(self, model, x_t, t, attn_mask=None) -> Dict[str, torch.Tensor]:
        raise "Not impl yet"


class RSDE:
    def __init__(self, sde: SDE):
        self.N = sde.N
        self.sde_obj = sde
        self.ode_sampling = sde.ode_sampling

    @property
    def T(self):
        return self.sde.T

    def sde(self, model, x_t, t, attn_mask) -> Dict[str, torch.Tensor]:
        """Create the drift and diffusion functions for the reverse SDE/ODE.
        SDE:
            dx = (-1/2 * beta * x_t - beta * score) * dt + sqrt(beta) * dw
        ODE:
            dx = (-1/2 * beta * x_t - 1/2 * beta * score) * dt
        """
        sde_params = self.sde_obj.sde(x_t, t)
        drift_par, diffusion = sde_params['drift'], sde_params['diffusion']  # -1/2 * beta * x_t, sqrt(beta)

        scores = self.sde_obj.calc_score(model, x_t, t, attn_mask=attn_mask)
        score = scores['score']
        drift = drift_par - diffusion[:, None, None] ** 2 * score * (0.5 if self.ode_sampling else 1.)
        # Set the diffusion function to zero for ODEs.
        diffusion = 0. if self.ode_sampling else diffusion
        return {
            "score": score,
            "drift": drift,
            "drift_par": drift_par,
            "diffusion": diffusion,
            "x_0": scores["x_0"]
        }
