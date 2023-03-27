import torch
import numpy as np

from typing import Dict

from .sde import RSDE


class EulerSolver:
    def __init__(self, rsde: RSDE, ode_sampling=False):
        self.ode_sampling: bool = ode_sampling
        self.rsde = rsde

    def step(self, model, x_t, t, attn_mask) -> Dict[str, torch.Tensor]:
        dt = -1. / self.rsde.N
        z = torch.randn_like(x_t)
        rsde_params = self.rsde.sde(model, x_t, t, attn_mask)
        drift, diffusion = rsde_params['drift'], rsde_params['diffusion']
        x_mean = x_t + drift * dt
        if not self.ode_sampling:
            x = x_mean + diffusion[:, None, None] * np.sqrt(-dt) * z
        else:
            x = x_mean
        return {
            "x_t": x,
            "x_mean": x_mean,
            "score": rsde_params['score'],
            "x_0": rsde_params['x_0'],
            "diffusion": diffusion,
            "drift": drift,
            "drift_par": rsde_params["drift_par"]
        }
