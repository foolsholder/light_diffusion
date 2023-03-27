import torch
import numpy as np
from typing import Dict, Union, List, Optional

from .sde import SDE, RSDE

class BetaLinear(SDE):
    def __init__(
        self,
        N: int,
        beta_min: float,
        beta_max: float,
        prediction: str,
        ode_sampling: bool
    ):
        """Construct a Variance Preserving SDE.
        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        super().__init__(N=N, ode_sampling=ode_sampling)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.prediction = prediction

    @property
    def T(self):
        return 1

    def sde(self, x, t) -> Dict[str, torch.Tensor]:
        """
        sde: dx = drift * dt + diffusion * dw
        drift = -1/2 * beta * x_t
        diffusion = sqrt(beta)
        """
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None].to(x.device) * x
        diffusion = torch.sqrt(beta_t).to(x.device)
        return {
            "drift": drift,
            "diffusion": diffusion
        }

    def marginal_params_tensor(self, x, t) -> Dict[str, torch.Tensor]:
        """
        x_t = x_0 * alpha + eps * std
        beta(s) = (beta_max - beta_min) * s + beta_min
        alpha_real = exp(-integrate(beta(s) ds)) = exp(-1/2 * (beta_max - beta_min) * t**2 - beta_min * t)
        here alpha = sqrt(alpha_real) in order to multiply x without sqrt
        std = sqrt(1 - alpha_real)
        """
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        log_gamma_coeff = log_mean_coeff * 2
        alpha = torch.exp(log_mean_coeff[:, None, None]).to(x.device)
        std = torch.sqrt(1. - torch.exp(log_gamma_coeff)).to(x.device)[:, None, None]
        return {
            "alpha": alpha,
            "std": std
        }

    def marginal_forward(self, x, t) -> Dict[str, torch.Tensor]:
        params = self.marginal_params_tensor(x, t)
        alpha, std = params['alpha'], params['std']
        mean = alpha * x
        noise = torch.randn_like(mean)
        return {
            "mean": mean,
            "x_t": mean + noise * std,
            "noise": noise,
            "noise_t": noise * std,
            "score": -noise / std,
        }


    def prior_sampling(self, shape) -> torch.Tensor:
        return torch.randn(*shape)

    def calc_score(self, model, x_t, t, attn_mask=None) -> Dict[str, torch.Tensor]:
        """
        x_0 - prediction x_0(x_t, t)
        eps = (x_t - sqrt(alpha_t) * x_0) / std
        score = (-x_t + sqrt(alpha_t) * x_0) / std**2
        """
        # x_0 = model.predict_x0(x_t=x_t, time_t=t, mask=mask)
        params = self.marginal_params_tensor(x_t, t)

        if self.prediction == "x_0":
            x_0 = model(x_t=x_t, time_t=t, attention_mask=attn_mask)
            eps_theta = (x_t - params["alpha"] * x_0) / params["std"]
            score = -eps_theta / params["std"]
        elif self.prediction == "x_0_x_t":
            x_0 = x_t + model(x_t=x_t, time_t=t, attention_mask=attn_mask)
            eps_theta = (x_t - params["alpha"] * x_0) / params["std"]
            score = -eps_theta / params["std"]
        elif self.prediction == "x_0_a_x_t":
            x_0 = params["alpha"] * x_t + model(x_t=x_t, time_t=t, attention_mask=attn_mask)
            eps_theta = (x_t - params["alpha"] * x_0) / params["std"]
            score = -eps_theta / params["std"]
        elif self.prediction == "eps":
            eps_theta = model(x_t=x_t, time_t=t, attention_mask=attn_mask)
            x_0 = (x_t - params["std"] * eps_theta) / params["alpha"]
            score = -eps_theta / params["std"]
        elif self.prediction == "eps_a_x_t":
            eps_theta = params["std"] * x_t + model(x_t=x_t, time_t=t, attention_mask=attn_mask)
            x_0 = (x_t - params["std"] * eps_theta) / params["alpha"]
            score = -eps_theta / params["std"]
        elif self.prediction == "score":
            score = model(x_t=x_t, time_t=t, attention_mask=attn_mask)
            eps_theta = -score * params["std"]
            x_0 = (x_t - params["std"] * eps_theta) / params["alpha"]

        return {
            "score": score,
            "x_0": x_0,
            "eps_theta": eps_theta
        }
