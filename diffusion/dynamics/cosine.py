import torch
import numpy as np
from typing import Dict, Union, List, Optional

from .sde import SDE, RSDE

class CosineSD(SDE):
    def __init__(
        self,
        d: int = 1,
        N: int = 1000,
        prediction: str = 'x_0',
        ode_sampling: bool = False
    ):
        """
        Args:
          N: number of discretization steps
        """
        super().__init__(N=N, ode_sampling=ode_sampling)
        self.d = d
        self.t_thr = 0.95
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
        beta_t = self.beta_t(t)
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
        alpha, std = self.alpha_std(t)
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

    def beta_t(self, t):
        t = torch.clip(t, 0, self.t_thr)
        tan = torch.tan(np.pi * t / 2)
        beta_t = np.pi * self.d ** 2 * tan * (1 + tan ** 2) / (1 + self.d ** 2 * tan ** 2)
        return beta_t

    def alpha_std(self, t):
        t = t[:, None, None]
        tan = torch.tan(np.pi * t / 2)
        alpha_t = 1 / torch.sqrt(1 + tan ** 2 * self.d ** 2)
        std_t = torch.sqrt(1 - alpha_t ** 2)
        return torch.clip(alpha_t, 0, 1), torch.clip(std_t, 0, 1)