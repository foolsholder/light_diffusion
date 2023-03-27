from dataclasses import dataclass

@dataclass
class SDECfg:
    _target_: str
    N: int
    beta_min: float
    beta_max: float
    prediction: str
    ode_sampling: bool