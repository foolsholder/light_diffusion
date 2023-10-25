from .base_contextual import ContextualDenoising
from diffusion.models.contextual_denoising.planner import PlannerEstimator, bert_config_slava


class PlanerContextualDenoising(ContextualDenoising):
    def __init__(self, *args, **kwargs) -> None:
        score_estimator = PlannerEstimator(bert_config_slava)
        super().__init__(*args, **kwargs)
        self.score_estimator = score_estimator
        self.noisy_part_encoder.restore_decoder()
