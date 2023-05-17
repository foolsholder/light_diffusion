from dataclasses import dataclass
from torch import Tensor, FloatTensor

@dataclass
class EncoderOutput:
    normed: FloatTensor
    true: FloatTensor