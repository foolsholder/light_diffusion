import torch

import numpy as np

from torch import nn, Tensor
from typing import Dict
from collections import OrderedDict


def dict_to_device(dct: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in dct.items()}


def calc_model_grads_norm(model: torch.nn.Module, p: float = 2):
    grads = []
    for par in model.parameters():
        if par.requires_grad and par.grad is not None:
            # print(par.grad)
            grads += [torch.sum(par.grad ** p)]
    return torch.pow(sum(grads), 1. / p)


def calc_model_weights_norm(model: torch.nn.Module, p: float = 2):
    weights = []
    for par in model.parameters():
        weights += [torch.sum(par.data ** p)]
    return torch.pow(sum(weights), 1. / p)


def calc_model_params_count(model):
    total_params = sum(
        param.numel() for param in model.parameters()
    )
    return total_params


def filter_losses(dct: Dict[str, Tensor]) -> Dict[str, Tensor]:
    return {key: value for key, value in dct.items() if 'loss' in key}
