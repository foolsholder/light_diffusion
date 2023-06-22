import math
import warnings
from typing import List

from torch import nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupSecondZeroLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        secound_group_zero_epochs: int = 0,
        warmup_start_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.secound_group_zero_epochs = secound_group_zero_epochs
        super().__init__(optimizer, last_epoch)

    def get_group_lr(
        self,
        last_epoch: int,
        warmup_epochs: int,
        zero_epochs: int,
        group_lr: float,
        base_lr: float
    ) -> List[float]:
        if last_epoch < zero_epochs:
            return 0
        last_epoch -= zero_epochs
        if last_epoch == 0:
            return self.warmup_start_lr
        if last_epoch < warmup_epochs:
            return group_lr + (base_lr - self.warmup_start_lr) / (warmup_epochs - 1)
        return base_lr

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.",
                UserWarning,
            )

        assert len(self.base_lrs) == 2, len(self.base_lrs)

        lrs = [
            self.get_group_lr(
                self.last_epoch, self.warmup_epochs, zero_epochs,
                group["lr"], base_lr
            ) for base_lr, group, zero_epochs \
                in zip(self.base_lrs, self.optimizer.param_groups, [0, self.secound_group_zero_epochs])
        ]
        #print(lrs)

        return lrs

    def _get_closed_form_lr(self) -> List[float]:
        return self.get_lr()
