import logging

import torch
from torch import nn


logger = logging.getLogger(__name__)
LRSchedulerBase = getattr(torch.optim.lr_scheduler, '_LRScheduler', None)


class NoOpScheduler(LRSchedulerBase):
    """ Dummy scheduler for transformers.Trainer. The real scheduler is defined in CollaborativeOptimizer.scheduler """

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    def print_lr(self, *args, **kwargs):
        if self.optimizer.scheduler:
            return self.optimizer.scheduler.print_lr(*args, **kwargs)

    def step(self):
        logger.debug("Called NoOpScheduler.step")
        self._last_lr = self.get_lr()

    def state_dict(self):
        return {}

    def load_state_dict(self, *args, **kwargs):
        logger.debug("Called NoOpScheduler.load_state_dict")


class IgnoreGradManipulations(nn.Module):
    """ Wrapper for model that blocks gradient manipulations in huggingface Trainer (e.g. zero_grad, clip_grad) """
    def __init__(self, module, override_clipping: bool = True, override_zero_grad: bool = True):
        super().__init__()
        self.module = module
        self.override_clipping = override_clipping
        self.override_zero_grad = override_zero_grad

    def forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)

    def zero_grad(self, set_to_none: bool = False) -> None:
        if self.override_zero_grad and all(param.grad.isfinite().all() for param in self.parameters()):
            logger.debug("Successfully bypassed zero_grad")
        else:
            self.module.zero_grad(set_to_none=set_to_none)

    def clip_grad_norm_(self, max_norm: float, norm_type: int = 2):
        """ ignore clip_grad_norm on each step, clip in optimizer instead """
        if self.override_clipping:
            logger.debug("Successfully bypassed clip_grad_norm_")
        else:
            return torch.nn.utils.clip_grad_norm_(self.module.parameters(), max_norm, norm_type=norm_type)
