import contextlib
from typing import Type, Iterable, Dict, Union, Optional

import torch

from .conduit import DeviceConduit
from .wrapper import OptimizerWrapper


class OffloadOptimizer(OptimizerWrapper):
    r""" A wrapper that stores optimizer statistics and performs updates on the offloaded device (e.g. CPU RAM). """

    def __init__(
            self, param_groups: Union[Iterable[torch.nn.Parameter], Iterable[Dict]],
            optim_cls: Type[torch.optim.Optimizer],  *args, full_sync: bool = True,
            conduit: Optional[DeviceConduit] = None, **kwargs):
        param_groups = list(param_groups)
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]
        super().__init__(optim_cls(param_groups, *args, **kwargs))
        self.full_sync = full_sync
        self.conduit = conduit if conduit is not None else DeviceConduit([
            param for group in param_groups for param in group["params"]
        ], device=param_groups[0]["params"][0].device)

    @contextlib.contextmanager
    def _use_offloaded_params(self, *,
                              sync_params_before: bool, sync_grads_before: bool,
                              sync_params_after: bool, sync_grads_after: bool):

        original_param_groups = [group["params"] for group in self.param_groups]
        original_params = [param for group in self.param_groups for param in group["params"]]

        try:
            with torch.no_grad():
                self.conduit.move_to_host(original_params, params=sync_params_before, grads=sync_grads_before)

            offset = 0
            flat_offload_parameters = self.conduit.host_parameters
            for group in self.param_groups:
                group["params"] = flat_offload_parameters[offset: offset + len(group["params"])]
                offset += len(group["params"])

            yield self.param_groups
        finally:
            for group, original_param_group in zip(self.param_groups, original_param_groups):
                group["params"] = original_param_group

            with torch.no_grad():
                self.conduit.move_to_device(original_params, params=sync_params_after, grads=sync_grads_after)

    def add_param_group(self, param_group: dict) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not support add_param_group.")

    def step(self, closure=None, *args, **kwargs):
        assert closure is None, "closure not supported in cpu offload mode"
        with self._use_offloaded_params(sync_params_before=self.full_sync, sync_grads_before=True,
                                        sync_params_after=True, sync_grads_after=self.full_sync):
            return self.optim.step(*args, **kwargs)

    def zero_grad(self, set_to_none: bool = False, *args, **kwargs):
        if not self.full_sync:
            torch.optim.Optimizer.zero_grad(self, set_to_none)
        with self._use_offloaded_params(sync_params_before=self.full_sync, sync_grads_before=self.full_sync,
                                        sync_params_after=self.full_sync, sync_grads_after=self.full_sync):
            return super().zero_grad(*args, set_to_none=False, **kwargs)

    def state_dict(self):
        with self._use_offloaded_params(sync_params_before=self.full_sync, sync_grads_before=self.full_sync,
                                        sync_params_after=False, sync_grads_after=False):
            return self.optim.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        with self._use_offloaded_params(sync_params_before=False, sync_grads_before=False,
                                        sync_params_after=True, sync_grads_after=self.full_sync):
            return self.optim.load_state_dict(state_dict)
