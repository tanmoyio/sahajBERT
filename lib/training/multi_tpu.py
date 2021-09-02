from contextlib import nullcontext
from typing import Optional

import torch
from hivemind import CollaborativeOptimizer, TrainingAverager

import torch_xla.core.xla_model as xm

from lib.training.conduit import DeviceConduit


class TPUCollaborativeOptimizer(CollaborativeOptimizer):
    def __init__(self, opt, conduit: Optional[DeviceConduit] = None, **kwargs):
        self.opt = opt
        if xm.is_master_ordinal():
            self.conduit = conduit if conduit is not None else DeviceConduit([
                param for group in opt.param_groups for param in group["params"]
            ], device=opt.param_groups[0]["params"][0].device)
            super().__init__(opt, **kwargs)

    def parameters(self):
        return [param for group in self.opt.param_groups for param in group["params"]]

    def step(self, *args, **kwargs):
        if xm.is_master_ordinal():
            super().step(*args, **kwargs)
            device_params = [param for group in self.param_groups for param in group["params"]]
            self.conduit.move_to_device(device_params, params=True, grads=False)

        with torch.no_grad():
            scale = 1.0 if xm.is_master_ordinal() else 0.0
            for param in self.parameters:
                param.mul_(scale)

            # all-reduce up 1 * master + 0 * everyone else => broadcast master weights
            xm.all_reduce(xm.REDUCE_SUM, self.parameters)

    def _make_averager(self, **kwargs):
        return TPUFriendlyAverager(
            self.opt,
            dht=self.dht,
            prefix=f"{self.prefix}_averaging",
            allreduce_timeout=self.averaging_timeout,
            client_mode=self.client_mode,
            conduit=self.conduit,
            **kwargs,
        )


class TPUFriendlyAverager(TrainingAverager):
    def __init__(self, *args, conduit: Optional[DeviceConduit] = None, **kwargs):
        super().__init__(*args, average_parameters=True, average_gradients=True,
                         average_opt_statistics=(), extra_tensors=(), **kwargs)
        self.conduit = conduit if conduit is not None else DeviceConduit([
            param for group in self.opt.param_groups for param in group["params"]
        ], device=self.opt.param_groups[0]["params"][0].device)

    def get_local_tensors_cpu(self):
      with torch.no_grad():
        self.conduit.move_to_host([param for group in self.opt.param_groups for param in group["params"]],
                                  params=True, grads=True)
        return [param for param in self.conduit.host_parameters
                ] + [param.grad for param in self.conduit.host_parameters]

    def set_local_tensors(self, updates):
      with torch.no_grad():
        host_tensors = [param for param in self.conduit.host_parameters
                        ] + [param.grad for param in self.conduit.host_parameters]
        assert len(host_tensors) == len(updates)
        for host_tensor, update in zip(host_tensors, updates):
            host_tensor[...] = update
        self.conduit.move_to_device([
            param for group in self.opt.param_groups for param in group["params"]
        ], params=True, grads=True)

    def step(self, data_lock=None, wait: bool = True, **kwargs):
        """
        Average optimizer weights and gradients with peers.

        :param data_lock: averager locks it when model parameters are modified. Otherwise it's assumed that no model
        modifications occur during averaging step
        """
        if not wait:
            return self.step_executor.submit(self.step, data_lock, wait=True, **kwargs)

        # if data_lock is supplied, tensors might change during averaging, so we need to copy them
        use_old_local_tensors = data_lock is not None
        if data_lock is None:
            data_lock = nullcontext()

        local_tensors_cpu = self.get_local_tensors_cpu()

        with self.lock_averager_step, torch.no_grad():
            # fill averager's tensors with current local tensors
            self.pending_updates_done.clear()
            with data_lock, self.get_tensors() as averaged_tensors:
                if use_old_local_tensors:
                    old_local_tensors = tuple(x.cpu().float().clone() for x in local_tensors_cpu)
                assert len(local_tensors_cpu) == len(
                    averaged_tensors
                ), "The number of optimized parameters should not change."
                for averaged_tensor, local_tensor in zip(averaged_tensors, local_tensors_cpu):
                    averaged_tensor[...] = local_tensor.cpu().float()
            self.pending_updates_done.set()

            # find a group and hopefully average tensors with peers, use batch sizes as weights
            gathered = super().step(**kwargs)
            if gathered is not None:
                # load averaged tensors back into model
                self.pending_updates_done.clear()
                with data_lock, self.get_tensors() as averaged_tensors:
                    if len(averaged_tensors) != len(local_tensors_cpu):
                        raise RuntimeError("The number of optimized parameters should not change.")

                    if use_old_local_tensors:
                        # since tensors might have changed, we subtract old_local_tensor and add averaged. This prevents
                        # losing local updates that might have occurred during averaging
                        for averaged_tensor, local_tensor, old_local_tensor in zip(
                            averaged_tensors, local_tensors_cpu, old_local_tensors
                        ):
                            averaged_tensor = averaged_tensor.to(
                                dtype=local_tensor.dtype, device=local_tensor.device, non_blocking=True
                            )
                            old_local_tensor = old_local_tensor.to(
                                dtype=local_tensor.dtype, device=local_tensor.device, non_blocking=True
                            )

                            local_tensor.add_(averaged_tensor - old_local_tensor)
                    else:
                        self.set_local_tensors(averaged_tensors)
                self.pending_updates_done.set()

            self.local_step += 1
            return gathered
