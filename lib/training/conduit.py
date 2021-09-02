import torch
import numpy as np
import multiprocessing as mp
from torch import nn


class DeviceConduit:
    """An auxiliary class for manipulating parameters and gradients without producing a ton of XLA graphs"""

    def __init__(self, parameters, device):
        print(f"Building device conduit for {device}")
        self.device, self.lock = device, mp.Lock()
        self.host_parameters = [nn.Parameter(param.data.cpu()) for param in parameters]
        for param in self.host_parameters:
            if param.grad is None:
                param.grad = torch.zeros_like(param)

        self.strides = np.cumsum([0] + [p.numel() for p in self.host_parameters])
        self.device_buffer = torch.zeros(self.strides[-1], device=self.device)
        self.host_buffer = self.device_buffer.detach().cpu()

    def move_to_device(self, device_parameters, params: bool = False, grads: bool = False):
        if params:
            self._move_tensors_to_device(self.host_parameters, device_parameters)
        if grads:
            self._move_tensors_to_device([hp.grad for hp in self.host_parameters],
                                         [dp.grad for dp in device_parameters])

    def move_to_host(self, device_parameters, params: bool = False, grads: bool = False):
        if params:
            self._move_tensors_to_host(device_parameters, self.host_parameters)
        if grads:
            self._move_tensors_to_host([dp.grad for dp in device_parameters],
                                       [hp.grad for hp in self.host_parameters])

    def _move_tensors_to_host(self, device_tensors, host_tensors):
        with mp.Lock():
            self.device_buffer.copy_(torch.cat([tensor.view(-1) for tensor in device_tensors], dim=0))
            self.host_buffer[...] = self.device_buffer.cpu()
            for host_tensor, host_update in zip(host_tensors, self._slice_into_tensors(self.host_buffer)):
                host_tensor[...] = host_update

    def _move_tensors_to_device(self, host_tensors, device_tensors):
        with mp.Lock():
            for i in range(len(self.host_parameters)):
                self.host_buffer[self.strides[i]: self.strides[i + 1]] = host_tensors[i].view(-1)
            self.device_buffer[...] = self.host_buffer.to(self.device)
            for device_tensor, device_update in zip(device_tensors, self._slice_into_tensors(self.device_buffer)):
                device_tensor.copy_(device_update)

    def _slice_into_tensors(self, buffer):
        tensors = []
        for i in range(len(self.host_parameters)):
            tensors.append(buffer[self.strides[i]: self.strides[i + 1]].view(self.host_parameters[i].shape))
        return tensors
