import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.utils.checkpoint import get_device_states, set_device_states


class LeanFFN(nn.Module):
    """
    A transformer FFN module that doesn't hog your GPU memory.
    Complete with pre-LayerNorm, residual connections and dropout.

    :param gated: use gated activations based on https://arxiv.org/abs/2002.05202 and https://arxiv.org/abs/2102.11972
      note: gated activations require 1.5x more parameters compared to their non-gated variants.
    """

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 activation=F.gelu,
                 gated: bool = False,
                 layer_norm_eps: float = 1e-12,
                 dropout: float = 0.0,
                 ):
        super().__init__()
        self.dense_i2h = nn.Linear(hidden_size, intermediate_size * 2 if gated else intermediate_size)
        self.dense_h2o = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.activation = activation
        self.dropout = dropout

    def forward(self, input):
        return _LeanFFN.apply(
            input,
            self.layer_norm.weight, self.layer_norm.bias,
            self.dense_i2h.weight, self.dense_i2h.bias,
            self.dense_h2o.weight, self.dense_h2o.bias,
            self.activation, self.dropout, self.training,
            self.layer_norm.eps
        )


class _LeanFFN(torch.autograd.Function):
    @staticmethod
    def _apply_activation(pre_activation: torch.Tensor, activation: callable, hid_size: int):
        if pre_activation.shape[-1] == hid_size:
            return activation(pre_activation)
        elif pre_activation.shape[-1] == 2 * hid_size:
            pre_gate, lin = pre_activation.split(pre_activation.shape[-1] // 2, dim=-1)
            return activation(pre_gate).mul_(lin)
        else:
            raise RuntimeError("The output size of FFN layer must be either 1x or 2x the intermediate_size.")

    @staticmethod
    @custom_fwd
    def forward(ctx, input, ln_weight, ln_bias, i2h_weight, i2h_bias, h2o_weight, h2o_bias,
                activation, dropout, training, ln_eps):
        ctx._activation, ctx._dropout, ctx._training, ctx._ln_eps = activation, dropout, training, ln_eps
        ctx._cpu_rng_state = torch.get_rng_state()
        ctx._device_rng_states = get_device_states(input)

        input_2d = input.view(-1, input.shape[-1])

        input_ln = F.layer_norm(input_2d, input.shape[-1:], ln_weight, ln_bias, ln_eps)

        pre_activation = F.linear(input_ln, i2h_weight, i2h_bias)
        hid_act = _LeanFFN._apply_activation(pre_activation, ctx._activation, h2o_weight.shape[1])

        out = F.linear(hid_act, h2o_weight, h2o_bias)
        out = F.dropout(out, dropout, training, inplace=True)
        out = out.add_(input_2d)
        ctx.save_for_backward(input, pre_activation, ln_weight, ln_bias, i2h_weight, h2o_weight)
        return out.view(*input.shape)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        grad_input = grad_ln_weight = grad_ln_bias = None
        grad_i2h_weight = grad_i2h_bias = grad_h2o_weight = grad_h2o_bias = None
        input, pre_activation, ln_weight, ln_bias, i2h_weight, h2o_weight = ctx.saved_tensors
        torch.set_rng_state(ctx._cpu_rng_state)
        set_device_states(*ctx._device_rng_states)

        input_2d = input.view(-1, input.shape[-1])
        grad_output_2d = grad_output.view(-1, grad_output.shape[-1])
        grad_hid_act = torch.mm(grad_output_2d, h2o_weight)

        with torch.enable_grad():
            # rematerialize activation
            pre_activation.requires_grad_(True)
            hid_act = _LeanFFN._apply_activation(pre_activation, ctx._activation, h2o_weight.shape[1])
            grad_hid, = torch.autograd.grad(hid_act, pre_activation, grad_hid_act)
            pre_activation.requires_grad_(False)

        grad_input_ln_2d = torch.mm(grad_hid, i2h_weight)

        with torch.enable_grad():
            # rematerialize input_ln
            input_2d.requires_grad_(True)
            input_ln_2d = F.layer_norm(input_2d, input.shape[-1:], ln_weight, ln_bias, ctx._ln_eps)

            if any(ctx.needs_input_grad[0:3]):
                grad_input_2d, grad_ln_weight, grad_ln_bias = torch.autograd.grad(
                    outputs=input_ln_2d,
                    inputs=[input_2d, ln_weight, ln_bias],
                    grad_outputs=grad_input_ln_2d)

            input_2d.requires_grad_(False)
            input_ln_2d = input_ln_2d.detach_()

        if ctx.needs_input_grad[0]:
            grad_input_2d = grad_input_2d.add_(grad_output_2d)
            grad_input = grad_input_2d.view(*grad_output.shape)
        if ctx.needs_input_grad[3]:
            grad_i2h_weight = grad_hid.t().mm(input_ln_2d)
        if ctx.needs_input_grad[4]:
            grad_i2h_bias = grad_hid.sum(0)
        if ctx.needs_input_grad[5]:
            grad_h2o_weight = grad_output_2d.t().mm(hid_act)
        if ctx.needs_input_grad[6]:
            grad_h2o_bias = grad_output_2d.sum(0)

        return (grad_input, grad_ln_weight, grad_ln_bias, grad_i2h_weight,
                grad_i2h_bias, grad_h2o_weight, grad_h2o_bias, None, None, None, None)
