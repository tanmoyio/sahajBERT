"""
Auxiliary modules for implementing Rotary Position Embeddi g
Original paper: https://arxiv.org/abs/2104.09864
Based on reference implementation from https://blog.eleuther.ai/rotary-embeddings
"""

import torch
import torch.nn as nn


class RotaryEmbeddings(nn.Module):
    """ Applies rotary position embeddings to a tensor, uses caching to improve performance """
    def __init__(self, dim: int, base: int = 10_000):
        super().__init__()
        self.dim, self.base = dim, base

    def forward(self, x: torch.Tensor, offset: int = 0):
        """
        :param x: tensor of shape [batch_size, seq_len, nhead, hid_size]
        :param offset: add this value to all position indices
        """
        print('!!!A')
        seq_len = x.shape[1]
        print('!!!B')
        cos, sin = getattr(self, 'cos', None), getattr(self, 'sin', None)
        print('!!!C')
        if cos is None or seq_len + offset >= cos.shape[0] or x.dtype != cos.dtype or x.device != cos.device:
            print('!!!D')
            cos, sin = get_auxiliary_tensors(seq_len + offset, self.dim, x.dtype, x.device, self.base)
            print('!!!E')
            self.register_buffer('cos', cos)
            print('!!!F')
            self.register_buffer('sin', sin)
            print('!!!G')

        return rotate(x, cos[None, offset: seq_len + offset, None, :], sin[None, offset: seq_len + offset, None, :])


@torch.no_grad()
def get_auxiliary_tensors(seq_len: int, dim: int, dtype: torch.dtype, device: torch.device, base: int):
    """
    Compute auxiliary sine and cosine tensors for rotary position embedding
    :returns: a tuple of (cos, sin) tensors of shape [seq_len, hid_size]
    """
    print('?-1')
    _buf = torch.linspace(0, -1 + 2 / dim, dim // 2, dtype=torch.float32, device=device)
    print('?-2')
    inv_freq = torch.pow(base, _buf, out=_buf).repeat(2)
    print('?-3')
    time_ix = torch.arange(seq_len, dtype=inv_freq.dtype, device=device)
    print('?-4')

    freqs = (time_ix[:, None] * inv_freq[None, :])
    print('?-5')
    cos = torch.cos(freqs)
    print('?-6')
    sin = torch.sin(freqs, out=freqs)
    print('?-7')
    try:
        return cos.to(dtype), sin.to(dtype)
    finally:
        print('?-8')


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """ rotate pairwise coordinate using precomputed cos & sin tensors """
    dim = x.shape[-1]
    print('H')
    x_left, x_right = x.split(split_size=dim // 2, dim=x.ndim - 1)
    print('I')
    x_rotated = torch.cat([x_right.neg(), x_left], dim=x.ndim - 1)
    print('J')
    return x * cos + x_rotated * sin
