import numpy as np
import torch
from einops import rearrange
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, dim): # e.g., dim=64 in each head
        super().__init__()
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.scale_factor = dim ** -0.5  # 1/np.sqrt(dim)

    def forward(self, x, mask=None):
        assert x.dim() == 3, '3D tensor must be provided' # b,n,d

        qkv = self.to_qkv(x)  # [b, n, dim*3 ]

        # rearrange tensor to [3, b, n, dim] and then extract q,k,v 
        q, k, v = tuple(rearrange(qkv, 'b n (d k) -> k b n d ', k=3))

        scaled_qkt_prod = torch.einsum('b i d , b j d -> b i j', q, k) * self.scale_factor  # shape = [b, n, n]

        if mask is not None:
            assert mask.shape == scaled_qkt_prod.shape[1:]
            scaled_qkt_prod = scaled_qkt_prod.masked_fill(mask, -np.inf)

        attention = torch.softmax(scaled_qkt_prod, dim=-1)

        return torch.einsum('b i j , b j d -> b i d', attention, v)