import torch
import torch.nn as nn

class GlobalAvgPool(nn.Module):
    def forward(self, x):
        return x.mean(dim=-2)

class Head(nn.Sequential):
    def __init__(self, dim, classes, p_drop=0.):
        super().__init__(
            nn.LayerNorm(dim),
            nn.GELU(),
            GlobalAvgPool(),
            nn.Dropout(p_drop),
            nn.Linear(dim, classes)
        )