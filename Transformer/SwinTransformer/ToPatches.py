import torch
import torch.nn as nn
import torch.nn.functional as F

class ToPatches(nn.Module):
    def __init__(self, in_channels, dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        patch_dim = in_channels * patch_size**2
        self.proj = nn.Linear(patch_dim, dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        x = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).movedim(1, -1)
        x = self.proj(x)
        x = self.norm(x)
        return x