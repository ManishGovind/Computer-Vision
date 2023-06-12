import torch
from torch import nn

class MyNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 32),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)