import torch
import torch.nn as nn

class AddPositionEmbedding(nn.Module):
    def __init__(self, dim, num_patches):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.Tensor(num_patches, dim))
    
    def forward(self, x):
        return x + self.pos_embedding