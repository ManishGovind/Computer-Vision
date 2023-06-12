from torch import nn
from TransformerBlock import TransformerBlock

class Encoder(nn.Module):
   def __init__(self, dim, num_layers=6, heads=8, dim_head=None):
       super().__init__()
       self.block_list = [TransformerBlock(dim, heads, dim_head) for _ in range(num_layers)]
       self.layers = nn.ModuleList(self.block_list)

   def forward(self, x, mask=None):
       for layer in self.layers:
           x = layer(x, mask)
       return x