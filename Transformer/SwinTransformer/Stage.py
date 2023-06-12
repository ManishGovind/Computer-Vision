import torch
import torch.nn as nn
from PatchMerging import PatchMerging
from TransformerBlock import TransformerBlock

class Stage(nn.Sequential):
    def __init__(self, num_blocks, in_dim, out_dim, head_dim, shape, window_size, p_drop=0.):
        if out_dim != in_dim:
            layers = [PatchMerging(in_dim, out_dim, shape)]
            shape = (shape[0] // 2, shape[1] // 2)
        else:
            layers = []
        
        shift_size = window_size // 2
        layers += [TransformerBlock(out_dim, head_dim, shape, window_size, 0 if (num % 2 == 0) else shift_size,
                                    p_drop) for num in range(num_blocks)]
        
        super().__init__(*layers)