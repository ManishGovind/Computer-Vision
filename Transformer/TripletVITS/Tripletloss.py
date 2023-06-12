

import torch
import torch.nn as nn

import numpy as np


# Define the triplet loss function
class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor_embedding, positive_embedding, negative_embedding):
        distance_pos = torch.sum(torch.pow(anchor_embedding - positive_embedding, 2), dim=1)
        distance_neg = torch.sum(torch.pow(anchor_embedding - negative_embedding, 2), dim=1)
        loss = torch.mean(torch.clamp(distance_pos - distance_neg + self.margin, min=0))
        return loss