# Define the ViT model architecture

import torch.nn as nn
from transformers import ViTModel
import torch.nn.functional  as F 


class ViTEmbedding(nn.Module):
    def __init__(self):
        super(ViTEmbedding, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.fc = nn.Linear(768, 256)

    def forward(self, x):
        x = self.vit(x)
        x = x.last_hidden_state[:, 0]
        x = self.fc(x)
        return x

# Define the triplet network architecture
class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, anchor, positive, negative):
        anchor_embedding = self.embedding_net(anchor)
        positive_embedding = self.embedding_net(positive)
        negative_embedding = self.embedding_net(negative)
        return anchor_embedding, positive_embedding, negative_embedding


    def forward2(self, anchor, positive):
        output_anchor = self.embedding_net(anchor)
        output_positive = self.embedding_net(positive)
        dist = F.pairwise_distance(output_anchor, output_positive)
        return dist