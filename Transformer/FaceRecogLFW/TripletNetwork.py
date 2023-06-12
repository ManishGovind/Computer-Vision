import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 5), nn.PReLU(),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(32, 16, 5), nn.PReLU(),
        nn.MaxPool2d(2, stride=2))
        self.fc = nn.Sequential(nn.Linear(16 * 53 * 53, 256),
            nn.PReLU(),
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.Linear(256, 128)  # 128 size embedding
        )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def forward2(self, x1, x2): # for similar, dissimilar
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        dist = (output1 - output2).pow(2).sum(dim=1).sqrt()
        return dist
    
    def get_embedding(self, x):
        return self.embedding_net(x)
