import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision.datasets import LFWPeople
from TripletNetwork import ViTEmbedding, TripletNet
from Tripletloss import TripletLoss
from Utils import TripletfaceDataset
from LFWTestDataset import LFWTestDataset
import TrainTestTriplet
import sys

import numpy as np




def main():
    # Define the training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-4
    # Split the dataset into training and validation sets
    #train_dataset = LFWPeople('./data/train', split='train', download=True,
    #                                                                transform = transforms.Compose([
    #                                                                    transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.0)),
    #                                                                    transforms.ToTensor(),
    #                                                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    #                                                                ]))
    #test_dataset = LFWPeople('./data/test', split='test', download=True,
    #                                                                transform = transforms.Compose([
    #                                                                    transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.0)),
    #                                                                    transforms.ToTensor(),
    #                                                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    #                                                                ]))
    n_classes = 10
 
    triplet_train_dataset = TripletfaceDataset(data_dir='D:\mgovind\LFW\LFW\lfw_funneled', transform = transforms.Compose([
                                                                        transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.0)),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                                                    ]))
    triplet_test_dataset = LFWTestDataset( pairs_file = 'D:\mgovind\LFW\LFW\pairsDevTest.txt',data_dir='D:\mgovind\LFW\LFW\lfw_funneled', transform = transforms.Compose([
                                                                        transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.0)),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                                                    ]))
   
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset,batch_size=batch_size, shuffle=False, **kwargs)
    # Set up the network and training parameters
    margin = 0.75
    embedding_net = ViTEmbedding().to(device)
    model = TripletNet(embedding_net).to(device)
    criterion = TripletLoss(margin)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.to(device)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 10000
    log_interval = 100
    TrainTestTriplet.train(triplet_train_loader, triplet_test_loader, model, criterion,
    optimizer, scheduler, n_epochs, device,  log_interval)
 
if __name__ == "__main__":
    sys.exit(int(main() or 0))
