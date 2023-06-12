import Utils
import sys
import torchvision
from TripletNetwork import EmbeddingNet, TripletNet
from TripletLoss import TripletLoss
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
import numpy as np
import TrainTestTriplet
import matplotlib.pyplot as plt


def main():
    cuda = torch.cuda.is_available()


    data_dir = 'D:/DeepLearning/data/LFW/lfw_funneled'  # original LFW
    data_dir_face = 'D:/DeepLearning/data/LFWFaceOnly/lfw_cut'  # face detected LFW
    batch_size = 64
    transform = torchvision.transforms.Compose([
        #torchvision.transforms.ToPILImage(),  # only for Dataset using Dlib
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_loader = Utils.get_train_dataloader(data_dir, batch_size, transform=transform)
    #train_loader = Utils.get_train_dataloader(data_dir_face, batch_size, transform=transform)
    for batch_idx, (data) in enumerate(train_loader):
        numpy_image = data[0][0] .permute(1, 2, 0).numpy()
        # display the numpy array as an image using matplotlib
        plt.imshow(numpy_image)
        plt.show()
        break


    pairs_file = 'D:/DeepLearning/data/LFW/pairsDevTest.txt'
 
    test_loader = Utils.get_test_loader(pairs_file,data_dir, batch_size, transform=transform)
    #test_loader = Utils.get_test_loader(pairs_file,data_dir_face, batch_size, transform=transform)
    margin = 0.3
    embedding_net = EmbeddingNet()
    model = TripletNet(embedding_net).cuda()
    loss_fn = TripletLoss(margin)
    lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 500
    log_interval = 100
    TrainTestTriplet.train(train_loader, test_loader, model, loss_fn, 
    optimizer, scheduler, n_epochs, cuda, log_interval)
  
    #train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, model)
    #plot_embeddings(train_embeddings_cl, train_labels_cl)
    #val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, model)
    #plot_embeddings(val_embeddings_cl, val_labels_cl)



if __name__ == "__main__":
    sys.exit(int(main() or 0))
