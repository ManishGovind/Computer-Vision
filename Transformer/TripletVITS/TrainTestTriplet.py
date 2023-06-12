import torch
import numpy as np

def train(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs,device ,  log_interval):
    for epoch in range(0, n_epochs):
        model.train()
        losses = []
        total_loss = 0

        for batch_idx, (data) in enumerate(train_loader):
            #target = target.cuda()
            optimizer.zero_grad()
            outputs = model(data[0].to(device),data[1].to(device), data[2].to(device))
            loss = loss_fn(outputs[0],outputs[1], outputs[2])
            losses.append(loss.item())
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
                print(message)
                losses = []

        total_loss /= (batch_idx + 1)
        scheduler.step()
        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, 
                    n_epochs, total_loss)

        val_acc = test_epoch(val_loader, model , device)
        val_acc /= len(val_loader)
        message += '\nEpoch: {}/{}. Validation set: Accuracy : {:.4f}'.format(epoch + 1, n_epochs, val_acc)
        print(message)

def test_epoch(val_loader, model , device ):
    with torch.no_grad():
        model.eval()
        val_loss = 0
        acc = 0
        for batch_idx, (data) in enumerate(val_loader):
            zz = data
            dist = model.forward2(data[0].to(device),data[1].to(device))
            
            for i in range(dist.shape[0]):
                if dist[i] < 1.2 and (data[2][i] == 0):
                    acc += 1
                if dist[i] > 2 and (data[2][i] == 1):
                    acc += 1
    return acc






