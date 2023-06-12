from audioop import avg
import torch
import numpy as np

def train(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, 
    cuda, log_interval):
    for epoch in range(0, n_epochs):
        model.train()
        losses = []
        total_loss = 0

        for batch_idx, (data) in enumerate(train_loader):
            #target = target.cuda()
            optimizer.zero_grad()
            outputs = model(data[0].cuda(),data[1].cuda(), data[2].cuda())
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
                #val_acc, acc_pos, acc_neg, count, avg_pos_dist, avg_neg_dist = test_epoch(val_loader, model, cuda)
                #val_acc /= count
                #message = '\nEpoch: {}/{}. Test set: Accuracy: {:.4f} Pos Count= {} Neg Count= {} total count={} pos_dist={} neg_dist={}'.format(epoch + 1, n_epochs, val_acc, acc_pos, acc_neg, count, avg_pos_dist, avg_neg_dist)
                #print(message)

        total_loss /= (batch_idx + 1)
        scheduler.step()
        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, 
                    n_epochs, total_loss)
        val_acc, acc_pos, acc_neg, count, avg_pos_dist, avg_neg_dist = test_epoch(val_loader, model, cuda)
        val_acc /= count
        message += '\nEpoch: {}/{}. Test set: Accuracy: {:.4f} Pos Count= {} Neg Count= {} total count={} pos_dist={} neg_dist={}'.format(epoch + 1, n_epochs, val_acc, acc_pos, acc_neg, count, avg_pos_dist, avg_neg_dist)
        print(message)

def test_epoch(val_loader, model, cuda):
    with torch.no_grad():
        model.eval()
        val_loss = 0
        acc = 0
        acc_pos = 0
        acc_neg = 0
        count = 0
        avg_pos_distance = 0
        avg_neg_distance = 0
        for batch_idx, (data) in enumerate(val_loader):
            recog_threshold = 1.3
            dist = model.forward2(data[0].cuda(),data[1].cuda())
            for i in range(dist.shape[0]):
                count = count + 1
                if data[2][i] == 1:
                    avg_pos_distance = avg_pos_distance + dist[i]
                if data[2][i] == 0:
                    avg_neg_distance = avg_neg_distance + dist[i]
                if dist[i] < recog_threshold and (data[2][i] == 1):
                    acc = acc + 1
                    acc_pos = acc_pos + 1
                if dist[i] > recog_threshold and (data[2][i] == 0):
                    acc = acc + 1
                    acc_neg = acc_neg + 1
    return acc, acc_pos, acc_neg, count, avg_pos_distance/500, avg_neg_distance/500