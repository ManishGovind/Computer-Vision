import sys
import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

from ignite.engine import Events, create_supervised_trainer,create_supervised_evaluator
import ignite.metrics
import ignite.contrib.handlers

from SwinTransformer import SwinTransformer

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# pip install pytorch-ignite

# https://juliusruseckas.github.io/ml/swin-cifar10.html

def get_optimizer(model, learning_rate, weight_decay):
    param_dict = {pn: p for pn, p in model.named_parameters()}
    parameters_decay, parameters_no_decay = model.separate_parameters()
    
    optim_groups = [
        {"params": [param_dict[pn] for pn in parameters_decay], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in parameters_no_decay], "weight_decay": 0.0},
    ]
    optimizer = optim.AdamW(optim_groups, lr=learning_rate)
    return optimizer

def dataset_show_image(dset, idx):
    X, Y = dset[idx]
    title = "Ground truth: {}".format(dset.classes[Y])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.imshow(np.moveaxis(X.numpy(), 0, -1))
    ax.set_title(title)
    plt.show()

def main():
    DATA_DIR='./data'

    IMAGE_SIZE = 32

    NUM_CLASSES = 10
    NUM_WORKERS = 8
    BATCH_SIZE = 32
    EPOCHS = 2 #100

    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-1

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device:", DEVICE)

    train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float)])

    train_dset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transform)
    test_dset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transforms.ToTensor())
    
    dataset_show_image(test_dset, 1)
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=BATCH_SIZE, shuffle=True,num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    model = SwinTransformer(NUM_CLASSES, IMAGE_SIZE,
                        num_blocks_list=[4, 4], dims=[128, 128, 256],
                        head_dim=32, patch_size=2, window_size=4,
                        emb_p_drop=0., trans_p_drop=0., head_p_drop=0.3)
    model.to(DEVICE)
    print("Number of parameters: {:,}".format(sum(p.numel() for p in model.parameters())))
    
    loss = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    trainer = create_supervised_trainer(model, optimizer, loss, device=DEVICE)
    
    lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE,steps_per_epoch=len(train_loader), epochs=EPOCHS)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, lambda engine: lr_scheduler.step());
    ignite.metrics.RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    val_metrics = {"accuracy": ignite.metrics.Accuracy(), "loss": ignite.metrics.Loss(loss)}
    evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=DEVICE)
    history = defaultdict(list)
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        #print("eoch completed called")
        train_state = engine.state
        epoch = train_state.epoch
        max_epochs = train_state.max_epochs
        train_loss = train_state.metrics["loss"]
        history['train loss'].append(train_loss)
    
        evaluator.run(test_loader)
        val_metrics = evaluator.state.metrics
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        history['val loss'].append(val_loss)
        history['val acc'].append(val_acc)
    
        print("{}/{} - train: loss {:.3f}; val: loss {:.3f} accuracy {:.3f}".format(
            epoch, max_epochs, train_loss, val_loss, val_acc))

    trainer.run(train_loader, max_epochs=EPOCHS)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    xs = np.arange(1, len(history['train loss']) + 1)
    ax.plot(xs, history['train loss'], '.-', label='train')
    ax.plot(xs, history['val loss'], '.-', label='val')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend()
    ax.grid()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    xs = np.arange(1, len(history['val acc']) + 1)
    ax.plot(xs, history['val acc'], '-')
    ax.set_xlabel('epoch')
    ax.set_ylabel('val acc')
    ax.grid()
    plt.show()
    
    
    print("all done..")

if __name__ == "__main__":
    sys.exit(int(main() or 0))
