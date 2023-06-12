

import random
import tqdm
import numpy as np
import torch
import torch.optim as optim
#from AutoRegressiveWrapper import AutoRegressiveWrapper
from TrainValidateWrapper import TrainValidateWrapper
from models.SimpleTransformer import SimpleTransformer
import Utils
import sys
import math
import os


# ------constants------------
NUM_BATCHES = int(1e5)
BATCH_SIZE = 32
GRADIENT_ACCUMULATE_EVERY = 1
LEARNING_RATE = 0.5e-4
VALIDATE_EVERY = 5000
SEQ_LENGTH = 197 # 14x14 + 1 for cls_token
RESUME_TRAINING = True # set to false to start training from beginning


# following commented functions are for character level modeling----------
#def decode_token(token): # convert token to character
# return str(chr(max(32, token)))
#def decode_tokens(tokens): # convert sequence of characters to tokens
# return ''.join(list(map(decode_token, tokens)))
#------------------------------------------------------------------------

def count_parameters(model): # count number of trainable parameters in the model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def configure_optimizers(mymodel):
     """
     This long function is unfortunately doing something very simple and is being
    very defensive:
     We are separating out all parameters of the model into two buckets: those that
    will experience
     weight decay for regularization and those that won't (biases, and
    layernorm/embedding weights).
     We are then returning the PyTorch optimizer object.
     """
     # separate out all parameters to those that will and won't experience regularizing weight decay
     decay = set()
     no_decay = set()
     whitelist_weight_modules = (torch.nn.Linear, )
     blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
     for mn, m in mymodel.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            # random note: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times. but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if pn.endswith('bias'):
                # all biases will not be decayed
                 no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
            # weights of whitelist modules will be weight decayed
    
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif fpn.startswith('model.token_emb'):
                no_decay.add(fpn)
            # validate that we considered every parameter
     param_dict = {pn: p for pn, p in mymodel.named_parameters()}
     inter_params = decay & no_decay
     union_params = decay | no_decay
     assert len(inter_params) == 0, "parameters %s made it into both decay/no_decaysets!" % (str(inter_params), )
     assert len(param_dict.keys() - union_params) == 0, "parameters %s were notseparated into either decay/no_decay set!"  % (str(param_dict.keys() -union_params), )

     # create the pytorch optimizer object
     optim_groups = [
     {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.1},
     {"params": [param_dict[pn] for pn in sorted(list(no_decay))],"weight_decay": 0.0},
     ]
     optimizer = torch.optim.AdamW(optim_groups, lr=LEARNING_RATE, betas=(0.9,0.95))
     return optimizer


def main():

    vision_model = SimpleTransformer(
    dim = 768, # embedding
    num_unique_tokens = 10, # for CIFAR-10, use 100 for CIFAR-100
    num_layers = 12,
    heads = 8,
    max_seq_len = SEQ_LENGTH,
    ).cuda()
    model = TrainValidateWrapper(vision_model)
    model.cuda()
    pcount = count_parameters(model)
    print("count of parameters in the model = ", pcount/1e6, " million")

    train_loader, val_loader, testset = Utils.get_loaders_cifar(dataset_type="CIFAR10", img_width=224, img_height=224, batch_size=BATCH_SIZE)
    #optim = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE) # optimizer
    optim = configure_optimizers(model)

    # --------training---------
    if RESUME_TRAINING == False:
        start = 0
    else:
        checkpoint_data = torch.load('D:/mgovind/VIT/VisualTransformer/checkpoint/visiontrans_model.pt')
        model.load_state_dict(checkpoint_data['state_dict'])
        optim.load_state_dict(checkpoint_data['optimizer'])
        start = checkpoint_data['epoch']

    for i in tqdm.tqdm(range(start,NUM_BATCHES), mininterval = 10., desc ='training'):
        model.train()
        total_loss = 0
        for __ in range(GRADIENT_ACCUMULATE_EVERY):
            x, y = next(train_loader)
            x = x.cuda()
            y = y.cuda()
            loss = model(x, y)
            loss.backward()
        if (i%100 == 0):
            print(f'training loss: {loss.item()} -- iteration = {i}')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad()
        if i % VALIDATE_EVERY == 0:
            model.eval()
            # ---------save the latest model---------
            print("----------saving model-----------------")
            checkpoint_data = {
            'epoch': i,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict()
            }
          
            torch.save(checkpoint_data, "D:/mgovind/VIT/VisualTransformer/checkpoint/visiontrans_model.pt")

            model.eval()
            val_count = 300
            total_count = 0
            count_correct = 0
            with torch.no_grad():
                for v in range(0,val_count):
                    x, y = next(val_loader)
                    x = x.cuda()
                    y = y .cuda()
                    count_correct = count_correct + model.validate(x,y)
                    total_count = total_count + x.shape[0]
                    accuracy = (count_correct/total_count)*100
                print("\n-------------Test Accuracy = ", accuracy,"\n")
                # revert model to training mode
            model.train()
        if i > 30000:
            optim.param_groups[0]['lr'] = 0.25e-4
    
    
if __name__ == "__main__":
    sys.exit(int(main() or 0))
