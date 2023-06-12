import torch
#from LFWDataset import LFWDataset
from LFWTRipletDataSet2WoutDlib import LFWTripletDataset
#from LFWTripletDatasetWithDlib import LFWTripletDataset
from LFWTestDataset import LFWTestDataset
#from LFWTestDatasetDlib import LFWTestDataset

def get_train_dataloader(data_dir, batch_size, transform=None):
    #dataset = LFWDataset(data_dir, transform=transform)
    dataset = LFWTripletDataset(data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    return dataloader

def get_test_loader(pairs_file, data_dir, batch_size, transform=None):
    dataset = LFWTestDataset(pairs_file, data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return dataloader



