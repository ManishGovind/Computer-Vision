import torch
import torchvision
import numpy as np
import os
from PIL import Image

class LFWTestDataset(torch.utils.data.Dataset):

    def __init__(self, pairs_file, data_dir, transform=None):
        self.pairs_file = pairs_file
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Load the image paths and labels from the pairs file
        with open(pairs_file, 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    label = 1
                    name1, idx1, idx2 = parts
                    path1 = os.path.join(data_dir, name1, f'{name1}_{idx1.zfill(4)}.jpg')
                    path2 = os.path.join(data_dir, name1, f'{name1}_{idx2.zfill(4)}.jpg')
                elif len(parts) == 4:
                    label = 0
                    name1, idx1, name2, idx2 = parts
                    path1 = os.path.join(data_dir, name1, f'{name1}_{idx1.zfill(4)}.jpg')
                    path2 = os.path.join(data_dir, name2, f'{name2}_{idx2.zfill(4)}.jpg')
                else:
                    raise ValueError(f'Invalid line: {line.strip()}')
                self.image_paths.append((path1, path2))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path1, path2 = self.image_paths[index]
        label = self.labels[index]

        image1 = Image.open(path1).convert('RGB')
        image2 = Image.open(path2).convert('RGB')

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, label


