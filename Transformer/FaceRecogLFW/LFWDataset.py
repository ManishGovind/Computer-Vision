import torch
import torchvision
import numpy as np
import os
from PIL import Image

class LFWDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for person_dir in os.listdir(data_dir): # each person folder in the data folder
            person_path = os.path.join(data_dir, person_dir)
            if not os.path.isdir(person_path):
                continue

            image_paths = [os.path.join(person_path, f) for f in os.listdir(person_path) if f.endswith('.jpg')] # list of image filenames for a person
            if len(image_paths) < 2: # number of images per person < num
                continue

            self.image_paths += image_paths
            self.labels += [len(self.labels)] * len(image_paths)

        self.num_classes = len(set(self.labels))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        anchor_path = self.image_paths[index]
        anchor_label = self.labels[index]

        positive_paths = [p for p, l in zip(self.image_paths, self.labels) if l == anchor_label and p != anchor_path]
        if not positive_paths:
            return self.__getitem__((index + 1) % len(self))

        positive_path = np.random.choice(positive_paths)

        negative_paths = [p for p, l in zip(self.image_paths, self.labels) if l != anchor_label]
        negative_path = np.random.choice(negative_paths)

        anchor_image = Image.open(anchor_path).convert('RGB')
        positive_image = Image.open(positive_path).convert('RGB')
        negative_image = Image.open(negative_path).convert('RGB')

        if self.transform is not None:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image
