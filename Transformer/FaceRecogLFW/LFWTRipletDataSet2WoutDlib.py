import os
import dlib
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random

class LFWTripletDataset(torch.utils.data.Dataset):
    def __init__(self, lfw_folder_path, transform):
        self.transform = transform
        image_dict = {}
        for foldername in os.listdir(lfw_folder_path):
            folder_path = os.path.join(lfw_folder_path, foldername)
            if os.path.isdir(folder_path):
                person_name = foldername 
                image_dict[person_name] = []
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)

                    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')] # list of image filenames for a person
                    if len(image_paths) < 2: # number of images per person < num
                        continue
                    image_dict[person_name].append(file_path)

        triplet_list = []
        for person in image_dict:
            person_images = image_dict[person]
            for i, anchor_path in enumerate(person_images):
                for j in range(i+1, len(person_images)):
                    pos_path = person_images[j]
                    neg_person = person
                    while neg_person == person:
                        neg_person = random.choice(list(image_dict.keys()))
                        zz = list(image_dict[neg_person])
                        if len(zz) < 2:
                            neg_person = person
                            continue
                    neg_path = random.choice(list(image_dict[neg_person]))
                    triplet = (anchor_path, pos_path, neg_path)
                    triplet_list.append(triplet)
  
        self.triplets = triplet_list

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_path, positive_path, negative_path = self.triplets[idx]
        anchor_image = Image.open(anchor_path).convert('RGB')
        positive_image = Image.open(positive_path).convert('RGB')
        negative_image = Image.open(negative_path).convert('RGB')

        if self.transform is not None:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image
