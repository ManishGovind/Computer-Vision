import os
import dlib
import numpy as np
import torch
from torch.utils.data import Dataset

class LFWTripletDataset(Dataset):
    def __init__(self, lfw_folder_path, transform):
        image_dict = {}
        self.transform = transform
        for foldername in os.listdir(lfw_folder_path):
            folder_path = os.path.join(lfw_folder_path, foldername)
            if os.path.isdir(folder_path):
                person_name = foldername.replace('_', ' ')
                image_dict[person_name] = []
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    image_dict[person_name].append(file_path)

        triplet_list = []
        for person in image_dict:
            person_images = image_dict[person]
            for i, anchor_path in enumerate(person_images):
                for j in range(i+1, len(person_images)):
                    pos_path = person_images[j]
                    neg_person = person
                    while neg_person == person:
                        neg_person = np.random.choice(list(image_dict.keys()))
                    neg_path = np.random.choice(image_dict[neg_person])
                    triplet = (anchor_path, pos_path, neg_path)
                    triplet_list.append(triplet)

        self.triplets = np.array(triplet_list)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_path, pos_path, neg_path = self.triplets[idx]
        anchor_img = dlib.load_rgb_image(anchor_path)
        pos_img = dlib.load_rgb_image(pos_path)
        neg_img = dlib.load_rgb_image(neg_path)
        return (torch.from_numpy(anchor_img.transpose(2, 0, 1)),
                torch.from_numpy(pos_img.transpose(2, 0, 1)),
                torch.from_numpy(neg_img.transpose(2, 0, 1)))

