import os
import dlib
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class LFWTripletDataset(Dataset):
    def __init__(self, lfw_folder_path, transform):
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        image_dict = {}
        detector = dlib.get_frontal_face_detector()
        for foldername in os.listdir(lfw_folder_path):
            folder_path = os.path.join(lfw_folder_path, foldername)
            if os.path.isdir(folder_path):
                person_name = foldername.replace('_', ' ')
                image_dict[person_name] = []
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    img = dlib.load_rgb_image(file_path)
                    # use detector to find the face in the image
                    dets = detector(img, 1)
                    if len(dets) == 1: # if one face is detected
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

        detector = dlib.get_frontal_face_detector()
        anchor_dets = detector(anchor_img, 1)
        pos_dets = detector(pos_img, 1)
        neg_dets = detector(neg_img, 1)

        if len(anchor_dets) == 1 and len(pos_dets) == 1 and len(neg_dets) == 1:
            # crop the face region using the bounding box
            anchor_box = anchor_dets[0]
            pos_box = pos_dets[0]
            neg_box = neg_dets[0]
            anchor_img = dlib.crop_image(anchor_img, anchor_box)
            pos_img = dlib.crop_image(pos_img, pos_box)
            neg_img = dlib.crop_image(neg_img, neg_box)
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)
            return (anchor_img, pos_img, neg_img)
        else:
            # if face detection fails, return None
            return None
