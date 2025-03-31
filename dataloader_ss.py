# Self-Supervised Task's Data loader
# By: Maahum Khan
# Used Dr. Greenspan's ELEC 874 Assignment (2025) data_loader.py file for reference


import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils import data
from torchvision import tv_tensors
import torchvision.transforms as transforms
from torchvision.transforms import v2
import random


# Index the classes of the 15 objects in LINEMOD
class_index_label_map = {
    1: 'ape', 2: 'benchvise', 3: 'bowl', 4: 'cam', 5: 'can', 6: 'cat',
    7: 'cup', 8: 'driller', 9: 'duck', 10: 'eggbox', 11: 'glue',
    12: 'holepuncher', 13: 'iron', 14: 'lamp', 15: 'phone'}


class SelfSupervisedDataset(Dataset):
    def __init__(self, dataset_dir, list_dir, transform):
        self.dataset_dir = dataset_dir
        self.transform = transform

        # Load image file names from the txt file (train.txt or test.txt)
        with open(list_dir, 'r') as f:
            self.imgs_list = f.read().splitlines()

        self.num_imgs = len(os.listdir(self.dataset_dir))
        self.all_img_paths = [os.path.join(self.dataset_dir, f"{img}") for img in sorted(os.listdir(self.dataset_dir))]


    def __len__(self):
        return len(self.imgs_list)

    # Get a valid index given the index, difference, and sign
    def get_valid_index(self, current_index, diff, sign):
        if sign == '+':
            new_index = current_index + diff
            # If the new index goes beyond the bounds, move backward instead
            if new_index >= self.num_imgs:
                new_index = current_index - diff
            return new_index
        elif sign == '-':
            new_index = current_index - diff
            # If the new index goes below zero, move forward instead
            if new_index < 0:
                new_index = current_index + diff
            return new_index

    def __getitem__(self, idx):
        # Get current anchor image
        img_key = int(self.imgs_list[idx])
        anchor_img_path = self.all_img_paths[img_key]
        anchor_img = Image.open(anchor_img_path).convert("RGB")

        # Select a positive pair (nearby image keys)
        pos_diff = random.randrange(1,4)
        pos_sign = random.choice(['+', '-'])
        pos_index = self.get_valid_index(img_key, pos_diff, pos_sign)
        pos_img_path = self.all_img_paths[pos_index]
        positive_img = Image.open(pos_img_path).convert("RGB")

        # Select a negative pair (anything 20+ away but stays within index range)
        neg_sign = random.choice(['+','-'])
        if neg_sign == '+':
            if self.num_imgs - img_key > 20:
                neg_diff = random.randrange(20, self.num_imgs - img_key)
            else:
                neg_sign = '-'
                neg_diff = random.randrange(20, img_key)
        else:
            if img_key > 20:
                neg_diff = random.randrange(20, img_key)
            else:
                neg_sign = '+'
                neg_diff = random.randrange(20, self.num_imgs - img_key)
        neg_index = self.get_valid_index(img_key, neg_diff, neg_sign)
        neg_img_path = self.all_img_paths[neg_index]
        negative_img = Image.open(neg_img_path).convert("RGB")

        # Apply transformations and/or convert to tensors
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        else:
            to_tensor = transforms.ToTensor()
            anchor_img = to_tensor(anchor_img)
            positive_img = to_tensor(positive_img)
            negative_img = to_tensor(negative_img)

        return anchor_img, positive_img, negative_img
