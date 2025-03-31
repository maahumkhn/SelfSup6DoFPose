# Full Model Pose Regression Data Loader
# By: Maahum Khan

import torch
from torch.utils.data import Dataset, DataLoader
import os
import yaml
from PIL import Image
import torchvision.transforms as transforms

class PoseRegressDataset(Dataset):
    def __init__(self, dataset_dir, gt_dir, list_dir, transform):
        self.dataset_dir = dataset_dir
        self.transform = transform

        # Get list of images to use (either train.txt or test.txt)
        with open(list_dir, 'r') as f:
            self.imgs_list = [line.strip() for line in f.readlines()]

        # Load gt.yml file containing ground truth data
        with open(gt_dir, 'r') as f:
            self.gt_data = yaml.load(f, Loader=yaml.FullLoader)

        self.all_img_paths = [os.path.join(self.dataset_dir, f"{img}") for img in sorted(os.listdir(self.dataset_dir))]

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, idx):
        # Get image
        #img_name = self.imgs_list[idx].strip() # Strip newline
        img_key = int(self.imgs_list[idx])
        img_path = self.all_img_paths[img_key]
        image = Image.open(img_path).convert("RGB")

        # Get ground truth rotation and translation data for image
        gt = self.gt_data[img_key]
        rotation = torch.tensor(gt[0]['cam_R_m2c'], dtype=torch.float32)
        translation = torch.tensor(gt[0]['cam_t_m2c'], dtype=torch.float32)

        # Apply any transformations made
        if self.transform:
            image = self.transform(image)
        else:
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)

        return image, rotation, translation
