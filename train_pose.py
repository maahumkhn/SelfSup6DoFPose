# Pose Regression Layers Training File
# By: Maahum Khan

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloader_pose import PoseRegressDataset
from model import SelfSupervisedModel, PoseRegressionModel
import sys
import datetime
import torch
import torch.nn.functional as F
import torch.linalg
from scipy.spatial.transform import Rotation as R


# Convert a 3x3 rotation matrix to a quaternion — NOTE: HAVE NOT CHECKED IF IT'S CORRECT
def rotation_matrix_to_quaternion(rotation_matrix):
    # Ensure the rotation matrix has the correct shape (3, 3) for a single example
    if rotation_matrix.shape == (9,):  # This means it's a flattened 3x3 matrix
        rotation_matrix = rotation_matrix.reshape(3, 3)

    # If you have a batch, ensure it's in shape (N, 3, 3)
    elif rotation_matrix.ndim == 2 and rotation_matrix.shape[1] == 9:
        rotation_matrix = rotation_matrix.reshape(-1, 3, 3)

    # Now pass the correctly shaped matrix to R.from_matrix()
    r = R.from_matrix(rotation_matrix.detach().cpu().numpy())
    quaternions = r.as_quat()  # Convert to quaternion
    return torch.tensor(quaternions, dtype=torch.float32).to(rotation_matrix.device)

# Loss function — NOTE: I HAVE NOT FIXED THESE CALCULATIONS YET, I DON'T THINK THEY'RE CORRECT
def pose_loss(rotation_pred, translation_pred, rotation_gt, translation_gt, lambda_trans=0.05):
    # Compute geodesic loss for rotation
    R_pred = rotation_pred.view(-1, 3, 3)  # Reshape to matrix form
    R_gt = rotation_gt.view(-1, 3, 3)
    R_diff = torch.bmm(R_pred.transpose(1, 2), R_gt) - torch.eye(3, device=R_pred.device).unsqueeze(0)
    rot_loss = torch.norm(R_diff, p='fro', dim=(1, 2)).mean()  # Frobenius norm of (R^T R_gt - I)

    # Compute L2 loss for translation
    trans_loss = F.mse_loss(translation_pred, translation_gt)

    # Weighted sum
    total_loss = rot_loss + lambda_trans * trans_loss
    return total_loss


def train_pose(epochs, optimizer, model, dataset, scheduler, lossplot, device):
    dataloader = DataLoader(dataset, batch_size=args.b, shuffle=True, drop_last=True)
    total_losses = []

    print("Training Pose Regression Model...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        batch_count = 0
        for images, gt_rot, gt_trans in dataloader:
            batch_count += 1
            images, gt_rot, gt_trans = images.to(device), gt_rot.to(device), gt_trans.to(device)
            optimizer.zero_grad()
            pred_rot, pred_trans = model(images)

            loss = pose_loss(pred_rot, pred_trans, gt_rot, gt_trans, 0.01)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"---> Batch {batch_count}/{len(dataloader)} - Loss: {loss.item()}")

            '''# Print one example result at the end of each epoch
            example_idx = np.random.randint(0, images.shape[0])  # Select a random sample from last batch
            example_pred_rot = pred_rot[example_idx]
            example_gt_rot = gt_rot[example_idx]
            example_pred_trans = pred_trans[example_idx]
            example_gt_trans = gt_trans[example_idx]'''

        scheduler.step()
        total_losses += [epoch_loss / len(dataloader)]
        print('{}: Epoch {}, Training Loss {}'.format(datetime.datetime.now(), epoch, epoch_loss / len(dataloader)))


        '''print(f"\nExample Result at End of Epoch {epoch}:")
        print("Predicted Rotation (Quaternion):", rotation_matrix_to_quaternion(example_pred_rot))
        print("Ground Truth Rotation (Quaternion):", rotation_matrix_to_quaternion(example_gt_rot))
        print("Predicted Translation:", example_pred_trans.cpu().numpy())
        print("Ground Truth Translation:", example_gt_trans.cpu().numpy(), "\n")'''


    plt.figure(1, figsize=(12, 7))
    plt.plot(total_losses, label='Pose Regression Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc=1)
    plt.title('Pose Regression Training Loss Over Time')
    plt.savefig(lossplot)
    plt.show()

    return total_losses



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Directories and file names
    parser.add_argument("--data_dir", type=str, default="LINEMOD/data/05/bb_rgb", help="Pre-processed bounding box RGB images directory")
    parser.add_argument("--gt_dir", type=str, default="LINEMOD/data/05/gt.yml", help="Ground truth YAML file directory")
    parser.add_argument("--list_dir", type=str, default="LINEMOD/data/05/train.txt", help="Path to train.txt (list of training images)")

    parser.add_argument("--loss_plot", type=str, default="results/loss_plots/p_loss_1.png",  help="Pose regression training loss graph")
    # Hyperparameters
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension for model")
    parser.add_argument("--hid_dim", type=int, default=256, help="Hidden dimension for pose regression layers")
    parser.add_argument("--e", type=int, default=10, help="Epochs")
    parser.add_argument("--b", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    # Model pths for loading and saving model weights
    parser.add_argument("--ss_pth", type=str, default="results/models/ss_19.pth", help="Self-supervised model path")
    parser.add_argument("--p_pth", type=str, default="results/models/p_1.pth", help="Pose model path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        #transforms.RandomRotation(40),  # Random rotation
        #transforms.RandomHorizontalFlip(),  # Horizontal flip
        #transforms.RandomVerticalFlip(),  # Vertical flip
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Color jitter
        #transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random resized crop
        #transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),  # Affine transformations
        transforms.ToTensor(),  # Convert to tensor
    ])

    dataset = PoseRegressDataset(args.data_dir, args.gt_dir, args.list_dir, transform=transform)

    # Load pre-trained self-supervised model and freeze it
    self_sup_model = SelfSupervisedModel(embedding_dim=args.embed_dim).to(device)
    self_sup_model.load_state_dict(torch.load(args.ss_pth, map_location=device))
    for param in self_sup_model.parameters():
        param.requires_grad = False  # Freeze self-supervised layers

    # Initialize pose regression model
    # NOTE: Embedding dimension must match self-supervised model embedding dimension
    model = PoseRegressionModel(self_sup_model, args.ss_pth, args.embed_dim, args.hid_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma=0.1, last_epoch=-1)

    train_losses = train_pose(args.e, optimizer, model, dataset, scheduler, args.loss_plot, device)
    torch.save(model.state_dict(), args.p_pth)


