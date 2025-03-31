# Self-Supervised Model Training File
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
from dataloader_ss import SelfSupervisedDataset
from model import SelfSupervisedModel
import sys
import datetime

 # Define the Cosine Similarity loss
# Pushes positive pairs closer together and negative ones farther apart in latent space
def contrastive_loss(anchor, positive, negative, margin=0.5):
    cos_sim = nn.CosineSimilarity(dim=1)
    pos_sim = cos_sim(anchor, positive)
    neg_sim = cos_sim(anchor, negative)
    loss = torch.mean(torch.relu(margin + neg_sim - pos_sim))
    return loss

def train_ss(epochs, optimizer, model, dataset, scheduler, lossplot, device):
    dataloader = DataLoader(dataset, batch_size=args.b, shuffle=True, drop_last=True)
    total_losses = []

    print("Training Self-Supervised Model...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        batch_count = 0
        for anchor, positive, negative in dataloader:
            batch_count += 1
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            optimizer.zero_grad()
            anchor_embed = model(anchor)
            positive_embed = model(positive)
            negative_embed = model(negative)

            loss = contrastive_loss(anchor_embed, positive_embed, negative_embed)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"---> Batch {batch_count}/{len(dataloader)} - Loss: {loss.item()}")
        scheduler.step(epoch_loss)
        total_losses += [epoch_loss / len(dataloader)]
        print('{}: Epoch {}, Training Loss {}'.format(datetime.datetime.now(), epoch, epoch_loss / len(dataloader)))

    plt.figure(1,figsize=(12, 7))
    plt.plot(total_losses, label='Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc=1)
    plt.title('Training Loss Over Time')
    plt.savefig(lossplot)
    plt.show()

    return total_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Directory and file names
    parser.add_argument("--data_dir", type=str, default="LINEMOD/data/05/rgb", help="Pre-processed bounding box RGB images directory")
    parser.add_argument("--list_dir", type=str, default="LINEMOD/data/05/train.txt", help="Path to train.txt (list of training images)")
    parser.add_argument("--loss_plot", type=str, default="results/loss_plots/ss_rgb_1.png", help="Loss graph directory")
    parser.add_argument("--pth", type=str, default="results/models/ss_rgb_1.pth", help="Path to save trained model")
    # Hyperparameters
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension for model")
    parser.add_argument("--e", type=int, default=15, help="Number of epochs")
    parser.add_argument("--b", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        #transforms.RandomRotation(40),  # Random rotation
        #transforms.RandomHorizontalFlip(),  # Horizontal flip
        #transforms.RandomVerticalFlip(),  # Vertical flip
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Color jitter
        #transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random resized crop
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),  # Affine transformations
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = SelfSupervisedDataset(args.data_dir, args.list_dir, transform=transform)

    model = SelfSupervisedModel(embedding_dim=args.embed_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1, last_epoch=-1)

    train_losses = train_ss(args.e, optimizer, model, dataset, scheduler, args.loss_plot, device)
    torch.save(model.state_dict(), args.pth)