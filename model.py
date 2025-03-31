# Models file
# By: Maahum Khan
# Self-Supervised pretext task of relative viewpoint detection with a ResNet
# backbone + 6DoF pose regression head

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import os
import json
from torchvision import tv_tensors
from torchvision.transforms import v2

# Self-supervised model (ResNet18 Backbone for Relative Viewpoint Detection)
class SelfSupervisedModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SelfSupervisedModel, self).__init__()

        # Use ResNet18 backbone without final classification layer
        self.backbone = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Add an FC layer for embeddings for contrastive learning
        self.embedding_layer = nn.Linear(512, embedding_dim)


    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        embeddings = self.embedding_layer(features)
        return embeddings



# Pose Regression Model (Frozen Self-Supervised Layers + Supervised Pose Regression Head)
class PoseRegressionModel(nn.Module):
    def __init__(self, self_sup_model, ss_weights, embedding_dim=128, hidden_dim=256):
        super(PoseRegressionModel, self).__init__()

        # Use our frozen, pre-trained self-supervised model for feature extraction
        self.feature_extractor = self_sup_model
        if ss_weights is not None:
            self.feature_extractor.load_state_dict(torch.load(ss_weights))

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Pose regression layers for translation (3D) and rotation (3x3 matrix as a vector)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_translation = nn.Linear(hidden_dim, 3)
        self.fc_rotation = nn.Linear(hidden_dim, 9)


    def forward(self, x):
        # Pass through self-supervised model for feature extraction
        embeddings = self.feature_extractor(x)

        # Pass through ReLU activation hidden layers (combine ground truth & features info)
        x = F.relu(self.fc1(embeddings))
        x = F.relu(self.fc2(x))

        # Output regressed rotation & translation values
        rotation = self.fc_rotation(x)
        translation = self.fc_translation(x)

        return rotation, translation
