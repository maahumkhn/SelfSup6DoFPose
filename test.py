# Testing and evaluation file
# NOTE: THIS CODE HAS ERRORS, AND I HAVE NOT CORRECTED THE CALCULATIONS YET.
# By: Maahum Khan

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataloader_pose import PoseRegressDataset
from model import SelfSupervisedModel, PoseRegressionModel
import torch
import torch.nn.functional as F
import torch.linalg
import trimesh

# Function to load the 3D object from a PLY file
def load_3d_model(ply_path):
    try:
        mesh = trimesh.load_mesh(ply_path)
        # Return the 3D points of the model
        return mesh.vertices
    except Exception as e:
        print(f"Error loading PLY file: {e}")
        return None

# Convert a 3x3 rotation matrix to a quaternion
def rotation_matrix_to_quaternion(rotation_matrix):
    # Reshape from (batch_size, 9) -> (batch_size, 3, 3)
    batch_size = rotation_matrix.shape[0]
    rotation_matrix = rotation_matrix.view(64, 3, 3)  # Reshape tensor

    # Detach from computation graph, move to CPU, and convert to NumPy
    rotation_matrix_np = rotation_matrix.detach().cpu().numpy()

    # Convert rotation matrices to quaternions
    r = R.from_matrix(rotation_matrix_np)  # Convert to scipy Rotation object
    quaternions = r.as_quat()  # Get quaternion representation
    #return torch.tensor(quat, dtype=torch.float32).to(rotation_matrix.device)

# Function to apply the rotation (3x3 matrix) and translation to 3D points
def apply_pose_to_points(points, rotation_matrix, translation_vector):
    # Apply the rotation and translation to each 3D point
    rotated_points = np.dot(points, rotation_matrix.T) + translation_vector
    return rotated_points

# Function to compute ADD metric between predicted and ground truth poses
def compute_add_metric(pred_rotation, pred_translation, gt_rotation, gt_translation, object_3d_points):
    # Apply the predicted pose to the 3D points
    pred_points = apply_pose_to_points(object_3d_points, pred_rotation, pred_translation)
    # Apply the ground truth pose to the 3D points
    gt_points = apply_pose_to_points(object_3d_points, gt_rotation, gt_translation)
    # Compute the Euclidean distance between corresponding points
    distances = np.linalg.norm(pred_points - gt_points, axis=1)
    # Compute the ADD metric (average distance)
    add_metric = np.mean(distances)
    return add_metric

# Main testing function
def test_pose(model, dataset, ply_file, device):
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    model.eval()
    obj_3d_points = load_3d_model(ply_file)
    all_add_metrics = []
    num_samples = 5  # Number of samples to display
    sample_count = 0

    with torch.no_grad():
        for i, (images, gt_rot, gt_trans) in enumerate(dataloader):
            print("Shape of ground truth rotation:", gt_rot.shape)
            images, gt_rot, gt_trans = images.to(device), gt_rot.to(device), gt_trans.to(device)
            pred_rot, pred_trans = model(images)

            # Compute the ADD metric
            add = compute_add_metric(pred_rot, pred_trans.cpu().numpy(),
                                     gt_rot.cpu().numpy().reshape(3, 3), gt_trans.cpu().numpy(),
                                     obj_3d_points)
            all_add_metrics.append(add)

            # Print out results for a few samples
            if sample_count < num_samples:
                print(f"Sample {i + 1}:")
                print("Predicted Rotation (Quaternion):", rotation_matrix_to_quaternion(pred_rot))
                print("Ground Truth Rotation (Quaternion):",
                      rotation_matrix_to_quaternion(gt_rot.cpu().numpy().reshape(3, 3)))
                print("Predicted Translation:", pred_trans.cpu().numpy())
                print("Ground Truth Translation:", gt_trans.cpu().numpy())
                print(f"ADD Metric for Sample {i + 1}: {add:.4f}")
                print("-" * 20)

            sample_count += 1
    avg_add = np.mean(all_add_metrics)
    print(f"Average ADD Metric: {avg_add:.4f}")

    return avg_add


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Directories and file names
    parser.add_argument("--data_dir", type=str, default="LINEMOD/data/05/bb_rgb", help="Pre-processed bounding box RGB images directory")
    parser.add_argument("--gt_dir", type=str, default="LINEMOD/data/05/gt.yml", help="Ground truth YAML file directory")
    parser.add_argument("--list_dir", type=str, default="LINEMOD/data/05/train.txt", help="Path to train.txt (list of training images)")
    parser.add_argument("--model_dir", type=str, default="LINEMOD/data/05/models/obj_05.ply",  help="Model directory")
    # Model pths for loading model weights
    parser.add_argument("--ss_pth", type=str, default="results/models/ss_19.pth", help="Self-supervised model path")
    parser.add_argument("--p_pth", type=str, default="results/models/p_1.pth", help="Pose model path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = PoseRegressDataset(args.data_dir, args.gt_dir, args.list_dir, transform=transform)

    # Load pre-trained model
    self_sup_model = SelfSupervisedModel(embedding_dim=128).to(device)
    self_sup_model.load_state_dict(torch.load(args.ss_pth, map_location=device))
    model = PoseRegressionModel(self_sup_model, args.ss_pth, embedding_dim=128,hidden_dim=256).to(device)
    model.load_state_dict(torch.load(args.p_pth, map_location=device))

    # Evaluate the model
    avg_add = test_pose(model, dataset, args.model_dir, device)
    print(f"Final Average ADD Metric: {avg_add:.4f}")