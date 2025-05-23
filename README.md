# Self-Supervised Representation Learning Approach to 6DoF Pose Estimation — Maahum Khan
One of the greatest limitations in the field of 6 degree-of-freedom (6DoF) pose estimation is the difficulty of annotating data. In this project, I tried to create a self-supervised representation learning solution to the 6DoF pose estimation problem to accomplish two goals: 
1. Demonstrate that advancement in 6DoF pose estimation solutions do not need to be limited to the amount of annotated data available, and
2. For my own interests and experience, learn how 6DoF pose estimation, self-supervised learning, and representation learning work by going through the process of implementing them myself.

In this repository, I have provided the code I wrote to create this solution. At the moment, I have not been able to fix the testing file and see my correct evaluation results, and have not fixed errors in my ADD metric, rotational loss function, and rotational matrix to quaternion conversion calculations. For these reasons, I have not been able to accurately determine how well my model works.

## How The Model Works
First, the self-supervised layers of this model (currently made with a ResNet18 backbone, but different architectures will later be tested) is trained on the task of relative viewpoint detection. This means that, given an anchor image, a positive pair, and a negative pair, this part of the model learns to detect/understand the difference between the viewpoints from which an object is being viewed in different images. The contrastive loss function uses cosine similarity to organize the latent space such that positive pairs are mapped close to their anchor image since they have similar viewpoints, and negative pairs are mapped farther since they have different viewpoints. By understanding the viewpoints of an image based on both its features and the location of its embedding in the latent space, this aspect of the model is expected to make determining the 6DoF pose much easier and more accurate. The positive and negative pairs were made by taking advantage of the image filenames from the LINEMOD dataset that are numbered by viewpoint such that consecutive image numbers are neighboring viewpoints.

The self-supervised layers are followed by 2 fully-connected layers and a regression layer. These three layers are trained on small subset of labelled data, unlike the self-supervised layers that are trained on a larger amount of unlabelled data. These layers output a 3x3 rotational matrix and 3D translation value that it regresses. A function is also provided in test.py and train_pose.py that converts these rotational matrices to quaternions, but the function has not yet been double-checked to make sure it is correct. While training the pose regression layers, the images used pass through the self-supervised layers but they are frozen and not trained since they were already separately trained as described above. The images used are either bounding box centered 224x224 RGB images of the object, or resized/stretched out 224x224 bounding box RGB images. Bounding boxes were currently derived using the masks provided in the LINEMOD dataset.

In future work, I plan on testing more combinations of hyperparameters for training both models, trying different backbone architectures for the self-supervised layers, fixing the calculations in the pose regression layer training and testing, and replacing the mask-based preprocessing system with an object detection model finetuned for this solution.

## Files and Folders Information
The repository contains the following files and folders:
- Maahum Khan - ELEC 874 Project Presentation — Presentation slides, describes project and progress so far
- commands.txt — Contains default commands to type into Terminal to run the preprocessing.py, train_ss.py, train_pose.py, and test.py files
- dataloader_pose.py — Data loader for the full model (self-supervised layers + pose regression layers)
- dataloader_ss.py — Data loader for training only the self-supervised layers
- imglist_generator.py — File I used to create txt files with sorted lists of randomly selected images for training/tests, containing however many elements you choose
- model.py — Code for both the self-supervised model and the full pose regression model. Can be altered to include more layers, or change the self-supervised backbone
- preprocessing.py — Current version converts the full RGB images into bounding box centered 224x224 images. Function "resize_img" can be called instead to get resized 224x224 bounding box images.
- test.py — Test file. NOTE: FILE IS INCOMPLETE AND NOT CURRENTLY RUNNING. CALCULATIONS IN THIS HAVE NOT YET BEEN FIXED, AND MAY BE INCORRECT.
- train_pose.py — Training only the pose regression layers. NOTE: CALCULATIONS IN THIS HAVE NOT YET BEEN FIXED, AND MAY BE INCORRECT.
- train_ss.py — Training the self-supervised model only.

## Dataset
The dataset this code used and was based on is LINEMOD, originally obtained from [The DenseFusion GitHub (linked)](https://github.com/j96w/DenseFusion/tree/master?tab=readme-ov-file#datasets) by downloading their [LINEMOD dataset](https://drive.google.com/drive/folders/19ivHpaKm9dOrr12fzC8IDFczWRPFxho7) from Google Drive.

## Notes
Notes to keep in mind when using or looking at this project:
- Since my results (i.e. loss graphs and model .pth files) could not be uploaded due to large file size, please feel free to train your own model using commands provided in commands.txt (alter depending on your dataset and directory structures)
- Current code uploads and uses trained model weights/path from a directory in the project called "results/models/". Please alter the code according to your needs.
- Rotation loss, ADD metric, and rotational matrix to quaternion conversion calculations need to be double checked and likely need correction
- Current model results (as shown in progress presentation) are not good, future work still needs to be implemented
