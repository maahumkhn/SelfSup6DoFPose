# Pre-processing images (alternative to object detection)
# By: Maahum Khan

import numpy as np
import torch
import cv2
print("cv2 imported")
import torchvision.transforms as transforms
print("torchvision imported")
import argparse
print("argparse imported")
import os
print("os imported")

# Function that uses the mask image of an object to get
# the bounding box coordinates around it
def get_bounding_box(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Find mask contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if object is even in the mask
    if len(contours) == 0:
        return None

    # Get bounding box around the object and return
    x, y, w, h = cv2.boundingRect(contours[0])
    return x, y, w, h

# Function that pre-processes image by cropping to bounding box
# to focus on the object, and resizing to 22x224 image
def preprocess_img(img_path, mask_path):
    # Get bounding box and its center
    x, y, w, h = get_bounding_box(mask_path)

    # Check if object is even in the mask
    if x is None:
        raise ValueError("Pre-Processing: No object found in the mask")

    # Get bounding box center
    center_x = x + w // 2
    center_y = y + h // 2

    # Read image and its size
    image = cv2.imread(img_path)
    img_h, img_w, _ = image.shape

    # Define the cropping area 224x224 centered around BBox
    crop_x1 = max(center_x - 112, 0)
    crop_x2 = min(center_x + 112, img_w)
    crop_y1 = max(center_y - 112, 0)
    crop_y2 = min(center_y + 112, img_h)

    # Adjust the cropping if it's smaller than 224x224/near image boundaries
    if (crop_x2 - crop_x1) < 224:
        if crop_x1 == 0:
            crop_x2 = min(224, img_w)
        elif crop_x2 == img_w:
            crop_x1 = max(img_w - 224, 0)

    if (crop_y2 - crop_y1) < 224:
        if crop_y1 == 0:
            crop_y2 = min(224, img_h)
        elif crop_y2 == img_h:
            crop_y1 = max(img_h - 224, 0)

    # Crop the image
    cropped_img = image[crop_y1:crop_y2, crop_x1:crop_x2]
    if cropped_img.shape[0] != 224 or cropped_img.shape[1] != 224:
        print("IMAGE NOT 224x224!")

    return cropped_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_dir", type=str, default="LINEMOD/data/05/rgb")
    parser.add_argument("--mask_dir", type=str, default="LINEMOD/data/05/mask")
    parser.add_argument("--output_dir", type=str, default="LINEMOD/data/05/bb_rgb")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Get list of images (same order in RGB file and mask file)
    img_files = os.listdir(args.rgb_dir)

    # For every image:
    for img_file in img_files:
        img_path = os.path.join(args.rgb_dir, img_file)
        mask_path = os.path.join(args.mask_dir, img_file) # Corresponding mask has same file name

        # Make sure mask file exists
        if not os.path.exists(mask_path):
            print(f"Pre-processing: Mask file for {img_file} not found. Skipping!")
            continue

        try:
            # Pre-process the image
            processed_img = preprocess_img(img_path, mask_path)

            # Save processed image (cropped and resized) as a .png file
            output_file = os.path.join(args.output_dir, f"{os.path.splitext(img_file)[0]}.png")
            cv2.imwrite(output_file, processed_img)
            print(f"Processed and saved: {output_file}")

        except ValueError as e:
            print(f"Skipping {img_file}: {e}")
