import json
import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import shutil

# Load annotations from JSON file
image_folder = '/test'  # Folder with 17,000 images
mask_folder = '/Users/mehdienrahimi/Desktop/DataScienceTests/Project/Dataset/PBC_dataset_normal_DIB/Unet'  # Folder with JSON mask files
train_folder = '/Users/mehdienrahimi/Desktop/DataScienceTests/Project/Dataset/PBC_dataset_normal_DIB/Unet/train'  # New folder to store images and masks

base_dir = "/Users/mehdienrahimi/Desktop/DataScienceTests/Project/Dataset"
json_path = '/Users/mehdienrahimi/Desktop/DataScienceTests/Project/Dataset/PBC_dataset_normal_DIB/Unet/Unet.json'


with open(json_path) as f:
    annotations = json.load(f)

data =annotations['_via_img_metadata']
#################################################
# # here 1) we extract annoted images from json file 2) find them in all images folder calld test file 3) move them to another folder called train folde3. 
# for key,value in data.items():
#     filename = value['filename']
#     src_image_path = os.path.join(image_folder, filename)
#     dest_image_path = os.path.join(train_folder, filename)
#     shutil.move(src_image_path, dest_image_path)

# Create mask_folder if it doesn't exist

# Define paths
image_folder = '/Users/mehdienrahimi/Desktop/DataScienceTests/Project/Dataset/PBC_dataset_normal_DIB/Unet/train'  # Folder with 17,000 images
annotation_file = mask_folder  # JSON annotations file
mask_folder = '/Users/mehdienrahimi/Desktop/DataScienceTests/Project/Dataset/PBC_dataset_normal_DIB/Unet/label'  # New folder to store images and masks


# Define cell type mapping to unique integers
cell_type_mapping = {
    'BA': 1,
    'ERB': 2,
    'LY': 3,
    'SNE': 4,
    'EO': 5,
    'PMY': 6,
    'MO': 7,
    'Plat': 8,
    'Back': 0  # Artifacts labeled as background
}

# Process each image
for key, value in data.items():
    filename = value['filename']
    img_path = os.path.join(image_folder, filename)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    
    if img is None:
        print(f"Image {filename} not found in {image_folder}. Skipping.")
        continue
    
    h, w, _ = img.shape

    # Initialize mask
    mask = np.zeros((h, w), dtype=np.uint8)

    regions = value["regions"]
    for region in regions:
        shape_attributes = region.get("shape_attributes", {})
        region_attributes = region.get("region_attributes", {})
        cell_type = region_attributes.get("type", "Back")  # Default to background if not specified



        # Ensure the region is a cell type and not background
        if cell_type not in cell_type_mapping or cell_type == "Back":
            continue  # Skip background and undefined regions

        # Ensure the expected keys exist in shape_attributes
        if "all_points_x" not in shape_attributes or "all_points_y" not in shape_attributes:
            print(f"Missing 'all_points_x' or 'all_points_y' in region: {region}. Skipping this region.")
            continue

        x_points = shape_attributes["all_points_x"]
        y_points = shape_attributes["all_points_y"]

                # Draw filled contour for the region with the corresponding cell type value
        contours = []
        for x,y in zip(x_points,y_points):
            contours.append((x,y))
        contours = np.array(contours)

        cv2.fillPoly(mask,[contours],-1,255,1)




        if len(contours) == 0:
            print(f"Contours are empty for region: {region}. Skipping.")
            continue

        # Debugging: print contour and cell type information
        print(f"Processing {filename}, cell type: {cell_type}, contours: {contours}")


    # Save the mask
    mask_path = os.path.join(mask_folder, filename)
    cv2.imwrite(mask_path, mask)

print("Masks have been created and saved.")
