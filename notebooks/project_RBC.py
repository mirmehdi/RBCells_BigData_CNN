import os
from PIL import Image
classes_dir = '/Users/mehdienrahimi/Desktop/DataScienceTests/Project/Dataset/PBC_dataset_normal_DIB'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

image_size = (360,360)
pixels = []
labels = []

# Iterate through the classes
for class_idx, class_name in enumerate(os.listdir(classes_dir)):
    class_dir = os.path.join(classes_dir, class_name)
    
    if not os.path.isdir(class_dir):
        continue
    # Iterate through the images in the class directory
    for image_file in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_file)
        
        if image_file.startswith('.'):
            continue  # Skip system files like .DS_Store
        
        try:
            # Load the image
            image = Image.open(image_path)
            image = image.resize(image_size)
            
            
            # Flatten the image and store the pixel values
            flattened_image = np.array(image).flatten()
            pixels.append(flattened_image)
            
            # Store the label (class index)
            labels.append(class_idx)
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

# Convert lists to NumPy arrays
pixels_array = np.array(pixels)
labels_array = np.array(labels)