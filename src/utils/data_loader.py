import os
from PIL import Image
import pandas as pd


def load_dataset_labels(main_folder_path):
    """Load dataset labels based on folder names."""
    labels = [os.path.basename(f.path) for f in
              os.scandir(main_folder_path) if f.is_dir()]
    return labels


def load_images_with_labels(main_folder_path):
    """Load images from each subfolder and count them,
       returning both the images and a summary DataFrame."""
    images_with_labels = {}  # {label: images list}

    # Gather images from each label's folder
    labels = load_dataset_labels(main_folder_path)
    for label in labels:
        folder_path = os.path.join(main_folder_path, label)
        images = os.listdir(folder_path)
        images_with_labels[label] = images

    # Create lists for DataFrame creation
    label_list = []
    num_images = []

    for label, images in images_with_labels.items():
        label_list.append(label)
        num_images.append(len(images))
        # print(f"Label: {label}, ---> Number of Images: {len(images)}")

    # Convert lists to DataFrame
    data = pd.DataFrame({
        'Label': label_list,
        'Number of Images': num_images
    })

    return images_with_labels, data
