# data_analysis.py
from PIL import Image
import os
import pandas as pd


def analyze_image_sizes(labels, main_folder_path):
    """Analyze and report image sizes within each label's folder
       and return a DataFrame."""
    image_sizes = {}  # Dictionary to store size counts

    for label in labels:
        folder_path = os.path.join(main_folder_path, label)
        image_files = os.listdir(folder_path)

        # Initialize dictionary for this label
        if label not in image_sizes:
            image_sizes[label] = {}

        # Count image sizes
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            with Image.open(image_path) as img:
                size = img.size  # Get image size (width, height)
                if size in image_sizes[label]:
                    image_sizes[label][size] += 1
                else:
                    image_sizes[label][size] = 1

    # Prepare data for DataFrame creation
    rows = []
    for label, sizes in image_sizes.items():
        for size, count in sizes.items():
            width, height = size
            size_label = f"{width}x{height}"
            rows.append({'Label': label, 'Size': size_label, 'Count': count})

    # Convert the list of dictionaries to a DataFrame
    df_sizes = pd.DataFrame(rows)

    return df_sizes


def filter_images_by_size(labels, main_folder_path, target_size=(360, 363)):
    """Filter images by a specific size and return a DataFrame
       with the filtered image paths."""
    images_filtered = {label: [] for label in labels}

    for label in labels:
        folder_path = os.path.join(main_folder_path, label)
        image_files = os.listdir(folder_path)

        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            try:
                with Image.open(image_path) as img:
                    if img.size == target_size:
                        images_filtered[label].append(image_path)
            except IOError:
                print(f"Error opening image: {image_path}")

    # Create a DataFrame from the collected image paths
    rows = []
    for label, paths in images_filtered.items():
        for path in paths:
            rows.append({'Label': label, 'Image_Path': path})

    return pd.DataFrame(rows)
