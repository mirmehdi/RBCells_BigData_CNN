import os
import pandas as pd


def load_dataset_labels(main_folder_path):
    """
    Retrieves a list of directory names within a specified parent directory,
    treating each directory name as a dataset label.

    Parameters:
    - main_folder_path (str): The file path to the directory containing
      the dataset's subdirectories.

    Returns:
    - list: A list of strings where each string is a label derived from
      the names of subdirectories within the specified main directory.

    Example:
    Assuming a directory structure as follows:
        - /path/to/data/
            - basophil/
            - eosinophil/
    Calling `load_dataset_labels('/path/to/data/')` would return
    `['basophil', 'eosinophil']`.
    """
    labels = [os.path.basename(f.path) for f in
              os.scandir(main_folder_path) if f.is_dir()]
    return labels


def load_images_with_labels(main_folder_path):
    """
    Load and count images from each labeled subfolder within
    a specified directory, organizing the results into
    a dictionary and a pandas DataFrame.

    Parameters:
    - main_folder_path (str): The file path to the directory containing
      labeled subfolders.

    Returns:
    - tuple: A tuple containing two elements:
        1. A dictionary with each key being a label and the value being
           a list of image file names.
        2. A pandas DataFrame summarizing the number of images per label.

    Example:
    Assuming a directory structure as follows:
        - /path/to/data/
            - basophil/
                - basophil1.jpg
                - basophil2.jpg
            - eosinophil/
                - eosinophil1.jpg
    Calling `load_images_with_labels('/path/to/data/')` would return
    a dictionary and a DataFrame:
    Dictionary:
        {'basophil': ['basophil1.jpg', 'basophil2.jpg'],
         'eosinophil': ['eosinophil1.jpg']}
    DataFrame:
        Label      | Number of Images
        basophil   | 2
        eosinophil | 1
    """
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

    # Convert lists to DataFrame
    data = pd.DataFrame({
        'Label': label_list,
        'Number of Images': num_images
    })

    return images_with_labels, data
