# data_analysis.py
from PIL import Image
import os
import pandas as pd


def analyze_image_sizes(labels, main_folder_path):
    """
    Analyzes and reports the dimensions of images across different labeled
    directories within a specified folder.

    Parameters:
    - labels (list of str): A list of subdirectory names representing labels
      or classes in the dataset.
    - main_folder_path (str): The path to the main directory that contains the
      labeled subdirectories.

    Returns:
    - pandas.DataFrame: A DataFrame containing columns for 'Label', 'Size',
      and 'Count'. Each row represents the count of a specific image size
      (width x height) for a given label.

    Example:
    Assume a directory structure:
        - /path/to/data/
            - basophil/
                - basophil1.jpg (100x200)
                - basophil2.jpg (100x200)
            - eosinophil/
                - eosinophil1.jpg (150x200)
    Calling `analyze_image_sizes(['basophil', 'eosinophil'], '/path/to/data/')`
    would return:
        Label      | Size     | Count
        basophil   | 100x200  | 2
        eosinophil | 150x200  | 1
    """
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
    """
    Filters images by a specified size from labeled subdirectories and returns
    a DataFrame with the filtered image paths.

    Parameters:
    - labels (list of str): A list of subdirectory names representing labels
      or classes in the dataset.
    - main_folder_path (str): The path to the main directory that contains the
      labeled subdirectories.
    - target_size (tuple of int, int, optional): The desired dimensions
      (width, height) of the images to filter. Default is (360, 363).

    Returns:
    - pandas.DataFrame: A DataFrame icluding columns 'Label' and 'Image_Path'.
      Each row represents an image that meets the size criteria under its
      respective label.

    Example:
    Assume a directory structure:
        - /path/to/data/
            - basophil/
                - basophil1.jpg (360x363)
                - basophil2.jpg (400x400)
            - eosinophil/
                - eosinophil1.jpg (360x363)
    Calling `filter_images_by_size(['basophil', 'eosinophil'], '/path/to/data/'
    ,target_size=(360, 363))` would return:
        Label | Image_Path
        basophil    | /path/to/data/basophil/basophil1.jpg
        eosinophil  | /path/to/data/eosinophil/eosinophil1.jpg
    """
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


def count_images_by_label(df):
    """
    Counts the number of images per label in a DataFrame and returns
    a DataFrame with these counts.

    This function assumes the DataFrame has columns 'Label' and 'Image_Path'.
    It groups the data by 'Label', counts the occurrences, and formats the
    result into a readable DataFrame.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing the image data with columns
      'Label' and 'Image_Path'.

    Returns:
    - pandas.DataFrame: A new DataFrame with columns 'Label' and
      'Number of Images', where each row represents a label and the count of
      images associated with that label.

    Example Usage:
    >>> df_images = pd.DataFrame({
    ...    'Label': ['basophil', 'basophil', 'eosinophil'],
    ...    'Image_Path': ['path/to/image1', 'path/to/image2', 'path/to/image3']
    ... })
    >>> image_counts = count_images_by_label(df_images)
    >>> print(image_counts)
       Label  Number of Images
    0  basophil               2
    1  eosinophil             1
    """
    # Group the DataFrame by 'Label' and count the number of images
    # in each group
    image_counts = df.groupby('Label').count()

    # Rename the column for clarity
    image_counts.rename(columns={'Image_Path': 'Number of Images'},
                        inplace=True)

    # Reset the index to make 'Label' a column again
    image_counts.reset_index(inplace=True)

    return image_counts


def balance_data_by_downsampling(df, seed=1):
    """
    Balances the dataset by down-sampling all classes to the size of the
    smallest class.

    Parameters:
    - df (pandas.DataFrame): A DataFrame containing at least 'Label' and
      'Image_Path' columns, where 'Label' indicates the class of each image.
    - seed (int, optional): A seed for the random number generator used in
      sampling for reproducibility.

    Returns:
    - pandas.DataFrame: A new DataFrame where each class has been down-sampled
      to the same number of images, equivalent to the count of the smallest
      class in the original dataset.

    Example Usage:
    Assuming 'df_images' is a DataFrame with columns 'Label' and 'Image_Path':
        balanced_images = balance_data_by_downsampling(df_images)
        print(balanced_images)
    """

    # Determine the minimum image count across all labels
    min_image_count = df['Label'].value_counts().min()
    # print(f"Minimum image count for balancing: {min_image_count}")

    # Sample the same number of images from each label
    balanced_df = pd.DataFrame()
    for label in df['Label'].unique():
        sampled_df = df[df['Label'] == label].sample(n=min_image_count,
                                                     random_state=seed)
        balanced_df = pd.concat([balanced_df, sampled_df], ignore_index=True)

    # Reset index for clean formatting
    balanced_df.reset_index(drop=True, inplace=True)

    return balanced_df
