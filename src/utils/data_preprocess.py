import os
import pandas as pd
from PIL import Image


def crop_image(image_path, output_size=(360, 360)):
    """
    Crop an image to the desired output size.

    Args:
    image_path (str): Path to the image to be cropped.
    output_size (tuple): Desired output size (height, width).

    Returns:
    Image: Cropped image.
    """
    with Image.open(image_path) as img:
        # Calculate coordinates to crop the image to 360x360
        left = (img.width - output_size[1]) // 2
        top = (img.height - output_size[0]) // 2
        right = (img.width + output_size[1]) // 2
        bottom = (img.height + output_size[0]) // 2

        # Crop the center of the image
        img_cropped = img.crop((left, top, right, bottom))
        return img_cropped


def crop_images_and_update_df(df, output_dir):
    """
    Process each image in the DataFrame, crop it, save the new image,
    and update the DataFrame.

    Args:
    df (DataFrame): DataFrame containing 'Label' and 'Image_Path'.
    directory_to_save (str): Directory to save the cropped images.

    Returns:
    DataFrame: Updated DataFrame with new paths for the cropped images.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prepare a list to hold new data for DataFrame
    data = []

    for index, row in df.iterrows():
        # Generate new path for the cropped image
        file_name = f"{os.path.basename(row['Image_Path'])}"
        new_image_path = os.path.join(output_dir, file_name)

        # Crop the image and save it
        cropped_image = crop_image(row['Image_Path'])
        cropped_image.save(new_image_path)

        # Append new path along with the label to the data list
        data.append({'Label': row['Label'], 'Image_Path': new_image_path})

    # Create a new DataFrame with updated data
    new_df = pd.DataFrame(data)
    return new_df
