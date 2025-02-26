# main_script.py
# %%
'''
Here we get a path where this script located. It will be our base path and the rest will be relative path
'''
import os
import inspect

# Fallback mechanism to get the directory when __file__ is not defined
try:
    # This will work when running as a script
    script_directory = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # This will work in an interactive environment
    script_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


print(script_directory)
# %%
# Import everything from our custom imports module
# Please add your libraries to the import module
from utils.imports import *
# Please add any functions related to loading/working data to data_loader
# module
from utils.data_loader import load_dataset_labels, load_images_with_labels
# Please add any plot functions to the visualize module
from visualization.visualize import (plot_image_distribution,
                                     display_sample_images,
                                     create_interactive_bar_plot,
                                     visualize_images,
                                     dist_cell_area
                                     )
# Please add any data analysis functions to data_analysis module
from utils.data_analysis import (analyze_image_sizes,
                                 filter_images_by_size,
                                 count_images_by_label,
                                 balance_data_by_downsampling,
                                 segmentation_openCV,
                                 )


# Define the path to the dataset
# For me, PBC_dataset_normal_DIB is unzipped folder including 8 subfilders
main_folder_path = ('/Users/mehdienrahimi/Desktop/DataScienceTests/Project/Dataset/PBC_dataset_normal_DIB')

# # Load dataset labels
# # Here, we can save the name of classes that we need them a lot
# labels = load_dataset_labels(main_folder_path)
# # print("Subfolder names (labels):", labels)

# """
# Note: We can see some variations in the number of images per class,
# which could potentially lead to issues of class imbalance in
# a machine learning context, particularly for models such as CNNs that are
# commonly used for image classification tasks.
# This can cause classification models to be biased towards the classes with
# a higher number of instances,  potentially leading to poor model performance,
# especially for the minority class.
# """

# Load images and their counts
images_with_labels, image_counts_df = load_images_with_labels(main_folder_path)
# print(image_counts_df)

# Display a bar plot to explore visually the distribution of image classes
plot_image_distribution(
   df=image_counts_df,
   title='Distribution of Image Classes in the Dataset',
   xlabel='Cell Type (Label)',
   ylabel='Number of Images'
)

# Display sample images from each label
# display_sample_images(labels, main_folder_path)

# Get the DataFrame of image sizes
df_sizes = analyze_image_sizes(labels, main_folder_path)
# print(df_sizes)

# Display df_sizes dataFrame
create_interactive_bar_plot(
   df=df_sizes,
   x_column='Label',
   y_column='Count',
   color_column='Size',
   title='Distribution of Image Sizes Across Classes',
   xlabel='Cell Type',
   ylabel='Number of Images'
)

"""
Note: The results show that the most common image size across all cell types
is 360x363 pixels, with just small variations in other dimensions.
This size variance in each class indicates the need for standardization
or a specific focus on the main size categories to ensure consistent
analysis and model training. Thus, my approach here is to remove minor
image size groups from each class such that images are
the same size across all cell types.
"""
# Here, I remove minority groups of image sizes to have
# the same size images in all Classes
# Generate DataFrame of images of size 360x363
df_images = filter_images_by_size(labels, main_folder_path,
                                  target_size=(360, 363))

# Check the first few entries in the DataFrame
# print(df_images.head())
# print(df_images.info())

# ----------------------------------------------------------------------------
# Here, I have saved both CSV and JASON format of filtered dataframe
# You don't need to uncomment this part
# Save the DataFrame to a CSV and jason files for easy sharing
# data_folder_path = '/apr24_bds_int_blood_cells/src/data'

# Save the DataFrame to a CSV file
# csv_file_path = os.path.join(data_folder_path, 'uniform_image_sizes.csv')
# df_images.to_csv(csv_file_path, index=False)

# Save the DataFrame to a JSON file
# json_file_path = os.path.join(data_folder_path, 'uniform_image_sizes.json')
# df_images.to_json(json_file_path, orient='records')
# ----------------------------------------------------------------------------
image_counts = count_images_by_label(df_images)
# print(image_counts)

# Display image_counts dataFrame
plot_image_distribution(
    df=image_counts,
    title='Distribution of Image Classes (360x363) in the Dataset',
    xlabel='Cell Type (Label)',
    ylabel='Number of Images'
)

# Resolving Imbalance in Data with Down-Sampling
"""
Down-sampling is employed to address data imbalance across different classes.
This technique involves identifying the class with the minimum number of
images and then reducing the image counts in other classes to match this
minimum count.
"""
balanced_images = balance_data_by_downsampling(df_images)
# print(balanced_images)

balanced_image_counts = count_images_by_label(balanced_images)
# print(balanced_image_counts)

# Display image_counts dataFrame
plot_image_distribution(
    df=balanced_image_counts,
    title='Distribution of Balanced Image Classes in the Dataset',
    xlabel='Cell Type (Label)',
    ylabel='Number of Images'
)

# ----------------------------------------------------------------------------
# Here, I have saved both CSV and JASON format of balanced dataframe
# You don't need to uncomment this part
# Save the DataFrame to a CSV and jason files for easy sharing
# data_folder_path = '/apr24_bds_int_blood_cells/src/data'

# Save the DataFrame to a CSV file
# csv_file_path = os.path.join(data_folder_path, 'balanced_image_paths.csv')
# df_images.to_csv(csv_file_path, index=False)

# Save the DataFrame to a JSON file
# json_file_path = os.path.join(data_folder_path, 'balanced_image_paths.json')
# df_images.to_json(json_file_path, orient='records')
# ----------------------------------------------------------------------------

### Image segmentation part
df_segmentation, first_images = segmentation_openCV(main_folder_path)
visualize_images(first_images, main_folder_path)
dist_cell_area(df_segmentation)

