import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import plotly.express as px
import cv2

def plot_image_distribution(df, title, xlabel, ylabel):
    """
    Creates and displays a bar plot to visualize the distribution of image
    classes using data from a DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data to plot. It
      should have columns 'Label' and 'Number of Images'.
    - title (str): The title of the plot.
    - xlabel (str): The label for the x-axis.
    - ylabel (str): The label for the y-axis.

    Returns:
    - None: This function does not return any value but displays
      a matplotlib plot.

    Example Usage:
    Assuming 'df_images' is a DataFrame with the following columns:
        - Label: ['basophil', 'eosinophil']
        - Number of Images: [150, 200]
    Calling `plot_image_distribution(df_images, 'Distribution of Cell Images'
    , 'Cell Type', 'Number of Images')`
    would display a bar plot illustrating the distribution of basophil and
    eosinophil images in the dataset.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Label', y='Number of Images', data=df, palette='Set2')
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def display_sample_images(labels, main_folder_path):
    """
    Displays the first image from each labeled subdirectory in a 2x4 grid
    layout.

    Parameters:
    - labels (list of str): A list of subdirectory names, each representing a
      distinct class or label in the dataset.
    - main_folder_path (str): The path to the main directory containing the
      labeled subfolders.

    Returns:
    - None: This function does not return any value but displays a grid of
      images using matplotlib.

    Example Usage:
    Assume you have a directory '/path/to/data' with subdirectories 'basophil',
    'eosinophil', etc., each containing image files.
    Calling `display_sample_images(['basophil', 'eosinophil'],
    '/path/to/data')` would display the first image from each of the 'basophil'
    and 'eosinophil' subdirectories in a grid.
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()

    for i, label in enumerate(labels):
        folder_path = os.path.join(main_folder_path, label)
        # Take the first image in the folder for displaying
        image_file = os.listdir(folder_path)[0]
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path)
        axes[i].imshow(image)
        axes[i].set_title(label, size=15)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def create_interactive_bar_plot(df, x_column, y_column, color_column,
                                title, xlabel, ylabel):
    """
    Creates and displays an interactive bar plot using Plotly Express.
    The plot is customizable with hover data and labels provided as parameters.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data to plot.
    - x_column (str): The column name to be used for the x-axis.
    - y_column (str): The column name to be used for the y-axis and as text on
      the bars.
    - color_column (str): The column name to be used for color coding the bars.
    - title (str): The title of the plot.
    - xlabel (str): The label for the x-axis.
    - ylabel (str): The label for the y-axis.

    Returns:
    - None: This function does not return any value but displays a Plotly
      interactive plot.

    Example Usage:
    Assuming 'df' is a DataFrame with columns 'Labels', 'Number of Images',
    and 'Sizes':
    Calling `create_interactive_bar_plot(df_sales, 'Labels', 'Number of Images'
    , 'Sizes', 'Distribution of Image Size across Classes', 'Labels', 'Count')`
    would display a bar plot showing monthly sales, color-coded by region,
    with interactive hover effects.
    """
    # Define hover data internally within the function
    hover_data = {color_column: True, y_column: True}

    fig = px.bar(df, x=x_column, y=y_column, color=color_column, text=y_column,
                 title=title, labels={x_column: xlabel, y_column: ylabel},
                 hover_data=hover_data)

    # Customize the appearance of the plot
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide',
                      xaxis_tickangle=-45, width=1100, height=600)

    fig.show()

# subplot the images samples from each class
def visualize_images(first_images, main_folder_path):
    '''
    Input: first_images from data_analysis.segmentation_openCV() 
    Output: plot 8X3, save 
    '''
    if not first_images:
        print("No images to display.")
        return
    fig, axs = plt.subplots(nrows=len(first_images), ncols=3, figsize=(15, 4 * len(first_images)))
    for idx, (class_name, (img_gray, thresholded, contours)) in enumerate(first_images.items()):
        contour_image = cv2.cvtColor(thresholded.copy(), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)

        axs[idx, 0].imshow(img_gray, cmap='gray')
        axs[idx, 0].set_title(f'{class_name} - Original')
        axs[idx, 0].axis('off')

        axs[idx, 1].imshow(thresholded, cmap='gray')
        axs[idx, 1].set_title(f'{class_name} - Thresholded')
        axs[idx, 1].axis('off')

        axs[idx, 2].imshow(contour_image)
        axs[idx, 2].set_title(f'{class_name} - Contours')
        axs[idx, 2].axis('off')

    plt.tight_layout()
    save_path = '/Users/mehdienrahimi/apr24_bds_int_blood_cells/src/outputs/samples_segmentationOpenCV.jpg'
    plt.savefig(save_path)
    # plt.close(fig)  # Close the figure to free up memory
import matplotlib.pyplot as plt
import seaborn as sns

def dist_cell_area(df_segmentation):
    '''Input: df_segmentation from data_analysis.segmentation_openCV()
    Output: barplot of mean cell area, cell perimeter, and cell circularity vs. labels'''

    # Setup the figure and subplots
    plt.figure(figsize=(18, 12))  # Adjusted for better fit of all plots

    # First subplot: Mean Cell Area per Label
    plt.subplot(3, 2, 1)  # (row, column, index)
    mean_area = df_segmentation.groupby('Label')['CellArea'].mean().reset_index()
    sns.barplot(x='Label', y='CellArea', data=mean_area)
    plt.title('Mean Cell Area per Label')
    plt.xlabel('Label')
    plt.ylabel('Mean Cell Area')
    plt.xticks(rotation=45)

    # Second subplot: Standard Deviation of Cell Area per Label
    plt.subplot(3, 2, 2)
    std_area = df_segmentation.groupby('Label')['CellArea'].std().reset_index()
    sns.barplot(x='Label', y='CellArea', data=std_area)
    plt.title('Standard Deviation of Cell Area per Label')
    plt.xlabel('Label')
    plt.ylabel('Standard Deviation of Cell Area')
    plt.xticks(rotation=45)

    # Third subplot: Mean Cell Perimeter per Label
    plt.subplot(3, 2, 3)
    mean_perimeter = df_segmentation.groupby('Label')['Cell_perimeter'].mean().reset_index()
    sns.barplot(x='Label', y='Cell_perimeter', data=mean_perimeter)
    plt.title('Mean Cell Perimeter per Label')
    plt.xlabel('Label')
    plt.ylabel('Mean Cell Perimeter')
    plt.xticks(rotation=45)

    # Fourth subplot: Standard Deviation of Cell Perimeter per Label
    plt.subplot(3, 2, 4)
    std_perimeter = df_segmentation.groupby('Label')['Cell_perimeter'].std().reset_index()
    sns.barplot(x='Label', y='Cell_perimeter', data=std_perimeter)
    plt.title('Standard Deviation of Cell Perimeter per Label')
    plt.xlabel('Label')
    plt.ylabel('Standard Deviation of Cell Perimeter')
    plt.xticks(rotation=45)

    # Fifth subplot: Mean Cell Circularity per Label
    plt.subplot(3, 2, 5)
    mean_circularity = df_segmentation.groupby('Label')['cell_circularity'].mean().reset_index()
    sns.barplot(x='Label', y='cell_circularity', data=mean_circularity)
    plt.title('Mean Cell Circularity per Label')
    plt.xlabel('Label')
    plt.ylabel('Mean Cell Circularity')
    plt.xticks(rotation=45)

    # Sixth subplot: Standard Deviation of Cell Circularity per Label
    plt.subplot(3, 2, 6)
    std_circularity = df_segmentation.groupby('Label')['cell_circularity'].std().reset_index()
    sns.barplot(x='Label', y='cell_circularity', data=std_circularity)
    plt.title('Standard Deviation of Cell Circularity per Label')
    plt.xlabel('Label')
    plt.ylabel('Standard Deviation of Cell Circularity')
    plt.xticks(rotation=45)

    plt.tight_layout()

    # Save the plot and display it
    save_path = '/Users/mehdienrahimi/apr24_bds_int_blood_cells/src/outputs/CellMetricsDistribution_segmentationOpenCV.jpg'
    plt.savefig(save_path)
    plt.show()


