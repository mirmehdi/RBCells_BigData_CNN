import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import plotly.express as px


def plot_image_distribution(df, title, xlabel, ylabel):
    """Create a bar plot to explore visually the distribution
       of image classes."""
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Label', y='Number of Images', data=df, palette='Set2')
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def display_sample_images(labels, main_folder_path):
    """Displays the first image from each class in a grid."""
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
    """Create an interactive bar plot using Plotly Express
       with predefined hover data."""
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
