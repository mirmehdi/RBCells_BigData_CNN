import os
os.chdir('/Users/mehdienrahimi/apr24_bds_int_blood_cells/notebooks')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.DataLoad import DataLoad
from utils.Visualization import Visualization
from utils.segmentation_openCV import segmentation_openCV
import cv2

'''This is the main .py file to run '''


# Define the directory containing the data
classes_dir = '/Users/mehdienrahimi/Desktop/DataScienceTests/Project/Dataset/PBC_dataset_normal_DIB'
save_data_dir = '/Users/mehdienrahimi/apr24_bds_int_blood_cells/notebooks/outputs'
# Load the data

# data_loader = DataLoad(classes_dir)


# save_path = os.path.join(save_data_dir, 'data.csv')
# data_loader.save_data(save_path) 



# # # Visualize the data
# visualizer = Visualization(data_loader.df)
# save_path = os.path.join(save_data_dir, 'hist.png')
# visualizer.hist_plot(save_path)

# save_path = os.path.join(save_data_dir, 'img_size_dist.png')
# visualizer.img_size_dist(save_path)


## Imgae preprocessing 
# Example usage
data_loader = segmentation_openCV(classes_dir)


# Visualize images using the method from the data_loader instance
save_path = os.path.join(save_data_dir, 'dist_area.jpg')

data_loader.visualize_images(save_path)



# # # Visualize the data
visualizer = Visualization(data_loader.df)
save_path = os.path.join(save_data_dir, 'dist_area.png')
visualizer.dist_cell_area(save_path)