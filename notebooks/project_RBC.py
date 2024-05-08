import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.DataLoad import DataLoad
from utils.Visualization import Visualization
import os
os.chdir('/Users/mehdienrahimi/apr24_bds_int_blood_cells/notebooks')

# Define the directory containing the data
classes_dir = '/Users/mehdienrahimi/Desktop/DataScienceTests/Project/Dataset/PBC_dataset_normal_DIB'

# Load the data
data_loader = DataLoad(classes_dir)

# Visualize the data
# visualizer = Visualization(data_loader.df)
# visualizer.hist_plot()