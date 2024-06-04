import json
import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Load annotations from JSON file
base_dir = "/Users/mehdienrahimi/Desktop/DataScienceTests/Project/Dataset"
json_path = '/Users/mehdienrahimi/Desktop/DataScienceTests/Project/Dataset/PBC_dataset_normal_DIB/ann_eos.json'
with open(json_path) as f:
    annotations = json.load(f)


for key,value in annotations.items():
    filename = value['filename']
    img_path = f"{base_dir}/{'PBC_dataset_normal_DIB/eosinophil'}/{filename}"
    img = cv2.imread(img_path,cv2.IMREAD_COLOR)
    h,w,_ = img.shape

    mask = np.zeros((h,w))

    regions = value["regions"]
    for region in regions:
        shape_attributes = region["shape_attributes"]
        x_points = shape_attributes["all_points_x"]
        y_points = shape_attributes["all_points_y"]

        contours = []
        for x,y in zip(x_points,y_points):
            contours.append((x,y))
        contours = np.array(contours)

        cv2.drawContours(mask,[contours],-1,255,1)

    cv2.imwrite(f"{base_dir}/{'mask'}/{filename}",mask)
