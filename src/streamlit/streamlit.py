import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import pandas as pd

current_dir = os.getcwd() 



################################################### Preliminary analysis
st.title("Leukopy 🩸 - blood cell classifier")
st.sidebar.title("Navigation")
pages=["Home", "Preliminary analysis","Statistical Analysis", "Modelling", "Classification with Transfer learning", "Perspectives"]
page=st.sidebar.radio("Go to", pages)

# Home page
if page == "Home":
        # Add some text
    st.header("Blood-py : a deep learning-based software for the automatic detection and classification of peripheral blood cells. ")
    st.write("""
        Peripheral blood cells, including erythrocytes, leukocytes, and thrombocytes, play crucial roles in oxygen transport and immune 
             defense. They constitute about 45% of blood volume, with the remaining 55% being plasma. The analysis of peripheral blood cells serves as a critical diagnostic tool, with morphological examination being the cornerstone for identifying over 80% of haematological diseases, such as anaemia, leukaemia, or lymphoma. 

    """)
    # Load an image
    
    image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'classes_samples.png'))  # Replace with your image path
    st.image(image, caption='Leukopy - Blood Cell Classifier', use_column_width=True)
    
    # Add some text
    
    st.header("Problematic")
    st.write("""
        Traditional diagnostic methods for blood diseases rely on manual inspection by haematologists, which is laborious, time-consuming, and prone to subjectivity. Automated systems exist but often cannot match human expertise, particularly in detecting subtle morphological differences indicative of diseases like leukemia. 
        Traditional machine learning methods show promise but fail to generalize well to diverse datasets. 

    """)
        
    st.header("Aim")
    st.write("""
        To address these limitations, we propose using convolutional neural network (CNN) models for the automatic classification of peripheral blood cells. Our goal is to develop a model that can accurately identify and classify various blood cell types, reducing the need for manual intervention. This system aims to streamline
         the diagnostic process for haematological diseases, improving efficiency and accuracy in clinical practice.

    """)

################################################### Preliminary analysis

if page == "Preliminary analysis":
    st.header("DataSet_numbers")
    st.write("""
        - The dataset is imbalanced, with some classes having significantly more images than others.
        - Neutrophils and Eosinophils have the highest number of images, with 3330 and 3117 images, respectively.
        - Lymphocytes and Basophils have the lowest number of images, with 1214 and 1218 images, respectively.
        """)
    image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'distribution_image_classes.png')) 
    st.image(image, caption='Cell Classes', use_column_width=True)

    st.header("Image size")
    st.write("""
        Distribution of Image size
        """)
    image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'Image_sizes.jpg')) 
    st.image(image, caption='Image Size Distribution', use_column_width=True)

    
    ################################################### Segmentation

if page == "Segmentation":

    st.write("### Select you model of segmentation")

    if st.checkbox("### Unet Segmentation"):
        st.header("Unet ALgorith")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'Unet.jpg')) 
        st.image(image, caption='Unet segmentatoin Algorithm', use_column_width=True)


        st.header("sample of Unet Binary segmentation")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'Unet_seg_sample_S1.jpg')) 
        st.image(image, caption='A sample of Unet Segmentation: on left we see an original image in grayscale, in middle we see a true segmentation, and right side is the predicted binary segmentation',
         use_column_width=True)

        st.header("Artifact removal")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'artifact_removal.jpg')) 
        st.image(image, caption='delete regions with smaller sizes', use_column_width=True)

        st.header("Image with Multiple cell recognition as bad images")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'Multicell_recog.jpg')) 
        st.image(image, caption='Consider images with mor than one big regions as bad image', use_column_width=True)

        st.header("Detect Ouliers")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'Outliers_cells.jpg')) 
        st.image(image, caption='The outliers are cell type which has area more than three standard deviation from the mean of the cell area at each group ', use_column_width=True)

        st.header("Ouliers are Abnormal Cells")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'deadcells_oulier.jpg')) 
        st.image(image, caption='The outliers are abnormal cells and we do not want to classify them', use_column_width=True)



##

    if st.checkbox("### Normalization as segmentation method"):
        st.header("Unet ALgorith")
   

 ################################################### Segmentation

if page == "Classification with Transfer learning":
    st.header("Classification with Transfer Learning")

    model_options = [ "VGG16", "EfficientNetB0"]
    selected_model = st.selectbox("Select the pre-trained model for transfer learning:", model_options)

    st.write(f"You selected: {selected_model}")
    st.write("Add your classification implementation here for the selected model.")
