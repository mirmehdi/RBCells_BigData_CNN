import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


st.title("Leukopy 🩸 - blood cell classifier")
st.sidebar.title("Navigation")
pages=["Home", "Preliminary analysis","Statistical Analysis", "Modelling", "Prediction & Evaluation", "Perspectives"]
page=st.sidebar.radio("Go to", pages)

# Home page
if page == "Home":
        # Add some text
    st.header("Blood-py : a deep learning-based software for the automatic detection and classification of peripheral blood cells. ")
    st.write("""
        Peripheral blood cells, including erythrocytes, leukocytes, and thrombocytes, play crucial roles in oxygen transport and immune 
             defense. They constitute about 45% of blood volume, with the remaining 55% being plasma.
    """)
    # Load an image
    image = Image.open('/Users/mehdienrahimi/apr24_bds_int_blood_cells/src/outputs/classes_samples.png')  # Replace with your image path
    st.image(image, caption='Leukopy - Blood Cell Classifier', use_column_width=True)
    
    # Add some text
    st.header("Welcome to Leukopy!")
    st.write("""
        Leukopy is a powerful tool designed to classify different types of blood cells. 
        It leverages state-of-the-art machine learning algorithms to provide accurate 
        and reliable results. Use the navigation menu to explore various features of the application.
    """)

