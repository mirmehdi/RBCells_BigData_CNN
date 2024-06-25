import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


st.title("Leukopy 🩸 - blood cell classifier")
st.sidebar.title("Navigation")
pages=["Home", "Preliminary analysis","Statistical Analysis", "Modelling", "Prediction & Evaluation", "Perspectives"]
page=st.sidebar.radio("Go to", pages)