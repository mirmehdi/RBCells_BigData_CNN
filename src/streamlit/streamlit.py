import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

current_dir = os.getcwd() 



################################################### Introduction 
st.title("Blood-py 🩸 - blood cell classifier")
st.sidebar.title("Navigation")
pages=["Home", "Preliminary analysis", "Segmentation", "Statistical Analysis", "Classification with Transfer learning","Interactive Test", "Perspectives"]
page=st.sidebar.radio("Go to", pages)

# Home page
if page == "Home":
     
    image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'microscope_720.png'))  # Replace with your image path
    st.image(image, caption='Blood-py - Blood Cell Classifier', use_column_width=True)

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

    # explainatoin about figures
    st.header("DataSet_numbers")
    st.write("""
        - The dataset is imbalanced, with some classes having significantly more images than others.
        - Neutrophils and Eosinophils have the highest number of images, with 3330 and 3117 images, respectively.
        - Lymphocytes and Basophils have the lowest number of images, with 1214 and 1218 images, respectively.
        """)

    # dataset
    rawimg_features = pd.read_csv(os.path.join(current_dir, os.pardir, 'outputs', 'cell_largest_features.csv'))
    rawimg_features.drop('Image_Path',axis=1,inplace = True)
    st.header(" DataSet")

    st.dataframe(rawimg_features.head())


    # distribution of cell numbers
    image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'distribution_image_classes.png')) 
    st.image(image, caption='Cell Classes', use_column_width=True)
    # distribution of image sizes
    st.header("Image size")
    st.write("""
        Distribution of Image size
        """)
    image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'Image_sizes.jpg')) 
    st.image(image, caption='Image Size Distribution', use_column_width=True)


    # Plot histograms for each column (excluding the Cell_Type column)

    st.header("Cell Area")
    image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'boxplot_of_largest_cell_area_by_cell_type.png')) 
    st.image(image, caption='Cell area calculated on raw image', use_column_width=True)

    st.header("Cell Primeter")
    image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'boxplot_of_largest_cell_perimeter_by_cell_type_720.png')) 
    st.image(image, caption='Cell primeter calculated on raw image', use_column_width=True)

    st.header("Cell Circularity")
    image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'boxplot_of_largest_cell_perimeter_by_cell_type.png')) 
    st.image(image, caption='Cell circularity calculated on raw image', use_column_width=True)

    ################################################### Segmentation

if page == "Segmentation":

    st.write("### Select you model of segmentation")

    if st.checkbox("### Image segmentation with UNet"):
        if st.checkbox("### Details"):
            st.header("UNet Approach")
            st.write("""
                - Thresholding Approach:
                    - Aim: mploys a deep learning-based UNet model, a convolutional neural network architecture designed for biomedical image segmentation.
                - **Annotation**: 
                    - 350 images (chosen in a balanced format) was manually annotated using VGG annotation tool developed by  robotic group of Oxford Univerwity. 
                
                - **Segmentation Steps**:
                    - Annotation: 350 images (chosen in a balanced format) was manually annotated using VGG annotation tool developed by  robotic group of Oxford Univerwity. 
                    - Developed a UNet model (manually)
                    - Trained model on 0.9 of data and validated on 0.1 of 350 images
                
                - **further step for artifact removal**: 
                    - label connected regions
                    - find largest regions
                        - remove the region if the size is small 
                        - consider image as bad image if there are more than one big region
                - **Outlier detection**: 
                    - We calculate the area of each segmented image and classify them as "bad calls" if the z-score of it is three standard deviation beyond the mean of each class. 
                
                
                - **Advantages**:
                    - Accuracy: the UNet model provides high segmentation accuracy, especially for complex images.
                    - Robustness: it can generalise well to varied datasets due to its deep learning architecture.
                - **Limitations**:
                    - Complexity: this method is computationally intensive and requires significant resources for training.
                    - Data requirement: it needs a large annotated dataset for effective training.
""")



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

    if st.checkbox("### Thresholding-based segmentation"):
        if st.checkbox("### Details"):
            st.header("Thresholding Approach")
            st.write("""
                - Thresholding Approach:
                    - Aim: Distinguish cells from the background using contrast stretching and colour masking techniques.
                -Steps:
                    - Images in both RGB --> normalized to Grayscale --> contrast streching --> color mask 
                - Advantages:
                    - Simplicity: Easy to implement and computationally efficient.
                    - Speed: Processes images quickly, making it suitable for applications requiring rapid results.
                - Limitations:
                    - Accuracy: Struggles with complex images where cell boundaries are not well-defined.
                    - Generalisation: May not perform well on varied datasets with different lighting and staining conditions.

                """)
        
        st.header("Output for threshold segmentation")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'masked_img_normalization.png')) 
        st.image(image, caption='Output for threshold segmentation', use_column_width=True)

        



        


 ################################################### Statistical Analysis


if page == "Statistical Analysis":
    # unet seg
    st.write("### Select you model of segmentation")

    if st.checkbox("### Based on Image segmentation with UNet"):
        st.header("Ouliers are Abnormal Cells")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs','stats_UNet_masks', 'White_Area_distribution.png')) 
        st.image(image, caption='The distribution of cell area based on Unet model', use_column_width=True)

      
        st.header("Significance matrix")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs','stats_UNet_masks', 'significance_matrix_heatmap_White_Area.png')) 
        st.image(image, caption='Significance matrix comparing cell area between different cell types', use_column_width=True)

       
        st.header("Violon plot")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs','stats_UNet_masks', 'White_Area_violinplot.png')) 
        st.image(image, caption=' Violin plots of cell area from the different cell types', use_column_width=True)

        st.header("Ouliers are Abnormal Cells")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs','stats_threshold_masks', 'Mask_Area_distribution.png')) 
        st.image(image, caption='The distribution of cell area based on Unet model', use_column_width=True)




    # normalization seg.
    if st.checkbox("### Based on Thresholding-based segmentation"):
        st.header("Ouliers are Abnormal Cells")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs','stats_threshold_masks', 'Mask_Area_distribution.png')) 
        st.image(image, caption='The distribution of cell area based on Unet model', use_column_width=True)

      
        st.header("Significance matrix")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs','stats_threshold_masks', 'significance_matrix_heatmap_Mask_Area.png')) 
        st.image(image, caption='Significance matrix comparing cell area between different cell types', use_column_width=True)

       
        st.header("Violon plot")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs','stats_threshold_masks', 'Mask_Area_violinplot.png')) 
        st.image(image, caption=' Violin plots of cell area from the different cell types', use_column_width=True)


   

 ################################################### Classificatoin 
if page == "Interactive Test":
    import streamlit as st
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
    from tensorflow.keras.models import Model
    import numpy as np
    import cv2
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from PIL import Image

    st.title("Interactive Test with Transfer Learning")

    # Function to define and build the model architecture
    def build_model(num_classes):
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model

    # Load the pre-trained model
    def load_model_with_weights(model_path, num_classes):
        model = build_model(num_classes)
        model.load_weights(model_path)
        return model

    # Define cell type mapping - ensure this matches the training data
    cell_type_mapping = {
        0: 'BA',
        1: 'ERB',
        2: 'LY',
        3: 'SNE',
        4: 'EO',
        5: 'PMY',
        6: 'MO',
        7: 'PLATELET',
        8: 'UNKNOWN'  # Add an additional class to match the model's expectation
    }

    num_classes = len(cell_type_mapping)

    # Get the current directory
    current_dir = os.getcwd()
    st.write("Current Directory:", current_dir)

    # Adjust these paths to your directory structure
    model_path = os.path.join(current_dir, os.pardir, 'models', 'efficientnet_model.h5')
    clean_data_path = os.path.join(current_dir, os.pardir, 'data', 'clean_data.csv')

    # Print paths for debugging
    st.write("Model Path:", model_path)
    st.write("Clean Data Path:", clean_data_path)

    # Load the model
    try:
        model = load_model_with_weights(model_path, num_classes)
        st.write("Model loaded successfully.")
    except Exception as e:
        st.write(f"Error loading model: {str(e)}")

    # Load clean_data DataFrame
    try:
        clean_data = pd.read_csv(clean_data_path)
        st.write("Clean data loaded successfully.")
    except Exception as e:
        st.write(f"Error loading clean data: {str(e)}")

    # Function to preprocess the image
    def preprocess_image(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        image = cv2.resize(image, (256, 256))  # Resize to 256x256
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = tf.keras.applications.efficientnet.preprocess_input(image)  # Preprocess image
        return image

    # Function to predict the class of the image
    def predict(image):
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
        return cell_type_mapping[predicted_class], confidence

    # File uploader for user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Read the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess and predict the class of the uploaded image
        try:
            processed_image = preprocess_image(image)
            predicted_class, confidence = predict(processed_image)
            st.write(f"Predicted Class: {predicted_class}")
            st.write(f"Confidence: {confidence * 100:.2f}%")

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax.set_title(f"Predicted Label: {predicted_class}\nConfidence: {confidence * 100:.2f}%")
            ax.axis('off')
            st.pyplot(fig)
        except Exception as e:
            st.write(f"Error: {str(e)}")