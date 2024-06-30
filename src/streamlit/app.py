import streamlit as st
from PIL import Image
import os
import pandas as pd
import base64

# Function to convert image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Get the current directory
current_dir = os.path.dirname(__file__)

# Path to your GitHub icon
github_icon_path = os.path.join(current_dir, os.pardir, 'outputs', 'GitHub.png')
github_url = "https://github.com/DataScientest-Studio/apr24_bds_int_blood_cells.git"
github_icon_base64 = image_to_base64(github_icon_path)

# Custom CSS to position the GitHub icon
st.markdown(
    f"""
    <style>
        .css-1v0mbdj.e16nr0p33 {{
            display: flex;
            justify-content: space-between;
        }}
        .github-icon {{
            margin-left: auto;
            margin-right: 20px;
            margin-top: 1px;
        }}
    </style>
    <div class="github-icon">
        <a href="{github_url}" target="_blank">
            <img src="data:image/png;base64,{github_icon_base64}" width="40">
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation
st.sidebar.title("Dashboard Navigation")
pages = [
    "Overview", 
    "Data Exploration", 
    "Data Segmentation", 
    "Statistical Insights", 
    "Multiclassification Models", 
    "Interactive Testing", 
    "Future Directions"
]
page = st.sidebar.radio("Go to", pages)

# Add some space
st.sidebar.write(" ")

# Add contributors section with links to their GitHub profiles
st.sidebar.markdown("### Contributors:")
st.sidebar.markdown("""
- [Eliana Lousada](https://github.com/elianalousada)
- [Francisca Diala](https://github.com/Chykka)
- [Mahdi Habibi](https://github.com/mdhabibi)
- [Mehdi Seyedebrahimi](https://github.com/mirmehdi)
""")

# # Add mentor section with link to their GitHub profile
# st.sidebar.markdown("### Mentor:")
# st.sidebar.markdown("""
# - [Aziz Agrebi](https://github.com/aziz-agrebi)
# """)


if page == "Overview":
    st.markdown("<h2 style='text-align: center;'><span style='color:red'>Blood</span>Py 🩸 Automated Blood Cell Classifier</h2>", unsafe_allow_html=True)



# Home page
if page == "Overview":
    # Load and display the first image
    image_path_1 = os.path.join(current_dir, os.pardir, 'outputs', 'microscope_720.png')
    image = Image.open(image_path_1)  
    st.image(image, width=670)  

    # Add some text
    st.header("Context")
    st.markdown("""
    - **Peripheral blood cells**, including *erythrocytes*, *leukocytes*, and *thrombocytes*, play crucial roles in oxygen transport and immune defense.
    - They constitute about 45% of blood volume, with the remaining 55% being plasma.
    - The analysis of peripheral blood cells serves as a critical diagnostic tool, with morphological examination being the cornerstone for identifying over 80% of haematological diseases, such as *anaemia*, *leukaemia*, or *lymphoma*.
    """)


    # Load and display the second image
    image_path_2 = os.path.join(current_dir, os.pardir, 'outputs', 'samples.png')
    image = Image.open(image_path_2)  
    st.image(image, width=650)  

    #st.image(image, caption='Histology of the different types of blood cells.', use_column_width=True)

    # Add some text
    st.header("Challenges")
    st.write("""
    - Traditional diagnostic methods for blood diseases rely on manual inspection by haematologists, which is expensive, laborious, and prone to subjectivity.
    - Automated systems exist but often cannot match human expertise, particularly in detecting subtle morphological differences indicative of diseases like leukemia.
    - Traditional machine learning methods show promise but fail to generalize well to diverse datasets.""")


    st.header("Objective")
    st.write("""
        To address these limitations, we propose using convolutional neural network (CNN) models for the automatic classification of peripheral blood cells. Our goal is to develop a model that can accurately identify and classify various blood cell types, reducing the need for manual intervention. This system aims to streamline
        the diagnostic process for haematological diseases, improving efficiency and accuracy in clinical practice.
    """)

# Data Exploration
if page == "Data Exploration":
    st.header("Data Exploration")
    # dataset
    rawimg_features = pd.read_csv(os.path.join(current_dir, os.pardir, 'outputs', 'cell_largest_features.csv'))
    rawimg_features.drop('Image_Path',axis=1,inplace = True)
    st.header(" Dataset")

    st.write("""
        We utilised a dataset comprising 17,092 images of individual normal peripheral blood cells, acquired using the CellaVision DM96 analyzer at the Core Laboratory of the Hospital Clinic of Barcelona.
        You can access the original dataset [here](https://data.mendeley.com/datasets/snkd93bnjr/1).
    """)

    #st.dataframe(rawimg_features.head())
    # Samples
    image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'classes_samples.png')) 
    st.image(image, caption='Examples of Blood Cell Types', use_column_width=True)

    # Distribution of cell numbers
    st.header("Distributions of Image Classes in the Dataset")
    image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'distribution_image_classes.png'))
    st.image(image, use_column_width=True) 

    # distribution of image sizes
    st.header("Image Size Distribution")
    image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'Image_sizes2.png')) 
    st.image(image, use_column_width=True)


    # Plot histograms for each column (excluding the Cell_Type column)

    st.header("Cell Area")
    image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'boxplot_of_largest_cell_area_by_cell_type.png')) 
    st.image(image, use_column_width=True)

    st.header("Cell Perimeter")
    image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'boxplot_of_largest_cell_perimeter_by_cell_type_720.png')) 
    st.image(image, use_column_width=True)

    st.header("Cell Circularity")
    image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'boxplot_of_largest_cell_perimeter_by_cell_type.png')) 
    st.image(image, use_column_width=True)

    # explainatoin about figures
    st.header("Summary")
    st.write("""
        - The dataset is imbalanced, with some classes having significantly more images than others.
        - Neutrophils and Eosinophils have the highest number of images, with 3330 and 3117 images, respectively.
        - Lymphocytes and Basophils have the lowest number of images, with 1214 and 1218 images, respectively.
        """)
    
# Segmentation
if page == "Data Segmentation":
    st.header("Data Segmentation")
    st.write("### Select one of the Segmentation Methods")

    if st.checkbox("### Thresholding-Based Segmentation"):
        if st.checkbox("### Details"):
            st.header("Thresholding Approach")
            st.write("""
                - **Thresholding Approach**:
                    - Aim: Distinguish cells from the background using contrast stretching and colour masking techniques.
                    - Images in RGB --> normalized to Grayscale --> contrast streching --> color mask. 
                - **Advantages**:
                    - Simplicity: easy to implement and computationally efficient.
                    - Speed: processes images quickly, making it suitable for applications requiring rapid results.
                - **Limitations**:
                    - Accuracy: struggles with complex images where cell boundaries are not well-defined.
                    - Generalisation: may not perform well on varied datasets with different lighting and staining conditions.

                """)
        
        st.header("Samples of Threshold Segmentation")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'masked_img_normalization.png')) 
        st.image(image, use_column_width=True)    

    if st.checkbox("### Image Segmentation with U-Net"):
        if st.checkbox("### Details"):
            st.header("U-Net Approach")
            st.write("""
                - **Thresholding Approach**:
                    - Aim: Employing a deep learning-based UNet model, a convolutional neural network architecture designed for biomedical image segmentation.
                - **Annotation**: 
                    - 350 images (chosen in a balanced format) were manually annotated using the VGG annotation tool developed by the robotic group of Oxford Univerwity. 
                - **Segmentation Steps**:
                    - Develop a UNet model.
                    - Train model on 0.9 of data and validated on 0.1 of 350 images.
                - **Additional steps for artifact removal**: 
                    - Label connected regions.
                    - Find largest regions:
                        - remove the region, if the size is small.
                        - consider image as a bad image if there are more than one big region.
                - **Outlier detection**: 
                    - We calculate the area of each segmented image and classify them as "bad cells" if the z-score of its area is three standard deviation beyond the mean of each class. 
                
                
                - **Advantages**:
                    - Accuracy: the UNet model provides high segmentation accuracy, especially for complex images.
                    - Robustness: it can generalise well to varied datasets due to its deep learning architecture.
                - **Limitations**:
                    - Complexity: this method is computationally intensive and requires significant resources for training.
                    - Data requirement: it needs a large annotated dataset for effective training.
""")



        st.header("U-Net Architecture")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'Unet.jpg')) 
        st.image(image, use_column_width=True)


        st.header("Samples of Semantic Segmentation")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'Unet_seg_sample_S1.jpg')) 
        st.image(image, caption='A sample of Unet Segmentation: on left we see an original image in grayscale, in middle we see a true segmentation, and right side is the predicted binary segmentation',
         use_column_width=True)

        st.header("Artifact removal")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'artifact_removal.jpg')) 
        st.image(image, caption='Remove regions with smaller sizes', use_column_width=True)

        st.header("Image with Multiple cell recognition as bad images")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'Multicell_recog.jpg')) 
        st.image(image, caption='Consider images with mor than one big regions as bad image', use_column_width=True)

        st.header("Detect Ouliers")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'Outliers_cells.jpg')) 
        st.image(image, caption='The outliers are cell type which has area more than three standard deviation from the mean of the cell area at each group ', use_column_width=True)

        st.header("Samples of Abnormal Cells")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'deadcells_oulier.jpg')) 
        st.image(image, caption='The outliers are abnormal cells and we do not want to classify them', use_column_width=True)

# Statistical Analysis
if page == "Statistical Insights":
    st.header("Statistical Insights on Segmented Images")
    st.write("### Select one of the Segmentation Methods")
    # normalization seg.
    if st.checkbox("### Segmented Images with Thresholding Method"):
        st.header("Cell area distribution")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs','stats_threshold_masks', 'Mask_Area_distribution.png')) 
        st.image(image,use_column_width=True)

      
        st.header("Significance matrix")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs','stats_threshold_masks', 'significance_matrix_heatmap_Mask_Area.png')) 
        st.image(image, use_column_width=True)

       
        st.header("Violin plot")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs','stats_threshold_masks', 'Mask_Area_violinplot.png')) 
        st.image(image, use_column_width=True)    

    # unet seg
    if st.checkbox("### Segmented Images with U-Net"):
        st.header("Cell area distribution")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs','stats_UNet_masks', 'White_Area_distribution.png')) 
        st.image(image, use_column_width=True)

      
        st.header("Significance matrix")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs','stats_UNet_masks', 'significance_matrix_heatmap_White_Area.png')) 
        st.image(image, use_column_width=True)

       
        st.header("Violin plot")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs','stats_UNet_masks', 'White_Area_violinplot.png')) 
        st.image(image, use_column_width=True)

# Classificatoin 
if page == "Multiclassification Models":
    st.header("Multiclassification Models")

    model_options = ["VGG16-Classifier for Original Images", "VGG16-Classifier for Masked Images","EfficientNetB0-Classifier for Masked Images"]
    selected_model = st.selectbox("Select one the Models:", model_options)

    #st.write(f"You selected: {selected_model}")
    #st.write("Add your classification implementation here for the selected model.")


    if selected_model == "EfficientNetB0-Classifier for Masked Images":

        cls_report = pd.read_csv(os.path.join(current_dir, os.pardir, 'outputs', 'classification_report_unet_seg.csv')) 
        st.header("Transfer learning with EfficientNetB0")
        #st.header("Algorithm")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'efficient.png')) 
        st.image(image, use_column_width=True)
        

        st.header("Classification report")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'cls_report_eff.png')) 
        st.image(image, use_column_width=True)


        st.header("Heatmap presentation")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'heat_map_cls_Unet_seg_efficientnetB0.png')) 
        st.image(image, use_column_width=True)

        st.header("Loss of EfficientNetB0 model")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'Loss_Curve_EfficientNetB0.png')) 
        st.image(image, use_column_width=True)

        st.header("Accuracy of EfficientNetB0 model")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'Accuracy_Curve_EfficientNetB0.png')) 
        st.image(image, use_column_width=True)


    if selected_model == "VGG16-Classifier for Masked Images":

        st.header(" Transfer learning with VGG16")
        #st.header("Algorithm")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'vgg.png')) 
        st.image(image, use_column_width=True)

        st.header("Classification report")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'screenshot_2024-06-27_at_17.33.21_720.png')) 
        st.image(image, caption='', use_column_width=True)

        st.header("Heatmap presentation")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'cm_masked_vgg16.png')) 
        st.image(image, use_column_width=True)

        st.header("Loss/Accuracy plot")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'train_loss_plots_vgg16.png')) 
        st.image(image, use_column_width=True)


    if selected_model == "VGG16-Classifier for Original Images":


        st.header(" Transfer learning with VGG16")
        #st.header("Algorithm")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'vgg.png')) 
        st.image(image, use_column_width=True)

        st.header("Classification report")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'screenshot_2024-06-27_at_17.34.32.png')) 
        st.image(image, caption='', use_column_width=True)

        st.header("Heatmap presentation")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'cm_unmasked_vgg16.png')) 
        st.image(image, use_column_width=True)

        st.header("Loss/Accuracy plot")
        image = Image.open(os.path.join(current_dir, os.pardir, 'outputs', 'train_loss_plots_vgg16_unmasked_720.png')) 
        st.image(image, caption='', use_column_width=True)

#  Interactive Test


if page == "Interactive Testing":
    st.header("Interactive Testing")
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

    #st.title("Interactive Test with Transfer Learning")

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
    

    # Adjust these paths to your directory structure
    model_path = os.path.join(current_dir, os.pardir, 'models', 'efficientnet_model.h5')
    clean_data_path = os.path.join(current_dir, os.pardir, 'data', 'clean_data.csv')

    # Print paths for debugging
    #st.write("Model Path:", model_path)
    

    # Load the model
    try:
        model = load_model_with_weights(model_path, num_classes)
        st.write("")
    except Exception as e:
        st.write("")

    # Load clean_data DataFrame
    try:
        clean_data = pd.read_csv(clean_data_path)
        st.write("")
    except Exception as e:
        st.write("")

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

#Perspectives

if page == "Future Directions":
     
    st.header("Future Directions:")

        # Add some text
    st.header("Summary:")
    st.write("""
       -  Deep learning solution for automated detection and classification of peripheral blood cells.
       -  Utilizes convolutional neural networks, particularly U-Net for segmentation and pre-trained models for feature extraction.
       -  Enhances accuracy and efficiency in haematological diagnostics, addressing challenges of manual inspection.
    """)

    st.header("Future Perspectives:")
    st.write("""
       - **Expanded Datasets**: To improve the generalisability and robustness of the model, we plan to incorporate more diverse and comprehensive datasets, including images with varying staining techniques and pathological conditions.
       - **Clinical Validation**: Collaborating with haematologists and clinical laboratories to validate the system on real-world data, ensuring its practical applicability and reliability in clinical settings.
       - **User Interface Development**: Developing a user-friendly interface for BloodPy, enabling seamless integration into clinical workflows and providing intuitive visualisations and reports for haematologists.

    """)