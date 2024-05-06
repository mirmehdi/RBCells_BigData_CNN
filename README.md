# Peripheral Blood Cell Classification Project

## Overview
This project aims to develop an automatic recognition system for classifying peripheral blood cell images into different cell types. The dataset used in this project consists of microscopic images of individual normal cells, acquired from the Core Laboratory at the Hospital Clinic of Barcelona. The dataset is organized into eight groups: neutrophils, eosinophils, basophils, lymphocytes, monocytes, immature granulocytes, erythroblasts, and platelets or thrombocytes.

## Dataset Description
- **Total Images:** 17,092
- **Image Size:** 360 x 363 pixels (JPG format)
- **Annotation:** Expert clinical pathologists annotated the images.
- **Categories:** Neutrophils, eosinophils, basophils, lymphocytes, monocytes, immature granulocytes (promyelocytes, myelocytes, and metamyelocytes), erythroblasts, and platelets or thrombocytes.
- **Image Source:** Individuals without infection, hematologic or oncologic disease, and free of any pharmacologic treatment at the moment of blood collection.

## Environment Setup
To run this project, you will need to set up a Python environment that contains all the necessary libraries and dependencies. This project uses Conda for environment management.

### Using the provided Conda environment file
1. **Clone the repository** or download the project files.
2. **Navigate to the project directory** where `environment.yml` is located.
3. **Create the Conda environment** by running the following command in your terminal:

   ```bash
   conda env create -f environment.yml

This command will create a new Conda environment with all the dependencies specified in the environment.yml file.
4. **Activate the newly created environment** by running:
   ```bash
   conda activate bloodCNN

5. **Verify that the environment has been set up correctly** and that all dependencies are installed by running:
   ```bash
   conda list
   
By following these steps, you should have a fully configured environment ready to run the project scripts.