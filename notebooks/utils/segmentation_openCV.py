import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class segmentation_openCV:
    def __init__(self, classes_dir, image_int_size=(360, 360), dataframe_path='data.csv'):
        self.classes_dir = classes_dir
        self.image_int_size = image_int_size  # Not currently used in image processing
        self.dataframe_path = dataframe_path
        self.image_sizes = []
        self.cell_areas = []
        self.labels = []
        self.first_images = {}  # Initialize the dictionary for first images
        self.load_data()

    def load_data(self):
        try:
            class_names = [d for d in os.listdir(self.classes_dir) if os.path.isdir(os.path.join(self.classes_dir, d)) and not d.startswith('.')]
            for class_idx, class_name in enumerate(class_names):
                class_dir = os.path.join(self.classes_dir, class_name)
                image_files = [f for f in os.listdir(class_dir) if not f.startswith('.')]
                for image_file in image_files:
                    image_path = os.path.join(class_dir, image_file)
                    try:
                        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        if img_gray is None:
                            raise ValueError("Image couldn't be read")
                        blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
                        _, thresholded = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
                        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        if class_name not in self.first_images:  # Save the first image only
                            self.first_images[class_name] = (img_gray, thresholded, contours)

                        max_area = max([cv2.contourArea(contour) for contour in contours], default=0)

                        self.labels.append(class_idx)
                        self.image_sizes.append(img_gray.size)
                        self.cell_areas.append(max_area)

                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")

            self.labels = np.array(self.labels)
            self.image_sizes = np.array(self.image_sizes)
            self.cell_areas = np.array(self.cell_areas)

            self.df = pd.DataFrame({
                'Label': self.labels,
                'ImageSize': self.image_sizes,
                'CellArea': self.cell_areas
            })

        except Exception as e:
            print(f"Error loading data: {e}")

    def visualize_images(self, path):
        if not self.first_images:
            print("No images to display.")
            return
        fig, axs = plt.subplots(nrows=len(self.first_images), ncols=3, figsize=(15, 4 * len(self.first_images)))
        for idx, (class_name, (img_gray, thresholded, contours)) in enumerate(self.first_images.items()):
            contour_image = cv2.cvtColor(thresholded.copy(), cv2.COLOR_GRAY2BGR)
            cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)

            axs[idx, 0].imshow(img_gray, cmap='gray')
            axs[idx, 0].set_title(f'{class_name} - Original')
            axs[idx, 0].axis('off')

            axs[idx, 1].imshow(thresholded, cmap='gray')
            axs[idx, 1].set_title(f'{class_name} - Thresholded')
            axs[idx, 1].axis('off')

            axs[idx, 2].imshow(contour_image)
            axs[idx, 2].set_title(f'{class_name} - Contours')
            axs[idx, 2].axis('off')

        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)  # Close the figure to free up memory

    def save_data(self, path=None):
        if path is None:
            path = self.dataframe_path
