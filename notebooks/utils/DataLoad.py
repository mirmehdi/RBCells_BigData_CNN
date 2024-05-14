import os
from PIL import Image
import numpy as np
import pandas as pd

class DataLoad:
    def __init__(self, classes_dir, image_int_size=(360, 360), dataframe_path='data.csv'):
        self.classes_dir = classes_dir
        self.image_int_size = image_int_size
        # self.pixel_data_path = pixel_data_path
        self.dataframe_path = dataframe_path
        self.image_size = []
        # self.pixels = []  # Maintain this if you still need to process or save the pixel data separately
        self.labels = []
        self.load_data()

    def load_data(self):
        try:
            paths = os.listdir(self.classes_dir)
            paths = [path for path in paths if not path.startswith('.')]  # Skip hidden files
            for class_idx, class_name in enumerate(paths):
                class_dir = os.path.join(self.classes_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue

                for image_file in os.listdir(class_dir):
                    if image_file.startswith('.'):
                        continue  # Skip system files like .DS_Store

                    image_path = os.path.join(class_dir, image_file)
                    try:
                        image = Image.open(image_path)
                        # image = imagopene.resize(self.image_int_size)
                        # flattened_image = np.array(image).flatten()
                        # self.pixels.append(flattened_image)  # Only store pixels if needed elsewhere
                        self.labels.append(class_idx)
                        self.image_size.append(image.size)  # Append a tuple (width, height)

                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")

            self.labels = np.array(self.labels)
            self.image_size = np.array(self.image_size)

            # Create a DataFrame without pixel data
            self.df = pd.DataFrame({
                'Label': self.labels,
                'ImageSize_D1': [size[0] for size in self.image_size],
                'ImageSize_D2': [size[1] for size in self.image_size]
            })

        except Exception as e:
            print(f"Error loading data: {e}")

    def save_data(self, path):
        # Save the DataFrame without pixel data as a CSV file
        self.df.to_csv(path, index=False)
        print(f"Dataframe saved to {self.dataframe_path}")

# Example usage
print('Loading of data is done.')
