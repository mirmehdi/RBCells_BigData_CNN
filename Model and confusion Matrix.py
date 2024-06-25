# src/evaluation.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
X_train = np.load('data/X_train.npy')
X_test = np.load('data/X_test.npy')
y_train = np.load('data/y_train.npy')
y_test = np.load('data/y_test.npy')

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
class_names = label_encoder.classes_  # Array of class names
num_classes = len(class_names)  # Number of unique classes

IMG_SIZE = X_train.shape[1]

# Define the model
model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Save the model
model.save('models/blood_cell_model.h5')

# Predict using the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Ensure class names are strings
class_names = [str(name) for name in class_names]

# Classification report
print(classification_report(y_test, y_pred_classes, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()




import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load data using memory mapping
X_train = np.load('data/X_train.npy', mmap_mode='r')
X_test = np.load('data/X_test.npy', mmap_mode='r')
y_train = np.load('data/y_train.npy', mmap_mode='r')
y_test = np.load('data/y_test.npy', mmap_mode='r')

# Create the model (assuming you already have a saved model)
model = tf.keras.models.load_model('models/blood_cell_model.h5')

# Predict using the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()




import matplotlib.pyplot as plt

# Create a figure with two subplots
plt.figure(figsize=(12, 4))

# Plot training and validation loss
plt.subplot(121)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss by Epoch')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

# Plot training and validation accuracy
plt.subplot(122)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy by Epoch')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')

# Display the plots
plt.show()



import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

# Define the ConfusionMatrix class
class ConfusionMatrix:
    def __init__(self, fig_size=(8, 6), fmt='d', title_fontsize=14, label_fontsize=12,
                x_ticklabels_rotation=45, y_ticklabels_rotation=0, cmap='Blues',
                title='Confusion Matrix', xlabel='Predicted', ylabel='Actual',
                save_filename='confusion_matrix.png'):
        self.fig_size = fig_size
        self.fmt = fmt
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title_fontsize = title_fontsize
        self.label_fontsize = label_fontsize
        self.x_ticklabels_rotation = x_ticklabels_rotation
        self.y_ticklabels_rotation = y_ticklabels_rotation
        self.cmap = cmap
        self.save_filename = save_filename

    def plot_confusion_matrix(self, cm, class_names):
        plt.figure(figsize=self.fig_size)
        sns.heatmap(cm, annot=True, fmt=self.fmt, cmap=self.cmap, xticklabels=class_names, yticklabels=class_names)
        plt.title(self.title, fontsize=self.title_fontsize)
        plt.xlabel(self.xlabel, fontsize=self.label_fontsize)
        plt.ylabel(self.ylabel, fontsize=self.label_fontsize)
        plt.xticks(rotation=self.x_ticklabels_rotation, ha='right')
        plt.yticks(rotation=self.y_ticklabels_rotation)
        plt.tight_layout()  # Adjusts the plot to ensure everything fits without overlapping
        plt.savefig(self.save_filename)
        plt.show()

# Correct model path
model_path = 'models/blood_cell_model.h5'  # Replace with your actual model path

try:
    model = load_model(model_path)
except FileNotFoundError:
    print(f"Model file not found at {model_path}")
    # Handle the error as appropriate for your application
    # Exiting the script if model not found
    exit(1)

# Compile the model with necessary configurations
model.compile(optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

# Assuming you have your data ready in X_test and y_test
# Ensure y_test is one-hot encoded
num_classes = 8  # Adjust this number according to your dataset
if y_test.ndim == 1:
    y_test = to_categorical(y_test, num_classes=num_classes)

# Evaluate the model to build the metrics
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

# Get predictions and convert them to class labels
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Generate the classification report
class_names = ['basophil', 'eosinophil', 'erythroblast', 'ig', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet']  # Replace with your class names
report = classification_report(y_true_classes, y_pred_classes, target_names=class_names)
print(report)

# Compute the confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)

# Create an instance of the ConfusionMatrix class and plot the confusion matrix
conf_matrix = ConfusionMatrix(title='Confusion Matrix for Blood Cell Classification',
                            xlabel='Predicted Label',
                            ylabel='True Label',
                            cmap='Blues',
                            save_filename='confusion_matrix.png')

conf_matrix.plot_confusion_matrix(cm, class_names)


# Point on warning in output:
# This shows that the model has been evaluated, and the accuracy and loss metrics have been calculated.
# To avoid seeing this warning, we can simply train or evaluate the model immediately after compiling it, which is already done in your current code.
# The results and confusion matrix indicate that the model is performing well, with an overall accuracy of approximately 95.83% and detailed precision, recall, and F1-score metrics for each class.



from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class ConfusionMatrix:
    def __init__(self,
                fig_size=(8, 6),
                fmt='d',
                title_fontsize=14,
                label_fontsize=12,
                x_ticklabels_rotation=0,
                y_ticklabels_rotation=0,
                cmap='Blues',
                title='Confusion Matrix',
                xlabel='Predicted',
                ylabel='Actual',
                save_filename='confusion_matrix.png'):
        self.fig_size = fig_size
        self.fmt = fmt
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title_fontsize = title_fontsize
        self.label_fontsize = label_fontsize
        self.x_ticklabels_rotation = x_ticklabels_rotation
        self.y_ticklabels_rotation = y_ticklabels_rotation
        self.cmap = cmap
        self.save_filename = save_filename

    def plot(self, y_true, y_pred):
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=self.fig_size)
        sns.heatmap(cm, annot=True, fmt=self.fmt, cmap=self.cmap, 
                    xticklabels=True, yticklabels=True)

        # Set plot labels and title
        plt.title(self.title, fontsize=self.title_fontsize)
        plt.xlabel(self.xlabel, fontsize=self.label_fontsize)
        plt.ylabel(self.ylabel, fontsize=self.label_fontsize)

        # Rotate tick labels if specified
        plt.xticks(rotation=self.x_ticklabels_rotation)
        plt.yticks(rotation=self.y_ticklabels_rotation)

        # Save the plot
        plt.savefig(self.save_filename)
        plt.show()

# Example usage
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]  # True labels
y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]  # Predicted labels

# Create an instance of the ConfusionMatrix class
conf_matrix = ConfusionMatrix(title='Confusion Matrix for Binary Classification',
                            xlabel='Predicted Label',
                            ylabel='True Label',
                            cmap='Blues',
                            save_filename='binary_confusion_matrix.png')

# Plot the confusion matrix
conf_matrix.plot(y_true, y_pred)
