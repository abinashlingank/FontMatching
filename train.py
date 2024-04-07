# Importing all the neccessary libraries
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D , UpSampling2D ,Conv2DTranspose
import cv2
from predict import predict_label_for_image

# Directory containing images
dataset_dir = "/content/dataset"

# Load images and labels
images = []
labels = []

# Iterate through each folder in the data directory
for folder in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, folder)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".png"):
                img = Image.open(os.path.join(folder_path, filename)).convert("L").resize((400, 100))
                img_array = np.array(img)
                images.append(img_array)
                labels.append(folder)

# Convert lists to numpy arrays
images = np.array(images)

# Convert string labels to numerical labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Convert numerical labels to one-hot encoded format
one_hot_labels = np.eye(len(np.unique(encoded_labels)))[encoded_labels]

# Print number of samples loaded
print("Samples loaded:", len(images))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(images, one_hot_labels, test_size=0.2, random_state=42)

# Initialize CNN model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(100, 400, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(encoded_labels)), activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train[..., np.newaxis], y_train, epochs=30, batch_size=32, validation_split=0.2)

# Save the model
model.save('fontMatchingModel.h5')

# Evaluate the model on test set
loss, accuracy = model.evaluate(X_test[..., np.newaxis], y_test)
print("Validation Accuracy:", accuracy)