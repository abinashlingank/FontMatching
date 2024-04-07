#Import necessary libraries
import cv2
import numpy as np
import tensorflow as tf

#Load the trained model
loadedModel=tf.keras.models.load_model("FontMatchingModel")

# Function to preprocess input image for inference
def preprocessImage(imagePath):
    img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (32, 32))
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img

# Function to perform inference
def classifyFont(imagePath, model):
    img = preprocessImage(imagePath)
    prediction = model.predict(img)
    fontIndex = np.argmax(prediction)
    confidence = np.max(prediction)
    return fontIndex, confidence

# Path to input image
imagePath = "path/to/input/image.jpg"

# Perform inference
fontIndex, confidence = classifyFont(imagePath, loadedModel)
print("Predicted font:", fontIndex, "with confidence:", confidence)