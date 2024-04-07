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
def predict_label_for_image(cropped_img, model):
    # Resize the image to 400x100 pixels
    resized_img = cv2.resize(cropped_img, (400, 100))

    # Convert the resized image to grayscale
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    # Reshape the grayscale image to match the input shape of the model
    input_img = gray_img.reshape((1, 100, 400, 1))

    # Predict the label for the new image
    predicted_label = model.predict(input_img)

    # Decode the one-hot encoded label to get the predicted class
    predicted_class = np.argmax(predicted_label)

    return predicted_class