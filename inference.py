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
from cv2 import cv2_imshow

import pytesseract

# Load the image
image = cv2.imread("/content/drive/MyDrive/Fonts/sample.png")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to preprocess the image
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,11)

# Define a function to draw rectangles around detected words
def draw_boxes_on_text(img):
    # Return raw information about the detected texts
    raw_data = pytesseract.image_to_data(img)

    for count, data in enumerate(raw_data.splitlines()):
        if count > 0:
            data = data.split()
            if len(data) == 12:
                x, y, w, h, content = int(data[6]), int(data[7]), int(data[8]), int(data[9]), data[11]
                cv2.rectangle(image, (x, y), (w+x, h+y), (255, 255, 0), 1)
                cropped_image = image[y:h+y, x:w+x]
                predicted_label = predict_label_for_image(cropped_image, model)
                cv2.putText(image, str(fonts_data[predicted_label]), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255) , 1)

    return image

# Draw rectangles around detected words
image_with_boxes = draw_boxes_on_text(thresh)

# Display the result

cv2_imshow(image_with_boxes)