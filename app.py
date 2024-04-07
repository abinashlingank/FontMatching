
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pytesseract
from tensorflow.keras.models import load_model

# Load the trained model
model_path = "fontMatchingModel.h5"
model = load_model(model_path)

fonts_data=['Arimo', 'Dancing_Script', 'Fredoka', 'Noto_Sans', 'Open_Sans', 'Oswald', 'Patua_One', 'PT_Serif', 'Roboto', 'Ubuntu']

# Function to predict label for a given image
def predict_label_for_image(cropped_img, model):
    resized_img = cv2.resize(cropped_img, (400, 100))
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    input_img = gray_img.reshape((1, 100, 400, 1))
    predicted_label = model.predict(input_img)
    predicted_class = np.argmax(predicted_label)
    return predicted_class

# Function to process the uploaded image
def process_image(image):
    # Convert image to OpenCV format
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Preprocess the image
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU, 11)
    
    # Detect text using Tesseract OCR
    raw_data = pytesseract.image_to_data(thresh)
    
    # Draw rectangles around detected words and predict font using the model
    for count, data in enumerate(raw_data.splitlines()):
        if count > 0:
            data = data.split()
            if len(data) == 12:
                x, y, w, h, content = int(data[6]), int(data[7]), int(data[8]), int(data[9]), data[11]
                cv2.rectangle(img_array, (x, y), (w+x, h+y), (255, 255, 0), 1)
                cropped_image = img_array[y:h+y, x:w+x]
                predicted_label = predict_label_for_image(cropped_image, model)
                cv2.putText(img_array, str(fonts_data[predicted_label]), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255) , 1)
    
    return img_array

# Streamlit app
st.title("Font Matching (Font Style Detection)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Process the image and display the result
    processed_image = process_image(image)
    st.image(processed_image, caption='Processed Image', use_column_width=True)
