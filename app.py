from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
#loadedModel = tf.keras.models.load_model("FontMatchingModel")

# Function to preprocess input image for inference
def preprocessImage(imagePath):
    img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (32, 32))
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img

# Function to perform inference and draw bounding boxes
def detectAndLabelFonts(imagePath, model):
    img = cv2.imread(imagePath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Perform text detection (you can replace this with your text detection method)
    # For demonstration purposes, we'll just assume we have a single text region
    textRegion = gray
    
    # Perform font classification on the detected text region
    fontIndex, confidence = classifyFont(textRegion, model)
    fontLabel = fontNames[fontIndex]  # Replace fontNames with your list of font names
    
    # Draw bounding box around the detected text region
    h, w = textRegion.shape
    cv2.rectangle(img, (0, 0), (w, h), (0, 255, 0), 2)
    
    # Label the detected font with confidence
    label = f"{fontLabel}: {confidence:.2f}"
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
 
    # Save the result image
    resultImagePath = "static/result_image.jpg"
    cv2.imwrite(resultImagePath, img)
    
    return resultImagePath

# Function to perform font classification
def classifyFont(textRegion, model):
    textRegion = cv2.resize(textRegion, (32, 32))
    textRegion = np.expand_dims(textRegion, axis=0)
    textRegion = np.expand_dims(textRegion, axis=-1)
    prediction = model.predict(textRegion)
    fontIndex = np.argmax(prediction)
    confidence = np.max(prediction)
    return fontIndex, confidence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def uploadImage():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        # Save the uploaded file
        uploadedImagePath = "static/uploaded_image.jpg"
        file.save(uploadedImagePath)

        # Perform font detection and labeling
        resultImagePath = detectAndLabelFonts(uploadedImagePath, loadedModel)
        return render_template('result.html', result_image=resultImagePath)

if __name__ == '__main__':
    app.run(debug=True)