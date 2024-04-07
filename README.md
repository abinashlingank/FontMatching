# Font Matching

This repository contains Python implementations for font matching using computer vision techniques and a deep learning model. It provides recommendations for matching fonts with confidence levels based on input images containing the text "Hello, World!"

## Contents

1. [Introduction](#introduction)
2. [Implementation](#implementation)
3. [Generating Synthetic Data](#generating-synthetic-data)
4. [Model Architecture](#model-architecture)
5. [Training Dataset](#training-dataset)
6. [Streamlit Application](#streamlit-application)
7. [How to Run It with Colab?](#how-to-run-it-with-colab)

## Introduction

Font matching is a computer vision and deep learning project aimed at classifying the fonts used in images containing the text "Hello, World!" The goal is to provide recommendations for matching fonts with confidence levels, enhancing the user's ability to replicate font styles from uploaded images.

## Implementation

The implementation includes several key components:

### 1. Generating Synthetic Data

Synthetic data is generated using Python scripts to create images with the text "Hello, World!" in various font styles. This ensures a diverse training dataset for the font classification model.

### 2. Model Architecture

The font classification model is built using a Convolutional Neural Network (CNN) architecture. The model is trained on the synthetic data to classify fonts with high accuracy.

### 3. Training Dataset

The training dataset consists of synthetic images with the text "Hello, World!" in different fonts. These images are used to train the font classification model.
We have trained the model using Google Colab.
Refer the Python notebook from this link https://github.com/abinashlingank/FontMatching/blob/main/FontMatching.ipynb

### 4. Streamlit Application

A Streamlit web application is created for easy visualization and usage of the font matching system. Users can upload images containing text and receive font recommendations with confidence levels.

## Generating Synthetic Data

To generate synthetic data, run the generate_data.py script. This script creates images with the text "Hello, World!" in various font styles and saves them to the dataset directory.

```bash
python generate_data.py
```

## Model Architecture

The CNN architecture used for font classification consists of Conv2D layers, MaxPooling2D layers, Flatten layers, and Dense layers. The model is defined and trained in the train.py script.


![model_architecture](https://github.com/abinashlingank/FontMatching/assets/114637586/06e95bd4-a78f-4932-829f-1d563abfdbed)

## Training Dataset

The training dataset is located in the dataset directory. It contains synthetic images with the text "Hello, World!" in different fonts.

## Streamlit Application

## Streamlit Application

To run the Streamlit application locally, follow these steps:

1. Make sure you have Streamlit installed. If not, install it using the following command:

```bash
pip install streamlit
```
2. Run the following command to start the Streamlit application:

```bash
streamlit run app.py
```

This will launch the web application locally, allowing users to upload images and receive font recommendations.

## How to Run It with Colab?

1. Open a new notebook in Google Colab.
2. Download the model from the drive link https://drive.google.com/file/d/1RZuSYuPByXn0uNDyOqgeDkZdT5yyOKtm/view?usp=sharing
3. Download app.py file from the drive link https://drive.google.com/file/d/1r10_AOUg5bv94Dv9Ioo1ImvLQUDqYvyl/view?usp=sharing
4. Upload the downloaded model file and the app.py file to the Google Colab notebook.
5. Install streamlit using the command
```python
!pip install streamlit
```
7. To deploy the Streamlit application in Colab, run the following command in a cell:
```python
!wget -q -O - ipv4.icanhazip.com
```

```python
!streamlit run app.py & npx localtunnel --port 8501
```

8. In the final cell, it will produce a link
https://quick-worms-kiss.loca.lt/
9. Go to the link. Then, it will ask for the tunnel passcode, which is available in the last cell output (e.g.,34.125.239.107:8501).
10. Now upload any font image file into it.
11. Finally, the output is displayed.

This will start the Streamlit application server in the background, and you can access the application using the generated URL.
