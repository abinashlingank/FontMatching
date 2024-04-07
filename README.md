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

### 4. Streamlit Application

A Streamlit web application is created for easy visualization and usage of the font matching system. Users can upload images containing text and receive font recommendations with confidence levels.

## Generating Synthetic Data

To generate synthetic data, run the generate_data.py script. This script creates images with the text "Hello, World!" in various font styles and saves them to the dataset directory.

bash
python generate_data.py


## Model Architecture

The CNN architecture used for font classification consists of Conv2D layers, MaxPooling2D layers, Flatten layers, and Dense layers. The model is defined and trained in the train.py script.

## Training Dataset

The training dataset is located in the dataset directory. It contains synthetic images with the text "Hello, World!" in different fonts.

## Streamlit Application

## Streamlit Application

To run the Streamlit application locally, follow these steps:

1. Make sure you have Streamlit installed. If not, install it using the following command:

bash
pip install streamlit

2. Run the following command to start the Streamlit application:

bash
streamlit run app.py


This will launch the web application locally, allowing users to upload images and receive font recommendations.

## How to Run It with Colab?

1. Upload the repository to your Google Drive.
2. Open the font_matching_colab.ipynb notebook in Google Colab.
3. Mount your Google Drive by executing the following code snippet in a cell:
python
from google.colab import drive
drive.mount('/content/drive')

4. Navigate to the repository directory using the following command:
bash
cd /content/drive/MyDrive/path_to_repository

Replace path_to_repository with the actual path where the repository is located in your Google Drive.

5. Run the notebook cells in font_matching_colab.ipynb to generate synthetic data, train the model, and deploy the Streamlit application.
6. To deploy the Streamlit application in Colab, run the following command in a cell:
python
!pip install streamlit
!wget -q -O - ipv4.icanhazip.com
!streamlit run app.py & npx localtunnel --port 8501


This will start the Streamlit application server in the background, and you can access the application using the generated URL.
