#Importing necessary libraries
import cv2
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

#Loading data
def loadData(data):
    images=[]
    labels=[]
    for i in os.listdir(data):
        fontDir=os.path.join(data,i)
        for j in os.listdir(fontDir):
            imgPath=os.path.join(fontDir,j)
            #Reading the image data
            img=cv2.imread(imgPath,cv2.IMREAD_GRAYSCALE)
            #Resizing image to fixed size
            img=cv2.resize(img,(32,32))
            images.append(img)
            labels.append(i)
    return np.array(images),np.array(labels)

#Path
dataDir=""
images,labels=loadData(dataDir)  

#Splitting dataset into training and validation sets
trainImages,testImages,trainLabels,testLabels=train_test_split(images,labels,test_size=0.2,random_state=42)
            
#Model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32, 32, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)  # num_classes is the number of fonts
])
            
#Model training
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
model.fit(trainImages, trainLabels, epochs=10, testData=(testImages, testLabels))

#Save the trained model
model.save("FontMatchingModel")            