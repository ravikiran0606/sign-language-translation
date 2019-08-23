from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications import InceptionV3
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from imutils import paths
import pickle as pkl

import pyttsx3
import glob
import cv2
import numpy as np

def getListOfFiles(dirName):
    listOfFile = os.listdir(dirName)
    allFiles = list()

    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

if __name__ == "__main__":

    # get the image paths
    imagePaths = getListOfFiles("datasetnewtemp/")
    data = []
    labels = []

    # mobilenet features
    base_model = MobileNet(input_shape=(224, 224, 3), weights='imagenet',include_top=False)

    # load the dataset
    print("Data loading started")
    index_img = 1
    for image in imagePaths:
        label = os.path.split(os.path.split(image)[0])[1]
        labels.append(label)

        img = cv2.imread(image)
        img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
        outp = base_model.predict(img.reshape(1,img.shape[0],img.shape[1],img.shape[2]))
        outp = outp.reshape(1,outp.shape[0]*outp.shape[1]*outp.shape[2]*outp.shape[3])
        data.append(outp)
        print("Loading Image Num = " + str(index_img))
        index_img = index_img + 1

    data = np.array(data)
    labels = np.array(labels)

    le = LabelEncoder()
    labels = le.fit_transform(labels)
    data = data.reshape(data.shape[0], -1)
    (trainX, testX, trainY, testY ) = train_test_split(data, labels, test_size= 0.25, random_state=30)
    model = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
        
    # training the model
    print("Model training started")
    model.fit(trainX, trainY)

    # saving the model
    print("Saving model")
    with open('inception_knn_image_classifier2.pkl', 'wb') as f:
        pkl.dump(model, f)   

    with open('inception_knn_image_le2.pkl', 'wb') as f:
        pkl.dump(le, f)  
