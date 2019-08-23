from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import pickle as pkl
import os
import glob
import cv2
import numpy as np


classes = ['I', 'add', 'assist', 'assistance', 'boxing', 'control', 'friend',
       'ghost', 'good', 'gun', 'handcuffs', 'help', 'how', 'learn', 'lose',
       'me', 'measure', 'mirror', 'promise', 'skin', 'snake', 'some',
       'stand', 'sugar', 'there', 'thick', 'this', 'time', 'vacation',
       'varanasi', 'weight', 'what', 'where', 'you']

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
    imagePaths = getListOfFiles("datasetnew/")
    data = []
    labels = []

    # load the dataset
    print("Data loading started")
    for image in imagePaths:
        label = os.path.split(os.path.split(image)[0])[1]
        labels.append(label)

        img = cv2.imread(image)
        img = cv2.resize(img, (227, 227), interpolation = cv2.INTER_AREA)
        data.append(img)

    
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

    # testing and metric calculation
    print("Evaluation started")
    testImagePaths = getListOfFiles("test/")
    
    y_pred_list = []
    y_truth_list = []

    for image in testImagePaths:
        label = os.path.split(os.path.split(image)[0])[1]
        img = cv2.imread(image)
        img = cv2.resize(img, (227, 227), interpolation = cv2.INTER_AREA)
        pred_label = model.predict(img.reshape(1,-1))
        print(label, classes[pred_label[0]])
        y_pred_list.append(classes[pred_label[0]])
        y_truth_list.append(label)

    with open("y_pred_list.pkl","wb") as f:
        pkl.dump(y_pred_list,f)

    with open("y_truth_list.pkl","wb") as f:
        pkl.dump(y_truth_list,f)