from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import pickle as pkl
import pyttsx3
import os
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

def classify_webcam(model, le):
    c = 0
    cap = cv2.VideoCapture(0)
    res, score = '', 0.0
    i = 0
    mem = ''
    consecutive = 0
    sequence = ''
    resstring = ''
    prevword = ''
    speaker_text = pyttsx3.init()

    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        resimg = np.zeros((200,1200,3), np.uint8)

        if ret:
            c += 1
            img = cv2.resize(img, (227, 227), interpolation = cv2.INTER_AREA)
            a = cv2.waitKey(1) # waits to see if `esc` is pressed
            if i == 4:
                scores = model.predict_proba(img.reshape(1,-1))
                res = le.classes_[np.argmax(scores)]
                score = np.max(scores)
                if score >=0.95:
                    if res == "clear":
                        resstring = ''
                    elif res != prevword:
                        resstring = resstring + res + " "
                        prevword = res
                        speaker_text.say(res)
                    print(res, score)
                    print("\n")
                i = 1
            i += 1

            
            speaker_text.runAndWait()
            cv2.putText(resimg, '%s' % (resstring.upper()), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imshow("Translated Sentence", resimg)
            cv2.imshow("Video Screen", img)

            if a == 27: # when `esc` is pressed
                break

    cv2.destroyAllWindows() 
    cv2.VideoCapture(0).release()

if __name__ == "__main__":

    # get the image paths
    imagePaths = getListOfFiles("datasetnewtemp/")
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

    # webcam classification
    classify_webcam(model, le)