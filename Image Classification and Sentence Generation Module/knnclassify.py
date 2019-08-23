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

def classify_webcam():
    c = 0
    cap = cv2.VideoCapture(0)
    res, score = '', 0.0
    i = 0
    mem = ''
    consecutive = 0
    sequence = ''

    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)

        if ret:
            c += 1
            img = cv2.resize(img, (227, 227), interpolation = cv2.INTER_AREA)
            a = cv2.waitKey(1) # waits to see if `esc` is pressed
            if i == 4:
                scores = model.predict_proba(img.reshape(1,-1))
                res = classes[np.argmax(scores)]
                score = np.max(scores)
                i = 1
            i += 1

            print(res,score)
            print("\n")
            cv2.putText(img, '%s' % (res.upper()), (100,400), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 4)
            cv2.putText(img, '(score = %.5f)' % (float(score)), (100,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
            cv2.imshow("img", img)

            if a == 27: # when `esc` is pressed
                break

    cv2.destroyAllWindows() 
    cv2.VideoCapture(0).release()

if __name__ == "__main__":

    # loading the model
    print("Loading the model..")
    with open('knn_image_classifier.pkl', 'rb') as f:
        model = pkl.load(f)

    print("Model Loaded...")
    # webcam classification
    classify_webcam()