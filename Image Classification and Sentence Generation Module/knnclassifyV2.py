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

nextW = {"i":"am","how":"are","what":"is","there":"is"}

def getNextWord(g_word):
    if g_word in nextW.keys():
        return g_word + " " + nextW[g_word]
    else:
        return g_word

def classify_webcam(base_model, model, le):
    c = 0
    cap = cv2.VideoCapture(0)
    res, score = '', 0.0
    cres = ''
    i = 0
    kwindowsize = 2
    consecutive = 1
    resstring = ''
    prevword = ''
    prevcres = ''
    speaker_text = pyttsx3.init()
    prediction_threshold = 0.9

    while True:
        ret, img = cap.read()
        # img = cv2.flip(img, 1)
        resimg = np.zeros((200,1200,3), np.uint8)

        if ret:
            c += 1
            img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
            a = cv2.waitKey(1) # waits to see if `esc` is pressed
            if i == 4:
                outp = base_model.predict(img.reshape(1,img.shape[0],img.shape[1],img.shape[2]))
                outp = outp.reshape(1,outp.shape[0]*outp.shape[1]*outp.shape[2]*outp.shape[3])
                scores = model.predict_proba(outp)
                res = le.classes_[np.argmax(scores)]
                score = np.max(scores)

                # score above prediction threshold
                if score >=prediction_threshold:

                    # checking for consecutive predictions
                    if res == prevword:
                        consecutive = consecutive + 1
                    else:
                        consecutive = 1

                    # speaking out the result
                    if consecutive == kwindowsize:
                        cres = res
                        if cres == "nothing":
                            resstring = resstring
                        elif cres == "clear":
                            resstring = ''
                            speaker_text.say(cres)
                        elif cres != prevcres:
                            resstring = resstring + getNextWord(cres) + " "
                            speaker_text.say(getNextWord(cres))
                        prevcres = cres
                        consecutive = 1
                        
                    # setting the prev word for next loop
                    prevword = res

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
    # loading the model    
    print("Loading the model..")

    base_model = MobileNet(input_shape=(224, 224, 3), weights='imagenet',include_top=False)

    with open('trained_models/inception_knn_image_classifier.pkl', 'rb') as f:
        model = pkl.load(f) 

    with open('trained_models/inception_knn_image_le.pkl', 'rb') as f:
        le = pkl.load(f) 

    # webcam classification
    classify_webcam(base_model, model, le)