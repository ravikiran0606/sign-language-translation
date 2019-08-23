#!/usr/bin/env python
# coding: utf-8

# In[18]:


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
# import matplotlib.pylot as plt
# %matplotlib inline


# In[2]:


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


# In[3]:


imagePaths = getListOfFiles("datasetnew/")


# In[4]:


data = []
labels = []


# In[5]:

print("Data loading started")
for image in imagePaths:

    label = os.path.split(os.path.split(image)[0])[1]
    labels.append(label)

    img = cv2.imread(image)
    img = cv2.resize(img, (227, 227), interpolation = cv2.INTER_AREA)
    data.append(img)


# In[6]:


data = np.array(data)
labels = np.array(labels)


# In[7]:


le = LabelEncoder()


# In[8]:


labels = le.fit_transform(labels)


# In[9]:


data = data.reshape(data.shape[0], -1)


# In[10]:


data.shape


# In[11]:


(trainX, testX, trainY, testY ) = train_test_split(data, labels, test_size= 0.25, random_state=30)


# In[12]:


model = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)


# In[13]:

print("Model training started")
model.fit(trainX, trainY)


# In[14]:


# cv2.imshow("image", testX[0].reshape(227,227,3))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# yy = model.predict_proba(testX[0].reshape(1,-1))


# In[15]:


# le.classes_[7]


# In[16]:


# Saving the model
# le.classes_


# In[ ]:

print("Saving Model")
with open('knn_image_classifier_2.pkl', 'wb') as f:
    pkl.dump(model, f)   


# In[ ]:




