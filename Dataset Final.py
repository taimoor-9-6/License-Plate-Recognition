#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image 
import sys 
import os
import shutil
from random import randint


# In[2]:


from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from skimage.io import imread
from skimage.filters import threshold_otsu


# In[3]:


letters = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ]


# In[4]:


training_directory='./Characters'
training_directory2='./Dataset'
bordertype=cv2.BORDER_CONSTANT
value = [randint(255, 255), randint(255, 255), randint(255, 255)]


# In[5]:


def read_training_data(training_directory):
    image_data = []
    target_data = []
    for each_letter in letters:
        for each in range(10):
            image_path=os.path.join(training_directory,each_letter + '_' + str(each) + '.jpg')
            image_details=cv2.imread(image_path)
            gray=cv2.cvtColor(image_details,cv2.COLOR_BGR2GRAY)
            top = int(.25 * gray.shape[0])  # shape[0] = rows
            bottom = top
            left = int(.25 * gray.shape[1])  # shape[1] = cols
            right = left
            gray=cv2.copyMakeBorder(gray,top,bottom,left,right,bordertype,None,value)
            resized=cv2.resize(gray,(70,70),interpolation=cv2.INTER_AREA)
            ret,thresh=cv2.threshold(resized,127,255,cv2.THRESH_BINARY)
            image_path2=os.path.join(training_directory2,each_letter+'_'+str(each)+'.jpg')
            new_image=cv2.imwrite(image_path2,thresh)
            flat_bin_image=thresh.reshape(-1)
            image_data.append(flat_bin_image)
            target_data.append(each_letter)
    return(np.array(image_data),np.array(target_data))


# In[6]:


def cross_validation(model, num_of_fold, train_data, train_label):
    # this uses the concept of cross validation to measure the accuracy
    # of a model, the num_of_fold determines the type of validation
    # e.g if num_of_fold is 4, then we are performing a 4-fold cross validation
    # it will divide the dataset into 4 and use 1/4 of it for testing
    # and the remaining 3/4 for the training
    accuracy_result = cross_val_score(model, train_data, train_label,
                                      cv=num_of_fold)
    print("Cross Validation Result for ", str(num_of_fold), " -fold")

    print(accuracy_result * 100)


# In[7]:


print('reading data')
training_dataset_dir = './Characters'
image_data, target_data = read_training_data(training_dataset_dir)
print('reading data completed')


# In[8]:


svc_model = SVC(kernel='linear', probability=True)

cross_validation(svc_model, 4, image_data, target_data)

print('training model')

# let's train the model with all the input data
svc_model.fit(image_data, target_data)

import pickle
print("model trained.saving model..")
filename = './finalized_model.sav'
pickle.dump(svc_model, open(filename, 'wb'))
print("model saved")


# In[ ]:





# In[ ]:




