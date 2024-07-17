# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 16:10:39 2023

@author: dawso
"""

import cv2
import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.image as mpimg
import numpy as np

import glob

count = 500
arr = []
files = glob.glob('GD_Cars/*')
for f in files:
    if count == 0:
        break
    count = count - 1
    img = cv2.imread(str(f))
    resized = cv2.resize(img  , (500 , 500))
    gray = resized[:,:,0]
    gray = np.array(gray)
    
    singlearr = gray.flatten()
    singlearr = np.array(singlearr)
    arr.append(singlearr)
    

count = 500
files = glob.glob('GD_Bikes/*')
for f in files:
    if count == 0:
        break
    count = count - 1
    img = cv2.imread(str(f))
    resized = cv2.resize(img  , (500 , 500))
    gray = resized[:,:,0]
    gray = np.array(gray)
    
    singlearr = gray.flatten()
    singlearr = np.array(singlearr)
    arr.append(singlearr)

count = 700
test_set = []
files = glob.glob('Images/*')
for f in files:
    if count == 0:
        break
    count = count - 1
    img = cv2.imread(str(f))
    resized = cv2.resize(img  , (500 , 500))
    gray = resized[:,:,0]
    gray = np.array(gray)
    
    singlearr = gray.flatten()
    singlearr = np.array(singlearr)
    test_set.append(singlearr)
    
    
df2 = pd.DataFrame(data=test_set)
temp_ytest = ["Car", "Bike", "Car", "Car", "Car"]
    
list1 = []
for i in range(len(arr)):
    if i < len(arr)/2:
        list1.append("Car")
    else:
        list1.append("Bike")

list1 = np.array(list1)
arr = np.array(arr)
print(arr)
df = pd.DataFrame(data=arr)
df['target'] = list1
print(df)


X = df.drop('target',axis='columns')
y = df.target
 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.01)
 
model = RandomForestClassifier(n_estimators=100)
 
model.fit(X_train, y_train)
 
print(model.score(df2, temp_ytest))
 
y_predicted = model.predict(df2)
print(y_predicted)
 
cm = confusion_matrix(temp_ytest, y_predicted)
 
print(cm)




# =============================================================================
#     plt.imshow(img)
#     plt.figure()
# =============================================================================
# =============================================================================
# 
# digits = load_digits()
# 
# dir(digits)
# 
# # =============================================================================
# # plt.gray() 
# # for i in range(4):
# #     plt.matshow(digits.images[i]) 
# #     plt.figure()
# # =============================================================================
#     
# df = pd.DataFrame(digits.data)
# df.head()
# 
# 
# df['target'] = digits.target
# #df[0:12]
# 
# X = df.drop('target',axis='columns')
# y = df.target
# 
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
# 
# model = RandomForestClassifier(n_estimators=20)
# 
# model.fit(X_train, y_train)
# 
# print(model.score(X_test, y_test))
# 
# y_predicted = model.predict(X_test)
# 
# cm = confusion_matrix(y_test, y_predicted)
# 
# print(cm)
# =============================================================================
