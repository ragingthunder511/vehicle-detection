# -*- coding: utf-8 -*-
"""
Created on Thu May 18 00:51:39 2023

@author: dawso
"""

import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier

# Load the trained MLP classifier
classifier = MLPClassifier()
#classifier.load('helmet_classifier.pkl')

# Load the helmet cascade classifier
helmet_cascade = cv2.CascadeClassifier()

# Function to detect and classify helmets
def detect_helmet(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    helmets = helmet_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    helmets =[1,1,1]
    for (x, y, w, h) in helmets:
        helmet_roi = gray[y:y+h, x:x+w]
        helmet_roi = cv2.resize(helmet_roi, (64, 64))
        helmet_roi = helmet_roi.flatten().reshape(1, -1)

        # Classify the helmet using MLP classifier
        prediction = classifier.predict(helmet_roi)

        # Draw bounding box and label
        if prediction == 1:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, 'Helmet', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, 'No Helmet', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return image

# Load and process the test image
image = cv2.imread('2.png')
result = detect_helmet(image)

# Display the output image
cv2.imshow('Helmet Detection', result)
cv2.waitKey(0)
cv2.destroyAllWindows()