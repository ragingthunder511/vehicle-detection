# -*- coding: utf-8 -*-
"""
Created on Sun May  7 16:24:20 2023

@author: dawso
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

## TO CLear Folder #################################
import os
import glob


## Clear Old Images In Folder
files = glob.glob('Processed/*')
for f in files:
    os.remove(f)
    
path = "Biker_images/48540.jpg"

## Read File
img = mpimg.imread(path)
imgplot = plt.imshow(img)
plt.show()

## Crop Image
height = len(img)
width = len(img[0])
cropped_image = img[0 :  math.ceil(0.25*height) , 0 : width]

## Store cropped Image
plt.figure()
imgplot = plt.imshow(cropped_image)
plt.show()
cv2.imwrite('Processed/on1' + '.jpg', cropped_image)


# =============================================================================
# cv2.imshow("eknjndr", img)
# 
# 
# k = cv2.waitKey(1)
# 
# if k == ord('q'):
#      cv2.destroyAllWindows()
# =============================================================================
