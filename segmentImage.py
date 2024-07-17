# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 15:49:59 2022

@author: dawso
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tracker import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import glob

def subtractBackground(path, y_top, y_bottom, x_left, x_right, contourArea, crossLineLow, crossLineHigh, frameNumber) :
    files = glob.glob('Images/*')
    for f in files:
        os.remove(f)

    tracker = EuclideanDistTracker()

    Dict = {}

    # load a video
    video = cv2.VideoCapture(path)

    # You can set custom kernel size if you want.
    kernel = None

    # Initialize the background object.
    backgroundObject = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

    idx = 0
    counter = -1

    while True:
        # Read a new frame.
        counter = counter + 1
        ret, frame = video.read()

        # Check if frame is not read correctly.
        if not ret:
            break
        
        roi = frame.copy()
        roi = roi[y_top:y_bottom , x_left:x_right]
        

        # Apply the background object on the frame to get the segmented mask. 
        fgmask = backgroundObject.apply(roi)
        #initialMask = fgmask.copy()
        cv2.imshow('Background Mask', fgmask)
        
        
        # Perform thresholding to get rid of the shadows.
        _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
        noisymask = fgmask.copy()

        # Detect contours in the frame.
        contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a copy of the frame to draw bounding boxes around the detected cars.
        
        temp = roi.copy()
        
        detections = []
        # loop over each contour found in the frame.
        for cnt in contours:
            idx = idx + 1
            
            # Make sure the contour area is somewhat higher than some threshold to make sure its a car and not some noise.
            if cv2.contourArea(cnt) > contourArea:

                
                # Retrieve the bounding box coordinates from the contour.
                x, y, width, height = cv2.boundingRect(cnt)
                
                detections.append([x, y, width, height])
                
        
        boxes_ids = tracker.update(detections)
        
        #print(boxes_ids)
        cv2.line(temp, (0, crossLineLow), (x_right, crossLineLow), (255, 0, 0), 3)
        cv2.line(temp, (0, crossLineHigh), (x_right, crossLineHigh), (255, 0, 0), 3)
        
        
        
        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            cv2.putText(temp, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.rectangle(temp, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.imshow('Boxes', temp)
            
            vert_mid = y + y + h // 2
            
            if vert_mid > crossLineLow and vert_mid < crossLineHigh and Dict.get(id) == None :
                tempFrame = roi.copy()
                cropped_image = tempFrame[y : y+h , x : x+w]
                cv2.imwrite('Images/' + str(idx) + '.jpg', cropped_image)
                Dict[id] = True


        #cv2.imshow('Clean Mask', temp)
        k = cv2.waitKey(1)
        
        # Check if 'q' key is pressed.
        if k == ord('q'):
            
            # Break the loop.
            break

    # Release the VideoCapture Object.
    video.release()

    # Close the windows.q
    cv2.destroyAllWindows()
    

