# -*- coding: utf-8 -*-
"""
Created on Sun May  7 16:48:12 2023

@author: dawso
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read image.
img = cv2.imread('Processed/on1.jpg', cv2.IMREAD_COLOR)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray_blurred = cv2.blur(gray, (3, 3))

# Apply Hough transform on the blurred image.
detected_circles = cv2.HoughCircles(gray_blurred,
				cv2.HOUGH_GRADIENT, 1, 1, param1 = 100,
			param2 = 30, minRadius = 1, maxRadius = 100)


imgplot = plt.imshow(img)
plt.show()

# Draw circles that are detected.
if detected_circles is not None:

	# Convert the circle parameters a, b and r to integers.
	detected_circles = np.uint16(np.around(detected_circles))

	for pt in detected_circles[0, :]:
		a, b, r = pt[0], pt[1], pt[2]

		# Draw the circumference of the circle.
		cv2.circle(img, (a, b), r, (0, 255, 0), 2)

		# Draw a small circle (of radius 1) to show the center.
		# cv2.circle(img, (a, b), 1, (0, 0, 255), 1)
        
plt.figure()
plt.imshow(img)
plt.show()
        
        
# =============================================================================
# 		cv2.imshow("Detected Circle", img)
# 		cv2.waitKey(0)
# =============================================================================
