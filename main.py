# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 21:11:11 2023

@author: dawso
"""

from segmentImage import subtractBackground

path = 'Single_bike_cctv.mp4'
contourArea = 1000
y_top = 120
y_bottom = 359
x_left = 150
x_right = 600
crossLineLow = 100
crossLineHigh = 140


path2 = 'india_long.webm'
contourArea2 = 400
y_top2 = 0
y_bottom2 = 335
x_left2 = 0
x_right2 = 595
crossLineLow2 = 150
crossLineHigh2 = 220

# 120 350
subtractBackground(path, y_top, y_bottom, x_left, x_right, contourArea, crossLineLow, crossLineHigh, 352)
#subtractBackground(path2, y_top2, y_bottom2, x_left2, x_right2, contourArea2, crossLineLow2, crossLineHigh2)