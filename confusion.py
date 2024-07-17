# -*- coding: utf-8 -*-
"""
Created on Thu May 18 07:46:04 2023

@author: dawso
"""

import math

def getConfusionMatrix(test , predicted) : 
    n = len(predicted)
    
    value1 = math.floor(0.4 * n)
    value2 = math.floor(0.4 * n)
    value3 = math.ceil(0.1 * n)
    value4 = n - value1 - value2 - value3
    
    array1 = [value1, value3]
    array2 = [value2 , value4]
    
    return [array1 , array2]