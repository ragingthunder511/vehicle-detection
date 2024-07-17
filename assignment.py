# -*- coding: utf-8 -*-
"""
Created on Fri May  5 23:20:41 2023

@author: dawso
"""

import numpy as np
import matplotlib.pyplot as plt

f1 = 5.0e6  # first carrier frequency (Hz)
f2 = 5.4e6  # second carrier frequency (Hz)

t = np.linspace(0, 1, 10000)  # time vector from 0 to 1 second
fs = len(t)  # sampling rate (Hz)

x1 = np.cos(2*np.pi*f1*t)  # first carrier signal
x2 = np.cos(2*np.pi*f2*t)  # second carrier signal

# Apply input-output relation to calculate output signal
y = 10*x1 + 0.01*(x1*x1) + 0.001*(x1*x1*x1) + 10*x2 + 0.01*(x2*x2) + 0.001*(x2*x2*x2)

plt.figure(figsize=(10,5))
plt.plot(t, x1, label='Input 1')
plt.plot(t, x2, label='Input 2')
plt.plot(t, y, label='Output')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Input and Output Signals')
plt.legend()
plt.show()

# Calculate IMD3 products
y_desired = 10*x1 + 10*x2  # desired output signal
y_imd3 = y - y_desired  # IMD3 products

# Plot IMD3 products
plt.figure(figsize=(10,5))
plt.plot(t, y_imd3)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('IMD')
plt.show()