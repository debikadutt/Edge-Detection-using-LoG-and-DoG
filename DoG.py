import numpy as np
import cv2
import scipy
from scipy import signal
import matplotlib.pyplot as plt
import os

A=np.array([[ 0,0,-1,-1,-1,0,0],
            [ 0,-2,-3,-3,-3,-2,0],
            [ -1,-3,5,5,5,-3,-1],
            [-1,-3,5,16,5,-3,-1],
            [-1,-3,5,5,5,-3,-1],
            [0,-2,-3,-3,-3,-2,0],
            [0,0,-1,-1,-1,0,0]],
            dtype=np.float32)

img = cv2.imread('C:\Users\Debika Dutt\Documents\UB\CSE573\Bridge.jpg',0)
img.shape  

ratio = img.shape[0] / 500.0
new_width=int(img.shape[1]/ratio)
original = img.copy()

nimg=cv2.resize(img,(new_width,500))

I1 = scipy.signal.convolve2d(nimg,A)
I1=np.absolute(I1)
I1= (I1-np.min(I1))/float(np.max(I1)-np.min(I1))

cv2.imshow('image1.jpg',I1)

I1 = cv2.GaussianBlur(I1, (5, 5), 0)
cv2.imshow('image2.jpg',I1)


