import scipy as sp
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('C:\Users\Debika Dutt\Documents\UB\CSE573\Bridge.jpg',0)
LoG = nd.gaussian_laplace(img, 2)
thres = np.absolute(LoG).mean() * 0.75
output = sp.zeros(LoG.shape)
w = output.shape[1]
h = output.shape[0]

for y in range(1, h - 1):
    for x in range(1, w - 1):
        patch = LoG[y-1:y+2, x-1:x+2]
        p = LoG[y, x]
        maxP = patch.max()
        minP = patch.min()
        if (p > 0):
            zeroCross = True if minP < 0 else False
        else:
            zeroCross = True if maxP > 0 else False
        if ((maxP - minP) > thres) and zeroCross:
            output[y, x] = 1
cv2.imshow("img",output)
cv2.show() 