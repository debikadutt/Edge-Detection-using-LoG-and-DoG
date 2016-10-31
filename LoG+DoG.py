import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from scipy import signal
import scipy.ndimage as nd

def frame(image, kernel, function):
    offset = np.divide(np.subtract(kernel.shape, 1), 2)
    for i, j in np.ndindex(image.shape):
        i1, j1 = np.fmax(np.subtract((i,j), offset), (0,0))
        i2, j2 = np.fmin(np.add((i+1,j+1), offset), image.shape)
        res = image[i1:i2, j1:j2]
        k1 = np.add(np.subtract((i1,j1), (i,j)), offset)
        k2 = np.add(res.shape, k1)
        kernel2 = kernel[k1[0]:k2[0], k1[1]:k2[1]]
        function(i, j, res, kernel2)

def zero_cross(image, kernel, threshold):
    result = np.empty(image.shape, dtype=np.float32)
    def convolution(i, j, res, kern):
        c = np.sum(np.multiply(np.asfarray(res), kern))
        result[i,j] = c
    frame(image, kernel, convolution)

    output = np.empty(result.shape, dtype=np.uint8)
    def thresholding(i, j, res, kern):
        center = result[i,j]
        output[i,j] = 255
        if center > threshold:
            if np.count_nonzero(res < -threshold) > 0:
                output[i,j] = 0
        elif center < -threshold:
            if np.count_nonzero(res > threshold) > 0:
                output[i,j] = 0
    frame(result, np.ones((3,3),dtype=np.uint8), thresholding)
    return output


def laplacian_of_gaussian(image):
    kernel = np.array([[ 0,0,1,0,0],
                       [ 0,1,2,1,0],
                       [ 1,2,-16,2,1],
                       [0,1,2,1,0],
                       [0,0,1,0,0]],
                       dtype=np.float32)
    return zero_cross(image, kernel, 100)
    

def difference_of_gaussian(image):
    def gkernel(size = 11, sigma = 0.95):
        from cv2 import getGaussianKernel, CV_32F
        mat = getGaussianKernel(size, sigma, CV_32F)
        return np.dot(mat, np.transpose(mat))
        dog = gkernel(sigma=0.95) - gkernel(sigma=2.75)
        return zero_cross(image, dog, 2.75)

if __name__ == '__main__':
    image = cv2.imread('C:\Users\Debika Dutt\Documents\UB\CSE573\Bridge.jpg', 0)

    l = laplacian_of_gaussian(image)
    cv2.imshow('laplacian-of-gaussian', l)
    cv2.imwrite('laplacian-of-gaussian.png', l)
    
    d = difference_of_gaussian(image)
    cv2.imshow('difference-of-gaussian', d)
    cv2.imwrite('difference-of-gaussian.png', d)

    cv2.waitKey(0)