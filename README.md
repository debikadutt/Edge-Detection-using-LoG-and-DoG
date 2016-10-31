# Edge-Detection-using-LoG-and-DoG
CSE 573 Computer Vision and Image Processing : Implemented zero crossings of a gray scale image by using Laplacian of Gaussian and Difference of Gaussian
The edges in the image can be obtained by these steps:

*Applying LoG to the image

*Detection of zero-crossings in the image

*Threshold the zero-crossings to keep only those strong ones (large difference between the positive maximum and the negative minimum)

*The last step is needed to suppress the weak zero-crossings most likely caused by noise.
