'''
Image segmentation
-Group together the pixels that have similar attributes using image segmentation
-Object detection doesn't tell anything about the shape of the object
-It creates pixel-wise mask for each object in the image

Eg, shape of cancerous cells plays a vital rooole in determining severity of the cancer
-Semantic segmentation will seperate object from background
-Instance segmentation willl seperate eacg instance of the class seperately

REGION-BASED segmentation
-Segment different objects using pixel values.
-Pixel values will be different for objects and background if theres a sharp contrast between them
-Can use threshold to separate, either global threshold or local threshold
'''

from skimage.color import rgb2gray
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

image = plt.imread('fireworks.jpeg')
gray = rgb2gray(image)
#use mean of pixel values as threshold
gray_r = gray.reshape(gray.shape[0] * gray.shape[1])
for i in range(gray_r.shape[0]):
    if gray_r[i]>gray_r.mean():
        gray_r[i]=1
    else:
        gray_r[i]=0
gray = gray_r.reshape(gray.shape[0], gray.shape[1])
# plt.imshow(gray, cmap='gray')
# plt.show()

'''
EDGE detection segmentation
-can detect ion them using filters and convolutions
-Sobel weight matrix: it has 2 matrices, one for detecting horizontal edges and one for vertical
-Filter that can detect booth horizontal and vertical edges at same time is laplace operator
'''

gray1 = rgb2gray(image)
sobel_horizontal = np.array([np.array([1, 2, 1]), np.array([0, 0, 0]), np.array([-1, -2, -1])])
sobel_vertical = np.array([np.array([-1, 0, 1]), np.array([-2, 0, 2]), np.array([-1, 0, 1])])
kernel_laplace = np.array([np.array([1, 1, 1]), np.array([1, -8, 1]), np.array([1, 1, 1])])
out_h = ndimage.convolve(gray, sobel_horizontal, mode='reflect')
out_v = ndimage.convolve(gray, sobel_vertical, mode='reflect')
out_l = ndimage.convolve(gray, kernel_laplace, mode='reflect')
#plt.imshow(out_h, cmap='gray')
# plt.show()

'''
CLUSTERING based segmentation
-Dividing into groups such that data more similar are in 1 group
-One such algorithm is kmeans clustering
Algorithm:
 i. select k venters
 ii. randomnly assign each data point to any one of k clusters
 iii. calculate centers of clusters
 iv. calculate distance from all points to center of each cluster
 v. depending on distancem reassign points to nearest cluster
 vi. calculate center of newly formed clusters
 vii. repeat till cluster centers dont change

It works well on small images, on large images becomes too computationally expensive
'''

from sklearn.cluster import KMeans
pic = plt.imread('fireworks.jpeg')/255
pic_n = pic.reshape(pic.shape[0]*pic.shape[1], pic_shape[2])
kmeans = KMeans(n_clusters=5, random_state=0).fit(pic_n)
pic2 = kmeans.cluster_centers_[kmeans.labels_]
#cluster_centers_ will retuen cluster centers and labels_function will give label for each pixel
pic3 = pic2.reshape(pic.shape[0], pic.shape[1],pic.shape[2])
#plt.imshow(pic3)
#plt.show()

'''
Mask RCNN
FAIR: Facebook AI Research created them
'''
