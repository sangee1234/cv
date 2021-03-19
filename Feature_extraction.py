'''
This script helps with understanding basic feature extraction methods.

Edge: Where there is a sharp change in pixel intensity. If subtracting the value from either side returns a high value, it belongs to an edge.
Different kernels are available to do such operations. The above can be acheieved using Prewitt kernel.
i.e. -1 0 1
     -1 0 1
     -1 0 1

'''

import numpy as np
from skimage.io import imread, imshow
from skimage.filters import prewitt_h, prewitt_v
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.feature import hog


image = imread('fireworks.jpeg', as_gray=True)
edges_prewitt_hor = prewitt_h(image)
imshow(edges_prewitt_hor)
plt.show()

'''
HOG(Histogram of Oriented Gradients)
Feature Descriptor: it is simplified representation of image that contains only most importamt information about image.
-It finds shape/structure of object
-It finds edge and edge direction
Gradient(magnitude) and orientation(direction)
-will generate histogram for various regions of image.
Histogram is created by taking bins of orientation and putting gradient instead of frequency.
And then follwed by normalization.
mag = sqrt(x^2 + y^2)
or = tan-1(y/x)
'''

image = imread('fireworks.jpeg')
resized_img = resize(image,(128,64))
fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True) 

ax1.imshow(resized_img, cmap=plt.cm.gray) 
ax1.set_title('Input image') 

ax2.imshow(hog_image, cmap=plt.cm.gray) 
ax2.set_title('Histogram of Oriented Gradients')

plt.show()

'''
SIFT: Scale Invariant Feature Transform)
Helps to identify keypoints that are scale and rotation invariant.
Major advantage: not affected by size/ orientation
i. Apply gaussian blur, too retain only relevant information
ii. Need to search on multiple sclaes by creating a scale space
iii. DOG (diff bw 2 blurs) to do feature enhancement
iv. To find keypoints, go through image and find local minima and maxima
    i.e. by comparing pixel with all neighbouring pixel
v. Eliminate poor keypoints
vi. Assign orientation to each keypoint so that invariant to rotation
    creatre HOG, bin at which max will give the orientation
vii. Keypoint descriptor
    - use neighbouring pixels to get descriptor for each keypoint
    - since we use neighbouring pixels, they are invariant to illumination
    - description: by getting HOG of neighbouring pixels
'''

from scipy import ndimage
from skimage.color import rgb2gray
import cv2
rotated = ndimage.rotate(image, 95)  
rotated_gray = rgb2gray(rotated)
image_gray = rgb2gray(rotated_gray)
sift = cv2.xfeatures2d.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(image_gray,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(rotated_gray,None)

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(image_gray, keypoints_1, rotated_gray, keypoints_2, matches[:50], img2, flags=2)
plt.imshow(img3)
plt.show()