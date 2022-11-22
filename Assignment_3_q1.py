# Problem 1

# importing important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from sklearn.decomposition import PCA
import cv2

# reading the picture
picture = imread("/content/sample_data/hw3_1.jpeg")
print(picture.shape)
plt.figure(figsize = [12,9])
plt.imshow(picture)

# determining thr shape of the image
print(picture.shape)

# splitting the image into its blue yellow and red 
blue,green,red = cv2.split(picture)

# pca function
def pca(picture,n):
    # initializing global variables
    global values, vect
    # Calculating the covariance matrix of the image, as the diagonal elements equal to the variance.
    c_m = np.cov(picture, rowvar = False)
    # getting Eigen value and vector
    values, vect = np.linalg.eig(c_m)
    # obtaining the principle component
    princ_Comp = vect[:,:n]
    s = princ_Comp @ princ_Comp.T
    pic_pca = np.dot(s.transpose(),picture)
    # returning the PCA value for the picture
    return pic_pca

# plotting the image in red, yellow and green and with varied components
def plot(merged):
    fig = plt.figure(figsize=(10,15))
    plt.imshow(merged.astype('uint8'))
    plt.axis('off')
    #plt.title('Image distributed in red, yellow green and merged again with varied components')

# plotting the picture at n value 100
pcablue = pca(blue,100)
pcagreen = pca(green,100)
pcared = pca(red,100)

pic_merged = (cv2.merge((pcablue,pcagreen,pcared)))
plot(pic_merged)

plt.title('100 Components')

# plotting the picture at n value 200
pcablue = pca(blue,200)
pcagreen = pca(green,200)
pcared = pca(red,200)

pic_merged = (cv2.merge((pcablue,pcagreen,pcared)))
plot(pic_merged)

plt.title('200 Components')

# plotting the picture at n value 300
pcablue = pca(blue,300)
pcagreen = pca(green,300)
pcared = pca(red,300)

pic_merged = (cv2.merge((pcablue,pcagreen,pcared)))
plot(pic_merged)

plt.title('300 Components')

# plotting the picture at n value 400
pcablue = pca(blue,400)
pcagreen = pca(green,400)
pcared = pca(red,400)

pic_merged = (cv2.merge((pcablue,pcagreen,pcared)))
plot(pic_merged)

plt.title('400 Components')

# plotting the picture at n value 500
pcablue = pca(blue,500)
pcagreen = pca(green,500)
pcared = pca(red,500)

pic_merged = (cv2.merge((pcablue,pcagreen,pcared)))
plot(pic_merged)

plt.title('500 Components')

# plottiing the variance vs number of components(n) graph to analyze its increase in the accuracy
variance=[]
for i in range(len(values)):
    variance.append(values[i]/np.sum(values))

plt.grid()
plt.plot(np.cumsum(variance))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance")

# referred from 
# https://medium.com/@pranjallk1995/pca-for-image-reconstruction-from-scratch-cf4a787c1e36 
# https://analyticsindiamag.com/guide-to-image-reconstruction-using-principal-component-analysis/