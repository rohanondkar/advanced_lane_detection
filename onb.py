from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv

img = cv.imread('road.jpg', 0)
#canny with proper gradient
edges = cv.Canny(img, 100, 200)

#original image
plt.subplot(121), plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

#canny image
plt.subplot(122), plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

#show and end
plt.show()