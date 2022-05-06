import cv2
import numpy as np

# load image
image = cv2.imread('road.jpg')
# crop image
h,w = image.shape[:2]
image = image[200:h-20,20:550]
# create hsv
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# set lower and upper color limits
low_val = (0,0,0)
high_val = (179,45,96)
# Threshold the HSV image
mask = cv2.inRange(hsv, low_val,high_val)
# remove noise
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=np.ones((8,8),dtype=np.uint8))
# close mask
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones((20,20),dtype=np.uint8))

# improve mask by drawing the convexhull
ret, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    hull = cv2.convexHull(cnt)
    cv2.drawContours(mask,[hull],0,(255), -1)
# erode mask a bit to migitate mask bleed of convexhull
mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel=np.ones((5,5),dtype=np.uint8))

# remove this line, used to show intermediate result of masked road
road = cv2.bitwise_and(image, image,mask=mask)

# apply mask to hsv image
road_hsv = cv2.bitwise_and(hsv, hsv,mask=mask)
# set lower and upper color limits
low_val = (0,0,102)
high_val = (179,255,255)
# Threshold the HSV image
mask2 = cv2.inRange(road_hsv, low_val,high_val)
# apply mask to original image
result = cv2.bitwise_and(image, image,mask=mask2)

#show image
cv2.imshow("Result", result)
cv2.imshow("Road", road)
cv2.imshow("Mask", mask)
cv2.imshow("Image", image)

cv2.waitKey(0)
cv2.destroyAllWindows()