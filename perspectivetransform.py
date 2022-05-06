import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    #visualise circles on screen
    #optional
    cv2.circle(frame, (250, 75), 5, (255, 0, 0), -1)
    cv2.circle(frame, (250, 300), 5, (255, 0, 0), -1)
    cv2.circle(frame, (600, 75), 5, (255, 0, 0), -1)
    cv2.circle(frame, (600, 300), 5, (255, 0, 0), -1)

    pts1 = np.float32([[250, 75], [600, 75], [250, 300], [600, 300]])
    #This 4 points are the size of the new window where we want to display the image transformed
    pts2 = np.float32([[0, 0], [400, 0], [0, 600], [400, 600]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(frame, matrix, (400, 600))

    #shows frame
    cv2.imshow("Frame", frame)
    cv2.imshow("PT", result)

    #Escape key (s is the key)
    escape_key = cv2.waitKey(1)
    if escape_key == 27:
        break

cap.release()
cv2.destroyAllWindows()

