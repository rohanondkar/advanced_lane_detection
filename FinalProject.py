import cv2
import numpy as np

#Note I am making these comments with 4 hour of sleep in the last 48 hours so sorry for nonsense

# Get the video
cap = cv2.VideoCapture("project_video.mp4")
frm = 0

# setting up Parameter for top down
# values for this video provided by Andy. I need to ask him how much damn trial error he went through to get these because GAWD DAMN
pts1 = np.float32([[977, 620], [848, 544],
                   [494, 544], [405, 620]])
pts2 = np.float32([[848, 620], [848, 544],
                   [494, 544], [494, 620]])


#getting perspective from points in top down view
matrix = cv2.getPerspectiveTransform(pts1, pts2)
matrix_reverse = cv2.getPerspectiveTransform(pts2, pts1)

#used to make sure the pixels per feet remain consitent and ya know... like feet
y_scale = 3.7625

#yellow line boundary.
#Is is yellow or orange I genuinely cannot tell the difference
min_val = np.array([10, 50, 135])
max_val = np.array([30, 255, 255])
# trasholding values for white
tresh_min = 200
tresh_max = 255
# distance from center
dx1 = 313
dx2 = 400
dy = 2433
y_interval = 45
step = 4

while True:
    success, video = cap.read()
    output = np.copy(video)

    #Perspective Transform
    vid = cv2.warpPerspective(video, matrix, (len(video[0]), len(video)))
    vid = cv2.resize(vid, (0, 0), fx=1, fy=y_scale)

    #Get the yellow/ orange values
    hsv = cv2.cvtColor(vid, cv2.COLOR_BGR2HSV)
    range_mask = cv2.inRange(hsv, min_val, max_val)

    #Gte white values (maybe some catching bc lines aren't solid
    #I mean we need to switch lanes after all
    vidG = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    vidB = cv2.GaussianBlur(vidG, (3, 3), 0)
    ret, maskW = cv2.threshold(vidB, tresh_min, tresh_max, cv2.THRESH_BINARY)
    maskW = cv2.dilate(maskW, np.ones((3, 3), np.uint8))
    #gets the mask range
    total_mask = range_mask + maskW
    total_mask = cv2.erode(total_mask, np.ones((3, 3), np.uint8))
    total_mask = cv2.dilate(total_mask, np.ones((13, 13), np.uint8))

    #masking the region
    if frm == 0:
        max_x = len(total_mask[0])
        half_val_x = int(max_x / 2)
        max_y = len(total_mask)

        #values for finding distances from half and center
        x1 = half_val_x - dx1
        x2 = half_val_x - dx2
        x3 = half_val_x + dx2
        x4 = half_val_x + dx1

    blank = np.zeros_like(total_mask)
    poly = np.array([[x1, dy], [x2, 0], [x3, 0], [x4, dy]], int)
    cv2.fillConvexPoly(blank, poly, 255)
    total_mask = cv2.bitwise_and(total_mask, total_mask, mask=blank)

    #getting the left and right maskes
    left_mask = total_mask[0:max_y, 0:half_val_x]
    right_mask = total_mask[0:max_y, half_val_x:max_x]


    left_coordinate = [[0, 0]]
    right_coordinate = [[0, 0]]
    #setup ends here. Jesus Christ how much setup do we need

    #"This is where the fun begins"
    #-Anakin Skywalker age 9 before he killed billions of aliens on the Trade Federation ship
    #May the fourth be with you
    for i in range(0, max_y, y_interval):
        first = True
        #Go through the left hand side of the video
        for j in range(half_val_x - 1, 0, -step):
            if left_mask[i, j] == 255 and first:
                start_j = j
                first = False
            if (left_mask[i, j] == 0 or j <= 1) and first is False:
                end_j = j + step
                first = True

                true_j = int((start_j + end_j) / 2)
                left_coordinate = np.append(left_coordinate, [[true_j, i]], axis=0)
                break

        first = True
        # Go through the right hand side of the video
        for j in range(0, half_val_x, step):
            if right_mask[i, j] == 255 and first:
                start_j = j
                first = False
            if (right_mask[i, j] == 0 or j >= len(right_mask[0]) - 1) and first is False:
                end_j = j - step
                first = True

                true_j = int(half_val_x + (end_j + start_j) / 2)
                right_coordinate = np.append(right_coordinate, [[true_j, i]], axis=0)
                break
    #Return a new array with sub-arrays along an axis deleted.
    left_coordinate = np.delete(left_coordinate, 0, 0)
    right_coordinate = np.delete(right_coordinate, 0, 0)

    #found thanks to https://en.wikipedia.org/wiki/Radius_of_curvature and https://www.cuemath.com/radius-of-curvature-formula/
    #and a lotta crying when I saw calc in the formula
    def findCurvature(array):
        radius_x = []
        data_x = array[:, 0]
        data_y = array[:, 1]
        p = np.polyfit(data_x, data_y, 2)
        p1 = 2 * p[0]

        for xPoint in data_x:
            p2 = 2 * p[0] * xPoint + p[1]
            #** is used to pass a keyword, variable-length argument dictionary to a function
            radius = ((1 + p2 ** 2) ** 1.5) / (30.1 * abs(p1))
            radius_x = np.append(radius_x, radius)

        min_x = np.min(data_x)
        max_x = np.max(data_x)
        length_x = max_x - min_x
        theory_x = np.linspace(min_x, max_x, length_x)
        theory_y = p[0] * theory_x ** 2 + p[1] * theory_x + p[2]
        total_theory = np.stack([theory_x, theory_y], axis=-1)

        return radius_x.mean(), total_theory.astype(int)


    text = ""
    if len(right_coordinate) != 0 and len(left_coordinate) != 0:
        left_radius, left_coordinate_new = findCurvature(left_coordinate)
        right_radius, right_coordinate_new = findCurvature(right_coordinate)
        r = int((left_radius + right_radius) / 2)

        #requirement to be considered right
        if right_coordinate[-1, 0] - right_coordinate[0, 0] < 0:
            text = "Right"
        else:
            text = "Left"
        #max value of radius before straight
        #is a radius is too big, it's not a circle anymore
        if r > 7000:
            r = 0
            text = "Straight"
        #radius estimate
        cv2.putText(output, "Radius = " + str(int(r)) + "ft", (0, 25),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

    cv2.putText(output, text, (0, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

    coordinate = np.append(left_coordinate, np.flip(right_coordinate, axis=0), axis=0)


    cv2.polylines(vid, [coordinate], True, (31, 150, 47), 14)

    coordinate[:, 1] = coordinate[:, 1] / y_scale
    coordinate = np.array([coordinate], dtype=np.float32)
    coordinate_output = cv2.perspectiveTransform(coordinate, matrix_reverse)
    coordinate_output = np.array(coordinate_output[0, :, :], int)
    # drawes the lines and fills everything out
    cv2.fillPoly(output, [coordinate_output], (31, 150, 47))
    cv2.polylines(output, [coordinate_output], True, (31, 150, 47), 4)

    total_mask = cv2.cvtColor(total_mask, cv2.COLOR_GRAY2BGR)
    vid1 = total_mask
    vid2 = vid
    vidStack = np.hstack([vid1, vid2])

    final = output


    cv2.imwrite(r"output\frame" + str(frm) + ".jpg", final)
    frm += 1

    scale = 1
    final = cv2.resize(final, (0, 0), fx=scale, fy=scale)
    cv2.imshow("result", final)

    #LOOP breaking
    #but fr tho why don't it work?!?!?!?!?!?!?!?!
    #nvm it works
    if cv2.waitKey(1) == ord('e'):
        break
