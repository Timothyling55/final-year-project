import numpy as np
import imutils
import cv2

def nothing(x):
    pass

def createMask(map, lowerBound, upperBound, row, column):
    try:
        mask = cv2.inRange(map, lowerBound, upperBound)
    except:
        mask = np.zeros((column, row), 'uint8')
    return mask

def findObjRect(img):
    cnts = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    try:
        # find the contours in the edged image and keep the largest one
        c = max(cnts, key=cv2.contourArea)
        # compute the bounding box of the region and return it
        objDetected = cv2.minAreaRect(c)
        boxPoints = cv2.boxPoints(objDetected)
        boxPoints = np.intp(boxPoints)
        return boxPoints
    except:
        return 0

def getActualWidth(boxPoints, distance):
    if (boxPoints[1][0] - boxPoints[0][0]) >= (boxPoints[2][0] - boxPoints[1][0]):
        pixelWidth = boxPoints[1][0] - boxPoints[0][0]
    else:
        pixelWidth = boxPoints[2][0] - boxPoints[1][0]
    return (distance * pixelWidth) / avgFocalLength

def getActualHeight(boxPoints, distance):
    if (boxPoints[1][0] - boxPoints[0][0]) <= (boxPoints[2][0] - boxPoints[1][0]):
        pixelHeight = boxPoints[1][1] - boxPoints[0][1]
    else:
        pixelHeight = boxPoints[2][1] - boxPoints[1][1]
    return abs((distance * pixelHeight) / avgFocalLength)

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap2 = cv2.VideoCapture(0)
cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

cv_file = cv2.FileStorage("./MonoParam6.xml", cv2.FILE_STORAGE_READ)

Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
Focal_Length = cv_file.getNode("Focal_Length").mat()
Q = cv_file.getNode('Q').mat()
cv_file.release()

DISTANCE_BETWEEN_CAMERAS = 0.07 #meters
DISTANCE_CONSTANT = 03.5
dispValuesFiltered = []

# Disparity Paramters
win_size = 5
min_disp = -1
max_disp = 63  # min_disp * 9
num_disp = max_disp - min_disp  # Needs to be divisible by 16

avgFocalLength = (Focal_Length[0] + Focal_Length[1]) / 2

leftMatcher = cv2.StereoSGBM_create(minDisparity=min_disp,
                               numDisparities=num_disp,
                               blockSize=5,
                               uniquenessRatio=5,
                               speckleWindowSize=5,
                               speckleRange=5,
                               disp12MaxDiff=1,
                               P1=8 * 3 * win_size ** 2,  # 8*3*win_size**2,
                               P2=32 * 3 * win_size ** 2)  # 32*3*win_size**2)

cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
cv2.createTrackbar('Min Distance', 'disp', 0, 10, nothing)
cv2.createTrackbar('Max Distance', 'disp', 1, 10, nothing)

while cap.isOpened():

    successL, imgL = cap.read()
    successR, imgR = cap2.read()

    if successL and successR:
        Left_calibrated = cv2.remap(imgL, Left_Stereo_Map_x, Left_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT,
                                    0)
        Right_calibrated = cv2.remap(imgR, Right_Stereo_Map_x, Right_Stereo_Map_y, cv2.INTER_LANCZOS4,
                                     cv2.BORDER_CONSTANT, 0)
        Left_nice = Left_calibrated[20:460, 40:550] # cropping output image
        Right_nice = Right_calibrated[20:460, 40:550]
        Left_nice = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)
        Right_nice = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2GRAY)

        displ = leftMatcher.compute(Left_nice, Right_nice)  # .astype(np.float32)/16

        left_matcher_unfiltered = displ.astype(np.float32)
        left_matcher_unfiltered = (left_matcher_unfiltered / 16.0 - min_disp) / num_disp
        # left_matcher_unfiltered = cv2.normalize(left_matcher_unfiltered, 0, 255, cv2.NORM_MINMAX)
        filteredMap = left_matcher_unfiltered.copy()
        imgXSize = len(left_matcher_unfiltered[0])
        imgYSize = len(left_matcher_unfiltered)

        MinDistance = cv2.getTrackbarPos('Min Distance', 'disp')
        MaxDistance = cv2.getTrackbarPos('Max Distance', 'disp')

        if MaxDistance == 1 or MaxDistance == 0:
            maxDispChosen = 0.98
        elif MaxDistance == 2:
            maxDispChosen = 0.77
        elif MaxDistance == 3:
            maxDispChosen = 0.49
        elif MaxDistance == 4:
            maxDispChosen = 0.36
        elif MaxDistance == 5:
            maxDispChosen = 0.30
        elif MaxDistance == 6:
            maxDispChosen = 0.22
        elif MaxDistance == 7:
            maxDispChosen = 0.18
        elif MaxDistance == 8:
            maxDispChosen = 0.16
        elif MaxDistance == 9:
            maxDispChosen = 0.15
        elif MaxDistance == 10:
            maxDispChosen = 0.128
        else:
            maxDispChosen = 0.10


        if MinDistance == 1 or MinDistance == 0:
            minDispChosen = 1
        elif MinDistance == 2:
            minDispChosen = 0.77
        elif MinDistance == 3:
            minDispChosen = 0.49
        elif MinDistance == 4:
            minDispChosen = 0.36
        elif MinDistance == 5:
            minDispChosen = 0.30
        elif MinDistance == 6:
            minDispChosen = 0.22
        elif MinDistance == 7:
            minDispChosen = 0.18
        elif MinDistance == 8:
            minDispChosen = 0.16
        elif MinDistance == 9:
            minDispChosen = 0.15
        elif MinDistance == 10:
            minDispChosen = 0.128
        else:
            minDispChosen = 0.10
        # print(disparityChosen)
        temp = Left_calibrated.copy()
        mask = createMask(filteredMap, maxDispChosen, minDispChosen, imgXSize, imgYSize)
        objDetected = findObjRect(mask)
        dispArray = mask * filteredMap
        averageDisparity = dispArray[np.nonzero(dispArray)].mean()
        estimatedDistance = DISTANCE_CONSTANT*(avgFocalLength) * (DISTANCE_BETWEEN_CAMERAS / averageDisparity)
        try:
            estimatedWidth = getActualWidth(objDetected, averageDisparity)
            estimatedHeight = getActualHeight(objDetected, averageDisparity)
        except:
            pass
        # print(estimatedWidth)
        try:
            cv2.drawContours(temp, [objDetected], 0, (0, 255, 0), 2)
            cv2.putText(temp, "chosen disp = %0.2f chosen dist. = %0.1f m" % (averageDisparity, estimatedDistance),
                        (50, imgYSize-15), cv2.FONT_ITALIC,
                        0.7, (255, 255, 255), 1)
            cv2.putText(temp, "est. size (WxH) = %0.3f m x %0.3f m" % (estimatedWidth, estimatedHeight),
                        (50, imgYSize - 45), cv2.FONT_ITALIC,
                        0.7, (255, 255, 255), 1)
        except:
            pass
        cv2.imshow('disparity map', left_matcher_unfiltered)
        # cv2.imshow('image', Left_calibrated)
        cv2.imshow('mask', mask)
        cv2.imshow('original', temp)
        dispValuesFiltered = []
        k = cv2.waitKey(20) & 0xFF
        #objDetected = findObjRect(mask)
        if k == 27 or (k & 0xFF == ord('q')):
            break
cv2.destroyAllWindows()

