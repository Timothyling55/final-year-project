import cv2
import numpy as np
import imutils
import argparse
import cv2

def nothing(x):
    pass

def mouse_press(event, x, y, flags, param):
    global depthClicked
    global xStart, xEnd, yStart, yEnd
    global cropStartCoordinates
    global cropSize
    global dispValuesFiltered
    if event == cv2.EVENT_LBUTTONDOWN:
        xStart, yStart = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        dispValuesFiltered = []
        xEnd, yEnd = x+1, y+1

        cropped_section = left_matcher_unfiltered[yStart:yEnd, xStart:xEnd]
        xSize = abs(xStart-xEnd)
        ySize = abs(yStart-yEnd)
        croppedCenterPointX = int (xSize / 2)
        croppedCenterPointY = int (ySize / 2)
        cv2.rectangle(left_matcher_unfiltered, (xStart, yStart), (xEnd, yEnd), (0, 255, 255), 1)
        # cv2.rectangle(left_matcher_unfiltered, (croppedCenterPointX, croppedCenterPointY), (croppedCenterPointX+2, croppedCenterPointY+2), (255, 255, 255), 1)
        cropSize = [xSize, ySize]
        # depthClicked = cropped_section[croppedCenterPointY] [croppedCenterPointX]
        # depthClicked = np.average(cropped_section)
        depthClicked = cropped_section.max() - 0.05
        print (depthClicked)


def createMask(map, depth, row, column):
    mask = np.zeros((column, row), 'uint8')
    upperBound = depth + 0.1
    lowerBound = depth - 0.1

    xTrueStart = xStart if xStart <= xEnd else xEnd
    yTrueStart = yStart if yStart <= yEnd else yEnd

    for y in range (0, column):
        for x in range(0, row):
            if (y>=yTrueStart and y<(yTrueStart+cropSize[1])) and (x>=xTrueStart and x<(xTrueStart+cropSize[0])):
                if (map [y, x] > lowerBound and map [y, x] < upperBound):
                    # print(y, x)
                    mask [y, x] = 255
            else:
                mask[y, x] = 0
                continue
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
    if (boxPoints[1][0] - boxPoints[0][0]) > (boxPoints[2][0] - boxPoints[1][0]):
        pixelWidth = boxPoints[1][0] - boxPoints[0][0] + 0.1
    else:
        pixelWidth = boxPoints[2][0] - boxPoints[1][0]
    return (distance * pixelWidth) / avgFocalLength + 0.1


def getActualHeight(boxPoints, distance):
    if (boxPoints[1][0] - boxPoints[0][0]) < (boxPoints[2][0] - boxPoints[1][0]):
        pixelHeight = boxPoints[1][1] - boxPoints[0][1]
    else:
        pixelHeight = boxPoints[2][1] - boxPoints[1][1]
    return abs((distance * pixelHeight) / avgFocalLength)


# Test4 = MonoParam, Test5 = MonoParam3
cv_file = cv2.FileStorage("./MonoParam6.xml", cv2.FILE_STORAGE_READ)

Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
Focal_Length = cv_file.getNode("Focal_Length").mat()
Q = cv_file.getNode('Q').mat()
cv_file.release()

pathL = "./images/stereoLeft/"
pathR = "./images/stereoRight/"

DISTANCE_BETWEEN_CAMERAS = 0.07 #meters
DISTANCE_CONSTANT = 3.6

depthClicked = 0
cropSize = []
itemArray = []

# itemArray = ["Block22",
#              "DirectionBoard",
#              "LightPillars",
#              "PocketD",
#              "V4Mart",
#              "V4Stairs",
#              "V4Trees",
#              "ControlFlat",
#              "ControlFlat22"]
for i in range (1, 22):
    itemArray.append(str(i) + "m")

item = "PocketD"

lmbda = 80000
sigma = 1.3
visual_multiplier = 6

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
                               speckleRange=3,
                               disp12MaxDiff=1,
                               P1=8 * 3 * win_size ** 2,  # 8*3*win_size**2,
                               P2=32 * 3 * win_size ** 2)  # 32*3*win_size**2)
imgXSize = 640
imgYSize = 480
print (imgXSize, imgYSize)

cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
cv2.createTrackbar('Image', 'disp', 0, len(itemArray)-1, nothing)

while 1:
    item = itemArray[cv2.getTrackbarPos('Image', 'disp')]
    imgL = cv2.imread(pathL + "TestL" + item + ".png")
    imgR = cv2.imread(pathR + "TestR" + item + ".png")
    Left_calibrated = cv2.remap(imgL, Left_Stereo_Map_x, Left_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    Right_calibrated = cv2.remap(imgR, Right_Stereo_Map_x, Right_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT,0)
    Left_nice = cv2.cvtColor(Left_calibrated, cv2.COLOR_BGR2GRAY)
    Right_nice = cv2.cvtColor(Right_calibrated, cv2.COLOR_BGR2GRAY)
    displ = leftMatcher.compute(Left_nice, Right_nice)  # .astype(np.float32)/16
    left_matcher_unfiltered = displ.astype(np.float32)
    left_matcher_unfiltered = (left_matcher_unfiltered / 16.0 - min_disp) / num_disp
    # left_matcher_unfiltered = cv2.normalize(left_matcher_unfiltered, 0, 255, cv2.NORM_MINMAX)
    filteredMap = left_matcher_unfiltered.copy()

    imgXSize = len(left_matcher_unfiltered[0])
    imgYSize = len(left_matcher_unfiltered)
    # print(imgXSize, imgYSize)
    # imgL = cv2.fastNlMeansDenoisingColored(imgL,None,10,10,7,21)
    # imgR = cv2.fastNlMeansDenoisingColored(imgR,None,10,10,7,21)
    # imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)


    # out = Left_calibrated.copy()
    # out[:, :, 0] = Right_calibrated[:, :, 0]
    # out[:, :, 1] = Right_calibrated[:, :, 1]
    # out[:, :, 2] = Left_calibrated[:, :, 2]
    # cv2.imshow("Output image", out)
    cv2.imshow('disparity map', left_matcher_unfiltered)
    cv2.setMouseCallback('disparity map', mouse_press)
    cv2.imshow('image', Left_calibrated)
    cv2.setMouseCallback('image', mouse_press)
    k = cv2.waitKey(20) & 0xFF
    if depthClicked != 0:
        temp = Left_calibrated.copy()
        mask = createMask(filteredMap, depthClicked, imgXSize, imgYSize)
        # draw a red rectangle
        objDetected = findObjRect(mask)
        # print (objDetected)
        dispArray = mask * filteredMap
        averageDisparity = dispArray[np.nonzero(dispArray)].mean()
        # print(averageDisparity)
        estimatedDistance = DISTANCE_CONSTANT*(avgFocalLength) * (DISTANCE_BETWEEN_CAMERAS / averageDisparity)
        # estimatedDistance = avgFocalLength / averageDisparity
        try:
            estimatedWidth = getActualWidth(objDetected, estimatedDistance)
            estimatedHeight = getActualHeight(objDetected, estimatedDistance)
            # print(avgFocalLength)
            # print(estimatedWidth)
        except:
            estimatedWidth = 0
            estimatedHeight = 0
        try:
            cv2.drawContours(temp, [objDetected], 0, (0, 255, 0), 2)
            cv2.putText(temp, "avg disp = %0.6f est. dist. = %0.3f m" % (averageDisparity, estimatedDistance),
                        (50, imgYSize-15), cv2.FONT_ITALIC,
                        0.5, (255, 0, 0), 1)
            cv2.putText(temp, "est. size (WxH) = %0.3f m x %0.3f m" % (estimatedWidth, estimatedHeight),
                        (50, imgYSize - 30), cv2.FONT_ITALIC,
                        0.5, (255, 0, 0), 1)
        except:
            pass
        cv2.imshow('mask', mask)
        cv2.imshow('original', temp)
        #objDetected = findObjRect(mask)
    if k == 27 or (k & 0xFF == ord('q')):
        break
cv2.destroyAllWindows()

