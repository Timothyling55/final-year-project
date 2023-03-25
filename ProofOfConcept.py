import numpy as np
import cv2
import imutils
import time


# returns the focal length of camera
def getFocalLength(boxPoints):
    if (boxPoints[1][0] - boxPoints[0][0]) > (boxPoints[2][0] - boxPoints[1][0]):
        pixelWidth = boxPoints[1][0] - boxPoints[0][0]
    else:
        pixelWidth = boxPoints[2][0] - boxPoints[1][0]
    return (KNOWN_DISTANCE * pixelWidth) / KNOWN_WIDTH


# returns the width of the object in cm
def getActualWidth(boxPoints, distance):
    if (boxPoints[1][0] - boxPoints[0][0]) > (boxPoints[2][0] - boxPoints[1][0]):
        pixelWidth = boxPoints[1][0] - boxPoints[0][0]
    else:
        pixelWidth = boxPoints[2][0] - boxPoints[1][0]
    return (distance * pixelWidth) / KNOWN_FOCAL_LENGTH +0.7


# gets the centerpoint of the image (0 = x axis, 1 = y axis)
def getCenter(points, axis):
    # average of all rectangle points x coordinate = center
    center = (points[0][axis] + points[1][axis] + points[2][axis] + points[3][axis]) // 4
    return center


def getDisparity(boxPointsL, boxPointsR):
    objLeftmostPointL = min(boxPointsL[0][0], boxPointsL[1][0], boxPointsL[2][0], boxPointsL[3][0])
    objLeftmostPointR = min(boxPointsR[0][0], boxPointsR[1][0], boxPointsR[2][0], boxPointsR[3][0])
    return abs(objLeftmostPointL - objLeftmostPointR)


# returns the distance of camera to object in cm
def getDistance(disparity):
    try:
        # Distance = (KNOWN_FOCAL_LENGTH / KNOWN_SIZE_OF_CAMERA) * (KNOWN_DISTANCE_BETWEEN_CAMERA / disparity) + 8.2
        Distance = (KNOWN_FOCAL_LENGTH) * (KNOWN_DISTANCE_BETWEEN_CAMERA / disparity)
        if Distance != float('inf'):
            return Distance
        else:
            return 0
    except:
        return "No object detected"


# returns the combined masks of the detected color (can be changed
# based on how many colors you wanna detect)
def findMask(img, boundaries):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower1 = np.array(boundaries[0:3])
    upper1 = np.array(boundaries[3:6])
    mask = cv2.inRange(imgHSV, lower1, upper1)
    return mask


# finds all contours in the image, finds the largest one,
# and returns the bounding rectangle
def findObjRect(img):
    cnts = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    try:
        # find the contours in the edged image and keep the largest one
        c = max(cnts, key=cv2.contourArea)
        # compute the bounding box of the region and return it
        return cv2.minAreaRect(c)
    except:
        return 0


def cameraCalc(frame, windowName):
    imgBoxed = frame.copy()
    imgMask = findMask(frame, boundaries)
    objDetected = findObjRect(imgMask)
    if (objDetected != 0 and objDetected[1][0] > 10):
        if (objDetected != 0 and objDetected[1][0] > 10):
            boxPoints = cv2.boxPoints(objDetected)
            boxPoints = np.int0(boxPoints)
        # print(getFocalLength(boxPoints))
        return imgBoxed, boxPoints
    else:
        return frame, np.array([[0, 0], [0, 0], [0, 0], [0, 0]])


def drawOnImage(frame, boxPoints, distance):
    imgBoxed = frame.copy()
    objCenterX = getCenter(boxPoints, 0)
    objCenterY = getCenter(boxPoints, 1)
    # draw a red rectangle
    cv2.drawContours(imgBoxed, [boxPoints], 0, (0, 255, 0), 2)
    # print the position of the box on the box
    width = getActualWidth(boxPoints, distance)
    cv2.putText(imgBoxed, "%0.2f cm" % width,
                (objCenterX - 50, objCenterY), cv2.FONT_ITALIC,
                0.7, (255, 0, 0), 2)
    # print distance
    cv2.putText(imgBoxed, "Distance: %0.2f cm" % distance,
                (10, 20), cv2.FONT_ITALIC,
                0.7, (255, 0, 0), 2)
    return imgBoxed, width


# cameraL = cv2.VideoCapture(0,cv2.CAP_GSTREAMER)
# cameraR = cv2.VideoCapture(2,cv2.CAP_GSTREAMER)

cameraL = cv2.VideoCapture(1)
cameraL.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cameraR = cv2.VideoCapture(0)
cameraR.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

KNOWN_WIDTH = 13
KNOWN_DISTANCE = 70
KNOWN_FOCAL_LENGTH = 1493  # estimated after about 1000 loops at 40 cm distance

# in cm
# KNOWN_SIZE_OF_CAMERA = 0.8
KNOWN_DISTANCE_BETWEEN_CAMERA = 7

# HSV boundaries, format [lowerH, lowerS, lowerV, upperH, upperS, upperV]
boundaries = [0, 166, 184, 179, 255, 255]  # Red: 0, 0, 0, 16, 255, 255,
count = 0
totalFocalLength = 0
totalDistance = 0
totalWidth = 0
num = 0

cv_file = cv2.FileStorage("./MonoParam7.xml", cv2.FILE_STORAGE_READ)

Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()

# MAIN
while True:
    # Setup both Cameras
    retL, frameL = cameraL.read()
    retR, frameR = cameraR.read()

    if not retL:
        print("failed to grab frame (Left Camera)")
        break

    if not retR:
        print("failed to grab frame (Right Camera)")
        break

    frameL = cv2.remap(frameL, Left_Stereo_Map_x, Left_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    frameR = cv2.remap(frameR, Right_Stereo_Map_x, Right_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    # Calculate position of color relative to center of camera
    POVL, boxPointsL = cameraCalc(frameL, "Left POV")
    POVR, boxPointsR = cameraCalc(frameR, "Right POV")
    disparity = getDisparity(boxPointsL, boxPointsR)
    distance = getDistance(disparity)
    print(distance)
    ShownL, widthL = drawOnImage(frameL, boxPointsL, distance)
    ShownR, widthR = drawOnImage(frameR, boxPointsR, distance)
    cv2.imshow("Camera View", np.hstack([ShownL, ShownR]))

    k = cv2.waitKey(1)

    count += 1
    countStart = 100
    countMax = 200-countStart
    recordingDistance = 200
    if count >= countStart:
        # totalFocalLength += getFocalLength(boxPointsR) + getFocalLength(boxPointsL)
        if True: #(distance > recordingDistance-10) and (distance<recordingDistance+10):
            totalDistance += distance
            totalWidth += widthL + widthR
        else:
            count -= 1

    if count >= countMax+countStart:
        # avgFocalLength = (totalFocalLength / 2) / 1000
        avgDistance = totalDistance / countMax
        avgWidth = (totalWidth / 2) / countMax
        print("actual distance = ",recordingDistance,"m")
        # print ("estimated focal length = ", avgFocalLength)
        print ("estimated width = ", avgWidth)
        print ("estimated distance = ", avgDistance/100, "m")
        break

    if (k & 0xFF == ord('q')):
        break

cameraL.release()
cameraR.release()
cv2.destroyAllWindows()

print("Program closed")


