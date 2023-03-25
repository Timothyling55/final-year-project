import numpy as np
import cv2

# Check for left and right camera IDs
# These values can change depending on the system
pathL = "./images/stereoLeft/"
pathR = "./images/stereoRight/"
itemArray = []

for i in range (1, 22):
    itemArray.append(str(i) + "m.png")
# Reading the mapping values for stereo image rectification
cv_file = cv2.FileStorage("./MonoParam6.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()

mouseX = 0
mouseY = 0

kernel= np.ones((3,3),np.uint8)

def nothing(x):
    pass

KNOWN_DISTANCE_BETWEEN_CAMERA = 7
KNOWN_FOCAL_LENGTH = 1493
win_size = 5

cv2.namedWindow('disp', cv2.WINDOW_NORMAL)

cv2.createTrackbar('numDisparities', 'disp', 4, 17, nothing)
cv2.createTrackbar('blockSize', 'disp', 5, 50, nothing)
cv2.createTrackbar('preFilterType', 'disp', 0, 1, nothing)
cv2.createTrackbar('preFilterSize', 'disp', 3, 100, nothing)
cv2.createTrackbar('preFilterCap', 'disp', 5, 62, nothing)
cv2.createTrackbar('textureThreshold', 'disp', 10, 100, nothing)
cv2.createTrackbar('uniquenessRatio', 'disp', 7, 100, nothing)
cv2.createTrackbar('speckleRange', 'disp', 2, 100, nothing)
cv2.createTrackbar('speckleWindowSize', 'disp', 3, 100, nothing)
cv2.createTrackbar('disp12MaxDiff', 'disp', 5, 25, nothing)
cv2.createTrackbar('minDisparity', 'disp', 5, 25, nothing)
cv2.createTrackbar('Image', 'disp', 0, 10, nothing)

# Creating an object of StereoBM algorithm
stereo = cv2.StereoSGBM_create(P1=8 * 3 * win_size ** 2,
                               P2=32 * 3 * win_size ** 2)  # 32*3*win_size**2)


while True:
    imgL = cv2.imread(pathL + "TestL" + itemArray[cv2.getTrackbarPos('Image', 'disp')])
    imgR = cv2.imread(pathR + "TestR" + itemArray[cv2.getTrackbarPos('Image', 'disp')])
    # Proceed only if the frames have been captured
    if True:
        # imgL = cv2.fastNlMeansDenoisingColored(imgL, None, 10, 10, 7, 21)
        # imgR = cv2.fastNlMeansDenoisingColored(imgR, None, 10, 10, 7, 21)
        imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)


        # Applying stereo image rectification on the left image
        Left_nice = cv2.remap(imgL_gray,
                              Left_Stereo_Map_x,
                              Left_Stereo_Map_y,
                              cv2.INTER_LANCZOS4,
                              cv2.BORDER_CONSTANT,
                              0)

        # Applying stereo image rectification on the right image
        Right_nice = cv2.remap(imgR_gray,
                               Right_Stereo_Map_x,
                               Right_Stereo_Map_y,
                               cv2.INTER_LANCZOS4,
                               cv2.BORDER_CONSTANT,
                               0)

        # Updating the parameters based on the trackbar positions
        numDisparities = cv2.getTrackbarPos('numDisparities', 'disp') * 16
        blockSize = cv2.getTrackbarPos('blockSize', 'disp')
        preFilterType = cv2.getTrackbarPos('preFilterType', 'disp')
        preFilterSize = cv2.getTrackbarPos('preFilterSize', 'disp')
        preFilterCap = cv2.getTrackbarPos('preFilterCap', 'disp')
        textureThreshold = cv2.getTrackbarPos('textureThreshold', 'disp')
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'disp')
        speckleRange = cv2.getTrackbarPos('speckleRange', 'disp')
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'disp')
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'disp')
        minDisparity = cv2.getTrackbarPos('minDisparity', 'disp')


        # Setting the updated parameters before computing disparity map
        stereo.setNumDisparities(numDisparities)
        stereo.setBlockSize(blockSize)
        stereo.setPreFilterCap(preFilterCap)
        stereo.setUniquenessRatio(uniquenessRatio)
        stereo.setSpeckleRange(speckleRange)
        stereo.setSpeckleWindowSize(speckleWindowSize)
        stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setMinDisparity(minDisparity)


        # Calculating disparity using the StereoBM algorithm
        disparity = stereo.compute(Left_nice, Right_nice)
        # NOTE: Code returns a 16bit signed single channel image,
        # CV_16S containing a disparity map scaled by 16. Hence it
        # is essential to convert it to CV_32F and scale it down 16 times.

        # Converting to float32
        # closing = cv2.morphologyEx(disparity, cv2.MORPH_CLOSE, kernel)  # Apply an morphological filter for closing little "black" holes in the picture(Remove noise)
        #
        # # Colors map
        # dispc = (closing - closing.min()) * 255
        # disparity = dispc.astype(np.uint8)  # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function

        # Scaling down the disparity values and normalizing them
        disparity = (disparity / 16.0 - minDisparity) / numDisparities
        # print(disparity)
        # Displaying the disparity map
        cv2.imshow("Disparity", disparity)
        cv2.imshow('View', np.hstack([Left_nice, Right_nice]))
        # Close window using esc key
        if cv2.waitKey(1) == 27:
            break

    else:
        CamL = cv2.VideoCapture(CamL_id)
        CamR = cv2.VideoCapture(CamR_id)