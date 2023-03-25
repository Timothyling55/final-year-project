import cv2
import numpy as np

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap2 = cv2.VideoCapture(0)
cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

a = 9
b = 6
chessboard_size = (a, b)
objp = np.zeros((b * a, 3), np.float32)
objp[:, :2] = np.mgrid[0:a, 0:b].T.reshape(-1, 2)

num = 0
img_ptsL = []
img_ptsR = []
obj_pts = []
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
chessboardYCoordinatesDifference = []
cv_file = cv2.FileStorage("./MonoParam7.xml", cv2.FILE_STORAGE_READ)

Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()

while cap.isOpened():

    succes1, imgL = cap.read()
    succes2, imgR = cap2.read()

    Left_nice = cv2.remap(imgL, Left_Stereo_Map_x, Left_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    Right_nice = cv2.remap(imgR, Right_Stereo_Map_x, Right_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    # Left_nice = imgL
    # Right_nice  =imgR
    k = cv2.waitKey(1)
    # num += 1
    if k == 27:
        break
    elif k == ord('s'):  # wait for 's' key to save and exit
        imgL_gray = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)
        imgR_gray = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2GRAY)
        retL, cornersL = cv2.findChessboardCorners(Left_nice, (9, 6), None)
        retR, cornersR = cv2.findChessboardCorners(Right_nice, (9, 6), None)
        # print(cornersR)
        # print(cornersR[0][0][0])
        if retR and retL:
            obj_pts.append(objp)
            # takes original imahe, location of corners, looks for best corner location within range. Iterative, termination criteria needed
            cv2.cornerSubPix(imgL_gray, cornersL, (11, 11), (-1, -1), criteria)
            cv2.cornerSubPix(imgR_gray, cornersR, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(Left_nice, (9, 6), cornersL, retL)
            cv2.drawChessboardCorners(Right_nice, (9, 6), cornersR, retR)
            for y in range (0, len(cornersR)):
                chessboardYCoordinatesDifference.append(abs(cornersR[y][0][1] - cornersL[y][0][1]))
            totalChessboardError = sum(chessboardYCoordinatesDifference)
            averageChessboardError = totalChessboardError / len(cornersR)
            print ("Total Errors: ", str(totalChessboardError))
            print ("Average error of each point: ", str(averageChessboardError))
            out = Right_nice.copy()
            out[:, :, 0] = Right_nice[:, :, 0]
            out[:, :, 1] = Right_nice[:, :, 1]
            out[:, :, 2] = Left_nice[:, :, 2]
            cv2.imshow('cornersL', Left_nice)
            cv2.imshow('cornersR', Right_nice)
            cv2.imshow("Output image", out)
            cv2.waitKey(0)

    elif (k & 0xFF == ord('q')):
        break

    cv2.imshow('stereo view', np.hstack([Left_nice, Right_nice]))

cap.release()
cap2.release()
cv2.destroyAllWindows()

print("Program closed. Total pictures taken: " + str(num))
