# https://learnopencv.com/camera-calibration-using-opencv/
import cv2
import numpy as np
import glob
from tqdm import tqdm

# Define size of chessboard target.
a = 9
b = 6
chessboard_size = (a, b)
objp = np.zeros((b * a, 3), np.float32)
objp[:, :2] = np.mgrid[0:a, 0:b].T.reshape(-1, 2)

# Define arrays to save detected points
obj_points = []  # 3D points in real world space
img_points = []  # 3D points in image plane
# Prepare grid and points to display
objp = np.zeros((np.prod(chessboard_size), 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

pathL = "./images/stereoLeft/"
pathR = "./images/stereoRight/"

# Termination criteria for refining the detected corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

img_ptsL = []
img_ptsR = []
obj_pts = []

for i in tqdm(range(0, 5)): # change based on number of pictures taken for the calibration
    imgL = cv2.imread(pathL + "img%d.png" % i)
    imgR = cv2.imread(pathR + "img%d.png" % i)
    imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    outputL = imgL.copy()
    outputR = imgR.copy()

    # Looks for Chessboard corners. Parameters: Image, Pattern Size of corners
    retL, cornersL = cv2.findChessboardCorners(outputL, (9, 6), None)
    retR, cornersR = cv2.findChessboardCorners(outputR, (9, 6), None)

    if retR and retL:
        obj_pts.append(objp)
        # takes original imahe, location of corners, looks for best corner location within range. Iterative, termination criteria needed
        cv2.cornerSubPix(imgL_gray, cornersL, (11, 11), (-1, -1), criteria)
        cv2.cornerSubPix(imgR_gray, cornersR, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(outputL, (9, 6), cornersL, retL)
        cv2.drawChessboardCorners(outputR, (9, 6), cornersR, retR)
        cv2.imshow('cornersL',outputL)
        cv2.imshow('cornersR',outputR)
        cv2.waitKey(0)

        img_ptsL.append(cornersL)
        img_ptsR.append(cornersR)

print("Calculating left camera parameters ... ")
# Calibrating left camera
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_pts, img_ptsL, imgL_gray.shape[::-1], None, None)
hL, wL = imgL_gray.shape[:2]
new_mtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))

print("Calculating right camera parameters ... ")
# Calibrating right camera
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(obj_pts, img_ptsR, imgR_gray.shape[::-1], None, None)
hR, wR = imgR_gray.shape[:2]
new_mtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, (wR, hR), 1, (wR, hR))

print("Stereo calibration .....")
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
# Hence intrinsic parameters are the same

criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(obj_pts,
                                                                                    img_ptsL,
                                                                                    img_ptsR,
                                                                                    new_mtxL,
                                                                                    distL,
                                                                                    new_mtxR,
                                                                                    distR,
                                                                                    imgL_gray.shape[::-1],
                                                                                    criteria_stereo,
                                                                                    flags)

# Once we know the transformation between the two cameras we can perform stereo rectification
# StereoRectify function
rectify_scale = 1  # if 0 image croped, if 1 image not croped
rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv2.stereoRectify(new_mtxL, distL, new_mtxR, distR,
                                                                          imgL_gray.shape[::-1], Rot, Trns,
                                                                          rectify_scale, (0, 0))

print (Q)
focalLengthL = (mtxL[0, 0] +mtxL[1, 1])/2
focalLengthR = (mtxR[0, 0] +mtxR[1, 1])/2
focalLength = (focalLengthL, focalLengthR)

# Use the rotation matrixes for stereo rectification and camera intrinsics for undistorting the image
# Compute the rectification map (mapping between the original image pixels and
# their transformed values after applying rectification and undistortion) for left and right camera frames
Left_Stereo_Map = cv2.initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l,
                                              imgL_gray.shape[::-1], cv2.CV_16SC2)
Right_Stereo_Map = cv2.initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r,
                                               imgR_gray.shape[::-1], cv2.CV_16SC2)

print("Saving parameters ......")
cv_file = cv2.FileStorage("./MonoParam7.xml", cv2.FILE_STORAGE_WRITE)
cv_file.write("Left_Stereo_Map_x", Left_Stereo_Map[0])
cv_file.write("Left_Stereo_Map_y", Left_Stereo_Map[1])
cv_file.write("Right_Stereo_Map_x", Right_Stereo_Map[0])
cv_file.write("Right_Stereo_Map_y", Right_Stereo_Map[1])
cv_file.write("Focal_Length", focalLength)
cv_file.write('Q', Q)
cv_file.write('T', Trns)
print(Trns)
print(focalLength)
print("Saved Parameters!")
cv_file.release()


# Show Rectified image
cv2.imshow("Images before rectification", np.hstack([imgL, imgR]))

Left_nice = cv2.remap(imgL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
Right_nice = cv2.remap(imgR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

cv2.imshow("Images after rectification", np.hstack([Left_nice, Right_nice]))
cv2.waitKey(0)

out = Right_nice.copy()
out[:, :, 0] = Right_nice[:, :, 0]
out[:, :, 1] = Right_nice[:, :, 1]
out[:, :, 2] = Left_nice[:, :, 2]

cv2.imshow("Output image", out)
cv2.waitKey(0)