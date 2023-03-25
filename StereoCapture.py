import cv2
import numpy as np

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap2 = cv2.VideoCapture(0)
cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))


# cap.set(cv2.CAP_PROP_AUTO_WB, 0) # Disable automatic white balance
# cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4200) # Set manual white balance temperature to 4200K
# cap.set(cv2.CAP_PROP_EXPOSURE, 0.25)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
#
# cap2.set(cv2.CAP_PROP_AUTO_WB, 0) # Disable automatic white balance
# cap2.set(cv2.CAP_PROP_WB_TEMPERATURE, 4200) # Set manual white balance temperature to 4200K
# cap2.set(cv2.CAP_PROP_EXPOSURE, 0.25)
# cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
# cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

num = 0
item = "21m"
while cap.isOpened():

    succes1, imgL = cap.read()
    succes2, imgR = cap2.read()

    k = cv2.waitKey(1)

    if k == 27:
        break

    elif (k & 0xFF == ord('q')):
        break
    elif k == ord('s'):  # wait for 's' key to save and exit
        cv2.imwrite('./images/stereoLeft/img' + str(num) + '.png', imgL)
        cv2.imwrite('./images/stereoRight/img' + str(num) + '.png', imgR)
        print("Image no. " + str(num+1) + " saved!")
        num += 1

    # elif k == ord('s'):  # wait for 's' key to save and exit
    #     cv2.imwrite('./images/stereoLeft/TestPoorLighting.png', imgL)
    #     cv2.imwrite('./images/stereoRight/TestPoorLighting.png', imgR)
    #     print("Test image saved")
    #     num += 1


    # elif num >= 350:
    #     # elif k == ord('s'):
    #     cv2.imwrite('./images/stereoLeft/TestL'+item+'.png', imgL)
    #     cv2.imwrite('./images/stereoRight/TestR'+item+'.png', imgR)
    #     print("Test Image Saved")
    #     break
    # num += 1

    cv2.imshow('stereo view', np.hstack([imgL, imgR]))
    cv2.imshow('bw view', np.hstack([cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY), cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)]))
    # cv2.imshow('stereo view L', imgL)
    # cv2.imshow('stereo view R', imgR)


cap.release()
cap2.release()
cv2.destroyAllWindows()

