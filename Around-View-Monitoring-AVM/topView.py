#get topview
#input: image, setDimensions(four points with written by clockwise order)
#ouput: topview image

"""
This code is used to obtain the bird's view of
a robot with two cameras installed(front and back cameras)
"""

import cv2
from Camera.Undistortion import UndistortFisheye
from Camera.PerspectiveTransformation import BirdView

leftStream = cv2.VideoCapture(0)
leftCamera = UndistortFisheye("left_Camera")
leftBird = BirdView()

# need to designate 4 points
leftBird.setDimensions((186, 195), (484, 207), (588, 402), (97, 363)) #fisheyecalibration을 진행한 후인 Undistorted_Front_View.jpg의 4 개의 꼭짓점


while True:
    
    _, leftFrame = leftStream.read()
    leftView = leftCamera.undistort(leftFrame)
    topDown_left = leftBird.transfrom(leftView)

    cv2.imshow("Left Bird's Eye View", topDown_left)

    key = cv2.waitKey(1)
        if key == ord('q'):
            break

        if key == ord('p'): #pause
            cv2.waitKey(-1)

        if key == ord("s"):
            cv2.imwrite("Capture.jpg", topDown_left)
    
        # if key == ord("r"):
        #     leftCamera.reset()

cap.release()
cv2.destroyAllWindows()