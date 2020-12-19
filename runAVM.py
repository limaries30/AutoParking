#run AVM (image/video/stream)
#input: frontFrame, backFrame
#output: AVM image/video/stream

import cv2
import numpy as np
from Camera.AroundViewMonitoring import avm
import datetime


## frontStream = cv2.VideoCapture(0) streaming
## backStream = cv2.VideoCapture(1)

leftStream = cv2.VideoCapture("./Around-View-Monitoring-AVM/dataset/front_camera.avi") #Video
rightStream = cv2.VideoCapture("./Around-View-Monitoring-AVM/dataset/back_camera.avi")
# print(leftStream.get(3), leftStream.get(4))
# print(rightStream.get(3), rightStream.get(4))

avm = avm()

# frontFrame = cv2.imread("./Around-View-Monitoring-AVM/Front_View.jpg") #image
# backFrame = cv2.imread("./Around-View-Monitoring-AVM/Rear_View.jpg")
startTime = datetime.datetime.now()
while True:
    isGrabbed, leftFrame = leftStream.read() #video
    isGrabbed2, rightFrame = rightStream.read()
    if not isGrabbed or not isGrabbed2:
        break

    # frontFrame = cv2.imread("./Around-View-Monitoring-AVM/dataset/Front_View.jpg") #image
    # backFrame = cv2.imread("./Around-View-Monitoring-AVM/dataset/Rear_View.jpg")

    birdView = avm.runAVM(leftFrame, rightFrame)
    dst = cv2.resize(birdView, dsize=(240, 320), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("dst", dst)
    print(dst.shape)
    # cv2.imshow("leftFrame", leftFrame)
    # cv2.imshow("rightFrame", rightFrame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('p'):
        cv2.waitKey(-1) #wait until any key is pressed

endTime = datetime.datetime.now()
time = (endTime - startTime).total_seconds()
print ("Approximate elapsed time is %(fps)d: "%{"fps": time})

leftStream.release()
rightStream.release() #video
cv2.destroyAllWindows()


# img = cv2.imread('./Around-View-Monitoring-AVM/dataset/Front_View.jpg')
# cv2.imshow("what", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()