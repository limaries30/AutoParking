import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# print(cap.get(3), cap.get(4))
# ret = cap.set(3,320)
# ret = cap.set(4,240)

while(True):
    ret, frame = cap.read() 
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    if key == ord('p'):
        cv2.waitKey(-1)

    if(int(vidcap.get(1)) % 50 == 0):
    
        print('Saved frame number : ' + str(int(vidcap.get(1))))
        cv2.imwrite("Chessboards/frame%d.jpg" % count, frame)
        print('Saved frame%d.jpg' % count)
        count += 1

cap.release()
cv2.destroyAllWindows()