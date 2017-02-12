import numpy
import cv2
from matplotlib import pyplot as plt

#img = cv2.imread('d:\SidSpace\Laptop.png',1)

#cv2.imshow('image', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

cap = cv2.VideoCapture('test.mp4')

orb = cv2.ORB_create()

while (cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (800,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    kp = orb.detect(frame, None)

    kp, des = orb.compute(frame, kp)

    print(type(kp))
    print(kp)
    print(type(des))
    print(des)

    frame2 = cv2.drawKeypoints(frame, kp, None, color=(0,255,0), flags=0)

    cv2.imshow('frame', frame2)
    
    #plt.imshow(frame2),plt.show()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
