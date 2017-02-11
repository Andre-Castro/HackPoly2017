import numpy
import cv2

#img = cv2.imread('d:\SidSpace\Laptop.png',1)

#cv2.imshow('image', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)

while (True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
