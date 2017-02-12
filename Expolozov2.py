import numpy
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('one(fou4)-c.png', 0)
img2 = cv2.imread('one(thr33)-c.png', 0)

orb = cv2.ORB_create()

kp1 = orb.detect(img1,None)
kp1, des1 = orb.compute(img1, kp1)

kp2 = orb.detect(img2, None)
kp2, des2 = orb.compute(img2, kp2)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=0)
#img4 = cv2.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
#plt.imshow(img2),plt.show()
img3 = cv2.resize(img3, (800,600))
cv2.imshow('img3',img3)
print(type(img3))
