#import sys
#sys.path.append('/usr/local/lib/python3.5/site-packages')
import numpy as np
import cv2

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('Andre2999.mp4')
cv2.namedWindow('cap')
cv2.namedWindow('mask')

lb = 0
lg = 0
lr = 0

ub = 0
ug = 0
ur = 0

orb = cv2.ORB_create(nfeatures=225, edgeThreshold=34, WTA_K=3, patchSize=34)

# create trackbars for color change
cv2.createTrackbar('loR','cap',0,255,nothing)
cv2.createTrackbar('loG','cap',0,255,nothing)
cv2.createTrackbar('loB','cap',0,255,nothing)

cv2.createTrackbar('upR','cap',0,255,nothing)
cv2.createTrackbar('upG','cap',0,255,nothing)
cv2.createTrackbar('upB','cap',0,255,nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'cap',0,1,nothing)

while(cap.isOpened()):
    ret,frame = cap.read()
    frame = cv2.resize(frame, (800, 600))
    
    lower_pink = np.array([54.5,54.5,100], dtype = "uint8")
    upper_pink = np.array([94,75,255], dtype = "uint8")

    mask = cv2.inRange(frame,lower_pink,upper_pink)
    
    cv2.imshow('mask', mask)
    #cv2.imshow('frame', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    

    # get current positions of four trackbars
    lr = cv2.getTrackbarPos('loR','cap')
    lg = cv2.getTrackbarPos('loG','cap')
    lb = cv2.getTrackbarPos('loB','cap')
    ur = cv2.getTrackbarPos('upR','cap')
    ug = cv2.getTrackbarPos('upG','cap')
    ub = cv2.getTrackbarPos('upB','cap')
    s = 1
    
    if s == 0:
        #print(frame.shape)
        frame[:] = 0
    #frame[:] = [ub,ug,ur]

    #cv2.imshow('frame', frame)
    
    res = cv2.bitwise_and(frame,frame,mask= mask)

    kp = orb.detect(res,None)
    kp, des = orb.compute(res, kp)

    frame = cv2.drawKeypoints(frame, kp, None, color=(0,255,0), flags=0)
    
    cv2.imshow('res', frame)

cv2.destroyAllWindows()

##cap = cv2.VideoCapture('VID_20170211_163108.mp4')
####cap = cv2.imread('IMG_20170211_164029.jpg')
####print(cap[int(cap.shape[0]/2), int(cap.shape[1]/2),:])
##
##while(cap.isOpened()):
##    ret, frame = cap.read()
##
##    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
##
##    pink = np.array([50,22,169], dtype = "uint8")
##    punk = np.array([70,42,250], dtype = "uint8")
##
##    mask = cv2.inRange(frame,pink,punk)
##
##    res = cv2.bitwise_and(frame,frame,mask= mask)
##
##    cv2.imshow('frame',frame)
##    cv2.imshow('mask', mask)
##    cv2.imshow('res',res)
##    if cv2.waitKey(1) & 0xFF == ord('q'):
##        break
##
##cap.release()
##cv2.destroyAllWindows()
