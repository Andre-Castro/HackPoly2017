#import sys
#sys.path.append('/usr/local/lib/python3.5/site-packages')
import numpy as np
import cv2
import turtle

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

sigma = 0.0

orb = cv2.ORB_create(nfeatures=225, edgeThreshold=32, WTA_K=3, patchSize=32)

applicableContours = []

initial_list = []
list_kp = []
list_kp2 = []

frame2 = None
gloveDetect = False
avg_count = 0
avg_ySum = 0
avg_xSum = 0

reqx = 800

turtle.title("This is TURTLE!")
turtle.setup(width=800, height=600, startx=0, starty=0)

turtle.penup()
turtle.setx(-400 + 10)
turtle.sety(300 + 10)
turtle.pendown()

# create trackbars for color change
cv2.createTrackbar('loR','cap',0,255,nothing)
cv2.createTrackbar('loG','cap',0,255,nothing)
cv2.createTrackbar('loB','cap',0,255,nothing)

cv2.createTrackbar('upR','cap',0,255,nothing)
cv2.createTrackbar('upG','cap',0,255,nothing)
cv2.createTrackbar('upB','cap',0,255,nothing)

#cv2.createTrackbar('sigma','cap',0.1,5.0,nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'cap',0,1,nothing)

while(cap.isOpened()):
    max_y = 600
    
    ret,frame = cap.read()
    frame = cv2.resize(frame, (800, 600))
    
    lower_pink = np.array([56,35,121], dtype = "uint8")
    upper_pink = np.array([125,93,145], dtype = "uint8")

    frame = cv2.GaussianBlur(frame, (5,5),3)
    #cv2.imshow('frameBlur',frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)

    framesList = cv2.split(frame)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    framesList[0] = clahe.apply(framesList[0])

    frame = cv2.merge(framesList)

    frame = cv2.cvtColor(frame, cv2.COLOR_Lab2BGR)

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
    sigma = cv2.getTrackbarPos('sigma','cap')
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

    res = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    im2, contours, hierarachy = cv2.findContours(res, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    frame = cv2.drawContours(frame, contours, -1, (255,0,0), 3)

    #print(contours[0][0][0][0])
    #print(contours[0].shape)



    if frame is None:
        continue

    if frame2 is None:
        for currContour in range(len(contours)):
            area = cv2.contourArea(contours[currContour])
            if area > 350:
                applicableContours.append(currContour)
                gloveDetect = True
        if gloveDetect is True:            
            for specContour in applicableContours:
                extTop = tuple(contours[specContour][contours[specContour][:, :, 1].argmin()][0])
                if extTop[1] < max_y:
                    max_y = extTop[1];
                    reqx = extTop[0];
                               
##                for pointsList in contours[specContour]:
##                    for point in pointsList:
##                        if point[1] < max_y:
##                            max_y = point[1]
##                            reqx = point[0]
            #print(max_y)
            #print(reqx)
            turtle.penup()
            turtle.sety((300-max_y))
            turtle.setx(-(-400+reqx))
            turtle.pendown()

    #hull = []
    #for element in contours:
    #    hull.append(cv2.convexHull(element))
    #frame = cv2.drawContours(frame, hull, -1, (255,255,0), 3)

    if frame2 is not None:
        gloveDetect = False
        for currContour in range(len(contours)):
            area = cv2.contourArea(contours[currContour])
            if area > 350:
                applicableContours.append(currContour)
                gloveDetect = True
        if gloveDetect is True:
            for specContour in applicableContours:
                extTop = tuple(contours[specContour][contours[specContour][:, :, 1].argmin()][0])
                if extTop[1] < max_y:
                    max_y = extTop[1]
                    reqx = extTop[0]
            if avg_count == 2:
                avg_count = 0
                avg_ySum = avg_ySum / 2
                avg_xSum = avg_xSum / 2
                
                
                turtle.sety((300-avg_ySum))
                turtle.setx(-(-400+avg_xSum))

            else:
                if avg_count == 0:
                    avg_count = avg_count + 1
                    avg_ySum = avg_ySum + max_y
                    avg_xSum = avg_xSum + reqx

                else:
                    tempYCount = avg_ySum / avg_count
                    tempXCount = avg_xSum / avg_count

##                    if abs(max_y - tempYCount) >= 100 or abs(reqx - tempXCount) >= 100:
##                        avg_count = 0
##                        avg_ySum = max_y
##                        avg_xSum = reqx
##                    else:
                    avg_count = avg_count + 1
                    avg_ySum = avg_ySum + max_y
                    avg_xSum = avg_xSum + reqx

    frame = cv2.flip(frame, 1)
    cv2.imshow('res', frame)

    if gloveDetect is True:
        #print("Hello")
        frame2 = frame
    else:
        frame2 = None
    applicableContours = []

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
cap.release()
##cv2.destroyAllWindows()
