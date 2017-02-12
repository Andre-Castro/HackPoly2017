#import sys
#sys.path.append('/usr/local/lib/python3.5/site-packages')
import numpy as np
import cv2
import turtle
from tkinter import *
import time

def nothing(x):
    pass

#Initializes the GUI for drawing screen
master = Tk()

#Initializes Camera Capture
cap = cv2.VideoCapture(0)

#Initializes windows
cv2.namedWindow('cap')
cv2.namedWindow('mask')

#A Pause for calibration
calibrationPause = 20

#For sliders to test masking (Uncomment, for recalibration)
lb = 0
lg = 0
lr = 0

ub = 0
ug = 0
ur = 0

sigma = 0.0

#Initialize ORB feature detector
orb = cv2.ORB_create(nfeatures=225, edgeThreshold=32, WTA_K=3, patchSize=32)

#Initialize variables for drawing calculations
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

#Initialize Canvas Parameters
canvas_width = 800
canvas_height = 600

#Initialize Canvas
w = Canvas(master, width=canvas_width, height=canvas_height)
w.pack()

#More variables
currXPos = 0
currYPos = 0

##turtle.title("This is TURTLE!")
##turtle.setup(width=800, height=600, startx=0, starty=0)
##
##turtle.penup()
##turtle.setx(-400 + 10)
##turtle.sety(300 + 10)
##turtle.pendown()

# create trackbars for color change
#(Uncomment for mask recalibration)
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

#Start analyzing video capture and drawing
while(cap.isOpened()):
    max_y = 600 #Sets lowest point of canvas

    #Read each frame of capture
    ret,frame = cap.read()
    #Resize Frame
    frame = cv2.resize(frame, (800, 600))

    #Color range for glove mask
    lower_pink = np.array([56,37,94], dtype = "uint8")
    upper_pink = np.array([95,66,145], dtype = "uint8")

    #Apply Gaussian Blur to reduce noise
    frame = cv2.GaussianBlur(frame, (5,5),3)
    #cv2.imshow('frameBlur',frame)

    #Convert frame to enhance mask
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)

    framesList = cv2.split(frame)

    #Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    framesList[0] = clahe.apply(framesList[0])

    frame = cv2.merge(framesList)

    frame = cv2.cvtColor(frame, cv2.COLOR_Lab2BGR)

    #Set Mask
    mask = cv2.inRange(frame,lower_pink,upper_pink)

    #Shows Mask
    #(Comment if unnecessary)
    cv2.imshow('mask', mask)
    #cv2.imshow('frame', frame)

    #(If 'escape' input, exit)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    #(Uncomment for mask calibration)
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

    #Detect keypoints in the frame
    kp = orb.detect(res,None)
    #Compute descriptors
    kp, des = orb.compute(res, kp)

    #Draw keypoints onto frame
    frame = cv2.drawKeypoints(frame, kp, None, color=(0,255,0), flags=0)

    #Find contours and draw them
    res = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    im2, contours, hierarachy = cv2.findContours(res, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    frame = cv2.drawContours(frame, contours, -1, (255,0,0), 3)

    #print(contours[0][0][0][0])
    #print(contours[0].shape)

    #If this is a dead frame, skip
    if frame is None:
        continue

    #Frame2 is None, only if target glove is not on screen
    if frame2 is None:
        #Calibration Pause
        if calibrationPause == 0:
            pass
        else:
            calibrationPause = calibrationPause - 1

        #Check contours against an area threshold
        #Add to applicableContours only if area is greater than threshold
        for currContour in range(len(contours)):
            area = cv2.contourArea(contours[currContour])
            if area > 350:
                applicableContours.append(currContour)
                gloveDetect = True

        #If glove is detected, set initial position of target
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
##            turtle.penup()
##            turtle.sety((300-max_y))
##            turtle.setx(-(-400+reqx))
##            turtle.pendown()
            currXPos = canvas_width - reqx
            currYPos = max_y

    #hull = []
    #for element in contours:
    #    hull.append(cv2.convexHull(element))
    #frame = cv2.drawContours(frame, hull, -1, (255,255,0), 3)

    #If glove has been detected in previous frame, track motion to draw
    if frame2 is not None:
        #If glove moves out of screen, program must detect it again
        gloveDetect = False

        #Check against area threshold again to check if glove is still in frame
        for currContour in range(len(contours)):
            area = cv2.contourArea(contours[currContour])
            if area > 350:
                applicableContours.append(currContour)
                gloveDetect = True

        #Find position of next position
        #Uses averages of two consecutive frames for a little more precision
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
                
                
##                turtle.sety((300-avg_ySum))
##                turtle.setx(-(-400+avg_xSum))

                #draw on the pixel on screen (given by coordinates)
                w.create_line(currXPos, currYPos, canvas_width - avg_xSum, avg_ySum, fill="#476042")
                currXPos = canvas_width - avg_xSum
                currYPos = avg_ySum
                
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
    master.update_idletasks()
    master.update()
    time.sleep(0.01)

    k1 = cv2.waitKey(1) & 0xFF
    if k1 == ord('r'):
        w.delete("all")

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
