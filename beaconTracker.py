#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 22:11:04 2019

@author: ashay
"""
import cv2
import numpy as np
cap = cv2.VideoCapture(1)

pts = []
while (1):

    # Take each frame
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    hsv=cv2.adaptiveThreshold(hsv,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,2)
#
    lower_white = np.array([180])
    upper_white = np.array([255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(mask)
    
    contours, heih = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    count=0
    for c in contours:
       # calculate moments for each contour
       M = cv2.moments(c)
     
       # calculate x,y coordinate of center
       if M["m00"] != 0:
         cX = int(M["m10"] / M["m00"])
         cY = int(M["m01"] / M["m00"])
         count+=1
       else:
         cX, cY = 0, 0
       cv2.circle(mask, (cX, cY), 15, (120, 120, 120), 2)

    if(count<3):
        print("Unable to track all")
    else:
        print("tracking correct;y")
#    cv2.circle(hsv, maxLoc, 15, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Track Laser', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()