#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 23:21:29 2019

@author: ashay
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)
movement=[(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1)]
thresh=np.zeros((600,550))


def whiteKar(i,j):
    thresh[(i,j)]=0

def calcBound(i,j,moveCount):
    test=0
    cv2.circle(thresh, (i,j), 0, (0), 2)
    for test in range(0, len(movement)):
        if(thresh[tuple(map(lambda x, y: x + y, (i,j), movement[(moveCount-test)%len(movement)]))]==1):
            calcBound(tuple(map(lambda x, y: x + y, (i,j), movement[(moveCount-test)%len(movement)])),(moveCount-test)%len(movement))
            break
        if(test==len(movement)):
            break
            

while(1):    
    
    ret, frame = cap.read()
    
    frame=frame[240:840,250:800]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    gray=cv2.medianBlur(gray,3)
#    //75 decent, eyelash interfering
    ret,thresh = cv2.threshold(gray,75,255,cv2.THRESH_BINARY_INV)
    
    contours, heih = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    count=0
#    for c in contours:
#       M = cv2.moments(c)
#       if M["m00"] != 0:
#         cX = int(M["m10"] / M["m00"])
#         cY = int(M["m01"] / M["m00"])
#         count+=1
#       else:
#         cX, cY = 0, 0
#       cv2.circle(mask, (cX, cY), 15, (120, 120, 120), 2)
    
    cv2.drawContours(frame, contours, -1, (0,255,0), 1)
    try:
        maxC=contours[0]
    except:
        pass
    for c in contours:
        try:
            if(cv2.contourArea(c)/cv2.arcLength(c,True)>cv2.contourArea(maxC)/cv2.arcLength(maxC,True) and cv2.contourArea(c)/cv2.arcLength(c,True)>15):
                maxC=c
        except:
            pass
        
    c=maxC
    try:
        print(cv2.contourArea(c),cv2.contourArea(c)/cv2.arcLength(c,True))
    except:
        pass
    x,y,w,h = cv2.boundingRect(c)
    # draw the book contour (in green)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    
    cv2.imshow("TEST2",thresh)
    cv2.imshow("TEST",frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()