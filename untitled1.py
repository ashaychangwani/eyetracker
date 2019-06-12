#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 23:19:14 2019

@author: ashay
"""
import numpy as np
from matplotlib import pyplot as pltq

import sys,os
sys.path.append("/usr/local/lib/python3.7/site-packages")
import cv2

import random as rng
rng.seed(12345)

'''
src=cv2.imread('images/test1.jpg',0)
canny=cv2.Canny(src,20,100)
canny2=canny
contours, _ = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
minRect = [None]*len(contours)
minEllipse = [None]*len(contours)
for i, c in enumerate(contours):
    minRect[i] = cv2.minAreaRect(c)
    if c.shape[0] > 5:
        minEllipse[i] = cv2.fitEllipse(c)
# Draw contours + rotated rects + ellipses

drawing = np.zeros((canny.shape[0], canny.shape[1], 3), dtype=np.uint8)
for i, c in enumerate(contours):
    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    # contour
    cv2.drawContours(drawing, contours, i, color)
    # ellipse
    if c.shape[0] > 5:
        cv2.ellipse(drawing, minEllipse[i], color, 2)
    # rotated rectangle
    box = cv2.boxPoints(minRect[i])
    box = np.intp(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
    cv2.drawContours(drawing, [box], 0, color)
    cv2.drawContours(canny, [box], 0, color)
cv2.imshow('image',drawing)
cv2.imshow('og',canny)
cv2.imshow('og2',canny2)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
def findCent(box):
    x=0
    y=0
    for tup in box:
       x+=tup[0]
       y+=tup[1]
    x/=len(box)
    y/=len(box)
    return (int(x),int(y))
frame=cv2.imread('images/test3.jpg')

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray=cv2.medianBlur(gray,3)
#    //75 decent, eyelash interfering
ret,thresh = cv2.threshold(gray,59,255,cv2.THRESH_BINARY_INV)
contours, heih = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#contours, heih = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

ratMax=0
cMax=None
eMax=None
total=np.zeros(np.shape(frame),np.uint8)
for c in contours:
   if cv2.contourArea(c) > 1000:
       testTemp=np.zeros(np.shape(frame),np.uint8)
       gray=np.zeros(np.shape(frame),np.uint8)
       '''
       kpCnt=len(c)
       x=0
       y=0
       for kp in c:
           x=x+kp[0][0]
           y=y+kp[0][1]
       x1=(np.uint8(np.ceil(x/kpCnt)))
       y1=np.uint8(np.ceil(y/kpCnt))
       
       #cv2.bitwise_and()
       #cv2.drawContours(testTemp, c, -1,(0,0,255), 1)
       #rect = cv2.minAreaRect(c)
       #box = cv2.boxPoints(rect)
       #box = np.int0(box)
       #cv2.drawContours(test,[box],0,(0,0,255),2)
       #(x,y)=findCent(box)
       #dist = np.sqrt( (x - x1)**2 + (y - y1)**2 )
       '''
       
       cv2.fillPoly(testTemp,pts=[c],color=(255,255,255))
       ellipse=cv2.fitEllipse(c)
       cv2.ellipse(gray, ellipse, (255,255,255),-1)
       testTemp=cv2.bitwise_and(testTemp,gray)
       
       '''
       #cv2.circle(test,(x,y),1, (255, 0, 0), 3)
       #cv2.circle(test, (np.uint8(np.ceil(x/kpCnt)), np.uint8(np.ceil(y/kpCnt))), 1, (255, 255, 255), 3)
       #cv2.ellipse(testTemp2, ellipse, (0,255,255),-1)
       #cv2.fillPoly(testTemp2,pts=[c],color=(0,0,255))
       #total=cv2.bitwise_or(testTemp,total)
       '''
       
       total=cv2.bitwise_or(testTemp,total)
       
       rat=cv2.countNonZero(cv2.cvtColor(testTemp,cv2.COLOR_BGR2GRAY))/(3.14*ellipse[1][0]/2*ellipse[1][1]/2)
       print(rat)
       if(rat>ratMax):
           ratMax=rat
           cMax=c
           eMax=ellipse
       '''
       if dist<distMin: #ALSO ACCOUNT FOR THE NUMBER OF PIXELS FILLED VS UNFILLED
           distMin=dist
           cMain=c
       '''
       
cv2.drawContours(frame,cMax,-1,(0,255,0),1)
cv2.ellipse(frame, eMax, (0,255,255),1)
#cv2.drawContours(frame, contours, -1,(0,255,0), 1)
cv2.imshow('image',frame)
cv2.imshow('image2',total)
cv2.waitKey(0)
cv2.destroyAllWindows()

