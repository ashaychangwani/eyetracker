#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 20:34:54 2020

@author: ashay
"""


import cv2
import time
cap = cv2.VideoCapture(0)


cap.set(cv2.CAP_PROP_FPS,30)
cap.set(3,1280)
cap.set(4,720)


t1=time.time()
i=0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    #cv2.findContours()
    # Our operations on the frame come here
    i+=1
    # Display the resulting frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if i==150:
        break
t2=time.time()

print('Camera 1 FPS: ',150/(t2-t1))

cap.release()

cap = cv2.VideoCapture(1)


cap.set(cv2.CAP_PROP_FPS,30)
cap.set(3,1280)
cap.set(4,720)


t1=time.time()
i=0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    i+=1
    # Display the resulting frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if i==150:
        break
t2=time.time()

print('Camera 2 FPS: ',150/(t2-t1))





# When everything done, release the capture
cap.release()