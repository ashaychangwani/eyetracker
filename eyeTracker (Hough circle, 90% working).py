#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 23:21:29 2019

@author: ashay
"""

#X-60 TO 410
import cv2
import numpy as np
cap = cv2.VideoCapture(1)
test=None
while (1):
    ret, frame = cap.read()
    frame=frame[:,190:868]
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img,29)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    lower_white = np.array([0])
    upper_white = np.array([16])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    cv2.imshow('Input2', img)
#        
#    contours1, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#    contours2, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#    
#    c = max(contours1, key = cv2.contourArea)
#    M = cv2.moments(c)
#    if M["m00"] != 0:
#     cX = int(M["m10"] / M["m00"])
#     cY = int(M["m01"] / M["m00"])
#    else:
#     cX, cY = 0, 0
#    cv2.circle(frame, (cX, cY), 15, (120, 120, 120), 2)
#    
#    cv2.imshow('Input3', frame)
#
#    	# show the output image
#    
    try:
        circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,30,
                            param1=37,param2=37,minRadius=80,maxRadius=0)
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
    
    
    except:
        print("")
    cv2.imshow('Input2', frame)
    cv2.imshow('Input3', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

        
cap.release()
cv2.destroyAllWindows()








import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
cap = cv2.VideoCapture(1)
test=None
while (1):
    ret, frame = cap.read()
    frame=frame[:,190:868]
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = canny(img, sigma=2.0,
              low_threshold=0.55, high_threshold=0.8)
    
#    cv2.imshow('test',edges)
    
    
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    result = hough_ellipse(edges, accuracy=20, threshold=250,
                       min_size=100, max_size=120)
    result.sort(order='accumulator')
    
    # Estimated parameters for the ellipse
    best = list(result[-1])
    yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    orientation = best[5]
    
    # Draw the ellipse on the original image
    cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
    frame[cy, cx] = (0, 0, 255)
    # Draw the edge (white) and the resulting ellipse (red)
    edges = color.gray2rgb(img_as_ubyte(edges))
    cv2.imshow('Input2', edges)
    edges[cy, cx] = (250, 0, 0)
    
    fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4),
                                    sharex=True, sharey=True)
    ax1.set_title('Original picture')
    ax1.imshow(frame)

    ax2.set_title('Edge (white) and result (red)')
    ax2.imshow(edges)
    
    plt.show()
cap.release()
cv2.destroyAllWindows()