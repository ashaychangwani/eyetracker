#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 21:16:36 2019

@author: ashay
"""
from tkinter import *
import PIL
from PIL import Image,ImageTk
import cv2
import tensorflow as tf
import sys
import numpy as np
sys.path.append("/Users/ashay/Desktop/eyeTracker/models/research/object_detection")
sys.path.append("/Users/ashay/Desktop/eyeTracker/models/research")





camFrameWidth=1020
menuFrameWidth=1920-camFrameWidth

cam1FrameHeight=500
cam2FrameHeight=1080-cam1FrameHeight
menuFrameHeight=1080

lower_white = np.array([180])
upper_white = np.array([255])

warningMessage="Tracking beacons: "

cams=["Camera 1","Camera 2"]
cap=None
cap2=None
execStarted=False
def initCams():
    global cap,cap2,execStarted
    try:
        width, height = 800, 600
        cap = cv2.VideoCapture(int(camVar.get()[-1])-1)
        cap2 = cv2.VideoCapture(int(camVar2.get()[-1])-1)
        cap.set(3,640)
        cap.set(4,480)
        cap2.set(3,640)
        cap2.set(4,480)
        if not execStarted:    
            show_frame()
    except Exception as e:
        print(e)
    execStarted=True

def show_frame():
    _, frameDisp = cap.read()
    hsv = cv2.cvtColor(frameDisp, cv2.COLOR_BGR2GRAY)#
    mask = cv2.inRange(hsv, lower_white, upper_white)    
    contours, heih = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    count=0
    for c in contours:
       M = cv2.moments(c)
       if M["m00"] != 0:
         cX = int(M["m10"] / M["m00"])
         cY = int(M["m01"] / M["m00"])
         count+=1
       else:
         cX, cY = 0, 0
       cv2.circle(mask, (cX, cY), 15, (120, 120, 120), 2)
    beaconCount.set("Beacons being tracked: "+str(count))
    mask = cv2.flip(mask, -1)
    img = PIL.Image.fromarray(mask)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)

    
    _, frameDisp2 = cap2.read()    
    frameDisp2 = cv2.rotate(frameDisp2,rotateCode=cv2.ROTATE_90_CLOCKWISE)
    frameDisp2 = cv2.flip(frameDisp2, 1)
    cv2image2 = cv2.cvtColor(frameDisp2, cv2.COLOR_BGR2RGBA)
    img2 = PIL.Image.fromarray(cv2image2)
    imgtk2 = ImageTk.PhotoImage(image=img2)
    lmain2.imgtk = imgtk2
    lmain2.configure(image=imgtk2)
    lmain.after(10, show_frame)

root=Tk()
#root.attributes("-fullscreen", True)
root.bind('<Escape>', lambda e: root.quit())
root.geometry("1920x1080")
frame = Frame(root, width=menuFrameWidth, height=menuFrameHeight,bg='Red')
frame2 = Frame(root, width=camFrameWidth, height=cam1FrameHeight,bg='Blue')
frame2.pack_propagate(0)
frame3 = Frame(root, width=camFrameWidth, height=cam2FrameHeight,bg='Yellow')
frame3.pack_propagate(0)
frame.pack(side=LEFT)
lmain = Label(frame2)
lslave = Label(frame2,text="Front Camera",font=("Times New Roman",22))
camVar = StringVar(frame2)
camVar.set(cams[0])
lOptions = OptionMenu(frame2,camVar,*cams)

lmain2 = Label(frame3)
lslave2 = Label(frame3,text="Iris Camera",font=("Times New Roman",22))
camVar2 = StringVar(frame3)
camVar2.set(cams[1])
lOptions2 = OptionMenu(frame3,camVar2,*cams)


frame2.pack(anchor='ne')
frame3.pack(anchor='se')
lmain.place(x=camFrameWidth/2, y=cam1FrameHeight/2, anchor="center")
lmain2.place(x=camFrameWidth/2, y=cam2FrameHeight/2, anchor="center")
lslave.place(x=20, y=cam1FrameHeight/2, anchor="w")
lOptions.place(x=20, y=cam1FrameHeight/2+30, anchor="w")
lslave2.place(x=20, y=cam2FrameHeight/2, anchor="w")
lOptions2.place(x=20, y=cam2FrameHeight/2+30, anchor="w")
labelTitle = Label(frame, text="Eye Tracker",font=("Times New Roman",30))
labelTitle.place(x=menuFrameWidth/2,y=20,anchor="center")


beaconCount=StringVar()
beaconCount.set("Program not running")

labelTrackingTitle=Label(frame,text="Tracking Beacons:")
labelTrackingTitle.place(x=450,y=840,anchor="center")
labelTrackingCount=Label(frame,textvariable=beaconCount)
labelTrackingCount.place(x=450,y=900,anchor="center")

startCamera = Button(frame,command=initCams,text="Start cameras")
startCamera.place(x=menuFrameWidth/2,y=menuFrameHeight/2,anchor="center")


#try:    
#    show_frame()
#except:
#    pass
root.mainloop()
cap=None
cap2=None



