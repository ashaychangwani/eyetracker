#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 21:16:36 2019

@author: ashay
"""
'''
import PIL
from PIL import Image,ImageTk
import cv2
from tkinter import *
width, height = 800, 600
cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
cap.set(3,640)
cap.set(4,480)
cap2.set(3,640)
cap2.set(4,480)

root = Tk()
root.geometry("1880x1080")
frame = Frame(root,width=1000,height=1000)
root.bind('<Escape>', lambda e: root.quit())
lmain = Label(root)
lmain2 = Label(root)
lol2=Label(frame,text="lol2",font=("Times New Roman",30))
lol3=Label(frame,text="lol3")
lol2.pack(side=TOP)
lol3.pack(side=BOTTOM)
lmain.grid(row=0,column=2,sticky=E)
lmain2.grid(row=1,column=2,sticky=E)
frame.grid(row=0,column=0,columnspan=2,padx=300)


def show_frame():
    _, frameDisp = cap.read()
    frameDisp = cv2.flip(frameDisp, -1)
    cv2image = cv2.cvtColor(frameDisp, cv2.COLOR_BGR2RGBA)
    img = PIL.Image.fromarray(cv2image)
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

try:    
    show_frame()
    pass
except:
    pass
root.mainloop()
'''









from tkinter import *
import PIL
from PIL import Image,ImageTk
import cv2

camFrameWidth=1020
menuFrameWidth=1920-camFrameWidth

cam1FrameHeight=500
cam2FrameHeight=1080-cam1FrameHeight
menuFrameHeight=1080


def initCams():
    width, height = 800, 600
    cap = cv2.VideoCapture(1)
    cap2 = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    cap2.set(3,640)
    cap2.set(4,480)
    try:
        show_frame()
    except Exception as e:
        print(e)

def show_frame():
    _, frameDisp = cap.read()
    frameDisp = cv2.flip(frameDisp, -1)
    cv2image = cv2.cvtColor(frameDisp, cv2.COLOR_BGR2RGBA)
    img = PIL.Image.fromarray(cv2image)
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
lmain2 = Label(frame3)
lslave2 = Label(frame3,text="Iris Camera",font=("Times New Roman",22))
frame2.pack(anchor='ne')
frame3.pack(anchor='se')
lmain.place(x=camFrameWidth/2, y=cam1FrameHeight/2, anchor="center")
lmain2.place(x=camFrameWidth/2, y=cam2FrameHeight/2, anchor="center")
lslave.place(x=20, y=cam1FrameHeight/2, anchor="w")
lslave2.place(x=20, y=cam2FrameHeight/2, anchor="w")

labelTitle = Label(frame, text="Eye Tracker",font=("Times New Roman",30))
labelTitle.place(x=menuFrameWidth/2,y=20,anchor="center")

#labelTest=Label(root,text="THIS IS A TEST")
#labelTest.place(x=900,y=540,anchor="center")

startCamera = Button(frame,command=initCams,text="Start cameras")
startCamera.place(x=menuFrameWidth/2,y=menuFrameHeight/2,anchor="center")


#try:    
#    show_frame()
#except:
#    pass
root.mainloop()

