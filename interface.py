
"""
Created on Sun Mar 24 21:16:36 2019

@author: ashay
"""
from tkinter import *
import PIL
from PIL import Image,ImageTk
import cv2
import time
import tensorflow as tf
import csv
import sys
import numpy as np
import math
#from keras.models import Sequential
#from keras.layers import Dense
sys.path.append("/Users/ashay/Desktop/eyeTracker/models/research/object_detection")
sys.path.append("/Users/ashay/Desktop/eyeTracker/models/research")





camFrameWidth=1020
menuFrameWidth=1920-camFrameWidth

cam1FrameHeight=500
cam2FrameHeight=1080-cam1FrameHeight
menuFrameHeight=1080

lower_white = np.array([180])
upper_white = np.array([255])



oldContour=(0,0)
maxC=None

warningMessage="Tracking beacons: "

cams=["Camera 2","Camera 1"]
cap=None
cap2=None
execStarted=False

isCalibrating=False
gTruthX=0
gTruthY=0

#model=Sequential()

csvIndex=0
writer=None
writeFile=open('calib.csv', 'w') 
writer = csv.writer(writeFile)

def initCalib():
    global model,gTruthX,gTruthY
    root2=Tk()
    root2.attributes("-fullscreen", True)
    root2.bind('<Escape>', lambda q: root2.quit())
    root2.geometry("1920x1080")
    waitVar=BooleanVar()

    	
    def temp():
        global waitVar
        waitVar.set(True)
    def displayDots():
        global waitVar,isCalibrating,gTruthX,gTruthY
        isCalibrating=True
        canvas = Canvas(root2, width=1920, height=1080, borderwidth=0, highlightthickness=0, bg="black")
        canvas.place(x=1920/2,y=1080/2,anchor="center")
        quitButton=Button(canvas, text="Quit", command=root2.destroy)
        quitButton.place(x=1920/2,y=1000,anchor="center")
        for i in range (5):
            for j in range (5):   
                x=1920*i/4
                y=1080*j/4
                gTruthX=x
                gTruthY=y
                canvas.create_oval(x-10, y-10, x+10, y+10, outline="#f11",fill="#1f1", width=2)
                root.after(4000, temp)
                root.wait_variable(waitVar)
                                   
                root2.update()
               
#                print('waiting',nextPt.get())
#                while(test==False):
#                    pass
#                test=False
#                print('here')
        isCalibrating=False
    frame = Frame(root2, width=1920, height=1080,bg='Red')
    startCalib=Button(frame, text="Start Calibration", command=displayDots)
    startCalib.place(x=1920/2,y=950,anchor="center")
    frame.pack()
    
    
    
    root2.mainloop()
    
def initCams():
    global cap,cap2,execStarted
    try:
        width, height = 800, 600
        cap = cv2.VideoCapture(int(camVar.get()[-1])-1)
        cap2 = cv2.VideoCapture(int(camVar2.get()[-1])-1)
        cap.set(3,640)
        cap.set(4,480)
#        cap2.set(3,640)
#        cap2.set(4,480)
        if not execStarted:    
            show_frame()
    except Exception as e:
        print(e)
    execStarted=True

def distance(p1,p2):
    return math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))
def show_frame():
    global oldContour,maxC,csvIndex,gTruthX,gTruthY,writer,isCalibrating
    row=[]
    _, frameDisp = cap.read()
    hsv = cv2.cvtColor(frameDisp, cv2.COLOR_BGR2GRAY)#
    mask = cv2.inRange(hsv, lower_white, upper_white)    
    contours, heih = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    count=0
    row=[]
    for c in contours:
       M = cv2.moments(c)
       if M["m00"] != 0:
         cX = int(M["m10"] / M["m00"])
         cY = int(M["m01"] / M["m00"])
         count+=1
         cv2.circle(mask, (cX, cY), 15, (120, 120, 120), 2)
         row.append((cX,cY))
       else:
         cX, cY = 0, 0
         row.append((-1,-1))
    row=sorted(row , key=lambda k: [k[0], k[1]])
    for i in range (count,3):
        row.append((-1,-1))
    beaconCount.set("Beacons being tracked: "+str(count))
    mask = cv2.flip(mask, -1)
    img = PIL.Image.fromarray(mask)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)

    
    _, frameDisp2 = cap2.read()    
    
    frame=frameDisp2[310:810,220:770]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    gray=cv2.medianBlur(gray,3)
#    //75 decent, eyelash interfering
    ret,thresh = cv2.threshold(gray,71,255,cv2.THRESH_BINARY_INV)
    contours, heih = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(frame, contours, -1, (0,255,0), 1)
    test=False
    try:
        maxC=contours[0]
    except:
        pass
    for c in contours:
        try:
            if(cv2.contourArea(c)/cv2.arcLength(c,True)>cv2.contourArea(maxC)/cv2.arcLength(maxC,True) and cv2.contourArea(c)/cv2.arcLength(c,True)>13):
                maxC=c
                test=True
        except:
            pass
    if(test):
        c=maxC
        x,y,w,h = cv2.boundingRect(c)
        if(distance((x+w/2,y+h/2),oldContour)<90):
            cv2.circle(frame,(int(x+w/2),int(y+h/2)), 15, (120, 120, 120), 2)
            row.append((x+w/2,y+h/2))
        else:
            row=[]
        
        oldContour=(x+w/2,y+h/2)
    else:
        row=[]
        
    frameDisp2 = cv2.rotate(frame,rotateCode=cv2.ROTATE_90_CLOCKWISE)
    img2 = PIL.Image.fromarray(frameDisp2)
    imgtk2 = ImageTk.PhotoImage(image=img2)
    lmain2.imgtk = imgtk2
    lmain2.configure(image=imgtk2)
    if(isCalibrating):
        if(len(row)>3):
            row.append(tuple((gTruthX,gTruthY)))
            writer.writerow(row)
            print(row)
    else:
        print('not workingggg')
        print(row)
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
lmain2.place(x=camFrameWidth/2, y=cam2FrameHeight/2-20, anchor="center")
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

quitApp = Button(frame,command=root.destroy,text="Quit")
quitApp.place(x=menuFrameWidth/2,y=menuFrameHeight/2+60,anchor="center")

startCamera = Button(frame,command=initCams,text="Start cameras")
startCamera.place(x=menuFrameWidth/2,y=menuFrameHeight/2,anchor="center")


startCalibration = Button(frame,command=initCalib,text="Start calibration")
startCalibration.place(x=menuFrameWidth/2,y=menuFrameHeight/2+30,anchor="center")



waitVar=BooleanVar()
#try:    
#    show_frame()
#except:
#    pass
root.mainloop()
cap=None
cap2=None
writeFile.close()