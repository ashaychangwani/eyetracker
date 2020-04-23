#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 00:49:53 2019

@author: ashay
"""


"""
Created on Sun Mar 24 21:16:36 2019

@author: ashay
"""
from tkinter import Tk, Canvas, Button, Frame, Label, LEFT, OptionMenu, Scale, HORIZONTAL, BooleanVar, StringVar
import PIL
from PIL import ImageTk
import time
import csv
import numpy as np
from pandas import read_csv
import math
import threading
import tensorflow as tf
import traceback
from itertools import chain

from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split

from concurrent.futures import ThreadPoolExecutor

import sys,os
sys.path.append("/usr/local/lib/python3.7/site-packages")
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2

#########           Setting up the interface, constants            ###########


height=1080
width=1920
cHeight=480
cWidth=640


timePerDot=1

camFrameWidth=width*0.7
menuFrameWidth=width-camFrameWidth

cam1FrameHeight=0.5*height
cam2FrameHeight=height-cam1FrameHeight
menuFrameHeight=height

lower_white = np.array([220])
upper_white = np.array([255])






warningMessage="Tracking beacons: "

cams=["Camera 1","Camera 2"]
cap=None
cap2=None
execStarted=False

isCalibrating=False
gTruthX=0
gTruthY=0

keepLogging=True

gPredX=0
gPredY=0

executor=ThreadPoolExecutor(max_workers=1)          
'''Allows show_frame to work in parallel'''
HTCExecutor=ThreadPoolExecutor(max_workers=1)
ITCExecutor=ThreadPoolExecutor(max_workers=1)
calibExecutor=ThreadPoolExecutor(max_workers=20)

HTCFrame=None
ITCFrame=None

htScale=None
wdScale=None
htScale2=None
wdScale2=None
#model=Sequential()

writeFile=None
writer = None
writerLock=threading.Lock()

pointer=None

trainingComplete=False

sc=None

allFramesProcessed=False





#############               For finding missing beacon              ############
def slopeCalc(x1, y1, x2, y2): 
    return (y2-y1)/(x2-x1)




def to_matrix(l, n=2):
    return [l[i:i+n] for i in range(0, len(l), n)]

def to_list(l1):
    return(list(chain.from_iterable(l1)))
    






########            #########
    

'''
root2 is the new TKinter frame


"start calibration" runs the displayDots function
"start tracking" runs the displayTracker function

'''


def initCalib():
    global model,gTruthX,gTruthY,trainingComplete
    root2=Tk()
    root2.attributes("-fullscreen", True)
    root2.bind('<Escape>', lambda q: root2.quit())
    root2.geometry('%dx%d'%(width,height))
    waitVar=BooleanVar(master=root2)
    pointer=None

    	
    def setWaitVarTrue():
        try:
            nonlocal waitVar
            waitVar.set(True)
        except:
            print('setWaitVarTrue failed')
            
            
            
    def quitThisMethod():
        global isCalibrating,trainingComplete,writeFile
        isCalibrating=False
        
        writeFile.close()
        
    def displayDots(): 
        global isCalibrating,gTruthX,gTruthY,model,writeFile,writer,trainingComplete,timePerDot,allFramesProcessed
        nonlocal waitVar,pointer
        writeFile=open('calib.csv', 'a+')
        writer = csv.writer(writeFile)
        canvas = Canvas(root2, width=width, height=height, borderwidth=0, highlightthickness=0, bg="black")
        canvas.place(x=width/2,y=height/2,anchor="center")
        quitButton=Button(canvas, text="Quit", command=lambda:[root2.destroy(),quitThisMethod()])
        quitButton.place(x=width/2,y=height-80,anchor="center")
        pointer=canvas.create_oval(0,0,0,0,outline="#f11",fill="#1f1", width=2)
        isCalibrating=True
        executor.submit(startCalibrationProcessing)
        for i in range (5):
            for j in range (4):  
                x=width*i/4
                y=height*j/3
                canvas.delete(pointer)
                gTruthX=x
                gTruthY=y
                pointer=canvas.create_oval(x-10, y-10, x+10, y+10, outline="#f11",fill="#1f1", width=2)
                root2.after(timePerDot, setWaitVarTrue)
                root2.wait_variable(waitVar)
                
                
                try:
                    root2.update()
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(e)
                    print("EXCP HERE333",exc_type, fname, exc_tb.tb_lineno, traceback.print_exc())
                    print(traceback.format_exc())
                
        print("reached here1")
        canvas.delete(pointer)
        print("reached here2")
        isCalibrating=False
        print("reached here3")
        writeFile.close()
        print("reached here4")
        dataframe=read_csv("calib.csv")
        print("reached here5")
        
        X=dataframe.iloc[:,0:8].values
        print("reached here6")
        y=dataframe.iloc[:,8:10].values
        
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=0)

        model=Sequential()
        model.add(Dense(8,input_dim=8,kernel_initializer='normal', activation='sigmoid'))
        #model.add(Dropout(0.2))
        model.add(Dense(22,kernel_initializer='normal', activation='sigmoid'))
        #model.add(Dropout(0.2))
        model.add(Dense(2,kernel_initializer='normal'))
        #model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mse','accuracy'])
        model.compile(loss='mean_squared_error', optimizer='RMSProp',metrics=['mse'])
        print("reached here7")
        #with tf.device('/cpu:0'):
        model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=16,epochs=350)
        print("reached here8")
        
        #y_pred=model.predict(np.array(X_test))      UNUSED
        
        trainingComplete=True
        #canvas.destroy()
    
    def dispTracker():
        global row2,gPredX,gPredY
        nonlocal pointer
        canvas1=Canvas(root2, width=1920, height=1080, borderwidth=0, highlightthickness=0, bg="blue")
        canvas1.place(x=width/2,y=height/2,anchor="center")
        qb=Button(canvas1,text="quit",command=root2.destroy)
        qb.place(x=width/2,y=height-80,anchor="center")
        test=0
        
        while(trainingComplete):
            if gPredX<0:
                gPredX=0
            if gPredY<0:
                gPredY=0
            if gPredY>1:                #Check why neural network is giving less than 0 and more than 1
                gPredY=1
            if gPredX>1:
                gPredX=1
            test+=1
            root2.update()
            canvas1.delete(pointer)
            pointer=canvas1.create_oval(int((gPredX+1)/2*width-50), int((gPredY+1)/2*height-50), int((gPredX+1)/2*width+50), int((gPredY+1)/2*height+50), outline="#f11",fill="#1f1", width=2)
            
            
    frame = Frame(root2, width=width, height=height,bg='Red')
    startCalib=Button(frame, text="Start Calibration", command=displayDots)
    startCalib.place(x=width/2,y=height-130,anchor="center")
    startTracking=Button(frame, text="Start Tracking", command=dispTracker)
    startTracking.place(x=width/2,y=height-80,anchor="center")
    quitExec=Button(frame, text="Quit", command=root2.destroy)
    quitExec.place(x=width/2,y=height-180,anchor="center")
    frame.pack()
    root2.mainloop()
    
    
def startCalibrationProcessing():
    global HTCFrame, ITCFrame, calibExecutor, isCalibrating, gTruthX, gPredY
    
    while isCalibrating:
        calibExecutor.submit(ProcessingFn,HTCFrame.copy(),ITCFrame.copy(),gTruthX, gTruthY)
        time.sleep(1/30)
        
    if not isCalibrating:
        print('waiting for executor to shut down')
        #executor.shutdown(wait=True)
        print('executor shutdown')
        print('waiting for calibExecutor to shut down')
        calibExecutor.shutdown(wait=True)
        print('calibExecutor shutdown')

def ProcessingFn(frame,frame2,gTruthX, gTruthY):
    global writer,writerLock
    wTemp=wdScale2.get()-wdScale.get()
    hTemp=htScale2.get()-htScale.get()
    try:
        frameDisp = frame
        frameDisp2 = frame2
        
        
        hsv = cv2.cvtColor(frameDisp, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(hsv, lower_white, upper_white)   
        _,contours,heih = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        count=0
        row=[]
        for c in contours:
           if cv2.contourArea(c) > 50:
               M = cv2.moments(c)
               if M["m00"] != 0:
                 cX = int(M["m10"] / M["m00"])
                 cY = int(M["m01"] / M["m00"])
                 cv2.circle(mask, (cX, cY), 15, (120, 120, 120), 2)
                 if(count<3):
                     row.extend([(cX/cWidth),(cY/cHeight)])
                 count+=1
        row=to_matrix(row)
        row=sorted(row , key=lambda k: [k[0], k[1]])
        row=to_list(row)
        if(count==2):
            slope=slopeCalc(*tuple(row))
            
            if slope>=-0.4 and slope<=0.4:
                row.insert(2,-1)
                row.insert(2,-1)  
            elif slope>0:
                row.insert(4,-1)
                row.insert(4,-1)
            else:
                row.insert(0,-1)
                row.insert(0,-1)
            count+=1
        
        while (count<3):
            row.extend([-1,-1])
            count+=1
            
    
    
    
    
        
        frame=frameDisp2[wdScale.get():wdScale2.get(),htScale.get():htScale2.get()]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray=cv2.medianBlur(gray,3)
        ret,thresh = cv2.threshold(gray,thresholdSlider.get(),255,cv2.THRESH_BINARY_INV)
        _,contours, heih = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        
        foundC=False
        ratMax=0
        eMax=None
        cMax=None
        for c in contours:
            try:
                if cv2.contourArea(c) > 8000:
                    testTemp=np.zeros(np.shape(frame),np.uint8)
                    gray=np.zeros(np.shape(frame),np.uint8)
                                
                    cv2.fillPoly(testTemp,pts=[c],color=(255,255,255))
                    ellipse=cv2.fitEllipse(c)
                    cv2.ellipse(gray, ellipse, (255,255,255),-1)
                    testTemp=cv2.bitwise_and(testTemp,gray)
                   
                    rat=cv2.countNonZero(cv2.cvtColor(testTemp,cv2.COLOR_BGR2GRAY))/(3.14*ellipse[1][0]/2*ellipse[1][1]/2)
                    testTemp=None
                    gray=None
                    if(rat>ratMax and rat>0.75):
                       ratMax=rat
                       cMax=c
                       eMax=ellipse
                       foundC=True
            except Exception as e:        
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(e)
                print("exception in Processing Fn",exc_type, fname, exc_tb.tb_lineno, traceback.print_exc())
                print(traceback.format_exc())
        
                    
                    
        if(foundC):
            
            M = cv2.moments(cMax)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            xTemp=(eMax[0][0]+cX)/2
            yTemp=(eMax[0][1]+cY)/2
            row.extend([(xTemp/wTemp-0.5)*2,(yTemp/hTemp-0.5)*2])
            
        else:
            row=[]
        if(len(row)>3):
            try:
                row.extend([(gTruthX/width-0.5)*2,(gTruthY/height-0.5)*2])         
                writerLock.acquire()
                writer.writerow(row)
                writerLock.release()
            except Exception as e:        
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(e)
                print("exception in Processing Fn 2",exc_type, fname, exc_tb.tb_lineno, traceback.print_exc())
                print(traceback.format_exc())
        
            
    
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(e)
        print("EXCP HERE2",exc_type, fname, exc_tb.tb_lineno, traceback.print_exc())
        print(traceback.format_exc())



    
def initCams():
    global cap,cap2,execStarted
    try:
        cap = cv2.VideoCapture(int(camVar.get()[-1])-1)
        cap2 = cv2.VideoCapture(int(camVar2.get()[-1])-1)
        cap.set(3,cWidth)
        cap.set(4,cHeight)
        cap2.set(3,1280)
        cap2.set(4,720)
        if not execStarted:    
            HTCExecutor.submit(startLoggingCalibFrames_HTC)
            ITCExecutor.submit(startLoggingCalibFrames_ITC)
            time.sleep(1)
            executor.map(show_frame())
            execStarted=True
    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("EXCP HERE",exc_type, fname, exc_tb.tb_lineno)
        print(traceback.format_exc())

def distance(p1,p2):
    return math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))


def startLoggingCalibFrames_HTC():
    global cap, HTCFrame,keepLogging
   
    try:
        while keepLogging:
            _, frame=cap.read()
            HTCFrame=frame.copy()
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(e)
        print("EXCP in startLoggingCalibFrames_HTC",exc_type, fname, exc_tb.tb_lineno, traceback.print_exc())
        print(traceback.format_exc())



def startLoggingCalibFrames_ITC():
    global cap2, ITCFrame,keepLogging
   
    try:
        while keepLogging:
            _, frame=cap2.read()
            ITCFrame=frame.copy()
        
    
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(e)
        print("EXCP in startLoggingCalibFrames_ITC",exc_type, fname, exc_tb.tb_lineno, traceback.print_exc())
        print(traceback.format_exc())






'''
Runs constantly, constantly updates the values of gPred and accepts values of gTruth etc 
'''
def show_frame():
    global gTruthX,gTruthY,writer,isCalibrating,gPredX,gPredY,trainingComplete,model,HTCFrame,ITCFrame
    wTemp=wdScale2.get()-wdScale.get()
    hTemp=htScale2.get()-htScale.get()
    try:
        frameDisp = HTCFrame.copy()
        frameDisp2 = ITCFrame.copy()
        
        hsv = cv2.cvtColor(frameDisp, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(hsv, lower_white, upper_white)   
        
        _,contours,heih = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        count=0
        row=[]
        for c in contours:
           if cv2.contourArea(c) > 50:
               M = cv2.moments(c)
               if M["m00"] != 0:
                 cX = int(M["m10"] / M["m00"])
                 cY = int(M["m01"] / M["m00"])
                 cv2.circle(mask, (cX, cY), 15, (120, 120, 120), 2)
                 if(count<3):
                     row.extend([(cX/cWidth-0.5)*2,(cY/cHeight-0.5)*2])
                 count+=1
        row=to_matrix(row)
        row=sorted(row , key=lambda k: [k[0], k[1]])
        row=to_list(row)
        if(count==2):
            slope=slopeCalc(*tuple(row))
            
            if slope>=-0.4 and slope<=0.4:
                row.insert(2,-1)
                row.insert(2,-1)  
            elif slope>0:
                row.insert(4,-1)
                row.insert(4,-1)
            else:
                row.insert(0,-1)
                row.insert(0,-1)
            count+=1
        
        while (count<3):
            row.extend([-1,-1])
            count+=1
            
            
        if not isCalibrating:
            mask = cv2.flip(mask, -1)
            img = PIL.Image.fromarray(mask)
            imgtk = ImageTk.PhotoImage(image=img)
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)
    
    
    
    
        
        frame=frameDisp2[wdScale.get():wdScale2.get(),htScale.get():htScale2.get()]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray=cv2.medianBlur(gray,3)
        ret,thresh = cv2.threshold(gray,thresholdSlider.get(),255,cv2.THRESH_BINARY_INV)
        
        _,contours, heih = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        
        foundC=False
        ratMax=0
        cMax=None
        eMax=None
        for c in contours:
            try:
                if cv2.contourArea(c) > 8000:
                    testTemp=np.zeros(np.shape(frame),np.uint8)
                    gray=np.zeros(np.shape(frame),np.uint8)
                                
                    cv2.fillPoly(testTemp,pts=[c],color=(255,255,255))
                    ellipse=cv2.fitEllipse(c)
                    cv2.ellipse(gray, ellipse, (255,255,255),-1)
                    testTemp=cv2.bitwise_and(testTemp,gray)
                    
                    #total=cv2.bitwise_or(testTemp,total)
                   
                    rat=cv2.countNonZero(cv2.cvtColor(testTemp,cv2.COLOR_BGR2GRAY))/(3.14*ellipse[1][0]/2*ellipse[1][1]/2)
                    testTemp=None
                    gray=None
                    if(rat>ratMax and rat>0.75):
                       ratMax=rat
                       cMax=c
                       eMax=ellipse
                       foundC=True
            except:        
                pass
        
                    
        if(foundC):
            #print(ratMax,(3.14*eMax[1][0]/2*eMax[1][1]/2))
            cv2.drawContours(frame,cMax,-1,(0,255,0),1)
            cv2.ellipse(frame, eMax, (0,255,255),1)
            
            M = cv2.moments(cMax)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            xTemp=(eMax[0][0]+cX)/2
            yTemp=(eMax[0][1]+cY)/2
            cv2.circle(frame, (int(xTemp), int(yTemp)), 15, (120, 120, 120), 2)
            row.extend([(xTemp/wTemp-0.5)*2,(yTemp/hTemp-0.5)*2])
            
        else:
            row=[]
        if not isCalibrating:
            frameDisp2 = cv2.rotate(frame,rotateCode=cv2.ROTATE_90_CLOCKWISE)
            img2 = PIL.Image.fromarray(frameDisp2)
            imgtk2 = ImageTk.PhotoImage(image=img2)
            lmain2.imgtk = imgtk2
            lmain2.configure(image=imgtk2)
        
            
        if trainingComplete and len(row)==8:
            result=model.predict(np.array([row]))
            gPredX,gPredY=result[0][0],result[0][1]
            #print("RESULTANT COORD:",width*(gPredX+1)/2,"  ",height*(gPredY+1)/2)
        if keepLogging:
            lmain.after(1, callShowFrame)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(e)
        print("EXCP HERE2",exc_type, fname, exc_tb.tb_lineno, traceback.print_exc())
        print(traceback.format_exc())
    
    
    '''temporary
    if keepLogging:
        lmain.after(1, callShowFrame)'''
    
    
    
def callShowFrame():
    executor.map(show_frame())
    
    
def shutdownTime():
    global keepLogging,HTCExecutor,ITCExecutor,cap,cap2,root,executor
    keepLogging=False
    try:
        HTCExecutor.shutdown(wait=False)
        ITCExecutor.shutdown(wait=False)
        executor.shutdown(wait=False)
        cap.release()
        cap2.release()
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(e)
        print("EXCP in shutdownTime",exc_type, fname, exc_tb.tb_lineno, traceback.print_exc())
        print(traceback.format_exc())
    finally:
        root.destroy()
root=Tk()
#root.attributes("-fullscreen", True)
root.bind('<Escape>', lambda e: root.quit())
root.geometry('%dx%d'%(width,height))
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

thresholdSlider=Scale(frame3,from_=10,to=80, orient=HORIZONTAL)
lslave3 = Label(frame3,text="Boundary adjustment",font=("Times New Roman",16))
htScale=Scale(frame3,from_=0,to=1280, orient=HORIZONTAL)
htScale2=Scale(frame3,from_=0,to=1280, orient=HORIZONTAL)
wdScale=Scale(frame3,from_=0,to=720, orient=HORIZONTAL)
wdScale2=Scale(frame3,from_=0,to=720, orient=HORIZONTAL)
htScale.set(395)
htScale2.set(941)
wdScale.set(95)
wdScale2.set(720)
thresholdSlider.set(28)



frame2.pack(anchor='ne')
frame3.pack(anchor='se')
lmain.place(x=camFrameWidth/2, y=cam1FrameHeight/2, anchor="center")
lmain2.place(x=camFrameWidth/2, y=cam2FrameHeight/2-70, anchor="center")
lslave.place(x=20, y=cam1FrameHeight/2, anchor="w")
lOptions.place(x=20, y=cam1FrameHeight/2+30, anchor="w")
lslave2.place(x=20, y=50, anchor="w")
lOptions2.place(x=20, y=80, anchor="w")
labelTitle = Label(frame, text="Eye Tracker",font=("Times New Roman",30))
labelTitle.place(x=menuFrameWidth/2,y=30,anchor="center")
lslave3.place(x=20, y=110, anchor="w")
htScale.place(x=20, y=150, anchor="w")
htScale2.place(x=20, y=190, anchor="w")
wdScale.place(x=20, y=240, anchor="w")
wdScale2.place(x=20, y=280, anchor="w")
thresholdSlider.place(x=20, y=320, anchor="w")


beaconCount=StringVar()
beaconCount.set("Program not running")

labelTrackingTitle=Label(frame,text="Tracking Beacons:")
labelTrackingTitle.place(x=menuFrameWidth/2,y=height-300,anchor="center")
labelTrackingCount=Label(frame,textvariable=beaconCount)
labelTrackingCount.place(x=menuFrameWidth/2,y=height-260,anchor="center")

quitApp = Button(frame,command=shutdownTime,text="Quit")
quitApp.place(x=menuFrameWidth/2,y=menuFrameHeight/2+60,anchor="center")

startCamera = Button(frame,command=initCams,text="Start cameras")
startCamera.place(x=menuFrameWidth/2,y=menuFrameHeight/2,anchor="center")


startCalibration = Button(frame,command=initCalib,text="Start calibration")
startCalibration.place(x=menuFrameWidth/2,y=menuFrameHeight/2+30,anchor="center")




root.mainloop()
cap=None
cap2=None