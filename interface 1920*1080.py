
"""
Created on Sun Mar 24 21:16:36 2019

@author: ashay
"""
from tkinter import *
import PIL
from PIL import Image,ImageTk
import time
#import tensorflow as tf
import csv
import numpy as np
from pandas import *
import math
import tensorflow as tf
import traceback
from itertools import chain


from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

import sys,os
sys.path.append("/usr/local/lib/python3.7/site-packages")
import cv2
cv2.__version__



height=1080
width=1920


camFrameWidth=width*0.7
menuFrameWidth=width-camFrameWidth

cam1FrameHeight=0.5*height
cam2FrameHeight=height-cam1FrameHeight
menuFrameHeight=height

lower_white = np.array([220])
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

gPredX=0
gPredY=0



htScale=None
wdScale=None
htScale2=None
wdScale2=None
#model=Sequential()

csvIndex=0
writer=None
writeFile=None
writer = None

pointer=None

trainingComplete=False

sc=None


def slopeCalc(x1, y1, x2, y2):
    return (y2-y1)/(x2-x1)

def to_matrix(l, n=2):
    return [l[i:i+n] for i in range(0, len(l), n)]

def to_list(l1):
    return(list(chain.from_iterable(l1)))
    
def initCalib():
    global model,gTruthX,gTruthY,trainingComplete
    root2=Tk()
    root2.attributes("-fullscreen", True)
    root2.bind('<Escape>', lambda q: root2.quit())
    root2.geometry(widthxheight)
    waitVar=BooleanVar(master=root2)
    pointer=None

    	
    def temp():
        nonlocal waitVar
        waitVar.set(True)
    def quitThisMethod():
        global isCalibrating,trainingComplete
        isCalibrating=False
        writeFile.close()
        
    def displayDots():
        global isCalibrating,gTruthX,gTruthY,model,writeFile,writer,trainingComplete
        nonlocal waitVar,pointer
        writeFile=open('calib.csv', 'a+') 
        writer = csv.writer(writeFile)
        isCalibrating=True
        canvas = Canvas(root2, width=width, height=height, borderwidth=0, highlightthickness=0, bg="black")
        canvas.place(x=width/2,y=height/2,anchor="center")
        quitButton=Button(canvas, text="Quit", command=lambda:[root2.destroy(),quitThisMethod()])
        quitButton.place(x=width/2,y=height-80,anchor="center")
        pointer=None
        for i in range (5):
            for j in range (4):   
                x=width*i/4
                y=height*j/3
                canvas.delete(pointer)
                gTruthX=x
                gTruthY=y
                pointer=canvas.create_oval(x-10, y-10, x+10, y+10, outline="#f11",fill="#1f1", width=2)
                root2.after(6000, temp)
                root2.wait_variable(waitVar)
                root2.update()
                
        canvas.delete(pointer)
        isCalibrating=False
        writeFile.close()
        dataframe=read_csv("calib.csv")
        
        X=dataframe.iloc[:,0:8].values
        y=dataframe.iloc[:,8:10].values
        
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
            
        model=Sequential()
        model.add(Dense(8,input_dim=8,kernel_initializer='normal', activation='sigmoid'))
        #model.add(Dropout(0.2))
        model.add(Dense(22,kernel_initializer='normal', activation='sigmoid'))
        #model.add(Dropout(0.2))
        model.add(Dense(2,kernel_initializer='normal'))
        #model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mse','accuracy'])
        model.compile(loss='mean_squared_error', optimizer='RMSProp',metrics=['mse'])
        with tf.device('/cpu:0'):
            model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=16,epochs=350)
        
        y_pred=model.predict(np.array(X_test))
        mse = (np.square(y_pred - y_test)).mean(axis=0)
        
        
        
        '''
        
        batch 16-0.04,0.055
        batch 32-0.04,0.08
        batch 8- 0.04,0.05
        
        '''
        
        trainingComplete=True
        canvas.destroy()
    
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
                gPrexX=0
            if gPredY<0:
                gPrexY=0
            if gPredY>1:
                gPrexY=1
            if gPredX>1:
                gPrexX=1
            test+=1
#            root2.after(4000, temp)
            root2.update()
#            time.sleep(0.05)
            canvas1.delete(pointer)
            pointer=canvas1.create_oval(int((gPredX+1)/2*width-50), int((gPredY+1)/2*height-50), int((gPredX+1)/2*width+50), int((gPredY+1)/2*height+50), outline="#f11",fill="#1f1", width=2)
            
#            pointer=canvas1.create_oval(200+test,200+test,400+test,400+test, outline="#f11",fill="#1f1", width=2)
            
    frame = Frame(root2, width=width, height=height,bg='Red')
    startCalib=Button(frame, text="Start Calibration", command=displayDots)
    startCalib.place(x=width/2,y=height-130,anchor="center")
    startTracking=Button(frame, text="Start Tracking", command=dispTracker)
    startTracking.place(x=width/2,y=height-80,anchor="center")
    quitExec=Button(frame, text="Quit", command=root2.destroy)
    quitExec.place(x=width/2,y=height-180,anchor="center")
    frame.pack()
    
    
    
    root2.mainloop()
    
def initCams():
    global cap,cap2,execStarted
    try:
        cap = cv2.VideoCapture(int(camVar.get()[-1])-1)
        cap2 = cv2.VideoCapture(int(camVar2.get()[-1])-1)
        cap.set(3,640)
        cap.set(4,480)
        cap2.set(3,1280)
        cap2.set(4,720)
        if not execStarted:    
            show_frame()
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("EXCP HERE",exc_type, fname, exc_tb.tb_lineno)
        print(traceback.format_exc())
    execStarted=True

def distance(p1,p2):
    return math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))

def show_frame():
    global oldContour,maxC,csvIndex,gTruthX,gTruthY,writer,isCalibrating,gPredX,gPredY,trainingComplete,model

    try:
        row=[]
        _, frameDisp = cap.read()
        hsv = cv2.cvtColor(frameDisp, cv2.COLOR_BGR2GRAY)#
        mask = cv2.inRange(hsv, lower_white, upper_white)   
        contours,heih = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        count=0
        row=[]
        for c in contours:
           if cv2.contourArea(c) > 100:
               M = cv2.moments(c)
               if M["m00"] != 0:
                 cX = int(M["m10"] / M["m00"])
                 cY = int(M["m01"] / M["m00"])
                 cv2.circle(mask, (cX, cY), 15, (120, 120, 120), 2)
                 if(count<3):
                     row.extend([(cX/width-0.5)*2,(cY/height-0.5)*2])
                 count+=1
        row=to_matrix(row)
        row=sorted(row , key=lambda k: [k[0], k[1]])
        row=to_list(row)
        if(count==2):
            slope=slopeCalc(*tuple(row))
            if(slope<0.6):
                row.insert(2,-1)
                row.insert(2,-1)
            else:
                row.insert(4,-1)
                row.insert(4,-1)
            
        
        beaconCount.set("Beacons being tracked: "+str(count))
        while (count<3):
            row.extend([-1,-1])
            count+=1
        mask = cv2.flip(mask, -1)
        img = PIL.Image.fromarray(mask)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
    
    
    
    
    
        
        _, frameDisp2 = cap2.read()    
        
        frame=frameDisp2[wdScale.get():wdScale2.get(),htScale.get():htScale2.get()]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray=cv2.medianBlur(gray,3)
    #    //75 decent, eyelash interfering
        ret,thresh = cv2.threshold(gray,thresholdSlider.get(),255,cv2.THRESH_BINARY_INV)
        contours, heih = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        
        total=np.zeros(np.shape(frame),np.uint8)
        foundC=False
        '''
        try:
            maxC=contours[0]
        except:
            pass
        '''
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
                    if(rat>ratMax):
                       ratMax=rat
                       cMax=c
                       eMax=ellipse
                       foundC=True
            except:
                pass
        
                    
        if(foundC):
            print(ratMax,(3.14*eMax[1][0]/2*eMax[1][1]/2))
            cv2.drawContours(frame,cMax,-1,(0,255,0),1)
            cv2.ellipse(frame, eMax, (0,255,255),1)
            xTemp=eMax[0][0]
            yTemp=eMax[0][1]
            row.extend([(xTemp/width-0.5)*2,(yTemp/height-0.5)*2])
        else:
            row=[]
        frameDisp2 = cv2.rotate(frame,rotateCode=cv2.ROTATE_90_CLOCKWISE)
        img2 = PIL.Image.fromarray(frameDisp2)
        imgtk2 = ImageTk.PhotoImage(image=img2)
        lmain2.imgtk = imgtk2
        lmain2.configure(image=imgtk2)
        if(isCalibrating):
            if(len(row)>3):
                try:
                    #row.extend([gTruthX/width,gTruthY/height])
                    row.extend([(gTruthX/width-0.5)*2,(gTruthY/height-0.5)*2])
                    writer.writerow(row)
                    print(row)
                except Exception as e:
                    print(e)
        else:
            pass
        if trainingComplete and len(row)==8:
            result=model.predict(np.array([row]))
            gPredX,gPredY=result[0][0],result[0][1]
            print("RESULTANT COORD:",(gPredX+1)/2*width,"  ",(gPredY+1)/2*height)
            
        lmain.after(10, show_frame)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(e)
        print("EXCP HERE2",exc_type, fname, exc_tb.tb_lineno, traceback.print_exc())
        print(traceback.format_exc())

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

thresholdSlider=Scale(frame3,from_=40,to=80, orient=HORIZONTAL)
lslave3 = Label(frame3,text="Boundary adjustment",font=("Times New Roman",16))
htScale=Scale(frame3,from_=0,to=1280, orient=HORIZONTAL)
htScale2=Scale(frame3,from_=0,to=1280, orient=HORIZONTAL)
wdScale=Scale(frame3,from_=0,to=720, orient=HORIZONTAL)
wdScale2=Scale(frame3,from_=0,to=720, orient=HORIZONTAL)
#[180:855,370:850]
htScale.set(395)
htScale2.set(941)
wdScale.set(95)
wdScale2.set(720)
thresholdSlider.set(65)



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

quitApp = Button(frame,command=root.destroy,text="Quit")
quitApp.place(x=menuFrameWidth/2,y=menuFrameHeight/2+60,anchor="center")

startCamera = Button(frame,command=initCams,text="Start cameras")
startCamera.place(x=menuFrameWidth/2,y=menuFrameHeight/2,anchor="center")


startCalibration = Button(frame,command=initCalib,text="Start calibration")
startCalibration.place(x=menuFrameWidth/2,y=menuFrameHeight/2+30,anchor="center")




#try:    
#    show_frame()
#except:
#    pass
root.mainloop()
cap=None
cap2=None
writeFile.close()