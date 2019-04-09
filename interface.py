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
import tensorflow as tf
import sys
sys.path.append("/Users/ashay/Desktop/eyeTracker/models/research/object_detection")
sys.path.append("/Users/ashay/Desktop/eyeTracker/models/research")





camFrameWidth=1020
menuFrameWidth=1920-camFrameWidth

cam1FrameHeight=500
cam2FrameHeight=1080-cam1FrameHeight
menuFrameHeight=1080

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

#labelTest=Label(root,text="THIS IS A TEST")
#labelTest.place(x=900,y=540,anchor="center")

startCamera = Button(frame,command=initCams,text="Start cameras")
startCamera.place(x=menuFrameWidth/2,y=menuFrameHeight/2,anchor="center")


#try:    
#    show_frame()
#except:
#    pass
root.mainloop()
cap=None
cap2=None





'''

https://www.youtube.com/watch?v=kbdbZFT9NQI
from tkinter import *
import PIL
from PIL import Image,ImageTk
import cv2
import tensorflow as tf
import sys
cap=cv2.VideoCapture(1)
while True:
    ret,frame=cap.read()
    roi=frame
    img_inv=cv2.bitwise_not(frame)
    gray_roi= cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    hsv= cv2.cvtColor(img_inv, cv2.COLOR_BGR2HSV) 
    h,s,v = cv2.split(hsv)
    
    _, threshold = cv2.threshold(v, 130 , 255,cv2.THRESH_BINARY_INV)
    
    cv2.imshow("GROI",threshold)
    
    key=cv2.waitKey(30)
    if key==27:
        break
cv2.destroyAllWindows()





MY PERSONAL TRAINING THING


from utils import label_map_util
from utils import visualization_utils as vis_util
import numpy as np

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile("outputFolder/frozen_inference_graph.pb", 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap("images/data/object-detection.pbtxt")
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
cap=cv2.VideoCapture(1)
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      ret, image_np = cap.read()
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      print(scores)
      # Visualization of the results of a detection.
      if scores[0][0]>0.8:
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=8)

      cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break











#convert xml to csv


import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
 
 
def xml_to_csv(path):
    xml_list = []
    print(path + '/*.xml')
    for f in glob.glob(path):
        print(f)
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df
 
 
def main():
    for directory in ['train','test']:
        image_path = "/Users/ashay/Desktop/eyeTracker/irisImages/"+directory
        print(image_path)
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv('data/{}_labels.csv'.format(directory), index=None)
        print('Successfully converted xml to csv.')
 
 
main()


#Code to delete all files that don't have XML Equivalent
files=[]
for f in glob.glob("*.xml"):
    files.append(f)
s=""
for f in glob.glob("*.jpg"):
    s=f[:-3]+"xml"
    if s not in files:
        os.remove(f)
'''

