#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 18:06:03 2020

@author: ashay
"""

'''
import cv2
import time
cap = cv2.VideoCapture(0)

frame2=None
cap.set(cv2.CAP_PROP_FPS,30)
cap.set(3,1280)
cap.set(4,720)

def test():
    global frame2
    cv2.imshow('frame',frame2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return 0

t1=time.time()
i=0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    i+=1
    frame2=frame.copy()
    # Display the resulting frame
    test()
    
    if i==3000:
        break
t2=time.time()

print(t2-t1)





# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

'''
'''



import queue
import time, threading
from random import *

from concurrent.futures import ThreadPoolExecutor

executor=ThreadPoolExecutor(max_workers=50)    


q=queue.Queue()

for i in range (200000):
    q.put(i)
    
def test():
    t=q.get()
    #time.sleep(randint(0,6)/2)
    print(t,threading.current_thread())
    if q.empty():
        print('shuttin down')
        executor.shutdown(wait=True)
        print('complete')
    
while not q.empty():
    executor.submit(test)
    
#executor.shutdown(wait=True)
    
'''
'''
import cv2, queue, threading
from concurrent.futures import ThreadPoolExecutor



#lock=threading.Lock()
q=queue.Queue()
cap=cv2.VideoCapture('/Users/ashay/Desktop/test.mp4')
print('starting exec')
i=0
if (cap.isOpened()== False): 

  print("Error opening video stream or file")

while(cap.isOpened()):
    
    ret, frame=cap.read()                            
    if ret==True:
        q.put(frame)    
    else:
        break
    
print('done storing in q')

def test():
    global i,q,lock
    frame=q.get()
    #lock.acquire()
    i+=1
    print(i,threading.current_thread())
    cv2.imwrite('/Users/ashay/Desktop/Temp/img'+str(i)+'.jpg',frame)
    #lock.release()
    return
    
with ThreadPoolExecutor(max_workers=5) as executor:
    while not q.empty():
        executor.submit(test)

print('waiting')
cap.release()
cv2.destroyAllWindows()
    
'''
'''

import threading
from concurrent.futures import ThreadPoolExecutor
import queue

q=queue.Queue()
q2=queue.Queue()
executor=ThreadPoolExecutor(max_workers=4)
for i in range (100):
    q.put(i)
    q2.put(100-i)
    
def test(x,y):
    global q
    print(x,y)
    
    if q.empty():
        executor.shutdown(wait=True)
    
    

while not q.empty() and not q2.empty():
    executor.submit(test,q.get(),q2.get())


print('waiting')
executor.shutdown(wait=False)
print('done')
'''
import sys,os,traceback,time
from concurrent.futures import ThreadPoolExecutor


def startLoggingCalibFrames_HTC():
    while True:
        time.sleep(0.1)
        print('test)')


exec1=ThreadPoolExecutor(max_workers=1)

exec1.submit(startLoggingCalibFrames_HTC)

for i in range(10):
    if i==9:
        exec1.shutdown()
    print(i)
    time.sleep(0.5)