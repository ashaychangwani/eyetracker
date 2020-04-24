#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 17:55:06 2020

@author: ashay
"""

from pandas import read_csv
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split

from concurrent.futures import ThreadPoolExecutor
import sys,os,threading,time
sys.path.append("/usr/local/lib/python3.7/site-packages")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


result=""
lock=threading.Lock()


dataframe=read_csv("calib.csv")

X=dataframe.iloc[:,0:8].values
y=dataframe.iloc[:,8:10].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)

executor=ThreadPoolExecutor(max_workers=10)


def mlCode(activationF,epoch,batchSize,drop,dropPtage):
    global result,X_train,X_test,y_train,y_test
    
    model=Sequential()
    model.add(Dense(8,input_dim=8,kernel_initializer='normal', activation=activationF))
    if drop:
        model.add(Dropout(dropPtage))
    model.add(Dense(22,kernel_initializer='normal', activation=activationF))
    if drop:
        model.add(Dropout(dropPtage))
    model.add(Dense(2,kernel_initializer='normal'))
    #model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mse','accuracy'])
    model.compile(loss='mean_squared_error', optimizer='RMSProp',metrics=['mse'])
    with tf.device('/cpu:0'):
        model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=batchSize,epochs=epoch)
        
    res=model.evaluate(X_train,y_train)
    result=result+"\n Activation Function= "+activationF+"\t Epochs= "+str(epoch)+"\t Batch Size= "+str(batchSize)+"\t Drop? ="+str(drop)+"\t Dropout rate= "+str(dropPtage)+"\t Result= "+str(res[0])

    

epochs=[100,150,200,250,300,350]
actFn=['tanh','relu','sigmoid']
batch_size=[8,16,32,64,128,256]
drop=[True,False]
dropPtage=[0.1,0.15,0.2,0.25]

for e in epochs:
    for a in actFn:
        for b in batch_size:
            for d in drop:
                for dP in dropPtage:
                    mlCode(a,e,b,d,dP)
