#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 00:37:55 2019

@author: ashay
"""
from keras.models import Sequential
from keras.layers import Dense
from pandas import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

dataframe=read_csv("calib.csv",header=None)

#dataframe[6]=dataframe[6]*1920
#dataframe[7]=dataframe[7]*1080

X=dataframe.iloc[:,0:8].values
y=dataframe.iloc[:,8:10].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#
#sc=MinMaxScaler()
#X_train[:,6:8]=sc.fit_transform(X_train[:,6:8])
#X_test[:,6:8]=sc.transform(X_test[:,6:8])

model=Sequential()
model.add(Dense(4,input_dim=8,kernel_initializer='normal', activation='relu'))
model.add(Dense(4,kernel_initializer='normal', activation='relu'))
model.add(Dense(2,kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])


model.fit(X_train,y_train,batch_size=16,nb_epoch=300)

y_pred=model.predict(X_test)

