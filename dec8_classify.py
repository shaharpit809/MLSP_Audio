#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 14:19:06 2018

@author: flash
"""

import numpy as np

import scipy.io.wavfile as wav
import os
import speechpy
import librosa
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report


path_data = 'D:/MLSP Project/Data/Data'

#f,sr = librosa.load('/home/flash/Documents/IUBBooks/MLSP/Prooject/RAVDESS/Data/04/03-01-04-01-01-01-13.wav')

mslen = 22050

c_labels = os.listdir(path_data)
c_labels = sorted(c_labels)
print(type(c_labels))

data = []

max_fs = 0
labels = []

directories = os.listdir(path_data)

print(directories)

#data_set = np.zeros((1,19240))
#dataset = []
#labels = []
#count = 0
##for d in directories:
##    for file in os.listdir(os.path.join(path_data,d)):
##
##        #fs,signal = wav.read(os.path.join(path_data,d,file))
##        f,sr = librosa.load(os.path.join(path_data,d,file),duration=3)
##
##
##        #mfcc = speechpy.feature.mfcc(signal,fs,frame_length = 0.020,frame_stride = 0.01)
##        mfcc = librosa.feature.mfcc(f,sr)
##        inter = mfcc.flatten()  
##        inter = inter.reshape((1,inter.shape[0]))
##        if inter.shape[1] < 2600:
##            n = 2600 - inter.shape[1]
##            n_arr = np.zeros((1,n))
##            inter = np.hstack((inter , n_arr))
#s = []
#m = []
#c = []
#me = []
#co = []
#to =[]
#feature_all=np.empty((0,193))
#for d in directories:
#    for file in os.listdir(os.path.join(path_data,d)):
#        #X,sr = librosa.load('/home/flash/Documents/IUBBooks/MLSP/Prooject/RAVDESS/Data/01/03-01-01-01-01-01-01.wav',sr = None)
#        X,sr = librosa.load(os.path.join(path_data,d,file),duration=3)
#        stft = np.abs(librosa.stft(X))
#        #print("stft shape: ", np.shape(stft))
#        #s.append(np.shape(stft))
#        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T,axis=0)
#        #mfccs=mfccs.reshape(1,np.shape(mfccs)[0])
#        #print("mfccs shape: ", np.shape(mfccs))
#        #m.append(np.shape(mfccs))
#        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
#        #chroma = chroma.reshape(1,np.shape(chroma)[0])
#        #print("chroma shape: " , np.shape(chroma))
#        #c.append(np.shape(chroma))
#        mel = np.mean(librosa.feature.melspectrogram(X, sr=sr).T,axis=0)
#       # mel = mel.reshape(1,np.shape(mel)[0])
#        #print("mel shape:" , np.shape(mel))
#        #me.append(np.shape(mel))
#        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T,axis=0)
#        #contrast = contrast.reshape(1,np.shape(contrast)[0])
#        #print("contrast shape:",np.shape(contrast))
#        #co.append(np.shape(contrast))
#        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sr).T,axis=0)
#        #tonnetz = tonnetz.reshape(1,np.shape(tonnetz)[0])
#        
#        #print("tonnetz shape:",np.shape(tonnetz))
#        #to.append(np.shape(tonnetz))
#        features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
#        #features = features.reshape(np.shape(features)[0],1)
#        feature_all = np.vstack([feature_all,features])
#        labels.append(d)
#        #print(feature.shape)
#
#
import pickle
#feature_file = open('/home/flash/Documents/IUBBooks/MLSP/Prooject/RAVDESS/dec8/feature.pkl','wb')
#pickle.dump(feature_all,feature_file)
#label_file = open('/home/flash/Documents/IUBBooks/MLSP/Prooject/RAVDESS/dec8/label.pkl','wb')
#pickle.dump(labels,label_file)
f2 = open('feature.pkl','rb')
feature_all = pickle.load(f2)
f3 = open('label.pkl','rb')
labels = pickle.load(f3)
from copy import deepcopy
y = deepcopy(labels)
for i in range(len(y)):
    y[i] = int(y[i])

#from sklearn.preprocessing import OneHotEncoder
#enc = OneHotEncoder(handle_unknown='ignore')
#labels =np.asarray(labels)
#labels = labels.reshape(-1,1)
#enc.fit(labels)

n_labels = len(y)
n_unique_labels = len(np.unique(y))
one_hot_encode = np.zeros((n_labels,n_unique_labels))
f = np.arange(n_labels)
for i in range(len(f)):
    one_hot_encode[f[i],y[i]-1]=1


X_train,X_test,y_train,y_test = train_test_split(feature_all,one_hot_encode,test_size = 0.3,random_state=20)

from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout

model = Sequential()

model.add(Dense(X_train.shape[1],input_dim =X_train.shape[1],init='normal',activation ='relu'))

model.add(Dense(400,init='normal',activation ='relu'))

model.add(Dropout(0.2))

model.add(Dense(200,init='normal',activation ='relu'))

model.add(Dropout(0.2))

model.add(Dense(100,init='normal',activation ='relu'))

model.add(Dropout(0.2))

model.add(Dense(y_train.shape[1],init='normal',activation ='softmax'))

model.compile(loss = 'categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

model.fit(X_train,y_train,nb_epoch=200,batch_size = 5,verbose=1)


model.evaluate(X_test,y_test)

mlp_model = model.to_json()
with open('mlp_model_relu_adadelta.json','w') as j:
    j.write(mlp_model)
model.save_weights("mlp_relu_adadelta_model.h5")


# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")

model2 = Sequential()

model2.add(Dense(X_train.shape[1],input_dim =X_train.shape[1],init='normal',activation ='relu'))

model2.add(Dense(400,init='normal',activation ='tanh'))

model2.add(Dropout(0.2))

model2.add(Dense(200,init='normal',activation ='tanh'))

model2.add(Dropout(0.2))

model2.add(Dense(100,init='normal',activation ='sigmoid'))

model2.add(Dropout(0.2))

model2.add(Dense(y_train.shape[1],init='normal',activation ='softmax'))

model2.compile(loss = 'categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

model2.fit(X_train,y_train,nb_epoch=200,batch_size = 5,verbose=1)

mlp_model2 = model2.to_json()
with open('mlp_model_tanh_adadelta.json','w') as j:
    j.write(mlp_model2)
model2.save_weights("mlp_tanh_adadelta_model.h5")
