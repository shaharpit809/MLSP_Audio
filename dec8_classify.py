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
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout, Flatten, Embedding


path_data = '/home/flash/Documents/IUBBooks/MLSP/Prooject/RAVDESS/Data'

#f,sr = librosa.load('/home/flash/Documents/IUBBooks/MLSP/Prooject/RAVDESS/Data/04/03-01-04-01-01-01-13.wav')

mslen = 22050

c_labels = os.listdir(path_data)
c_labels = sorted(c_labels)
print(type(c_labels))

data = []

max_fs = 0
labels = []

emotions = ['neutral','calm','happy','sad','angry','fearful','disgust','surprised']
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
'''
feature_all=np.empty((0,193))
for d in directories:
    for file in os.listdir(os.path.join(path_data,d)):
        #X,sr = librosa.load('/home/flash/Documents/IUBBooks/MLSP/Prooject/RAVDESS/Data/01/03-01-01-01-01-01-01.wav',sr = None)
        X,sr = librosa.load(os.path.join(path_data,d,file),duration=3)
        stft = np.abs(librosa.stft(X))
        #print("stft shape: ", np.shape(stft))
        #s.append(np.shape(stft))
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T,axis=0)
        #mfccs=mfccs.reshape(1,np.shape(mfccs)[0])
        #print("mfccs shape: ", np.shape(mfccs))
        #m.append(np.shape(mfccs))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
        #chroma = chroma.reshape(1,np.shape(chroma)[0])
        #print("chroma shape: " , np.shape(chroma))
        #c.append(np.shape(chroma))
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sr).T,axis=0)
       # mel = mel.reshape(1,np.shape(mel)[0])
        #print("mel shape:" , np.shape(mel))
        #me.append(np.shape(mel))
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T,axis=0)
        #contrast = contrast.reshape(1,np.shape(contrast)[0])
        #print("contrast shape:",np.shape(contrast))
        #co.append(np.shape(contrast))
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sr).T,axis=0)
        #tonnetz = tonnetz.reshape(1,np.shape(tonnetz)[0])
        
        #print("tonnetz shape:",np.shape(tonnetz))
        #to.append(np.shape(tonnetz))
        features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
        #features = features.reshape(np.shape(features)[0],1)
        feature_all = np.vstack([feature_all,features])
        labels.append(d)
        #print(feature.shape)
'''
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

y_pred_model1 = model.predict(X_test)
y2 = np.argmax(y_pred_model1,axis=1)
y_test2 = np.argmax(y_test , axis = 1)

count = 0
for i in range(y2.shape[0]):
    if y2[i] == y_test2[i]:
        count+=1
        
        

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

model2.evaluate(X_test, y_test)


# mlp_model2 = model2.to_json()
# with open('mlp_model_tanh_adadelta.json','w') as j:
#     j.write(mlp_model2)
# model2.save_weights("mlp_tanh_adadelta_model.h5")


y_pred_model2 = model2.predict(X_test)
y22 = np.argmax(y_pred_model2,axis=1)
y_test22 = np.argmax(y_test , axis = 1)

count = 0
for i in range(y22.shape[0]):
    if y22[i] == y_test22[i]:
        count+=1
        
from xgboost import XGBClassifier
X_train2,X_test2,y_train2,y_test2 = train_test_split(feature_all,y,test_size = 0.3,random_state=20)        

model3 = XGBClassifier()
model3.fit(X_train2,y_train2)
model3.evals_result()
cross_val_score(model3, X_train2, y_train2, cv=5)
y_pred3 = model3.predict(X_test)

count = 0
for i in range(y_pred3.shape[0]):
    if y_pred3[i] == y_test2[i]:
        count+=1   
        
# clf = RandomForestClassifier(n_estimators=60,max_features=8,max_depth=None,min_samples_split=3,bootstrap=True,random_state=35)
# clf = clf.fit(X_train, y_train)
# #scores = cross_val_score(clf, X_train, y_train, cv=5)
# #print(scores.mean())
# y_pred = clf.predict(X_test)
# for i in range(np.shape(y_test))
        
X,sr = librosa.load('/home/flash/Documents/IUBBooks/MLSP/Prooject/MLSP_Audio/chunk-00.wav',sr = None)
stft = np.abs(librosa.stft(X))
#print("stft shape: ", np.shape(stft))

mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40),axis=1)


chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
mel = np.mean(librosa.feature.melspectrogram(X, sr=sr).T,axis=0)

contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr,fmin=0.5*sr* 2**(-6)).T,axis=0)

tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sr*2).T,axis=0)
features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
#features = features.reshape(np.shape(features)[0],1)
feature_all = np.vstack([feature_all,features])

print(features.shape)

x_chunk = np.array(features)
x_chunk = x_chunk.reshape(1,np.shape(x_chunk)[0])
y_chunk_model1 = model.predict(x_chunk)
index = np.argmax(y_chunk_model1)
print('Emotion:',emotions[index])


feature_all2=np.empty((0,193))
X2,sr = librosa.load('/home/flash/Documents/IUBBooks/MLSP/Prooject/MLSP_Audio/chunk-02.wav',sr = None)
stft2 = np.abs(librosa.stft(X2))
#print("stft shape: ", np.shape(stft))

mfccs2 = np.mean(librosa.feature.mfcc(y=X2, sr=sr, n_mfcc=40),axis=1)


chroma2 = np.mean(librosa.feature.chroma_stft(S=stft2, sr=sr).T,axis=0)
mel2 = np.mean(librosa.feature.melspectrogram(X2, sr=sr).T,axis=0)

contrast2 = np.mean(librosa.feature.spectral_contrast(S=stft2, sr=sr,fmin=0.5*sr* 2**(-6)).T,axis=0)

tonnetz2 = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X2),sr=sr*2).T,axis=0)
features2 = np.hstack([mfccs2,chroma2,mel2,contrast2,tonnetz2])
#features = features.reshape(np.shape(features)[0],1)
feature_all2 = np.vstack([feature_all2,features2])

print(features2.shape)

x_chunk2 = np.array(features2)
x_chunk2 = x_chunk2.reshape(1,np.shape(x_chunk2)[0])
y_chunk2_model1 = model.predict(x_chunk2)
index2 = np.argmax(y_chunk2_model1)
print('Emotion:',emotions[index2])

