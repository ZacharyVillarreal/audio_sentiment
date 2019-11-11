import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import librosa

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Conv1D, Conv2D, MaxPooling1D, Flatten, LSTM, Activation, Dropout, Dense

## import data
df = pd.read_csv('data/data.csv')
df = df[df['emotion'] != 'male_disgust']
df = df[df['emotion'] != 'female_disgust'].drop(columns='filename')
from sklearn.utils import shuffle
df = shuffle(df)

## create y
emotion = df['emotion']
encoder = LabelEncoder()
y = encoder.fit_transform(emotion)

## create x
scaler = StandardScaler()
X = scaler.fit(np.array(df.iloc[:, :-1], dtype = float))
X = scaler.transform(np.array(df.iloc[:, :-1], dtype = float))
X = X.reshape(X.shape[0], X.shape[1],1)

## train/test/split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

## categorical
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

## MODEL
model = keras.Sequential()

model.add(Conv1D(16, 5,padding='same',
                 input_shape=(25,1)))
model.add(Activation('relu'))
model.add(Conv1D(32, 5,padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Conv1D(64, 5,padding='same',))
model.add(Activation('relu'))
model.add(Conv1D(128, 5,padding='same'))
model.add(Activation('relu'))

model.add(MaxPooling1D(pool_size=(8)))

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(64))
model.add(Activation('relu'))
## unstacking rows of pixels in the image and lining them up
model.add(keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])))
## The second (and last) layer is a 10-node softmax layer that 
##    returns an array of 10 probability scores that sum to 1
model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(X_train, y_train, batch_size=16, epochs=24, validation_data=(X_test, y_test))


import pickle
pickle.dump(model, open('pickles/final_model.p','wb'))
pickle.dump(encoder, open('pickles/final_encoder.p','wb'))
pickle.dump(scaler, open('pickles/final_scaler.p','wb'))

